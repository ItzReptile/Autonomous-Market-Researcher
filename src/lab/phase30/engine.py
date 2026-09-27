"""
Phase 30: Narrow Deep Refinement Engine & Statistical Referee around Trial 106.
Dedicated refinement module for UNIV_10 (DOT, ATOM, LINK, UNIUSDT).

Supports:
1. Endogenous per-asset continuous factors with AST lookback and composition validation.
2. Per-asset position sizing (per_asset_position_size) to test tail-risk exposure reductions on ATOM/DOT.
3. Custom asset portfolio weighting (asset_weights) to test overweight/underweight asset distributions.
4. Full unchanged statistical referee gate stack: DSR >= 0.95, bootstrap p <= 0.05,
   Option A multi-regime consistency, per-asset drawdown <= -30% ceiling.
5. Air-gapped Q3 2024 holdout evaluation strictly gated to candidates clearing ALL dev gates.
"""
import ast
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab.audit.drawdown import (
    CURRENT_CONVENTION,
    geometric_cagr,
    max_drawdown,
    max_drawdown_nav,
    required_gross_for_drawdown_budget,
    scale_invariant_calmar,
)
from src.lab.audit.dsr import DeflatedSharpeCalculator, SharpeUnit, TrialVarianceMethod
from src.lab.audit.bootstrap import StationaryBootstrap
from src.lab.audit.cost_stress import CostStressTester
from src.lab.audit.metrics import MetricsCalculator
from src.lab.phase21.backtest_engine import BacktestResult
from src.lab.phase21.referee import Phase21Evaluation
from src.lab.phase26.data_engine import build_multi_asset_table
from src.lab.stage4.execution_model import (
    BASE_FEE_RATE,
    BASE_SLIPPAGE_RATE,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    causal_friction_ratios,
)
from src.lab.phase28.sandbox import (
    ExecutionResult,
    SignalSandbox,
    apply_adaptive_hysteresis_filter,
    apply_hysteresis_filter,
    check_code_composition,
    check_code_lookbacks,
    check_code_static,
)

CATALOG_PATH = PROJECT_ROOT / "data" / "universes" / "phase26_catalog.json"
PARQUET_DIR = PROJECT_ROOT / "data" / "universes" / "phase26"


@dataclass
class Phase30Proposal:
    universe_id: str
    code_str: str
    reasoning: str
    variant_name: str = ""
    priority_category: str = "TAIL_RISK_MITIGATION"  # or 'PARAMETER_PERTURBATION'
    predicted_confidence: float = 65.0
    confidence_justification: str = ""
    theta_enter: float = 0.50
    theta_exit: float = 0.20
    position_size: float = 1.0
    per_asset_position_size: Optional[Dict[str, float]] = None
    asset_weights: Optional[Dict[str, float]] = None
    # Percentile-based hysteresis on the factor's own rolling distribution.
    # Defaults OFF so every historical result stays reproducible bit for bit;
    # new hunts opt in explicitly.
    adaptive_hysteresis: bool = False
    enter_pct: float = 97.0
    exit_pct: float = 2.0
    # Positive-control diagnostics only. A proposal with this set is by
    # construction not a tradeable strategy and must run under a quarantined
    # referee. Default False; nothing in the normal hunt path sets it.
    allow_lookahead_for_control: bool = False

    def get_code_hash(self) -> str:
        """Returns a stable normalized hash of the generated Python code and configuration."""
        try:
            tree = ast.parse(self.code_str)
            dump_str = ast.dump(tree)
            cfg_str = f"{dump_str}_{self.theta_enter}_{self.theta_exit}_{self.position_size}_{self.per_asset_position_size}_{self.asset_weights}_{self.adaptive_hysteresis}_{self.enter_pct}_{self.exit_pct}"
            return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]
        except Exception:
            clean = "".join(self.code_str.split())
            cfg_str = f"{clean}_{self.theta_enter}_{self.theta_exit}_{self.position_size}_{self.per_asset_position_size}_{self.asset_weights}_{self.adaptive_hysteresis}_{self.enter_pct}_{self.exit_pct}"
            return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Phase30BacktestEngine:
    # Fees come from the ONE canonical schedule, not a local copy. This engine
    # sat at 5 bps while execution_model was at 10 and the real entry-tier
    # taker fee is 90 -- research was pricing fills at a nineteenth of reality
    # while deciding which candidates to promote.
    BASE_FEE_RATE = BASE_FEE_RATE
    BASE_SLIPPAGE_RATE = BASE_SLIPPAGE_RATE
    HOURS_PER_YEAR = 8760

    # NOTE ON FRICTION: this engine deliberately keeps no causal-normalisation
    # helper of its own. It calls causal_friction_ratios from
    # src.lab.stage4.execution_model -- the same function the live path uses --
    # so research and live cannot price friction differently. A second
    # implementation living here is exactly how they would drift apart, and
    # briefly did: the first version of this fix used an expanding mean while
    # the canonical one uses a trailing 720-bar window.
    #
    # RESOLVED 2026-09-22: the 5 vs 10 bps split between this engine and
    # execution_model is gone, and both were wrong anyway. The real Coinbase
    # Advanced entry tier is 50 bps maker / 90 bps taker, and this engine now
    # imports the single canonical schedule. Every historical result recorded
    # before that date was produced under 5 bps and is NOT comparable to
    # anything produced after it.


    def __init__(
        self,
        parquet_dir: Path = PARQUET_DIR,
        timeout_seconds: float = 5.0,
        drawdown_convention: str = CURRENT_CONVENTION,
    ):
        self.parquet_dir = parquet_dir
        self.sandbox = SignalSandbox(timeout_seconds=timeout_seconds)
        self._universe_cache: Dict[str, pl.DataFrame] = {}
        # "nav" from Phase 39 on. "cumsum" reproduces pre-Phase-39 numbers and
        # is the ONLY thing it should ever be used for; it is de-leverageable.
        if drawdown_convention not in ("nav", "cumsum"):
            raise ValueError(f"Unknown drawdown convention {drawdown_convention!r}")
        self.drawdown_convention = drawdown_convention
        if drawdown_convention == "cumsum":
            print("[Phase30BacktestEngine] WARNING: LEGACY cumsum drawdown selected. "
                  "This quantity scales with gross exposure, so the drawdown gate can "
                  "be passed by de-leveraging alone. Valid only for replaying "
                  "pre-Phase-39 results.")

    def get_universe_data(self, universe_id: str) -> pl.DataFrame:
        if universe_id not in self._universe_cache:
            p_path = self.parquet_dir / f"{universe_id}.parquet"
            if not p_path.exists():
                for alt_dir in [
                    PROJECT_ROOT / "data" / "universes" / "phase26",
                    PROJECT_ROOT / "data" / "universes" / "phase21",
                ]:
                    alt_path = alt_dir / f"{universe_id}.parquet"
                    if alt_path.exists():
                        p_path = alt_path
                        break
            if not p_path.exists():
                raise FileNotFoundError(f"Universe parquet not found: {p_path}")
            df_loaded = pl.read_parquet(p_path)
            # A parquet can silently span the embargo: the pre-existing
            # phase26 UNIV_05 table runs 2024-01-01..2024-06-30 and so crosses
            # it. Check the ACTUAL bars, not the filename or the catalog.
            if "open_time" in df_loaded.columns:
                from src.lab.audit.embargo import assert_timestamps_allowed

                assert_timestamps_allowed(
                    df_loaded["open_time"].to_list(),
                    context=f"get_universe_data({universe_id}) <- {p_path.name}",
                )
            self._universe_cache[universe_id] = df_loaded
        return self._universe_cache[universe_id]

    def run_generative_backtest(
        self,
        proposal: Phase30Proposal,
        custom_df: Optional[pl.DataFrame] = None,
        strategy_id: Optional[str] = None,
    ) -> Tuple[BacktestResult, Dict[str, Any]]:
        df_polars = custom_df if custom_df is not None else self.get_universe_data(proposal.universe_id)
        symbols = sorted([c.split("_")[0] for c in df_polars.columns if c.endswith("_close")])
        n_assets = len(symbols)
        n_bars = len(df_polars)

        if n_assets == 0:
            raise ValueError(f"No asset columns found in universe {proposal.universe_id}")

        close_mat = np.column_stack([df_polars[f"{s}_close"].to_numpy() for s in symbols])
        open_mat = np.column_stack([df_polars[f"{s}_open"].to_numpy() for s in symbols])
        high_mat = np.column_stack([df_polars[f"{s}_high"].to_numpy() for s in symbols])
        low_mat = np.column_stack([df_polars[f"{s}_low"].to_numpy() for s in symbols])
        vol_mat = np.column_stack([df_polars[f"{s}_volume"].to_numpy() for s in symbols])

        # ORDER FLOW (Phase 43). The source parquets have carried
        # <sym>_taker_buy_volume and <sym>_quote_volume since Phase 21, but the
        # factor contract never exposed them, so no factor in this project has
        # ever been able to see order flow. taker_buy / volume is the share of
        # a bar's volume that was AGGRESSIVE BUYING -- a genuinely signed,
        # directional quantity, unlike every volatility term the lineage has
        # leaned on.
        #
        # Degrades gracefully: tables without these columns (e.g. the stage4
        # Q4-2024 live table) get a neutral 0.5 split and a synthetic quote
        # volume, and the substitution is reported in exec_metadata rather than
        # silently passed off as real order flow.
        _have_flow = all(f"{s}_taker_buy_volume" in df_polars.columns for s in symbols)
        _have_quote = all(f"{s}_quote_volume" in df_polars.columns for s in symbols)
        if _have_flow:
            takerbuy_mat = np.column_stack(
                [df_polars[f"{s}_taker_buy_volume"].to_numpy() for s in symbols])
        else:
            takerbuy_mat = 0.5 * vol_mat
        if _have_quote:
            quotevol_mat = np.column_stack(
                [df_polars[f"{s}_quote_volume"].to_numpy() for s in symbols])
        else:
            quotevol_mat = vol_mat * close_mat

        ret_mat = np.diff(close_mat, axis=0) / np.maximum(1e-8, close_mat[:-1])
        ret_mat = np.vstack([np.zeros((1, n_assets)), ret_mat])

        # State-dependent friction, normalised CAUSALLY.
        #
        # This engine used to divide by the mean over the WHOLE evaluated
        # window, so the friction charged at bar t depended on volatility that
        # had not happened yet. It touched only costs and never the signal, so
        # no gate verdict rested on it, but it was real.
        #
        # The fix is to call the same function Stage 4 already used. That
        # matters more than the look-ahead itself: a causal implementation had
        # existed in execution_model.py all along and the LIVE path used it,
        # while this research engine - the thing that decides which candidates
        # get promoted - did not. Research and live were pricing friction
        # differently. One implementation now serves both.
        vol_ratio, spread_ratio = causal_friction_ratios(
            close_mat, high_mat, low_mat, ret_mat
        )

        timestamps = (
            df_polars["open_time"].to_numpy()
            if "open_time" in df_polars.columns
            else np.arange(n_bars, dtype=np.int64)
        )

        raw_factors_mat = np.zeros((n_bars, n_assets), dtype=np.float64)
        positions_mat = np.zeros((n_bars, n_assets), dtype=np.float64)
        exec_metadata: Dict[str, Any] = {"per_asset_status": {}, "errors": []}

        per_asset_sizes = proposal.per_asset_position_size or {}

        # Cross-asset context. The factor contract was strictly single-asset, so
        # a co-movement or divergence factor could not be written at all. These
        # columns are CONTEMPORANEOUS (bar t of the other assets at bar t), so
        # they add no lookahead: an equal-weight index of the peers is knowable
        # at the same instant as the asset's own close.
        norm_levels = close_mat / np.maximum(1e-12, close_mat[0])
        mkt_level = norm_levels.mean(axis=1)

        for i, sym in enumerate(symbols):
            if n_assets > 1:
                peer_cols = [j for j in range(n_assets) if j != i]
                peer_level = norm_levels[:, peer_cols].mean(axis=1)
                peer_disp = norm_levels[:, peer_cols].std(axis=1)
            else:
                # Single-asset evaluation: no peers exist. Degenerate on
                # purpose, and the per-asset drawdown gate no longer runs in
                # this mode precisely because a cross-asset factor would go
                # flat here and pass the gate for the wrong reason.
                peer_level = norm_levels[:, i]
                peer_disp = np.zeros(n_bars)

            df_asset = pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": open_mat[:, i],
                    "high": high_mat[:, i],
                    "low": low_mat[:, i],
                    "close": close_mat[:, i],
                    "volume": vol_mat[:, i],
                    "taker_buy_volume": takerbuy_mat[:, i],
                    "quote_volume": quotevol_mat[:, i],
                    "peer_close": peer_level,
                    "mkt_close": mkt_level,
                    "peer_dispersion": peer_disp,
                }
            )

            res: ExecutionResult = self.sandbox.execute_code(
                proposal.code_str, df_asset,
                allow_lookahead_for_control=getattr(
                    proposal, "allow_lookahead_for_control", False),
            )
            exec_metadata["per_asset_status"][sym] = res.status

            if res.status != "SUCCESS" or res.weights is None:
                exec_metadata["errors"].append(f"{sym}: {res.error_message}")
                raise RuntimeError(f"Sandbox execution failed for asset {sym}: {res.error_message}")

            raw_factors_mat[:, i] = res.weights

            p_size = per_asset_sizes.get(sym, proposal.position_size)
            if proposal.adaptive_hysteresis:
                # Judge each factor on its own scale. The fixed 0.50/0.20 pair
                # was calibrated for the Trial 106 family; in Phase 32 it left
                # 15 of 26 evaluated trials structurally unable to trade.
                positions_mat[:, i] = apply_adaptive_hysteresis_filter(
                    raw_factors_mat[:, i],
                    enter_pct=proposal.enter_pct,
                    exit_pct=proposal.exit_pct,
                    position_size=p_size,
                )
            else:
                positions_mat[:, i] = apply_hysteresis_filter(
                    raw_factors_mat[:, i],
                    theta_enter=proposal.theta_enter,
                    theta_exit=proposal.theta_exit,
                    position_size=p_size,
                )

        # Allocate portfolio weights
        if proposal.asset_weights and n_assets > 1:
            raw_w = np.array([proposal.asset_weights.get(s, 1.0 / n_assets) for s in symbols], dtype=np.float64)
            norm_w = raw_w / np.sum(raw_w) if np.sum(raw_w) > 0 else np.ones(n_assets) / float(n_assets)
            weights_mat = positions_mat * norm_w[None, :]
        else:
            weights_mat = positions_mat / float(n_assets)

        sum_abs = np.sum(np.abs(weights_mat), axis=1)
        over_leverage = sum_abs > 1.0
        if np.any(over_leverage):
            weights_mat[over_leverage] = weights_mat[over_leverage] / sum_abs[over_leverage, None]

        warmup = 24
        gross_returns = []
        net_returns = []
        flat_net_returns = []
        turnovers = []
        costs_bps = []

        cur_w = np.zeros(n_assets)
        total_trades = 0

        for t in range(warmup, n_bars - 1):
            target_w = weights_mat[t]
            traded_dw = np.abs(target_w - cur_w)
            sum_dw = float(np.sum(traded_dw))
            turnovers.append(sum_dw)
            if sum_dw > 1e-4:
                total_trades += 1

            fee_t = self.BASE_FEE_RATE
            slippage_t = self.BASE_SLIPPAGE_RATE * (0.5 * vol_ratio[t] + 0.5 * spread_ratio[t])
            friction_rate = fee_t + slippage_t

            cost_t = float(np.sum(traded_dw * friction_rate))
            flat_cost_t = float(np.sum(traded_dw * (self.BASE_FEE_RATE + self.BASE_SLIPPAGE_RATE)))

            bar_gross_ret = float(np.sum(target_w * ret_mat[t + 1]))
            bar_net_ret = bar_gross_ret - cost_t
            bar_flat_net_ret = bar_gross_ret - flat_cost_t

            gross_returns.append(bar_gross_ret)
            net_returns.append(bar_net_ret)
            flat_net_returns.append(bar_flat_net_ret)

            if sum_dw > 1e-6:
                costs_bps.append((cost_t / sum_dw) * 10000.0)
            else:
                costs_bps.append(10.0)

            cur_w = target_w

        net_arr = np.array(net_returns)
        gross_arr = np.array(gross_returns)
        flat_arr = np.array(flat_net_returns)
        to_arr = np.array(turnovers)

        n_valid = len(net_arr)
        mu_net = float(np.mean(net_arr)) if n_valid > 0 else 0.0
        sd_net = float(np.std(net_arr)) + 1e-8 if n_valid > 0 else 1.0
        sr_net = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_net / sd_net))

        mu_gross = float(np.mean(gross_arr)) if n_valid > 0 else 0.0
        sd_gross = float(np.std(gross_arr)) + 1e-8 if n_valid > 0 else 1.0
        sr_gross = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_gross / sd_gross))

        # Drawdown convention. From Phase 39 this is TRUE NAV drawdown: the
        # worst peak-to-trough decline of the compounded equity curve, as a
        # fraction of the peak. The previous cumsum form measured the decline
        # of a running SUM of returns, which is not a quantity any account
        # holds, is unbounded below, and is exactly proportional to gross
        # exposure. See src/lab/audit/drawdown.py for the measurement that
        # forced the change. Legacy form retained for replaying pre-Phase-39
        # results only.
        max_dd = float(max_drawdown(net_arr, convention=self.drawdown_convention))

        ann_to = float(np.sum(to_arr) * (self.HOURS_PER_YEAR / max(1, n_valid)))
        cagr = float(mu_net * self.HOURS_PER_YEAR)

        mean_cost = float(np.mean(costs_bps)) if costs_bps else 10.0
        max_cost = float(np.max(costs_bps)) if costs_bps else 10.0
        state_penalty = float(np.mean(flat_arr) - mu_net) * self.HOURS_PER_YEAR

        s_id = strategy_id or f"p30_{proposal.universe_id}_{int(time.time()*1000)%100000}"

        result = BacktestResult(
            strategy_id=s_id,
            universe_id=proposal.universe_id,
            n_bars=n_bars,
            gross_returns=gross_returns,
            net_returns=net_returns,
            turnovers=turnovers,
            costs_bps=costs_bps,
            annualized_sharpe_gross=sr_gross,
            annualized_sharpe_net=sr_net,
            cagr_net=cagr,
            max_drawdown=max_dd,
            annual_turnover=ann_to,
            mean_cost_bps=mean_cost,
            max_cost_bps=max_cost,
            state_friction_penalty=state_penalty,
            n_trades=total_trades,
            ignored_parameters=[],
            relevance_warning="",
        )

        exec_metadata["symbols"] = symbols
        exec_metadata["order_flow_real"] = bool(_have_flow)
        exec_metadata["quote_volume_real"] = bool(_have_quote)
        exec_metadata["n_assets"] = n_assets
        exec_metadata["weights_shape"] = weights_mat.shape

        return result, exec_metadata


class Phase30Referee:
    DSR_THRESHOLD = 0.95
    # Same strategy re-expressed, whatever the source text says. Distinct from
    # the clustering threshold (0.95), which asks "same search family"; this
    # asks "literally the same run".
    DUPLICATE_RHO_THRESHOLD = 0.999
    BOOTSTRAP_P_THRESHOLD = 0.05
    MIN_BREAKEVEN_BPS = 10.0
    # RETIRED as a gate at Phase 40, retained for reporting the gross multiplier
    # that brings a candidate to this risk budget. See MIN_CALMAR.
    MAX_DRAWDOWN_LIMIT = -0.30

    # Calmar floor: geometric CAGR / |NAV max drawdown|, over the full window.
    #
    # CALIBRATION (measured, not chosen round). The Phase 36 control dials
    # predictive accuracy to a KNOWN value, the only construction in this
    # project where skill is known rather than inferred:
    #
    #     accuracy   Sharpe   NAV DD    Calmar
    #       100%      5.221   -27.4%    122.69
    #        90%      3.363   -68.4%     11.63
    #        80%      3.015   -68.4%      8.69
    #        72%      2.663   -73.9%      5.91   <- Phase 36 "genuinely excellent"
    #        65%      1.625   -73.9%      2.05   <- weakest unambiguous skill
    #        60%      0.733   -86.7%      0.36
    #        50%      0.103   -86.7%     -0.20   <- coin flip, no skill
    #     buy & hold  1.108   -86.1%      0.90
    #
    # 2.0 is the level the weakest control with unambiguous genuine skill
    # reaches. It is 2.2x buy-and-hold, so "just be long" cannot pass, and far
    # above both the no-skill null and the marginal-skill band.
    #
    # It is deliberately NOT set at the DSR-implied level (~5.1 by
    # interpolation at the Sharpe 2.45 that K=141 demands). Two reasons: that
    # would duplicate DSR rather than constrain something new, and the control's
    # SHAPE (400x/yr turnover, -74% drawdown) is not representative of an
    # implementable strategy, so its ratio should anchor a floor and not a bar.
    MIN_CALMAR = 2.0

    # Per-asset sleeve drawdown is gated against the asset's OWN buy-and-hold
    # drawdown over the same window, not against a fixed number. R = 1.00 means
    # "must not do worse than not trading it at all". Adopted Phase 49; the
    # retired -30% constant and the evidence against it are recorded in
    # data/audit/POLICY_REVIEW_per_asset_drawdown_ceiling.json.
    PER_ASSET_DD_BENCHMARK_RATIO = 1.00

    def __init__(
        self,
        catalog_path: Path = CATALOG_PATH,
        seed_universe: Optional[str] = None,
        seed_memory: bool = True,
        k_estimator: str = "cluster",
        cluster_threshold: float = None,
        quarantine: bool = False,
    ):
        """
        seed_universe / seed_memory
            Load this universe's PRIOR trials, across phases, before scoring
            anything new.

            This phase originally ran with an empty memory, so its first trial
            was scored at K_eff = 1.0 -- the weakest possible Deflated Sharpe
            test -- despite 19 earlier trials on the same universe in Phase 29.
            Whichever variant was evaluated first therefore got the easiest
            gate, making the DSR outcome depend on ordering rather than on the
            strategy. Pass seed_universe to avoid repeating that.

            seed_memory=False reproduces the original (unseeded) behaviour and
            exists only for replaying historical results.
        """
        from src.lab.audit.referee_memory import CLUSTER_RHO_THRESHOLD

        if k_estimator not in ("cluster", "rho_avg"):
            raise ValueError(f"unknown k_estimator: {k_estimator}")
        self.k_estimator = k_estimator
        self.cluster_threshold = (
            CLUSTER_RHO_THRESHOLD if cluster_threshold is None else float(cluster_threshold)
        )
        if k_estimator == "rho_avg":
            print(
                "[Phase30Referee] WARNING: rho_avg K_eff selected. This estimator is "
                "gameable -- adding correlated variants lowers the penalty. Valid only "
                "for reproducing historical results."
            )

        self.catalog_path = catalog_path
        self._catalog: List[Dict[str, Any]] = []
        if self.catalog_path.exists():
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                self._catalog = json.load(f)

        self.engine = Phase30BacktestEngine()
        self.prior_trial_returns: List[np.ndarray] = []
        self.prior_code_hashes: Dict[str, int] = {}
        # slot in prior_trial_returns -> originating trial index, so a
        # duplicate can name the trial it duplicates rather than a slot number
        self.prior_trial_index_by_slot: Dict[int, int] = {}
        # Quarantine: this referee may SCORE but must never REMEMBER. Used for
        # positive-control diagnostics so a synthetic clairvoyant strategy can
        # be pushed through the real gates without contaminating any real
        # universe's multiple-testing counter or duplicate history.
        # Project-wide search footprint, resolved once so a runner cannot report
        # a result without it. K_eff prices multiple testing WITHIN a universe;
        # this prices the choice of universe, which DSR cannot see.
        from src.lab.audit.referee_memory import project_search_footprint

        self.project_footprint = project_search_footprint()
        print(f"[Phase30Referee] Project search footprint: "
              f"{self.project_footprint['universes_ever_searched']} distinct baskets "
              f"ever searched, {self.project_footprint['total_trials_logged']} trials "
              f"logged across {self.project_footprint['ledgers']} ledgers. A low K_eff "
              f"on a fresh basket is not a first look.")

        self.quarantine = bool(quarantine)
        if self.quarantine:
            print("[Phase30Referee] QUARANTINE MODE: results will NOT enter memory, "
                  "K_eff accounting, or duplicate history.")
        self._df_cache: Dict[Any, pl.DataFrame] = {}
        self.seeded_from: List[Dict[str, Any]] = []
        # Seeding is the DEFAULT. If no universe is named here, the first
        # evaluate() lazily seeds from the proposal's own universe, so a runner
        # cannot start with an empty multiple-testing counter by omission --
        # which is exactly how Phase 30 scored its first trial at K_eff = 1.
        self._seed_memory = bool(seed_memory)
        self._seeded = False
        if not self._seed_memory:
            print(
                "[Phase30Referee] WARNING: memory seeding DISABLED. K_eff will "
                "restart at 1 and the Deflated Sharpe gate will be far too "
                "lenient. Only valid for replaying historical results."
            )
            self._seeded = True  # nothing to do later

        if seed_universe and seed_memory:
            from src.lab.audit.referee_memory import collect_stored_returns

            series, provenance = collect_stored_returns(seed_universe, catalog_path=self.catalog_path)
            # A constant series carries no information and its correlation with
            # anything is undefined; including it drags rho down and inflates
            # K_eff, making the gate harsher for the wrong reason.
            kept = [
                (s, p) for s, p in zip(series, provenance)
                if float(np.std(s)) > 0.0 and bool(np.isfinite(s).all())
            ]
            self.prior_trial_returns = [s for s, _ in kept]
            self.seeded_from = [p for _, p in kept]
            print(
                f"[Phase30Referee] Seeded memory for {seed_universe}: "
                f"{len(kept)} prior trial(s) from "
                f"{len({p['ledger'] for p in self.seeded_from})} ledger(s) "
                f"({len(series) - len(kept)} degenerate series skipped)."
            )
            self._seeded = True

    def _ensure_seeded(self, universe_id: str) -> None:
        """Lazily seed from the universe actually being scored.

        Makes correct seeding the default rather than something a runner has to
        remember. Phase 30 forgot, and its first trial was scored at K_eff = 1.
        """
        if self._seeded or not self._seed_memory:
            return
        from src.lab.audit.referee_memory import collect_stored_returns

        series, provenance = collect_stored_returns(universe_id, catalog_path=self.catalog_path)
        kept = [
            (s, p) for s, p in zip(series, provenance)
            if float(np.std(s)) > 0.0 and bool(np.isfinite(s).all())
        ]
        self.prior_trial_returns = [s for s, _ in kept] + self.prior_trial_returns
        self.seeded_from = [p for _, p in kept]
        self._seeded = True
        print(
            f"[Phase30Referee] Auto-seeded memory for {universe_id}: "
            f"{len(kept)} prior trial(s) "
            f"({len(series) - len(kept)} degenerate series skipped)."
        )

    def compute_effective_k(self, new_returns: np.ndarray) -> Tuple[float, float, int]:
        """
        Effective number of independent searches.

        DEFAULT ESTIMATOR: cluster-count (self.k_estimator == "cluster").

        The previous rho-average form, K_eff = k/(1+(k-1)*rho), is bounded above
        by 1/rho, so flooding the family with near-duplicates RAISED rho and
        LOWERED the penalty. Measured on this project's own data, 50
        near-duplicates moved the live candidate from DSR 0.7683 (fail) to
        0.9725 (pass) -- searching more made the gate easier. Cluster counting
        is monotone: a duplicate joins an existing cluster or starts a new one,
        and can never reduce the count.

        "rho_avg" is still computed and returned for reporting continuity, but
        no longer drives the penalty unless k_estimator == "rho_avg" (retained
        only to reproduce historical results).
        """
        k = len(self.prior_trial_returns) + 1
        if k <= 1:
            return 1.0, 0.0, 1

        min_len = min(len(new_returns), min(len(r) for r in self.prior_trial_returns))
        truncated = [r[:min_len] for r in self.prior_trial_returns] + [new_returns[:min_len]]

        mat = np.column_stack(truncated)
        corr = np.corrcoef(mat, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        triu_idx = np.triu_indices(k, k=1)
        pair_corrs = corr[triu_idx]
        rho_avg = float(np.mean(pair_corrs)) if len(pair_corrs) > 0 else 0.0
        rho_avg = max(0.0, min(0.999, rho_avg))

        if getattr(self, "k_estimator", "cluster") == "rho_avg":
            k_eff = float(k / (1.0 + (k - 1) * rho_avg))
            return max(1.0, min(float(k), k_eff)), rho_avg, k

        from src.lab.audit.referee_memory import cluster_count_k

        n_clusters, _, _ = cluster_count_k(
            new_returns, self.prior_trial_returns, threshold=self.cluster_threshold
        )
        return float(n_clusters), rho_avg, k

    # A regime is a CRASH if the MARKET fell hard in it. Measured from an
    # equal-weight buy-and-hold of the universe's own assets, so the label is a
    # property of the window and is completely independent of the candidate
    # being scored -- a strategy cannot influence which gate it faces.
    #
    # BOTH conditions are required. Drawdown alone misclassifies: Q3 2021 has a
    # -36% basket drawdown but returned +84%, a volatile bull quarter, not a
    # crash. Demanding a materially negative total return as well separates
    # them cleanly.
    CRASH_RETURN_THRESHOLD = -0.20     # market ended the window down >= 20%
    CRASH_DRAWDOWN_THRESHOLD = -0.30   # and fell >= 30% peak-to-trough

    def classify_regime(self, assets: List[str], start: str, end: str) -> Dict[str, Any]:
        """Label a window CRASH or NORMAL from buy-and-hold behaviour alone."""
        key = (tuple(assets), start, end)
        if key not in self._df_cache:
            self._df_cache[key] = build_multi_asset_table(assets, start, end)
        df = self._df_cache[key]
        closes = np.column_stack([df[f"{a}_close"].to_numpy() for a in assets])
        nav = (closes / closes[0]).mean(axis=1)
        peak = np.maximum.accumulate(nav)
        max_dd = float(((nav - peak) / peak).min())
        total_ret = float(nav[-1] / nav[0] - 1.0)
        is_crash = (total_ret <= self.CRASH_RETURN_THRESHOLD
                    and max_dd <= self.CRASH_DRAWDOWN_THRESHOLD)
        return {"is_crash": bool(is_crash), "bh_total_return": round(total_ret, 4),
                "bh_max_drawdown": round(max_dd, 4)}

    def evaluate_multi_regime_consistency(
        self, proposal: Phase30Proposal
    ) -> Tuple[bool, List[str]]:
        u_info = next((c for c in self._catalog if c["universe_id"] == proposal.universe_id), None)
        if not u_info:
            return True, []
        assets = u_info.get("assets", [])
        if not assets:
            return True, []

        # Respread across the full 3-year dev span (governing spec). The old
        # trio (Q1/Q2/Q4 2023) all sat in the final third of the window and
        # contained no crash regime, so the gate was not testing regime
        # diversity at all. One quarter per year plus BOTH 2022 crashes.
        regimes = [
            ("Q3 2021", "2021-07-01", "2021-09-30"),
            ("Q2 2022 (Luna/3AC crash)", "2022-04-01", "2022-06-30"),
            ("Q4 2022 (FTX crash)", "2022-10-01", "2022-12-31"),
            ("Q2 2023", "2023-04-01", "2023-06-30"),
        ]

        srs = []
        dds = []
        breaches = []
        regime_detail: List[Dict[str, Any]] = []
        regime_errors: List[str] = []
        for q_name, s, e in regimes:
            try:
                cls = self.classify_regime(assets, s, e)
                b_key = (tuple(assets), s, e)
                if b_key not in self._df_cache:
                    self._df_cache[b_key] = build_multi_asset_table(assets, s, e)
                df_b = self._df_cache[b_key]
                res_b, _ = self.engine.run_generative_backtest(proposal, custom_df=df_b)
                srs.append(res_b.annualized_sharpe_net)
                dds.append(res_b.max_drawdown)
                regime_detail.append({"name": q_name, "is_crash": cls["is_crash"],
                                      "sharpe": res_b.annualized_sharpe_net,
                                      "drawdown": res_b.max_drawdown,
                                      "bh_total_return": cls["bh_total_return"],
                                      "bh_max_drawdown": cls["bh_max_drawdown"]})

                for sym in assets:
                    # Isolate the asset by WEIGHT, on the full multi-asset table,
                    # rather than by slicing the table down to one symbol.
                    #
                    # Slicing removed the peers, which silently zeroes any
                    # cross-asset factor: it would trade nothing here, post a 0%
                    # drawdown, and pass the per-asset gate for entirely the
                    # wrong reason. Weight isolation keeps the peer context the
                    # factor actually sees while measuring one asset's sleeve.
                    import copy as _copy

                    iso = _copy.copy(proposal)
                    iso.asset_weights = {a: (1.0 if a == sym else 0.0) for a in assets}
                    res_s, _ = self.engine.run_generative_backtest(iso, custom_df=df_b)

                    # BENCHMARK, NOT A CONSTANT (Phase 49). The sleeve must not
                    # lose more than simply HOLDING the same asset over the same
                    # window. See data/audit/POLICY_REVIEW_per_asset_drawdown_
                    # ceiling.json: the previous -30% was declared in Phase 21
                    # and never derived, and it was regime-inverted -- it demanded
                    # the sleeve lose under 45% of buy-and-hold during a crash but
                    # allowed 75% of it in a calm quarter, which is backwards. It
                    # also rejected sleeves that had REDUCED the loss versus doing
                    # nothing: 83% of historical breaches beat buy-and-hold.
                    #
                    # R = 1.00 is the do-no-harm standard and has ZERO free
                    # parameters, which is why it was chosen over a calibrated
                    # value: no calibration decision could be made while a
                    # specific candidate's outcome was already known.
                    bh_dd = self._buy_and_hold_drawdown(df_b, sym)
                    limit = self.PER_ASSET_DD_BENCHMARK_RATIO * bh_dd
                    if res_s.max_drawdown < limit:
                        breaches.append(
                            f"{sym} in {q_name} (sleeve {res_s.max_drawdown*100:.1f}% vs "
                            f"buy-and-hold {bh_dd*100:.1f}%, ratio "
                            f"{abs(res_s.max_drawdown)/max(1e-9, abs(bh_dd)):.2f}x)")
            except Exception as err:
                regime_errors.append(f"{q_name}: {type(err).__name__}: {err}")

        # FAIL CLOSED.
        #
        # This previously swallowed every exception and then returned
        # (True, []) whenever fewer than three regimes evaluated -- so a data
        # outage made the multi-regime AND per-asset drawdown gates both pass
        # with nothing actually checked, and any breach already found in the
        # regimes that DID load was discarded. A gate that cannot run has not
        # been satisfied; it is inconclusive, and inconclusive is not a pass.
        if len(srs) < len(regimes):
            detail = "; ".join(regime_errors) if regime_errors else "no regime data returned"
            found = f" Breaches already observed before the failure: {breaches}." if breaches else ""
            return False, [
                f"Multi-regime consistency INCONCLUSIVE: only {len(srs)}/{len(regimes)} regimes "
                f"could be evaluated, so the gate could not be applied. "
                f"Errors: {detail}.{found}"
            ]

        # SPLIT GATE.
        #
        # Requiring Sharpe >= 0.50 during an FTX-collapse quarter effectively
        # demands a short-biased or market-neutral strategy. A long-biased
        # strategy that earns in normal conditions and merely SURVIVES a crash
        # is a legitimate thing to want, and the old uniform rule could not
        # express that. So:
        #
        #   non-crash regimes -> must PERFORM: Sharpe >= 0.50 in
        #                        ceil(2/3 * n_non_crash) of them
        # WHY THESE CEILINGS STAY ABSOLUTE WHILE THE WHOLE-PERIOD GATE DOES NOT
        # (Phase 40). Calmar is the right basis for an EFFICIENCY test over the
        # full window. It is the wrong basis inside a crash window, where the
        # strategy is only asked to survive: returns there are expected to be
        # flat or negative, so CAGR/|DD| is negative and a ratio bar would
        # reject everything mechanically. A survival floor needs an absolute
        # shape.
        #
        # This does not reopen the de-leveraging loophole. De-leveraging to
        # survive a crash is legitimate risk management, not gaming, because
        # the SAME de-leveraged series must still clear the whole-period Calmar
        # floor -- and de-leveraging leaves Calmar flat or slightly worse. You
        # can resize your way out of a crash ceiling; you cannot resize your
        # way into efficiency.
        #
        #   crash regimes     -> must only SURVIVE: drawdown ceilings, no
        #                        Sharpe requirement at all
        #
        # Drawdown ceilings (basket and per-asset) apply to EVERY regime,
        # crash or not. Crash regimes are exempted from the return test, never
        # from the risk test.
        normal = [r for r in regime_detail if not r["is_crash"]]
        crash = [r for r in regime_detail if r["is_crash"]]
        worst_dd = min(dds)

        failures = []

        # A set with no normal regime cannot test performance at all. Fail
        # closed rather than pass a candidate on an all-crash technicality.
        if not normal:
            failures.append(
                f"Multi-regime consistency INCONCLUSIVE: every evaluated regime classified as a "
                f"crash ({[r['name'] for r in crash]}), so the performance requirement could not "
                f"be applied. The regime set must contain at least one non-crash window."
            )
        else:
            required_n50 = math.ceil((2.0 / 3.0) * len(normal))
            n_50 = sum(1 for r in normal if r["sharpe"] >= 0.50)
            if n_50 < required_n50:
                failures.append(
                    f"Multi-regime consistency failed: only {n_50}/{len(normal)} NON-CRASH regimes "
                    f"had Sharpe >= +0.50 (need {required_n50}); "
                    f"{[(r['name'], round(r['sharpe'], 3)) for r in normal]}."
                )
            worst_normal_sr = min(r["sharpe"] for r in normal)
            if worst_normal_sr < -2.50:
                failures.append(
                    f"Multi-regime consistency failed: worst NON-CRASH regime Sharpe "
                    f"{worst_normal_sr:.2f} < -2.50 floor."
                )

        # Risk gates: every regime, no exemptions.
        if worst_dd < -0.30:
            offender = min(regime_detail, key=lambda r: r["drawdown"]) if regime_detail else None
            where = f" (worst: {offender['name']})" if offender else ""
            failures.append(
                f"Multi-regime consistency failed: basket max drawdown {worst_dd*100:.1f}% "
                f"exceeded -30% ceiling{where}."
            )
        if breaches:
            failures.append(
                f"Per-asset drawdown gate failed: constituent assets lost more than simply "
                f"holding them (R={self.PER_ASSET_DD_BENCHMARK_RATIO:.2f} of buy-and-hold, "
                f"same window): {', '.join(breaches)}."
            )

        print(
            f"    [regimes] " + "  ".join(
                f"{r['name']}={'CRASH' if r['is_crash'] else 'normal'}"
                f"(SR {r['sharpe']:+.2f}, DD {r['drawdown']*100:+.1f}%)"
                for r in regime_detail
            )
        )

        passed = len(failures) == 0
        return passed, failures

    def _buy_and_hold_drawdown(self, df_window, sym: str) -> float:
        """NAV drawdown of simply holding `sym` over this window.

        The benchmark the sleeve is judged against. Cached per (window, symbol)
        because the multi-regime gate asks for it once per asset per regime.
        """
        key = ("bh_dd", id(df_window), sym)
        if key in self._df_cache:
            return self._df_cache[key]
        import numpy as _np

        close = _np.asarray(df_window[f"{sym}_close"].to_numpy(), dtype=float)
        rets = _np.zeros_like(close)
        rets[1:] = _np.diff(close) / _np.maximum(1e-12, close[:-1])
        dd = float(max_drawdown_nav(rets))
        self._df_cache[key] = dd
        return dd

    def check_duplicate(
        self,
        trial_index: int,
        proposal: Phase30Proposal,
        net_ret: np.ndarray,
    ) -> Tuple[bool, List[int]]:
        code_hash = proposal.get_code_hash()

        # 1. Exact AST code & config hash match
        if code_hash in self.prior_code_hashes:
            dup_trial = self.prior_code_hashes[code_hash]
            return True, [dup_trial]

        # 2. Same BEHAVIOUR under different source text. A hash cannot see
        #    this: Phase 31 produced four bitwise-identical return series under
        #    four distinct hashes.
        for p_idx, prior_r in enumerate(self.prior_trial_returns):
            min_l = min(len(net_ret), len(prior_r))
            if min_l <= 100:
                continue
            a, b = np.asarray(net_ret)[:min_l], np.asarray(prior_r)[:min_l]
            if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(corr) and corr >= self.DUPLICATE_RHO_THRESHOLD:
                # Name the trial duplicated. Priors from seeded cross-phase
                # history have no trial index in THIS run, so report -1 rather
                # than a slot number. The previous fallback, p_idx + 1, emitted
                # a slot dressed up as a trial index: plausible-looking and
                # wrong, which is how bad numbers end up quoted from a ledger.
                return True, [self.prior_trial_index_by_slot.get(p_idx, -1)]

        return False, []

    def evaluate(
        self,
        trial_index: int,
        proposal: Phase30Proposal,
        result: BacktestResult,
        is_duplicate: bool = False,
        duplicate_of_trials: Optional[List[int]] = None,
    ) -> Phase21Evaluation:
        self._ensure_seeded(proposal.universe_id)

        net_ret = np.array(result.net_returns)
        gross_ret = np.array(result.gross_returns)
        turnover = np.array(result.turnovers)
        n_obs = len(net_ret)

        failures = []
        is_dup = is_duplicate
        dup_trials = duplicate_of_trials or []

        if is_dup:
            failures.append(
                f"Proposal is mechanically identical or duplicate of prior trial(s) {dup_trials} on universe {proposal.universe_id}."
            )

        # 1. Effective K calculation
        k_eff, rho_avg, k_total = self.compute_effective_k(net_ret)

        # 2. Deflated Sharpe Ratio
        metrics = MetricsCalculator.calculate_metrics(returns=list(net_ret), periods_per_year=8760)
        period_sharpe = metrics.period_sharpe
        skew = metrics.skewness
        kurt = metrics.kurtosis

        dsr_res = DeflatedSharpeCalculator.calculate_dsr(
            observed_period_sharpe=period_sharpe,
            n_observations=n_obs,
            skewness=skew,
            kurtosis=kurt,
            k_trials=max(1, int(round(k_eff))),
            variance_method=TrialVarianceMethod.THEORETICAL_NULL,
            periods_per_year=8760,
            sharpe_unit=SharpeUnit.PERIOD,
        )
        dsr_passed = bool(dsr_res.dsr_value >= self.DSR_THRESHOLD)
        if not dsr_passed:
            failures.append(
                f"Deflated Sharpe Ratio failed: DSR={dsr_res.dsr_value:.4f} (threshold: >={self.DSR_THRESHOLD} under K_eff={k_eff:.1f})."
            )

        # 3. Stationary Block Bootstrap
        boot = StationaryBootstrap(mean_block_length=24.0, n_bootstraps=1000, seed=42 + trial_index)
        boot_res = boot.test_null_hypothesis(returns=list(net_ret), statistic="sharpe")
        boot_passed = bool(boot_res.p_value <= self.BOOTSTRAP_P_THRESHOLD)
        if not boot_passed:
            failures.append(
                f"Stationary bootstrap test of H0 (E[r] <= 0) failed: p-value={boot_res.p_value:.4f} (threshold: <={self.BOOTSTRAP_P_THRESHOLD})."
            )

        # 4. State-Dependent Cost Stress
        cost_tester = CostStressTester(periods_per_year=8760)
        cost_res = cost_tester.evaluate_returns_with_turnover(gross_returns=list(gross_ret), turnovers=list(turnover))
        breakeven_bps = float(cost_res.breakeven_fee_bps)
        cost_passed = bool(breakeven_bps >= self.MIN_BREAKEVEN_BPS)
        if not cost_passed:
            failures.append(
                f"State-dependent friction failed: Breakeven cost is {breakeven_bps:.1f} bps (threshold: >={self.MIN_BREAKEVEN_BPS} bps)."
            )

        # 5. Core economic constraints
        if result.cagr_net <= 0.0:
            failures.append(f"Net CAGR is non-positive: {result.cagr_net:.2%}.")
        # RISK EFFICIENCY GATE (Phase 40). Replaces the absolute drawdown
        # ceiling, which was de-leverageable: measured across 379 stored return
        # series, scaling gross exposure to 0.10x cut NAV drawdown to ~11% of
        # its original value in 378 of them, while Sharpe and DSR were
        # unchanged to four decimals. A ceiling anything can satisfy by trading
        # smaller is a leverage constraint, not a risk constraint.
        #
        # It was also wrong in the other direction: the Phase 36 control with
        # 72% genuine predictive accuracy (Sharpe 2.663, DSR-passing) has a
        # -73.9% drawdown and was REJECTED by the -30% ceiling.
        #
        # Calmar cannot be resized into. Over the same 379 series, 0.10x gross
        # left Calmar at a median 85.7% of its original value -- de-leveraging
        # HURTS it -- and improved it in only 21.8% of cases, by at most ~19%
        # in the top decile, which is genuine reduction of volatility drag
        # rather than gaming.
        calmar = scale_invariant_calmar(net_ret, 8760)
        calmar_passed = bool(calmar >= self.MIN_CALMAR)
        if not calmar_passed:
            failures.append(
                f"Risk efficiency failed: Calmar={calmar:.3f} "
                f"(threshold: >={self.MIN_CALMAR}; geometric CAGR / |NAV max drawdown|). "
                f"Cannot be fixed by resizing."
            )

        # 6. Multi-regime and per-asset consistency gate
        regime_passed, regime_failures = self.evaluate_multi_regime_consistency(proposal)
        if not regime_passed:
            failures.extend(regime_failures)

        is_confirmed = (len(failures) == 0)

        if is_confirmed:
            diag = "Statistically Confirmed: Refined alpha candidate exhibits persistent positive expected return surviving effective-trial DSR, multi-regime consistency, per-asset drawdown ceilings, and state-dependent friction."
        elif is_dup:
            diag = f"Deduplication Rejection: Code or return series is duplicate of prior trial(s) {dup_trials}."
        elif not cost_passed:
            diag = "Execution Friction Failure: Turnover frequency erodes gross edge below viable fee bounds."
        elif not dsr_passed:
            diag = "Multiple Testing Overfitting: Observed Sharpe does not exceed expected maximum sample variance under multiple trial selection."
        elif not boot_passed:
            diag = "Significance Depletion: Return distribution fails stationary bootstrap test under centered block shuffling."
        elif not regime_passed:
            diag = "Multi-Regime Inconsistency: Strategy fails cross-regime stability or constituent assets violate individual drawdown limits."
        else:
            diag = "Structural Drawdown Deficit: Risk-reward asymmetry exceeds tolerable portfolio preservation bounds."

        eval_record = Phase21Evaluation(
            trial_index=trial_index,
            strategy_id=result.strategy_id,
            universe_id=proposal.universe_id,
            is_confirmed_viable=is_confirmed,
            annualized_sharpe_net=result.annualized_sharpe_net,
            dsr_value=dsr_res.dsr_value,
            dsr_passed=dsr_passed,
            bootstrap_p_value=boot_res.p_value,
            bootstrap_passed=boot_passed,
            breakeven_bps=breakeven_bps,
            cost_passed=cost_passed,
            cagr_net=result.cagr_net,
            max_drawdown=result.max_drawdown,
            k_trials_total=k_total,
            k_eff=k_eff,
            rho_trials_avg=rho_avg,
            k_trials_universe=k_total,
            k_eff_universe=k_eff,
            rho_universe_avg=rho_avg,
            is_duplicate=is_dup,
            duplicate_of_trials=dup_trials,
            relevance_warning="",
            drawdown_convention=self.engine.drawdown_convention,
            scale_invariant_calmar=calmar,
            calmar_passed=calmar_passed,
            cagr_geometric=geometric_cagr(net_ret, 8760),
            deployment_gross_for_30pct_dd=required_gross_for_drawdown_budget(
                net_ret, self.MAX_DRAWDOWN_LIMIT),
            project_universes_searched=self.project_footprint["universes_ever_searched"],
            project_trials_logged=self.project_footprint["total_trials_logged"],
            failure_reasons=failures,
            qualitative_diagnosis=diag,
            is_null_universe=False,
        )

        # Update historical memory.
        #
        # A strategy that never traded has a constant return series. It is not
        # a search: it tested nothing. Worse, a zero-variance series correlates
        # with nothing (corrcoef is undefined, coerced to 0), so under cluster
        # counting each no-op would form its OWN cluster and inflate the
        # multiple-testing penalty for every later candidate. Phase 32 logged
        # 15 such trials out of 26 evaluated, pushing K from 53 to 78 while
        # only 11 real searches occurred.
        #
        # Seeding already filters degenerate series; accumulation must match,
        # or the live counter and the seeded counter disagree.
        if self.quarantine:
            # Structural, not conventional: the ONLY path that mutates referee
            # memory is disabled outright, so a control cannot seed K_eff or be
            # seen by duplicate detection even by mistake.
            return eval_record

        if float(np.std(net_ret)) > 0.0 and bool(np.isfinite(net_ret).all()):
            self.prior_trial_index_by_slot[len(self.prior_trial_returns)] = trial_index
            self.prior_trial_returns.append(net_ret)
        code_hash = proposal.get_code_hash()
        self.prior_code_hashes[code_hash] = trial_index

        return eval_record


def evaluate_holdout_q3_2024(proposal: Phase30Proposal) -> Dict[str, Any]:
    """Evaluates the candidate ONCE on the air-gapped Q3 2024 holdout dataset."""
    if not CATALOG_PATH.exists():
        return {"error": "Catalog not found"}
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    u_info = next((c for c in catalog if c["universe_id"] == proposal.universe_id), None)
    if not u_info:
        return {"error": f"Universe {proposal.universe_id} not in catalog"}

    symbols = u_info["assets"]
    start_date = "2024-07-01"
    end_date = "2024-09-30"

    # Consuming the sealed holdout is an explicit, logged act. The embargo
    # guard refuses this range to everything else.
    from src.lab.audit.embargo import sanctioned_holdout_evaluation

    try:
        with sanctioned_holdout_evaluation(
            f"holdout evaluation of {proposal.variant_name or proposal.universe_id}"
        ):
            df_holdout = build_multi_asset_table(symbols, start_date, end_date)
    except Exception as e:
        return {"error": f"Failed to load holdout data: {e}"}

    engine = Phase30BacktestEngine(parquet_dir=PARQUET_DIR)
    res, _ = engine.run_generative_backtest(proposal, custom_df=df_holdout)

    return {
        "universe_id": proposal.universe_id,
        "symbols": symbols,
        "holdout_window": f"{start_date} to {end_date} (Q3 2024)",
        "n_bars": len(df_holdout),
        "annualized_sharpe_net": round(res.annualized_sharpe_net, 4),
        "annualized_sharpe_gross": round(res.annualized_sharpe_gross, 4),
        "cagr_net": round(res.cagr_net, 4),
        "max_drawdown": round(res.max_drawdown, 4),
        "turnover_annual": round(res.annual_turnover, 2),
        "mean_cost_bps": round(res.mean_cost_bps, 2),
        "n_trades": res.n_trades,
        "viable_on_holdout": bool(res.annualized_sharpe_net > 0.0 and
                                  scale_invariant_calmar(res.net_returns, 8760) >= 2.0),
    }
