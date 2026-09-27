"""
Stage 4 Paper Trading Execution Engine for Phase 30 Trial 01 Survivor.
Executes the frozen strategy configuration bar-by-bar and logs every trade ticket.

Friction and order quantization are delegated to src.lab.stage4.execution_model
so this engine and the live daemon share one implementation and cannot drift.
"""
from dataclasses import asdict, dataclass
import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.lab.stage4.execution_model import (
    BASE_FEE_RATE,
    BASE_SLIPPAGE_RATE,
    DEFAULT_FILTERS,
    SymbolFilters,
    causal_friction_ratios,
    quantize_order,
    slippage_rate,
)

TRADE_LOG_PATH = PROJECT_ROOT / "data" / "paper_trading" / "trade_log.jsonl"


@dataclass
class TradeTicket:
    trade_id: int
    timestamp: str
    bar_index: int
    symbol: str
    side: str  # 'BUY' or 'SELL'
    traded_weight_delta: float
    position_before: float
    position_after: float
    fill_price: float
    fee_bps: float
    slippage_bps: float
    total_cost_bps: float
    trade_cost_usd: float
    portfolio_nav_before: float
    portfolio_nav_after: float
    # Quantization diagnostics (material at a $1k basis, negligible at $100k)
    requested_notional_usd: float = 0.0
    filled_notional_usd: float = 0.0
    filled_qty: float = 0.0
    quantization_error_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def legacy_full_sample_ratios(
    close_mat: np.ndarray,
    high_mat: np.ndarray,
    low_mat: np.ndarray,
    ret_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The original full-sample friction normalisation.

    Retained ONLY so previously published figures can be reproduced for
    comparison. It divides by the mean over the whole sample, which peeks at
    the future and is not reproducible in live trading. Do not use it for any
    forward-looking result.
    """
    n_bars, n_assets = close_mat.shape
    vol_24h = np.zeros_like(ret_mat)
    for t in range(24, n_bars):
        vol_24h[t] = np.std(ret_mat[t - 24 : t], axis=0)
    mean_vol = np.maximum(1e-6, np.mean(vol_24h, axis=0, keepdims=True))
    vol_ratio = vol_24h / mean_vol

    spread_proxy = (high_mat - low_mat) / np.maximum(1e-8, close_mat)
    mean_spread = np.maximum(1e-6, np.mean(spread_proxy, axis=0, keepdims=True))
    spread_ratio = spread_proxy / mean_spread
    return vol_ratio, spread_ratio


class Stage4PaperTradingEngine:
    """
    Executes the exact Phase 30 Trial 01 survivor:
    - 24h momentum + 16/24h volume ratio
    - Hysteresis: enter > 0.50, exit < 0.20
    - Position caps: uniform 0.70 on all four assets
    - Capital allocation: ATOM 15%, DOT 15%, LINK 35%, UNI 35%
    - State-dependent friction: 5 bps fee + dynamic slippage (causal baseline)
    - Exchange LOT_SIZE / MIN_NOTIONAL quantization
    """
    BASE_FEE_RATE = BASE_FEE_RATE
    BASE_SLIPPAGE_RATE = BASE_SLIPPAGE_RATE
    HOURS_PER_YEAR = 8760

    CAPITAL_WEIGHTS = {
        "ATOMUSDT": 0.15,
        "DOTUSDT": 0.15,
        "LINKUSDT": 0.35,
        "UNIUSDT": 0.35,
    }

    POSITION_CAPS = {
        "ATOMUSDT": 0.70,
        "DOTUSDT": 0.70,
        "LINKUSDT": 0.70,
        "UNIUSDT": 0.70,
    }

    def __init__(
        self,
        initial_capital_usd: float = 1000.0,
        trade_log_path: Path = TRADE_LOG_PATH,
        friction_mode: str = "causal",
        apply_quantization: bool = True,
        filters: Optional[Dict[str, SymbolFilters]] = None,
    ):
        if friction_mode not in ("causal", "legacy_full_sample"):
            raise ValueError(f"Unknown friction_mode: {friction_mode}")
        self.initial_capital_usd = initial_capital_usd
        self.trade_log_path = trade_log_path
        self.friction_mode = friction_mode
        self.apply_quantization = apply_quantization
        self.filters = filters if filters is not None else dict(DEFAULT_FILTERS)

    def compute_raw_factors(self, df_polars: pl.DataFrame, symbols: List[str]) -> np.ndarray:
        n_bars = len(df_polars)
        n_assets = len(symbols)
        raw_factors = np.zeros((n_bars, n_assets), dtype=np.float64)

        for i, sym in enumerate(symbols):
            close = df_polars[f"{sym}_close"].to_pandas()
            vol = df_polars[f"{sym}_volume"].to_pandas()

            # Dimension A: 24h Momentum
            term_mom = close.pct_change(24)

            # Dimension B: 16h/24h Volume Ratio
            vol_16h = vol.rolling(16, min_periods=16).mean()
            vol_24h = vol.rolling(24, min_periods=24).mean()
            term_vol = vol_16h / (vol_24h + 1e-8)

            # Canonical composition
            raw_score = 0.60 * term_mom + 0.40 * term_vol
            factor = np.tanh(raw_score).fillna(0.0)
            raw_factors[:, i] = factor.to_numpy()

        return raw_factors

    def run_simulation(
        self,
        df_polars: pl.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if symbols is None:
            symbols = sorted([c.split("_")[0] for c in df_polars.columns if c.endswith("_close")])
        n_assets = len(symbols)
        n_bars = len(df_polars)

        close_mat = np.column_stack([df_polars[f"{s}_close"].to_numpy() for s in symbols])
        high_mat = np.column_stack([df_polars[f"{s}_high"].to_numpy() for s in symbols])
        low_mat = np.column_stack([df_polars[f"{s}_low"].to_numpy() for s in symbols])

        ret_mat = np.diff(close_mat, axis=0) / np.maximum(1e-8, close_mat[:-1])
        ret_mat = np.vstack([np.zeros((1, n_assets)), ret_mat])

        if self.friction_mode == "causal":
            vol_ratio, spread_ratio = causal_friction_ratios(close_mat, high_mat, low_mat, ret_mat)
        else:
            vol_ratio, spread_ratio = legacy_full_sample_ratios(close_mat, high_mat, low_mat, ret_mat)

        timestamps = (
            df_polars["open_time"].to_numpy()
            if "open_time" in df_polars.columns
            else np.arange(n_bars, dtype=np.int64)
        )

        # 1. Compute raw continuous factors
        raw_factors = self.compute_raw_factors(df_polars, symbols)

        # 2. Stateful hysteresis per asset -> desired position
        positions_mat = np.zeros((n_bars, n_assets), dtype=np.float64)
        for i, sym in enumerate(symbols):
            cap = self.POSITION_CAPS.get(sym, 1.0)
            cur_pos = 0.0
            for t in range(n_bars):
                val = raw_factors[t, i]
                if abs(val) > 0.50:
                    cur_pos = float(np.sign(val) * cap)
                elif abs(val) < 0.20:
                    cur_pos = 0.0
                positions_mat[t, i] = cur_pos

        # 3. Portfolio capital weights
        cap_w = np.array([self.CAPITAL_WEIGHTS.get(s, 1.0 / n_assets) for s in symbols], dtype=np.float64)
        cap_w = cap_w / np.sum(cap_w)
        target_weights_mat = positions_mat * cap_w[None, :]

        # 4. Bar-by-bar execution and trade logging
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        trade_log_file = open(self.trade_log_path, "w", encoding="utf-8")

        warmup = 24
        cur_w = np.zeros(n_assets)
        portfolio_nav = self.initial_capital_usd
        trade_id = 1
        trade_tickets: List[TradeTicket] = []

        gross_returns: List[float] = []
        net_returns: List[float] = []
        turnovers: List[float] = []
        costs_bps: List[float] = []
        nav_history: List[float] = [portfolio_nav]
        timestamps_history: List[int] = [int(timestamps[warmup])]

        rejected_orders = 0
        total_quantization_error_usd = 0.0

        # The survivor is a hysteresis strategy: it trades only when the desired
        # position CHANGES state, never to rebalance a residual. Tracking the
        # previous desired weight (rather than the achieved weight) keeps a
        # quantization leftover from being re-attempted every single bar.
        last_desired_w = np.zeros(n_assets)

        for t in range(warmup, n_bars - 1):
            desired_w = target_weights_mat[t]
            intended_dw = np.where(
                np.abs(desired_w - last_desired_w) > 1e-9,
                desired_w - cur_w,
                0.0,
            )
            slip_t = slippage_rate(vol_ratio[t], spread_ratio[t])

            nav_before = portfolio_nav
            achieved_w = cur_w.copy()
            realized_dw = np.zeros(n_assets)
            bar_cost_usd = 0.0
            pending_tickets: List[TradeTicket] = []

            for i, sym in enumerate(symbols):
                delta_i = intended_dw[i]
                if abs(delta_i) <= 1e-4:
                    continue

                requested_notional = delta_i * nav_before
                price = float(close_mat[t, i])

                if self.apply_quantization:
                    filt = self.filters.get(sym)
                    if filt is None:
                        filled_notional = requested_notional
                        filled_qty = requested_notional / max(1e-12, price)
                        quant_err = 0.0
                        rejected = False
                    else:
                        fill = quantize_order(requested_notional, price, filt)
                        filled_notional = fill.filled_notional
                        filled_qty = fill.filled_qty
                        quant_err = fill.quantization_error_usd
                        rejected = fill.rejected_min_notional
                else:
                    filled_notional = requested_notional
                    filled_qty = requested_notional / max(1e-12, price)
                    quant_err = 0.0
                    rejected = False

                total_quantization_error_usd += quant_err
                if rejected or abs(filled_notional) < 1e-12:
                    rejected_orders += 1
                    continue

                filled_dw = filled_notional / max(1e-12, nav_before)
                realized_dw[i] = filled_dw
                achieved_w[i] = cur_w[i] + filled_dw

                fee_rate_i = self.BASE_FEE_RATE
                slip_rate_i = float(slip_t[i])
                cost_i = abs(filled_notional) * (fee_rate_i + slip_rate_i)
                bar_cost_usd += cost_i

                ts_str = datetime.datetime.fromtimestamp(
                    timestamps[t] / 1000.0, tz=datetime.timezone.utc
                ).isoformat()
                pending_tickets.append(
                    TradeTicket(
                        trade_id=trade_id,
                        timestamp=ts_str,
                        bar_index=t,
                        symbol=sym,
                        side="BUY" if filled_dw > 0 else "SELL",
                        traded_weight_delta=float(round(filled_dw, 6)),
                        position_before=float(round(cur_w[i], 6)),
                        position_after=float(round(achieved_w[i], 6)),
                        fill_price=float(round(price, 6)),
                        fee_bps=float(round(fee_rate_i * 10000.0, 2)),
                        slippage_bps=float(round(slip_rate_i * 10000.0, 2)),
                        total_cost_bps=float(round((fee_rate_i + slip_rate_i) * 10000.0, 2)),
                        trade_cost_usd=float(round(cost_i, 4)),
                        portfolio_nav_before=float(round(nav_before, 2)),
                        portfolio_nav_after=0.0,  # filled in below
                        requested_notional_usd=float(round(requested_notional, 4)),
                        filled_notional_usd=float(round(filled_notional, 4)),
                        filled_qty=float(round(filled_qty, 8)),
                        quantization_error_usd=float(round(quant_err, 6)),
                    )
                )
                trade_id += 1

            sum_dw = float(np.sum(np.abs(realized_dw)))
            turnovers.append(sum_dw)

            cost_frac = bar_cost_usd / max(1e-12, nav_before)
            bar_gross_ret = float(np.sum(achieved_w * ret_mat[t + 1]))
            bar_net_ret = bar_gross_ret - cost_frac

            portfolio_nav = nav_before * (1.0 + bar_net_ret)

            for tkt in pending_tickets:
                tkt.portfolio_nav_after = float(round(portfolio_nav, 2))
                trade_tickets.append(tkt)
                trade_log_file.write(json.dumps(tkt.to_dict()) + "\n")

            nav_history.append(portfolio_nav)
            timestamps_history.append(int(timestamps[t + 1]))

            gross_returns.append(bar_gross_ret)
            net_returns.append(bar_net_ret)
            # Only bars that actually traded contribute to realized friction.
            # The previous version appended a hardcoded 10.0 on no-trade bars,
            # which are the vast majority, so the reported mean was pinned near
            # 10 bps regardless of the real cost and barely moved when the fee
            # assumption changed.
            if sum_dw > 1e-6:
                costs_bps.append((cost_frac / sum_dw) * 10000.0)

            cur_w = achieved_w
            last_desired_w = desired_w

        trade_log_file.close()

        # Compute summary metrics
        net_arr = np.array(net_returns)
        gross_arr = np.array(gross_returns)
        to_arr = np.array(turnovers)
        n_valid = len(net_arr)

        mu_net = float(np.mean(net_arr)) if n_valid > 0 else 0.0
        sd_net = float(np.std(net_arr)) + 1e-8 if n_valid > 0 else 1.0
        sr_net = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_net / sd_net))

        mu_gross = float(np.mean(gross_arr)) if n_valid > 0 else 0.0
        sd_gross = float(np.std(gross_arr)) + 1e-8 if n_valid > 0 else 1.0
        sr_gross = float(np.sqrt(self.HOURS_PER_YEAR) * (mu_gross / sd_gross))

        cum_net = np.cumsum(net_arr)
        peak = np.maximum.accumulate(cum_net)
        dd = cum_net - peak
        max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

        ann_to = float(np.sum(to_arr) * (self.HOURS_PER_YEAR / max(1, n_valid)))
        cagr = float(mu_net * self.HOURS_PER_YEAR)
        total_pnl_usd = portfolio_nav - self.initial_capital_usd

        return {
            "strategy": "Phase 30 Trial 01 Survivor (T01_uniform_cap_070)",
            "symbols": symbols,
            "friction_mode": self.friction_mode,
            "quantization_applied": self.apply_quantization,
            "n_bars": n_valid,
            "net_sharpe": round(sr_net, 4),
            "gross_sharpe": round(sr_gross, 4),
            "cagr_net": round(cagr, 4),
            "max_drawdown": round(max_dd, 4),
            "annual_turnover": round(ann_to, 2),
            "mean_cost_bps": round(float(np.mean(costs_bps)), 2) if costs_bps else 0.0,
            "traded_bars": len(costs_bps),
            "total_trades": len(trade_tickets),
            "rejected_orders_min_notional": rejected_orders,
            "total_quantization_error_usd": round(total_quantization_error_usd, 4),
            "initial_capital_usd": self.initial_capital_usd,
            "ending_nav_usd": round(portfolio_nav, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "net_returns": net_returns,
            "gross_returns": gross_returns,
            "turnovers": turnovers,
            "costs_bps": costs_bps,
            "nav_history": nav_history,
            "timestamps": timestamps_history,
            "trade_tickets": trade_tickets,
        }
