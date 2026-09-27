"""
Stage 4 Parallel Baseline Suite.
Evaluates 4 honest reference comparison points on UNIV_10 in parallel
with the Phase 30 survivor over the exact same forward stream:
1. BUY_AND_HOLD (equal 25% weights, hold)
2. SIMPLE_MOMENTUM (20-day / 480h lookback, weekly rebalance to top performer)
3. SIMPLE_MEAN_REVERSION (20-day / 480h lookback, enter long at z <= -2, exit at z >= 0)
4. CASH (0% return, zero position)
"""
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from src.lab.stage4.execution_model import BASE_FEE_RATE, BASE_SLIPPAGE_RATE


class ParallelBaselinesRunner:
    # Sourced from the shared execution model so this legacy batch runner can
    # never drift from the live friction assumptions.
    BASE_FEE_RATE = BASE_FEE_RATE
    BASE_SLIPPAGE_RATE = BASE_SLIPPAGE_RATE
    HOURS_PER_YEAR = 8760

    def __init__(self, initial_capital_usd: float = 100000.0):
        self.initial_capital_usd = initial_capital_usd

    def _compute_state_friction(
        self,
        close_mat: np.ndarray,
        high_mat: np.ndarray,
        low_mat: np.ndarray,
        ret_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def run_all_baselines(
        self,
        df_polars: pl.DataFrame,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if symbols is None:
            symbols = sorted([c.split("_")[0] for c in df_polars.columns if c.endswith("_close")])
        n_assets = len(symbols)
        n_bars = len(df_polars)

        close_mat = np.column_stack([df_polars[f"{s}_close"].to_numpy() for s in symbols])
        high_mat = np.column_stack([df_polars[f"{s}_high"].to_numpy() for s in symbols])
        low_mat = np.column_stack([df_polars[f"{s}_low"].to_numpy() for s in symbols])

        ret_mat = np.diff(close_mat, axis=0) / np.maximum(1e-8, close_mat[:-1])
        ret_mat = np.vstack([np.zeros((1, n_assets)), ret_mat])

        vol_ratio, spread_ratio = self._compute_state_friction(close_mat, high_mat, low_mat, ret_mat)

        warmup = 24
        results = {}

        # -------------------------------------------------------------
        # 1. BUY AND HOLD (Equal 25% weights across all assets)
        # -------------------------------------------------------------
        bnh_weights = np.ones((n_bars, n_assets)) / float(n_assets)
        results["BUY_AND_HOLD"] = self._simulate_weights(
            "BUY_AND_HOLD", bnh_weights, ret_mat, vol_ratio, spread_ratio, warmup, n_bars, n_assets
        )

        # -------------------------------------------------------------
        # 2. SIMPLE MOMENTUM (20-day / 480h lookback, weekly rebalance)
        # -------------------------------------------------------------
        mom_weights = np.zeros((n_bars, n_assets))
        lookback_mom = 480
        rebal_freq = 168
        cur_winner = 0

        for t in range(lookback_mom, n_bars):
            if (t - lookback_mom) % rebal_freq == 0:
                p_now = close_mat[t]
                p_old = close_mat[t - lookback_mom]
                cum_ret = (p_now - p_old) / np.maximum(1e-8, p_old)
                cur_winner = int(np.argmax(cum_ret))
            mom_weights[t, cur_winner] = 1.0

        results["SIMPLE_MOMENTUM"] = self._simulate_weights(
            "SIMPLE_MOMENTUM", mom_weights, ret_mat, vol_ratio, spread_ratio, warmup, n_bars, n_assets
        )

        # -------------------------------------------------------------
        # 3. SIMPLE MEAN REVERSION (20-day / 480h z-score, enter <= -2, exit >= 0)
        # -------------------------------------------------------------
        mr_weights = np.zeros((n_bars, n_assets))
        lookback_mr = 480
        active_pos = np.zeros(n_assets)

        for t in range(lookback_mr, n_bars):
            window = close_mat[t - lookback_mr : t]
            mu = np.mean(window, axis=0)
            sigma = np.std(window, axis=0) + 1e-8
            z = (close_mat[t] - mu) / sigma

            for i in range(n_assets):
                if z[i] <= -2.0:
                    active_pos[i] = 1.0
                elif z[i] >= 0.0:
                    active_pos[i] = 0.0

            # Equal scale active positions
            n_active = np.sum(active_pos)
            if n_active > 0:
                mr_weights[t] = (active_pos / float(n_active)) * (active_pos / float(n_assets))
            else:
                mr_weights[t] = 0.0

        results["SIMPLE_MEAN_REVERSION"] = self._simulate_weights(
            "SIMPLE_MEAN_REVERSION", mr_weights, ret_mat, vol_ratio, spread_ratio, warmup, n_bars, n_assets
        )

        # -------------------------------------------------------------
        # 4. CASH (Zero position, 0% return)
        # -------------------------------------------------------------
        results["CASH"] = {
            "strategy": "CASH",
            "net_sharpe": 0.0,
            "gross_sharpe": 0.0,
            "cagr_net": 0.0,
            "max_drawdown": 0.0,
            "annual_turnover": 0.0,
            "mean_cost_bps": 0.0,
            "total_trades": 0,
            "ending_nav_usd": self.initial_capital_usd,
            "total_pnl_usd": 0.0,
            "net_returns": [0.0] * (n_bars - warmup - 1),
            "nav_history": [self.initial_capital_usd] * (n_bars - warmup),
        }

        return results

    def _simulate_weights(
        self,
        name: str,
        weights_mat: np.ndarray,
        ret_mat: np.ndarray,
        vol_ratio: np.ndarray,
        spread_ratio: np.ndarray,
        warmup: int,
        n_bars: int,
        n_assets: int,
    ) -> Dict[str, Any]:
        cur_w = np.zeros(n_assets)
        portfolio_nav = self.initial_capital_usd
        trades = 0

        gross_returns = []
        net_returns = []
        turnovers = []
        costs_bps = []
        nav_history = [portfolio_nav]

        for t in range(warmup, n_bars - 1):
            target_w = weights_mat[t]
            traded_dw = np.abs(target_w - cur_w)
            sum_dw = float(np.sum(traded_dw))
            turnovers.append(sum_dw)
            if sum_dw > 1e-4:
                trades += 1

            fee_t = self.BASE_FEE_RATE
            slippage_t = self.BASE_SLIPPAGE_RATE * (0.5 * vol_ratio[t] + 0.5 * spread_ratio[t])
            friction_rate = fee_t + slippage_t

            cost_t = float(np.sum(traded_dw * friction_rate))
            bar_gross_ret = float(np.sum(target_w * ret_mat[t + 1]))
            bar_net_ret = bar_gross_ret - cost_t

            portfolio_nav = portfolio_nav * (1.0 + bar_net_ret)
            nav_history.append(portfolio_nav)
            gross_returns.append(bar_gross_ret)
            net_returns.append(bar_net_ret)

            if sum_dw > 1e-6:
                costs_bps.append((cost_t / sum_dw) * 10000.0)
            else:
                costs_bps.append(10.0)

            cur_w = target_w

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
            "strategy": name,
            "net_sharpe": round(sr_net, 4),
            "gross_sharpe": round(sr_gross, 4),
            "cagr_net": round(cagr, 4),
            "max_drawdown": round(max_dd, 4),
            "annual_turnover": round(ann_to, 2),
            "mean_cost_bps": round(float(np.mean(costs_bps)), 2),
            "total_trades": trades,
            "ending_nav_usd": round(portfolio_nav, 2),
            "total_pnl_usd": round(total_pnl_usd, 2),
            "net_returns": net_returns,
            "nav_history": nav_history,
        }
