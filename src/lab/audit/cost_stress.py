"""
Transaction Cost Stress Testing.
Evaluates strategy performance across standard fee tiers: 0, 5, 10, 25, 50, 100 bps.
Computes performance degradation curves and breakeven transaction costs.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass(frozen=True)
class CostPointResult:
    fee_bps: float
    fee_rate: float
    annualized_sharpe: float
    cagr: float
    max_drawdown: float
    profit_factor: float


@dataclass(frozen=True)
class CostStressResult:
    cost_curve: List[CostPointResult]
    breakeven_fee_bps: float          # Fee in bps where Sharpe drops to <= 0
    sharpe_decay_per_10bps: float     # Linear rate of Sharpe decay per 10 bps
    survives_10bps: bool              # True if Sharpe > 0 at 10 bps (typical VIP0/taker spot fee)
    survives_25bps: bool              # True if Sharpe > 0 at 25 bps (high friction)


class CostStressTester:
    """
    Stress-tests portfolio returns across transaction cost tiers.
    Standard evaluation grid: 0, 5, 10, 25, 50, 100 bps.

    FEE ACCOUNTING DEFINITION:
      - fee_rate is the ONE-WAY PROPORTIONAL FEE RATE applied to traded notional.
        10 bps = 0.0010 = 0.10% per side.
      - turnover_t is the one-way traded notional fraction at bar t:
        turnover_t = sum(|target_val_i - pre_trade_val_i|) / portfolio_val.
      - A round-trip transaction (buy 100% then sell 100%) incurs 2 * fee_rate
        (e.g., 20 bps total round-trip friction at a 10 bps fee rate).
      - Breakeven fee >= 10 bps means the strategy survives >= 10 bps per side
        (>= 20 bps round trip) of trading friction before Sharpe drops to <= 0.
      - Slippage is modeled separately in the execution simulator at bar execution.
    """
    DEFAULT_FEE_GRID_BPS = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]

    def __init__(
        self,
        fee_grid_bps: Optional[List[float]] = None,
        periods_per_year: int = MetricsCalculator.HOURLY_CRYPTO,
    ):
        self.fee_grid_bps = sorted(fee_grid_bps or self.DEFAULT_FEE_GRID_BPS)
        self.periods_per_year = periods_per_year

    def evaluate_returns_with_turnover(
        self,
        gross_returns: List[float],
        turnovers: List[float],
    ) -> CostStressResult:
        """
        Evaluates returns under cost grid using gross bar returns and traded turnover fractions.
        Net return at bar t: r_net,t = r_gross,t - turnover_t * fee_rate.
        Here fee_rate is per-side on one-way traded notional fraction (turnover_t).
        """
        n = len(gross_returns)
        if n != len(turnovers):
            raise ValueError(f"Length mismatch: {n} returns vs {len(turnovers)} turnovers")
        if n < 2:
            raise ValueError(f"Insufficient observations: N={n}")

        curve: List[CostPointResult] = []
        for bps in self.fee_grid_bps:
            fee_rate = bps / 10_000.0
            net_returns = [r - (turn * fee_rate) for r, turn in zip(gross_returns, turnovers)]
            m = MetricsCalculator.calculate_metrics(
                returns=net_returns,
                periods_per_year=self.periods_per_year,
            )
            curve.append(
                CostPointResult(
                    fee_bps=bps,
                    fee_rate=fee_rate,
                    annualized_sharpe=m.annualized_sharpe,
                    cagr=m.cagr,
                    max_drawdown=m.max_drawdown,
                    profit_factor=m.profit_factor,
                )
            )

        return self._summarize_curve(curve)

    def evaluate_runner(
        self,
        simulator_runner: Callable[[float], List[float]],
    ) -> CostStressResult:
        """
        Evaluates returns by re-running a simulation function that takes fee_rate and returns net returns.
        """
        curve: List[CostPointResult] = []
        for bps in self.fee_grid_bps:
            fee_rate = bps / 10_000.0
            net_returns = simulator_runner(fee_rate)
            m = MetricsCalculator.calculate_metrics(
                returns=net_returns,
                periods_per_year=self.periods_per_year,
            )
            curve.append(
                CostPointResult(
                    fee_bps=bps,
                    fee_rate=fee_rate,
                    annualized_sharpe=m.annualized_sharpe,
                    cagr=m.cagr,
                    max_drawdown=m.max_drawdown,
                    profit_factor=m.profit_factor,
                )
            )

        return self._summarize_curve(curve)

    def _summarize_curve(self, curve: List[CostPointResult]) -> CostStressResult:
        # Determine breakeven fee where Sharpe crosses <= 0
        breakeven_bps = 0.0
        # Check if first point is already <= 0
        if curve[0].annualized_sharpe <= 0:
            breakeven_bps = 0.0
        else:
            crossed = False
            for i in range(len(curve) - 1):
                p1 = curve[i]
                p2 = curve[i + 1]
                if p1.annualized_sharpe > 0 and p2.annualized_sharpe <= 0:
                    # Linear interpolation
                    slope = (p2.annualized_sharpe - p1.annualized_sharpe) / (p2.fee_bps - p1.fee_bps)
                    if abs(slope) > 1e-12:
                        breakeven_bps = p1.fee_bps + (-p1.annualized_sharpe / slope)
                    else:
                        breakeven_bps = p1.fee_bps
                    crossed = True
                    break
            if not crossed:
                # Still positive at highest grid point, extrapolate from last segment
                p_last_2 = curve[-2]
                p_last = curve[-1]
                slope = (p_last.annualized_sharpe - p_last_2.annualized_sharpe) / (p_last.fee_bps - p_last_2.fee_bps)
                if slope < -1e-12:
                    breakeven_bps = p_last.fee_bps + (-p_last.annualized_sharpe / slope)
                else:
                    breakeven_bps = float("inf")

        # Linear regression slope for Sharpe decay per 10 bps
        bps_vals = np.array([p.fee_bps for p in curve])
        sr_vals = np.array([p.annualized_sharpe for p in curve])
        if len(bps_vals) >= 2 and np.ptp(bps_vals) > 0:
            # Fit line: sr = slope * bps + intercept
            slope, _ = np.polyfit(bps_vals, sr_vals, 1)
            sharpe_decay_per_10bps = float(abs(slope) * 10.0)
        else:
            sharpe_decay_per_10bps = 0.0

        survives_10 = any(p.fee_bps == 10.0 and p.annualized_sharpe > 0 for p in curve)
        survives_25 = any(p.fee_bps == 25.0 and p.annualized_sharpe > 0 for p in curve)

        return CostStressResult(
            cost_curve=curve,
            breakeven_fee_bps=max(0.0, breakeven_bps),
            sharpe_decay_per_10bps=sharpe_decay_per_10bps,
            survives_10bps=survives_10,
            survives_25bps=survives_25,
        )
