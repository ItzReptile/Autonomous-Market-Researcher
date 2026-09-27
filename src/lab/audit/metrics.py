from dataclasses import dataclass
import math
from typing import List, Optional
import polars as pl


@dataclass(frozen=True)
class PerformanceMetrics:
    total_observations: int
    periods_per_year: int
    cagr: float
    annualized_volatility: float
    annualized_sharpe: float
    period_sharpe: float
    annualized_sortino: float
    period_sortino: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    skewness: float
    kurtosis: float  # Pearson kurtosis (normal distribution = 3.0)


class MetricsCalculator:
    """
    Deterministic performance metrics calculator with explicit conventions.
    Annualization assumes discrete compounding for CAGR and sqrt(T) scaling for vol/Sharpe.
    """
    HOURLY_CRYPTO = 8760   # 24 * 365
    DAILY_CRYPTO = 365
    DAILY_EQUITY = 252

    @classmethod
    def calculate_metrics(
        cls,
        returns: List[float],
        periods_per_year: int = HOURLY_CRYPTO,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        n = len(returns)
        if n < 2:
            raise ValueError(f"Insufficient observations to compute metrics: N={n} (minimum 2 required).")

        # Mean and standard deviation
        mean_r = sum(returns) / n
        excess_returns = [r - (risk_free_rate / periods_per_year) for r in returns]
        mean_excess = sum(excess_returns) / n

        var_sample = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_sample = math.sqrt(max(0.0, var_sample))

        # Volatility
        ann_vol = std_sample * math.sqrt(periods_per_year)

        # Sharpe ratio
        if std_sample > 1e-12:
            period_sharpe = mean_excess / std_sample
            ann_sharpe = period_sharpe * math.sqrt(periods_per_year)
        else:
            period_sharpe = 0.0
            ann_sharpe = 0.0

        # Downside deviation for Sortino (per-period)
        downside_sq_sum = sum(min(0.0, r) ** 2 for r in excess_returns)
        downside_std = math.sqrt(downside_sq_sum / (n - 1)) if n > 1 else 0.0

        if downside_std > 1e-12:
            period_sortino = mean_excess / downside_std
            ann_sortino = period_sortino * math.sqrt(periods_per_year)
        else:
            period_sortino = 0.0
            ann_sortino = 0.0

        # Equity curve & Max Drawdown
        equity = 1.0
        peak = 1.0
        max_dd = 0.0

        for r in returns:
            equity *= (1.0 + r)
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0.0 else 0.0
            if dd < max_dd:
                max_dd = dd

        # CAGR
        # Cumulative return: equity - 1.0
        if equity > 0.0:
            cagr = (equity ** (periods_per_year / n)) - 1.0
        else:
            cagr = -1.0  # Total loss

        # Calmar Ratio
        if abs(max_dd) > 1e-12:
            calmar = cagr / abs(max_dd)
        else:
            calmar = 0.0

        # Win Rate
        non_zero_trades = [r for r in returns if abs(r) > 1e-12]
        win_trades = [r for r in non_zero_trades if r > 0.0]
        win_rate = len(win_trades) / len(non_zero_trades) if non_zero_trades else 0.0

        # Profit Factor
        gross_profit = sum(r for r in returns if r > 0.0)
        gross_loss = abs(sum(r for r in returns if r < 0.0))
        if gross_loss > 1e-12:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0.0 else 0.0

        # Skewness and Kurtosis (Pearson convention)
        m2 = sum((r - mean_r) ** 2 for r in returns) / n
        if m2 > 1e-12:
            m3 = sum((r - mean_r) ** 3 for r in returns) / n
            m4 = sum((r - mean_r) ** 4 for r in returns) / n
            skewness = m3 / (m2 ** 1.5)
            kurtosis = m4 / (m2 ** 2.0)
        else:
            skewness = 0.0
            kurtosis = 3.0  # Normal default

        return PerformanceMetrics(
            total_observations=n,
            periods_per_year=periods_per_year,
            cagr=cagr,
            annualized_volatility=ann_vol,
            annualized_sharpe=ann_sharpe,
            period_sharpe=period_sharpe,
            annualized_sortino=ann_sortino,
            period_sortino=period_sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            skewness=skewness,
            kurtosis=kurtosis,
        )
