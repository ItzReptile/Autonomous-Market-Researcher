"""
Stationary Block Bootstrap for Financial Time Series and Panels.
Reference: Politis, D. N., & Romano, J. P. (1994).
"The Stationary Bootstrap." Journal of the American Statistical Association, 89(428), 1303-1313.
"""
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    observed_statistic: float
    p_value: float
    null_mean: float
    null_std: float
    bootstrap_samples_count: int
    mean_block_length: float
    confidence_interval_95: Tuple[float, float]
    statistic_name: str
    seed: Optional[int]


class StationaryBootstrap:
    """
    Stationary Block Bootstrap for time-series and synchronous multi-asset panels.
    Preserves serial autocorrelation and cross-sectional asset dependencies.
    """

    def __init__(
        self,
        mean_block_length: float = 24.0,  # e.g., 24 hours for hourly data
        n_bootstraps: int = 1000,
        seed: Optional[int] = 42,
    ):
        if mean_block_length <= 0:
            raise ValueError(f"mean_block_length must be > 0, got {mean_block_length}")
        if n_bootstraps < 10:
            raise ValueError(f"n_bootstraps must be >= 10, got {n_bootstraps}")

        self.mean_block_length = float(mean_block_length)
        self.n_bootstraps = n_bootstraps
        self.seed = seed

    def generate_resampled_indices(self, n_observations: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generates resampled time indices [i_0, ..., i_{N-1}] under the Politis-Romano stationary scheme.
        """
        n = n_observations
        p = 1.0 / self.mean_block_length

        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(0, n)

        for t in range(1, n):
            if rng.random() < p:
                indices[t] = rng.integers(0, n)
            else:
                indices[t] = (indices[t - 1] + 1) % n

        return indices

    def resample_panel(
        self,
        returns_panel: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Synchronously resamples rows of a 2D panel (N observations x M assets).
        Preserves cross-sectional correlation across all M assets at time t.
        """
        if returns_panel.ndim == 1:
            returns_panel = returns_panel[:, np.newaxis]

        n_obs, n_assets = returns_panel.shape
        indices = self.generate_resampled_indices(n_obs, rng)
        return returns_panel[indices, :]

    def test_null_hypothesis(
        self,
        returns: Union[List[float], np.ndarray],
        statistic: str = "sharpe",
    ) -> BootstrapResult:
        """
        Tests null hypothesis H_0: E[r_t] <= 0 (zero or negative mean return) against H_1: E[r_t] > 0,
        using the specified test statistic (e.g. per-period Sharpe ratio).

        Procedure:
          1. Compute observed test statistic on raw returns.
          2. Impose the null by centering returns: r_tilde = r - mean(r).
          3. Generate B stationary bootstrap replicates of centered returns.
          4. Compute test statistic on each centered bootstrap replicate.
          5. Empirical p-value = (1 + count(bootstrap_stats >= obs_stat)) / (B + 1)
             using the Davison & Hinkley (1997) finite-sample corrected formula.
        """
        arr = np.asarray(returns, dtype=np.float64)
        if arr.ndim > 1:
            arr = arr.ravel()
        n = len(arr)
        if n < 2:
            raise ValueError(f"Insufficient observations for bootstrap: N={n}")

        rng = np.random.default_rng(self.seed)

        # Helper to compute statistic
        def _calc_stat(data: np.ndarray) -> float:
            m = np.mean(data)
            s = np.std(data, ddof=1)
            if statistic.lower() == "sharpe":
                return float(m / s) if s > 1e-12 else 0.0
            elif statistic.lower() == "mean":
                return float(m)
            else:
                raise ValueError(f"Unknown statistic: {statistic}. Choose 'sharpe' or 'mean'.")

        obs_stat = _calc_stat(arr)

        # Impose null hypothesis: center data to mean 0
        arr_centered = arr - np.mean(arr)

        bootstrap_stats = np.empty(self.n_bootstraps, dtype=np.float64)
        for b in range(self.n_bootstraps):
            indices = self.generate_resampled_indices(n, rng)
            sample_centered = arr_centered[indices]
            bootstrap_stats[b] = _calc_stat(sample_centered)

        # Calculate p-value: Davison & Hinkley (1997) finite-sample corrected formula
        exceedances = np.sum(bootstrap_stats >= obs_stat)
        p_val = float((1.0 + exceedances) / (self.n_bootstraps + 1.0))

        null_mean = float(np.mean(bootstrap_stats))
        null_std = float(np.std(bootstrap_stats, ddof=1))
        ci_lower = float(np.percentile(bootstrap_stats, 2.5))
        ci_upper = float(np.percentile(bootstrap_stats, 97.5))

        return BootstrapResult(
            observed_statistic=obs_stat,
            p_value=p_val,
            null_mean=null_mean,
            null_std=null_std,
            bootstrap_samples_count=self.n_bootstraps,
            mean_block_length=self.mean_block_length,
            confidence_interval_95=(ci_lower, ci_upper),
            statistic_name=statistic,
            seed=self.seed,
        )
