"""
Deflated Sharpe Ratio (DSR) Implementation.
Reference: Bailey, D. H., & López de Prado, M. (2014).
"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
Journal of Portfolio Management, 40(5), 94-107.
"""
from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional
import numpy as np
from scipy.special import ndtri
from scipy.stats import norm


EULER_MASCHERONI = 0.577215664901532860606512090082402431042


class TrialVarianceMethod(str, Enum):
    """
    Explicit methodology for determining sigma_trials in the expected maximum Sharpe formula.
    """
    EMPIRICAL_TRIALS = "EMPIRICAL_TRIALS"    # Sample std across >= 2 observed trial Sharpes
    THEORETICAL_NULL = "THEORETICAL_NULL"    # Asymptotic standard error under null: 1 / sqrt(N - 1)
    FIXED_BENCHMARK = "FIXED_BENCHMARK"      # User-supplied explicit reference value (for literature replication)


class SharpeUnit(str, Enum):
    """
    Explicit unit specification for Sharpe ratios.
    Eliminates ambiguity between per-period (e.g. hourly) and annualized statistics.
    """
    PERIOD = "PERIOD"          # Per-period (e.g., hourly) Sharpe ratio: mean(r_t) / std(r_t)
    ANNUALIZED = "ANNUALIZED"  # Annualized Sharpe ratio: SR_period * sqrt(periods_per_year)


@dataclass(frozen=True)
class DSRResult:
    observed_period_sharpe: float
    observed_annualized_sharpe: float
    expected_max_period_sharpe: float
    expected_max_annualized_sharpe: float
    dsr_value: float              # Statistical confidence under DSR model (between 0.0 and 1.0)
    psr_value: float              # Probabilistic Sharpe Ratio against benchmark SR_0
    k_trials: int
    n_observations: int
    skewness: float
    kurtosis: float               # Pearson kurtosis (normal = 3.0)
    se_sharpe: float              # Standard error of Sharpe ratio
    sigma_trials: float           # Value of sigma_trials used for SR*
    trial_variance_method: TrialVarianceMethod  # Methodology used to determine sigma_trials
    effective_trial_rank: Optional[float] = None  # Diagnostic participation ratio (if matrix provided)
    z_score: Optional[float] = None               # Test statistic (observed_period_sharpe - expected_max_period_sharpe) / se_sharpe


class DeflatedSharpeCalculator:
    """
    Computes the Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR)
    under the Bailey & López de Prado (2014) framework.

    CRITICAL INVARIANT:
    All core estimations and test statistics are computed on PER-PERIOD returns
    and per-period Sharpe ratios. Annualization is applied only for reporting.

    METHODOLOGY NOTE:
    DSR is a statistical hurdle that quantifies the probability that the observed
    Sharpe ratio exceeds the expected maximum Sharpe ratio under the null hypothesis
    given K trials and non-normal return characteristics. Passing DSR is a necessary
    statistical condition, NOT an economic proof or guarantee of live alpha.
    """

    @staticmethod
    def calculate_expected_max_sharpe(
        k_trials: int,
        trial_sharpe_std: float,
        benchmark_period_sharpe: float = 0.0,
    ) -> float:
        """
        Calculates the expected maximum Sharpe ratio among K independent trials under the null.
        Formula (Bailey & López de Prado 2014, Eq. 8):
        SR* = benchmark + sigma_trials * [ (1 - gamma) * Phi^{-1}(1 - 1/K) + gamma * Phi^{-1}(1 - 1/(K*e)) ]
        """
        if k_trials < 1:
            raise ValueError(f"k_trials must be >= 1, got {k_trials}")
        if k_trials == 1:
            return float(benchmark_period_sharpe)
        if trial_sharpe_std < 0:
            raise ValueError(f"trial_sharpe_std must be >= 0, got {trial_sharpe_std}")
        if trial_sharpe_std < 1e-12:
            return float(benchmark_period_sharpe)

        gamma = EULER_MASCHERONI
        e = math.e

        p1 = 1.0 - (1.0 / k_trials)
        p2 = 1.0 - (1.0 / (k_trials * e))

        # Numerical bounds for inverse CDF
        p1 = min(max(p1, 1e-15), 1.0 - 1e-15)
        p2 = min(max(p2, 1e-15), 1.0 - 1e-15)

        z1 = float(ndtri(p1))
        z2 = float(ndtri(p2))

        expected_max = benchmark_period_sharpe + trial_sharpe_std * ((1.0 - gamma) * z1 + gamma * z2)
        return expected_max

    @classmethod
    def calculate_dsr(
        cls,
        observed_period_sharpe: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
        k_trials: int = 1,
        trial_sharpe_std: Optional[float] = None,
        trial_sharpes: Optional[List[float]] = None,
        variance_method: Optional[TrialVarianceMethod] = None,
        benchmark_period_sharpe: float = 0.0,
        periods_per_year: int = 8760,
        trial_returns_matrix: Optional[np.ndarray] = None,
        sharpe_unit: SharpeUnit = SharpeUnit.PERIOD,
        trial_sharpes_unit: SharpeUnit = SharpeUnit.PERIOD,
    ) -> DSRResult:
        """
        Calculates the Deflated Sharpe Ratio.

        Parameters:
            observed_period_sharpe: Per-period estimated Sharpe ratio (mu / sigma).
            n_observations: Number of return observations (N >= 2).
            skewness: Sample skewness (gamma_3).
            kurtosis: Pearson kurtosis (gamma_4, normal = 3.0).
            k_trials: Number of trials executed (K >= 1).
            trial_sharpe_std: Explicit standard deviation of period Sharpes (for FIXED_BENCHMARK).
            trial_sharpes: Explicit list of period Sharpes across trials (for EMPIRICAL_TRIALS).
            variance_method: Methodology enum (EMPIRICAL_TRIALS, THEORETICAL_NULL, FIXED_BENCHMARK).
            benchmark_period_sharpe: Benchmark period Sharpe under null (SR_0, default 0.0).
            periods_per_year: Annualization factor (default 8760 for 1h crypto).
            trial_returns_matrix: Optional (N x K) matrix of trial returns to compute diagnostic effective rank.
            sharpe_unit: Explicit unit of observed_period_sharpe (must be SharpeUnit.PERIOD).
            trial_sharpes_unit: Explicit unit of trial_sharpes (must be SharpeUnit.PERIOD).
        """
        if sharpe_unit == SharpeUnit.ANNUALIZED:
            raise ValueError(
                f"DeflatedSharpeCalculator.calculate_dsr requires per-period Sharpe (sharpe_unit=SharpeUnit.PERIOD). "
                f"Received sharpe_unit=SharpeUnit.ANNUALIZED. Convert upstream using: "
                f"observed_period_sharpe = observed_annualized_sharpe / math.sqrt(periods_per_year)."
            )
        elif sharpe_unit != SharpeUnit.PERIOD:
            raise ValueError(f"Unknown SharpeUnit: {sharpe_unit}. Must be SharpeUnit.PERIOD.")

        if trial_sharpes is not None and trial_sharpes_unit == SharpeUnit.ANNUALIZED:
            raise ValueError(
                f"DeflatedSharpeCalculator.calculate_dsr requires trial_sharpes in per-period units (trial_sharpes_unit=SharpeUnit.PERIOD). "
                f"Received trial_sharpes_unit=SharpeUnit.ANNUALIZED. Convert upstream using: "
                f"[s / math.sqrt(periods_per_year) for s in trial_sharpes]."
            )

        if n_observations < 2:
            raise ValueError(f"Insufficient observations: N={n_observations} (minimum 2 required).")
        if k_trials < 1:
            raise ValueError(f"k_trials must be >= 1, got {k_trials}")

        # Compute standard error of Sharpe ratio (Mertens 1996 / Lo 2002)
        # Var(SR) = (1 / (N - 1)) * (1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2)
        sr = observed_period_sharpe
        var_term = 1.0 - (skewness * sr) + (((kurtosis - 1.0) / 4.0) * (sr ** 2))
        if var_term < 1e-12:
            var_term = 1e-12
        se_sr = math.sqrt(var_term / (n_observations - 1))

        # Resolve Trial Variance Methodology Unambiguously
        if variance_method is None:
            if trial_sharpes is not None and len(trial_sharpes) >= 2:
                variance_method = TrialVarianceMethod.EMPIRICAL_TRIALS
            elif trial_sharpe_std is not None:
                variance_method = TrialVarianceMethod.FIXED_BENCHMARK
            else:
                variance_method = TrialVarianceMethod.THEORETICAL_NULL

        if variance_method == TrialVarianceMethod.FIXED_BENCHMARK:
            if trial_sharpe_std is None:
                raise ValueError("FIXED_BENCHMARK requires explicit trial_sharpe_std parameter.")
            computed_std = float(trial_sharpe_std)
        elif variance_method == TrialVarianceMethod.EMPIRICAL_TRIALS:
            if trial_sharpes is None or len(trial_sharpes) < 2:
                raise ValueError("EMPIRICAL_TRIALS requires trial_sharpes list with at least 2 entries.")
            computed_std = float(np.std(trial_sharpes, ddof=1))
            k_trials = max(k_trials, len(trial_sharpes))
        elif variance_method == TrialVarianceMethod.THEORETICAL_NULL:
            # Under null of i.i.d. returns with true mean 0:
            # Var(SR_period) = 1 / (N - 1), so standard deviation sigma_trials = 1 / sqrt(N - 1)
            computed_std = 1.0 / math.sqrt(n_observations - 1)
        else:
            raise ValueError(f"Unknown TrialVarianceMethod: {variance_method}")

        # Expected maximum Sharpe under null among K trials
        expected_max_period = cls.calculate_expected_max_sharpe(
            k_trials=k_trials,
            trial_sharpe_std=computed_std,
            benchmark_period_sharpe=benchmark_period_sharpe,
        )

        # Probabilistic Sharpe Ratio (PSR) against benchmark SR_0
        if se_sr > 1e-12:
            psr_stat = (sr - benchmark_period_sharpe) / se_sr
            psr_value = float(norm.cdf(psr_stat))
            # Deflated Sharpe Ratio (DSR) against expected maximum SR*
            dsr_stat = (sr - expected_max_period) / se_sr
            dsr_value = float(norm.cdf(dsr_stat))
            z_score = float(dsr_stat)
        else:
            psr_value = 1.0 if sr >= benchmark_period_sharpe else 0.0
            dsr_value = 1.0 if sr >= expected_max_period else 0.0
            z_score = 0.0 if sr == expected_max_period else (float('inf') if sr > expected_max_period else float('-inf'))

        # Optional Diagnostic: Effective Trial Rank (Participation Ratio)
        effective_rank = None
        if trial_returns_matrix is not None and trial_returns_matrix.shape[1] > 1:
            effective_rank = cls.calculate_effective_trial_rank(trial_returns_matrix)

        ann_factor = math.sqrt(periods_per_year)
        return DSRResult(
            observed_period_sharpe=sr,
            observed_annualized_sharpe=sr * ann_factor,
            expected_max_period_sharpe=expected_max_period,
            expected_max_annualized_sharpe=expected_max_period * ann_factor,
            dsr_value=dsr_value,
            psr_value=psr_value,
            k_trials=k_trials,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis,
            se_sharpe=se_sr,
            sigma_trials=computed_std,
            trial_variance_method=variance_method,
            effective_trial_rank=effective_rank,
            z_score=z_score,
        )

    @staticmethod
    def calculate_effective_trial_rank(trial_returns_matrix: np.ndarray) -> float:
        """
        Computes the participation ratio / effective rank of the trial correlation matrix:
        N_eff = (tr(C))^2 / tr(C^2) = K^2 / sum(C_{ij}^2).

        NOTE: This is strictly an exploratory diagnostic to measure trial redundancy.
        It is NOT a direct substitute for the formal K in the DSR calculation.
        """
        if trial_returns_matrix.ndim != 2:
            raise ValueError("trial_returns_matrix must be a 2D array (N observations x K trials)")
        n_obs, k_cols = trial_returns_matrix.shape
        if k_cols < 2:
            return float(k_cols)

        # Correlation matrix
        corr = np.corrcoef(trial_returns_matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        # sum of squared elements
        sum_sq = float(np.sum(corr ** 2))
        if sum_sq < 1e-12:
            return float(k_cols)

        effective_rank = (k_cols ** 2) / sum_sq
        return float(min(max(effective_rank, 1.0), float(k_cols)))
