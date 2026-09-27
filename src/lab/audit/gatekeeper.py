"""
Statistical Gatekeeper: Automated Defense Against Overfitting and p-Hacking.
Synthesizes standard metrics, Deflated Sharpe Ratio, stationary bootstrap,
cost stress testing, and protected holdout evaluation into a unified audit verdict.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .bootstrap import BootstrapResult, StationaryBootstrap
from .cost_stress import CostStressResult, CostStressTester
from .dsr import DeflatedSharpeCalculator, DSRResult, SharpeUnit, TrialVarianceMethod
from .holdout_vault import ProtectedHoldoutVault
from .ledger import ExperimentLedger
from .metrics import MetricsCalculator, PerformanceMetrics


class GatekeeperVerdict(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True)
class GatekeeperCriteria:
    min_observations: int = 240         # Minimum 10 days for hourly bars
    min_cagr: float = 0.0               # Must have positive cumulative growth
    max_acceptable_drawdown: float = -0.30  # Max 30% drawdown
    min_dsr: float = 0.95               # Statistical confidence threshold under DSR model (not guarantee of live alpha)
    max_bootstrap_p_value: float = 0.05 # p <= 0.05 on centered stationary bootstrap (evidence against H0: E[r] <= 0)
    min_breakeven_bps: float = 10.0     # Must survive >= 10 bps one-way fee per side (20 bps round-trip on traded notional)
    warn_trial_count_k: int = 50        # Flag warning if K >= 50


@dataclass(frozen=True)
class StatisticalAuditReport:
    strategy_id: str
    verdict: GatekeeperVerdict
    passed: bool
    failure_reasons: List[str]
    warnings: List[str]
    metrics: PerformanceMetrics
    dsr_result: DSRResult                 # Conservative global DSR
    bootstrap_result: BootstrapResult
    cost_stress_result: CostStressResult
    holdout_metrics: Optional[PerformanceMetrics]
    multiple_testing_warning: bool
    plain_english_summary: str
    dsr_result_family: Optional[DSRResult] = None
    dsr_result_global: Optional[DSRResult] = None


class StatisticalGatekeeper:
    """
    Automated statistical referee preventing the autonomous agent from fooling itself.
    """

    def __init__(
        self,
        criteria: Optional[GatekeeperCriteria] = None,
        ledger: Optional[ExperimentLedger] = None,
        periods_per_year: int = MetricsCalculator.HOURLY_CRYPTO,
    ):
        self.criteria = criteria or GatekeeperCriteria()
        self.ledger = ledger
        self.periods_per_year = periods_per_year

    def audit_strategy(
        self,
        strategy_id: str,
        gross_returns: List[float],
        turnovers: List[float],
        hypothesis_family: str,
        k_trials_override: Optional[int] = None,
        trial_sharpes_override: Optional[List[float]] = None,
        holdout_vault: Optional[ProtectedHoldoutVault] = None,
        holdout_evaluator: Optional[Callable[[Any], PerformanceMetrics]] = None,
        bootstrap_seed: int = 42,
    ) -> StatisticalAuditReport:
        """
        Conducts a full statistical audit of a strategy candidate.
        """
        n = len(gross_returns)
        failures: List[str] = []
        warnings: List[str] = []

        # 1. Observation count check
        if n < self.criteria.min_observations:
            failures.append(
                f"Insufficient sample size: N={n} bars (minimum required: {self.criteria.min_observations})."
            )

        # 2. Baseline Performance Metrics (with 10 bps default fee applied)
        default_fee_rate = 0.0010  # 10 bps
        net_returns_10bps = [r - (turn * default_fee_rate) for r, turn in zip(gross_returns, turnovers)]

        if n >= 2:
            metrics = MetricsCalculator.calculate_metrics(
                returns=net_returns_10bps,
                periods_per_year=self.periods_per_year,
            )
        else:
            raise ValueError(f"Cannot compute metrics with N={n} < 2 observations.")

        if metrics.cagr <= self.criteria.min_cagr:
            failures.append(
                f"Net CAGR at 10 bps fee is non-positive: {metrics.cagr:.2%} (threshold: > {self.criteria.min_cagr:.2%})."
            )

        if metrics.max_drawdown < self.criteria.max_acceptable_drawdown:
            failures.append(
                f"Max drawdown exceeds limit: {metrics.max_drawdown:.2%} (limit: >= {self.criteria.max_acceptable_drawdown:.2%})."
            )

        # 3. Multiple Testing & Deflated Sharpe Ratio (DSR)
        # Determine family K and global research K from ledger or override
        if self.ledger is not None:
            k_family = max(1, self.ledger.get_trial_count_for_family(hypothesis_family))
            family_sharpes = self.ledger.get_trial_sharpes_for_family(hypothesis_family)
            k_global = max(k_family, self.ledger.get_global_trial_count())
            global_sharpes = self.ledger.get_global_trial_sharpes()
        else:
            k_family = max(1, k_trials_override or 1)
            family_sharpes = trial_sharpes_override
            k_global = k_family
            global_sharpes = family_sharpes

        dsr_result_family = DeflatedSharpeCalculator.calculate_dsr(
            observed_period_sharpe=metrics.period_sharpe,
            n_observations=n,
            skewness=metrics.skewness,
            kurtosis=metrics.kurtosis,
            k_trials=k_family,
            variance_method=TrialVarianceMethod.THEORETICAL_NULL,
            periods_per_year=self.periods_per_year,
            sharpe_unit=SharpeUnit.PERIOD,
        )

        dsr_result_global = DeflatedSharpeCalculator.calculate_dsr(
            observed_period_sharpe=metrics.period_sharpe,
            n_observations=n,
            skewness=metrics.skewness,
            kurtosis=metrics.kurtosis,
            k_trials=k_global,
            variance_method=TrialVarianceMethod.THEORETICAL_NULL,
            periods_per_year=self.periods_per_year,
            sharpe_unit=SharpeUnit.PERIOD,
        )

        # For conservative discovery gatekeeping, evaluate against global research K
        dsr_result = dsr_result_global

        multiple_testing_warning = False
        if k_global >= self.criteria.warn_trial_count_k:
            multiple_testing_warning = True
            warnings.append(
                f"High global trial count (K_global={k_global}, K_family={k_family} in '{hypothesis_family}'). "
                f"Nominal Sharpe {metrics.annualized_sharpe:.2f} requires strong deflated evidence."
            )

        if dsr_result.dsr_value < self.criteria.min_dsr:
            failures.append(
                f"Deflated Sharpe Ratio failed: DSR_global={dsr_result_global.dsr_value:.4f} "
                f"(DSR_family={dsr_result_family.dsr_value:.4f}, minimum required: {self.criteria.min_dsr:.4f} "
                f"given K_global={k_global} trials, expected max Sharpe={dsr_result_global.expected_max_annualized_sharpe:.2f})."
            )

        # 4. Stationary Block Bootstrap
        bootstrap = StationaryBootstrap(
            mean_block_length=24.0,
            n_bootstraps=1000,
            seed=bootstrap_seed,
        )
        bootstrap_result = bootstrap.test_null_hypothesis(
            returns=net_returns_10bps,
            statistic="sharpe",
        )

        if bootstrap_result.p_value > self.criteria.max_bootstrap_p_value:
            failures.append(
                f"Bootstrap test of H_0 (E[r_t] <= 0) failed: p-value={bootstrap_result.p_value:.4f} "
                f"(threshold: <= {self.criteria.max_bootstrap_p_value:.4f})."
            )

        # 5. Cost Stress Testing
        cost_tester = CostStressTester(periods_per_year=self.periods_per_year)
        cost_result = cost_tester.evaluate_returns_with_turnover(
            gross_returns=gross_returns,
            turnovers=turnovers,
        )

        if cost_result.breakeven_fee_bps < self.criteria.min_breakeven_bps:
            failures.append(
                f"Cost stress test failed: Breakeven fee is {cost_result.breakeven_fee_bps:.1f} bps "
                f"(minimum required: {self.criteria.min_breakeven_bps:.1f} bps)."
            )

        # 6. Protected Holdout Evaluation (if provided)
        holdout_metrics = None
        if holdout_vault is not None and holdout_evaluator is not None:
            if holdout_vault.is_locked:
                failures.append("Holdout vault is exhausted/locked; cannot conduct final validation.")
            else:
                holdout_metrics = holdout_vault.evaluate(
                    strategy_evaluator=holdout_evaluator,
                    experiment_id=strategy_id,
                    caller_info="statistical_gatekeeper",
                )
                if holdout_metrics.cagr <= 0.0 or holdout_metrics.annualized_sharpe <= 0.0:
                    failures.append(
                        f"Holdout out-of-sample test failed: OOS Sharpe={holdout_metrics.annualized_sharpe:.2f}, "
                        f"CAGR={holdout_metrics.cagr:.2%}."
                    )

        # Determine Verdict
        passed = len(failures) == 0
        if not passed:
            verdict = GatekeeperVerdict.FAIL
        elif len(warnings) > 0:
            verdict = GatekeeperVerdict.WARNING
        else:
            verdict = GatekeeperVerdict.PASS

        # Generate Plain English Summary
        summary_lines = [
            f"Gatekeeper Audit for '{strategy_id}': VERDICT = {verdict.value}.",
            f"Observed Annualized Sharpe: {metrics.annualized_sharpe:.2f} (period Sharpe: {metrics.period_sharpe:.4f}).",
            f"Trials: K_global={k_global}, K_family={k_family} | Expected max Sharpe under null: {dsr_result.expected_max_annualized_sharpe:.2f}.",
            f"Deflated Sharpe Ratio (DSR): {dsr_result.dsr_value:.4f} (pass: >={self.criteria.min_dsr:.2f}).",
            f"Stationary Bootstrap p-value: {bootstrap_result.p_value:.4f} (pass: <={self.criteria.max_bootstrap_p_value:.2f}).",
            f"Cost Stress Breakeven: {cost_result.breakeven_fee_bps:.1f} bps (pass: >={self.criteria.min_breakeven_bps:.1f} bps).",
        ]
        if holdout_metrics:
            summary_lines.append(
                f"Air-gapped OOS Holdout: Sharpe={holdout_metrics.annualized_sharpe:.2f}, CAGR={holdout_metrics.cagr:.2%}."
            )
        if failures:
            summary_lines.append(f"Failures ({len(failures)}): " + " | ".join(failures))
        if warnings:
            summary_lines.append(f"Warnings ({len(warnings)}): " + " | ".join(warnings))
        summary_lines.append(
            "Statistical Note: Passing gatekeeper hurdles provides evidence against evaluated null hypotheses under historical data; "
            "it is not a guarantee of live trading profitability."
        )

        return StatisticalAuditReport(
            strategy_id=strategy_id,
            verdict=verdict,
            passed=passed,
            failure_reasons=failures,
            warnings=warnings,
            metrics=metrics,
            dsr_result=dsr_result,
            bootstrap_result=bootstrap_result,
            cost_stress_result=cost_result,
            holdout_metrics=holdout_metrics,
            multiple_testing_warning=multiple_testing_warning,
            plain_english_summary="\n".join(summary_lines),
            dsr_result_family=dsr_result_family,
            dsr_result_global=dsr_result_global,
        )
