"""
Protected Holdout Vault for Air-Gapped Out-of-Sample Evaluation.
Enforces a finite, configurable query budget to prevent adaptive overfitting
and p-hacking on untouched holdout datasets.
"""
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional

from .metrics import PerformanceMetrics


class HoldoutVaultExhaustedError(RuntimeError):
    """Raised when evaluation is attempted on an exhausted holdout vault."""
    pass


@dataclass(frozen=True)
class HoldoutAccessRecord:
    timestamp_utc: int
    experiment_id: str
    caller_info: str
    evaluations_remaining_after: int
    annualized_sharpe: float
    cagr: float


class ProtectedHoldoutVault:
    """
    Air-gapped holdout evaluator with strict, finite query budget.
    Ensures researchers cannot repeatedly query out-of-sample data.
    """

    def __init__(
        self,
        holdout_data: Any,
        initial_budget: int = 5,
        vault_id: str = "vault_primary",
    ):
        if initial_budget < 1:
            raise ValueError(f"initial_budget must be >= 1, got {initial_budget}")

        self._holdout_data = holdout_data  # Kept private
        self.initial_budget = int(initial_budget)
        self.evaluations_remaining = int(initial_budget)
        self.vault_id = str(vault_id)
        self._access_log: List[HoldoutAccessRecord] = []

    @property
    def is_locked(self) -> bool:
        return self.evaluations_remaining <= 0

    @property
    def access_log(self) -> List[HoldoutAccessRecord]:
        return list(self._access_log)

    def evaluate(
        self,
        strategy_evaluator: Callable[[Any], PerformanceMetrics],
        experiment_id: str,
        caller_info: str = "autonomous_agent",
    ) -> PerformanceMetrics:
        """
        Executes an evaluation of a strategy on the private holdout dataset.
        Consumes exactly 1 evaluation from the remaining budget.
        Raises HoldoutVaultExhaustedError if the budget is 0.
        """
        if self.is_locked:
            raise HoldoutVaultExhaustedError(
                f"Holdout vault '{self.vault_id}' budget exhausted ({self.initial_budget}/{self.initial_budget} used). "
                "Further evaluation is permanently locked."
            )

        self.evaluations_remaining -= 1
        now_utc = int(time.time() * 1_000_000)

        # Execute evaluation inside vault boundary
        metrics = strategy_evaluator(self._holdout_data)

        # Record access in immutable log
        record = HoldoutAccessRecord(
            timestamp_utc=now_utc,
            experiment_id=experiment_id,
            caller_info=caller_info,
            evaluations_remaining_after=self.evaluations_remaining,
            annualized_sharpe=metrics.annualized_sharpe,
            cagr=metrics.cagr,
        )
        self._access_log.append(record)

        return metrics
