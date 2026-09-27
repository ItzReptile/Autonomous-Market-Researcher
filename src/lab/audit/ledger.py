"""
Experiment Ledger and Multiple-Testing Trial Accounting.
Ensures every trial is logged in an append-only ledger and tracks trial count K
for Deflated Sharpe Ratio calculation to prevent multiple testing inflation.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional


class EvaluationTier(str, Enum):
    EXPLORATORY = "EXPLORATORY"
    VALIDATION = "VALIDATION"
    FINAL_HOLDOUT = "FINAL_HOLDOUT"


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    parent_experiment_id: Optional[str]
    hypothesis_family: str
    parameter_changes: Dict[str, Any]
    trial_number: int
    cumulative_family_trials: int     # K within hypothesis family
    cumulative_global_trials: int     # Global K across all discovery trials
    dataset_id: str
    code_git_commit: str
    environment_hash: str
    evaluation_tier: EvaluationTier
    timestamp_utc: int
    observed_sharpe: float            # Annualized Sharpe (maintained for backwards compatibility)
    dsr_value: float
    metadata: Dict[str, Any]
    economic_mechanism_id: Optional[str] = None
    observed_period_sharpe: Optional[float] = None
    observed_annualized_sharpe: Optional[float] = None


class ExperimentLedger:
    """
    Append-only experiment ledger.
    Tracks experiment lineage and prevents the researcher from resetting K.
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path
        self._records: List[ExperimentRecord] = []
        if self.ledger_path and self.ledger_path.exists():
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        self._records.clear()
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                record = ExperimentRecord(
                    experiment_id=d["experiment_id"],
                    parent_experiment_id=d.get("parent_experiment_id"),
                    hypothesis_family=d["hypothesis_family"],
                    parameter_changes=d.get("parameter_changes", {}),
                    trial_number=d["trial_number"],
                    cumulative_family_trials=d["cumulative_family_trials"],
                    cumulative_global_trials=d.get("cumulative_global_trials", len(self._records) + 1),
                    dataset_id=d["dataset_id"],
                    code_git_commit=d.get("code_git_commit", "unknown"),
                    environment_hash=d.get("environment_hash", "unknown"),
                    evaluation_tier=EvaluationTier(d["evaluation_tier"]),
                    timestamp_utc=d["timestamp_utc"],
                    observed_sharpe=d.get("observed_sharpe", 0.0),
                    dsr_value=d.get("dsr_value", 0.0),
                    metadata=d.get("metadata", {}),
                    economic_mechanism_id=d.get("economic_mechanism_id"),
                    observed_period_sharpe=d.get("observed_period_sharpe"),
                    observed_annualized_sharpe=d.get("observed_annualized_sharpe", d.get("observed_sharpe")),
                )
                self._records.append(record)

    def record_experiment(
        self,
        hypothesis_family: str,
        parameter_changes: Dict[str, Any],
        dataset_id: str,
        evaluation_tier: EvaluationTier = EvaluationTier.EXPLORATORY,
        parent_experiment_id: Optional[str] = None,
        code_git_commit: str = "dev",
        environment_hash: str = "unknown",
        observed_sharpe: float = 0.0,
        dsr_value: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        custom_experiment_id: Optional[str] = None,
        economic_mechanism_id: Optional[str] = None,
        observed_period_sharpe: Optional[float] = None,
        observed_annualized_sharpe: Optional[float] = None,
    ) -> ExperimentRecord:
        """
        Appends a new experiment record. Automatically advances trial_number, cumulative family K, and cumulative global K.
        """
        prior_family_records = [r for r in self._records if r.hypothesis_family == hypothesis_family]
        trial_number = len(prior_family_records) + 1
        cumulative_k_family = trial_number
        cumulative_k_global = len(self._records) + 1

        if observed_annualized_sharpe is not None and observed_period_sharpe is None:
            observed_period_sharpe = observed_annualized_sharpe / math.sqrt(8760)
        elif observed_period_sharpe is not None and observed_annualized_sharpe is None:
            observed_annualized_sharpe = observed_period_sharpe * math.sqrt(8760)

        if observed_annualized_sharpe is not None:
            observed_sharpe = observed_annualized_sharpe
        elif observed_sharpe != 0.0 and observed_period_sharpe is None:
            observed_period_sharpe = observed_sharpe / math.sqrt(8760)
            observed_annualized_sharpe = observed_sharpe

        now_utc = int(time.time() * 1_000_000)
        if custom_experiment_id:
            exp_id = custom_experiment_id
        else:
            id_payload = f"{hypothesis_family}:{trial_number}:{now_utc}:{json.dumps(parameter_changes, sort_keys=True)}"
            exp_id = f"exp_{hashlib.sha256(id_payload.encode('utf-8')).hexdigest()[:16]}"

        record = ExperimentRecord(
            experiment_id=exp_id,
            parent_experiment_id=parent_experiment_id,
            hypothesis_family=hypothesis_family,
            parameter_changes=parameter_changes,
            trial_number=trial_number,
            cumulative_family_trials=cumulative_k_family,
            cumulative_global_trials=cumulative_k_global,
            dataset_id=dataset_id,
            code_git_commit=code_git_commit,
            environment_hash=environment_hash,
            evaluation_tier=evaluation_tier,
            timestamp_utc=now_utc,
            observed_sharpe=observed_sharpe,
            dsr_value=dsr_value,
            metadata=metadata or {},
            economic_mechanism_id=economic_mechanism_id,
            observed_period_sharpe=observed_period_sharpe,
            observed_annualized_sharpe=observed_annualized_sharpe,
        )

        self._records.append(record)

        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a", encoding="utf-8") as f:
                d = asdict(record)
                d["evaluation_tier"] = record.evaluation_tier.value
                f.write(json.dumps(d) + "\n")

        return record

    def get_trial_count_for_family(self, hypothesis_family: str) -> int:
        return sum(1 for r in self._records if r.hypothesis_family == hypothesis_family)

    def get_trial_sharpes_for_family(self, hypothesis_family: str) -> List[float]:
        return [r.observed_sharpe for r in self._records if r.hypothesis_family == hypothesis_family]

    def get_trial_period_sharpes_for_family(self, hypothesis_family: str) -> List[float]:
        return [
            r.observed_period_sharpe if r.observed_period_sharpe is not None
            else (r.observed_sharpe / math.sqrt(8760))
            for r in self._records if r.hypothesis_family == hypothesis_family
        ]

    def get_global_trial_count(self) -> int:
        """Returns the total number of discovery trials across all families."""
        return len(self._records)

    def get_global_trial_sharpes(self) -> List[float]:
        """Returns observed annualized Sharpes across all discovery trials in the ledger."""
        return [r.observed_sharpe for r in self._records]

    def get_global_trial_period_sharpes(self) -> List[float]:
        """Returns observed per-period Sharpes across all discovery trials in the ledger."""
        return [
            r.observed_period_sharpe if r.observed_period_sharpe is not None
            else (r.observed_sharpe / math.sqrt(8760))
            for r in self._records
        ]

    def get_trial_count_for_mechanism(self, economic_mechanism_id: str) -> int:
        """Returns the number of trials exploring a specific economic mechanism."""
        return sum(1 for r in self._records if r.economic_mechanism_id == economic_mechanism_id)

    def get_records(self) -> List[ExperimentRecord]:
        return list(self._records)
