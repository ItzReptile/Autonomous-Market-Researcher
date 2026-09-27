"""
Audit Package: Statistical Gatekeeper and Overfitting Defense.
"""
from .metrics import PerformanceMetrics, MetricsCalculator
from .dsr import DeflatedSharpeCalculator, DSRResult, TrialVarianceMethod
from .ledger import ExperimentLedger, ExperimentRecord, EvaluationTier
from .bootstrap import StationaryBootstrap, BootstrapResult
from .cost_stress import CostStressTester, CostStressResult, CostPointResult
from .holdout_vault import ProtectedHoldoutVault, HoldoutVaultExhaustedError
from .gatekeeper import StatisticalGatekeeper, StatisticalAuditReport, GatekeeperVerdict

__all__ = [
    "PerformanceMetrics",
    "MetricsCalculator",
    "DeflatedSharpeCalculator",
    "DSRResult",
    "TrialVarianceMethod",
    "ExperimentLedger",
    "ExperimentRecord",
    "EvaluationTier",
    "StationaryBootstrap",
    "BootstrapResult",
    "CostStressTester",
    "CostStressResult",
    "CostPointResult",
    "ProtectedHoldoutVault",
    "HoldoutVaultExhaustedError",
    "StatisticalGatekeeper",
    "StatisticalAuditReport",
    "GatekeeperVerdict",
]
