"""
BEHAVIORAL ADAPTATION TRACKER (Phase 10 / RESEARCHER-V5).
Directly measures whether Qwen actually changes its empirical research behavior
after experiencing failures and falsifications across consecutive cycles (N -> N+1).
Strict invariant: Textual acknowledgment alone does NOT count as behavioral adaptation.
Tracks:
  1. Avoidance of previously falsified hypotheses
  2. Family and mechanism pivots
  3. Causal premise and regime adjustments
  4. Cadence/turnover friction adaptations
  5. Termination of uninformative repetitions
  6. Selection of discriminative experiments
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..experiments.dsl import HypothesisProposal, StrategyType
from ..experiments.loop import IterationCycleRecord


@dataclass
class CycleAdaptationRecord:
    prior_cycle_num: int
    current_cycle_num: int
    prior_failure_mode: str
    prior_family: str
    current_family: str
    avoided_falsified_hypothesis: bool
    family_pivoted: bool
    regime_premise_adjusted: bool
    friction_adapted: bool
    stopped_repetition: bool
    adaptation_score: float  # 0.0 to 1.0
    adaptation_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehavioralAdaptationTracker:
    """
    Evaluates empirical research adaptation across consecutive cycles.
    """

    @classmethod
    def evaluate_cycle_transition(
        cls,
        prior_cycle: IterationCycleRecord,
        curr_cycle: IterationCycleRecord,
    ) -> Optional[CycleAdaptationRecord]:
        """
        Evaluates behavioral change from prior cycle to current cycle if prior cycle failed.
        """
        # Only evaluate if prior cycle experienced a failure or falsification
        prior_failed = not prior_cycle.passed_gatekeeper or (
            prior_cycle.claim_empirically_supported is not None and not prior_cycle.claim_empirically_supported
        )
        if not prior_failed:
            return None

        p_prop = prior_cycle.proposal
        c_prop = curr_cycle.proposal

        # Identify prior failure mode
        p_reasons = []
        if prior_cycle.run_result and prior_cycle.run_result.audit_report_out_of_sample:
            p_reasons = prior_cycle.run_result.audit_report_out_of_sample.failure_reasons
        elif prior_cycle.review and prior_cycle.review.critique_notes:
            p_reasons = prior_cycle.review.critique_notes

        fail_str = " ".join(p_reasons).lower()
        is_fee_fail = "fee" in fail_str or "cost" in fail_str or (
            prior_cycle.run_result
            and prior_cycle.run_result.audit_report_out_of_sample
            and prior_cycle.run_result.audit_report_out_of_sample.cost_stress_result.breakeven_fee_bps < 10.0
        )

        # 1. Avoided Falsified Hypothesis
        p_fam = p_prop.hypothesis_family
        c_fam = c_prop.hypothesis_family
        p_params = p_prop.parameters or {}
        c_params = c_prop.parameters or {}
        exact_param_match = (p_prop.strategy_type == c_prop.strategy_type and p_params == c_params)
        avoided_falsified = (p_fam != c_fam or not exact_param_match)

        # 2. Family Pivoted
        fam_pivoted = (p_fam != c_fam) or (p_prop.strategy_type != c_prop.strategy_type)

        # 3. Regime / Premise Adjusted
        p_regime = p_prop.target_market_regime or "all_market_regimes"
        c_regime = c_prop.target_market_regime or "all_market_regimes"
        regime_adjusted = (p_regime != c_regime) or (p_prop.economic_mechanism_id != c_prop.economic_mechanism_id)

        # 4. Friction Adapted (If prior failed fee drag, did current increase rebalance or buffer?)
        p_reb = p_params.get("rebalance_interval_bars", 12)
        c_reb = c_params.get("rebalance_interval_bars", 12)
        p_cash = p_prop.risk_rules.cash_buffer if p_prop.risk_rules else 0.0
        c_cash = c_prop.risk_rules.cash_buffer if c_prop.risk_rules else 0.0
        if is_fee_fail:
            friction_adapted = (c_reb > p_reb) or (c_cash > p_cash) or (c_reb >= 24) or (c_cash >= 0.10)
        else:
            friction_adapted = True

        # 5. Stopped Uninformative Repetition
        stopped_rep = not exact_param_match and (p_prop.economic_mechanism_id != c_prop.economic_mechanism_id or fam_pivoted)

        # Compute composite adaptation score (0.0 to 1.0)
        criteria = [avoided_falsified, fam_pivoted, regime_adjusted, friction_adapted, stopped_rep]
        adaptation_score = round(sum(1.0 for c in criteria if c) / len(criteria), 2)

        notes_parts = []
        if avoided_falsified:
            notes_parts.append("Avoided repeat of falsified parameter tuple")
        if fam_pivoted:
            notes_parts.append(f"Pivoted family ({p_fam} -> {c_fam})")
        if is_fee_fail and friction_adapted:
            notes_parts.append(f"Adapted cadence to counter fee drag (reb: {p_reb} -> {c_reb}, cash: {p_cash:.0%} -> {c_cash:.0%})")
        if regime_adjusted:
            notes_parts.append(f"Conditioned on distinct premise/regime ({c_regime})")

        return CycleAdaptationRecord(
            prior_cycle_num=prior_cycle.iteration,
            current_cycle_num=curr_cycle.iteration,
            prior_failure_mode="Fee Drag / Cost Stress" if is_fee_fail else "Gatekeeper Rejection",
            prior_family=p_fam,
            current_family=c_fam,
            avoided_falsified_hypothesis=avoided_falsified,
            family_pivoted=fam_pivoted,
            regime_premise_adjusted=regime_adjusted,
            friction_adapted=friction_adapted,
            stopped_repetition=stopped_rep,
            adaptation_score=adaptation_score,
            adaptation_notes="; ".join(notes_parts) or "No material adaptation detected",
        )

    @classmethod
    def audit_program_adaptation(cls, cycles: List[IterationCycleRecord]) -> Dict[str, Any]:
        records: List[CycleAdaptationRecord] = []
        for i in range(len(cycles) - 1):
            prior_c = cycles[i]
            curr_c = cycles[i + 1]
            rec = cls.evaluate_cycle_transition(prior_c, curr_c)
            if rec:
                records.append(rec)

        if not records:
            return {
                "total_failure_transitions": 0,
                "mean_adaptation_score": 1.0,
                "behavioral_adaptation_rate_pct": 100.0,
                "adaptation_records": [],
            }

        total = len(records)
        adapted_count = sum(1 for r in records if r.adaptation_score >= 0.60)
        mean_score = sum(r.adaptation_score for r in records) / total
        adapt_rate = (adapted_count / total) * 100.0

        return {
            "total_failure_transitions": total,
            "adapted_transitions_count": adapted_count,
            "mean_adaptation_score": round(mean_score, 2),
            "behavioral_adaptation_rate_pct": round(adapt_rate, 2),
            "adaptation_records": [r.to_dict() for r in records],
        }
