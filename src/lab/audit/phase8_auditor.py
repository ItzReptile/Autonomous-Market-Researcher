"""
Phase 8 Longitudinal Researcher Learning Auditor.
Computes comprehensive cognitive and empirical learning scorecards:
  * Evidence-Conditioned Adaptation Rate
  * Failure-Family Learning & Working Belief Updates
  * Cross-Archetype Shared Failure Inference Rate
  * Temporal Prediction Calibration (Early, Middle, Late epochs)
  * Mistake Recurrence and Correction Rates
  * Attractor Classification
  * Authoritative Level 2+ Gate Promotion Verification
"""
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..experiments.loop import IterationCycleRecord, ResearchLoopSummary
from ..experiments.longitudinal_learner import (
    AdaptationLevel,
    EvidenceConditionedClassifier,
    FailureFamilyTracker,
    SharedFailureInferenceDetector,
    MistakeTracker,
    AttractorClassifier,
)


@dataclass
class Phase8Scorecard:
    timestamp: float
    total_cycles: int
    researcher_id: str
    evidence_adaptation_summary: Dict[str, Any]
    failure_family_summary: Dict[str, Any]
    shared_failure_inferences: List[Dict[str, Any]]
    temporal_calibration_epochs: Dict[str, Any]
    mistake_recurrence_summary: Dict[str, Any]
    attractor_summary: Dict[str, Any]
    provenance_distribution: Dict[str, int]
    gate_promotion_audit: Dict[str, Any]
    final_verdict: str
    verdict_rationale: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class Phase8Auditor:
    """
    Evaluates longitudinal research performance against pre-registered Level 2+ promotion thresholds.
    """

    @classmethod
    def audit_loop(
        cls,
        summary: ResearchLoopSummary,
        researcher_id: str = "RESEARCHER-V1",
        vault_queries_spent: int = 0,
        human_interventions: int = 0,
    ) -> Phase8Scorecard:
        cycles = summary.cycles
        total = len(cycles)

        # 1. Evidence-Conditioned Adaptation
        adaptation_results = []
        scores = []
        for i, c in enumerate(cycles):
            prior = cycles[:i]
            res = EvidenceConditionedClassifier.classify(
                proposal=c.proposal,
                worker_reasoning=c.worker_reasoning,
                prior_cycles=prior,
            )
            adaptation_results.append(res)
            scores.append(res.score)

        level_counts = {
            AdaptationLevel.LEVEL_1_INDEPENDENT.value: sum(1 for r in adaptation_results if r.level == AdaptationLevel.LEVEL_1_INDEPENDENT),
            AdaptationLevel.LEVEL_2_AWARE_NON_DISCRIMINATIVE.value: sum(1 for r in adaptation_results if r.level == AdaptationLevel.LEVEL_2_AWARE_NON_DISCRIMINATIVE),
            AdaptationLevel.LEVEL_3_CONDITIONED.value: sum(1 for r in adaptation_results if r.level == AdaptationLevel.LEVEL_3_CONDITIONED),
            AdaptationLevel.LEVEL_4_CAUSAL_CHANGE.value: sum(1 for r in adaptation_results if r.level == AdaptationLevel.LEVEL_4_CAUSAL_CHANGE),
        }
        l3_l4_count = level_counts[AdaptationLevel.LEVEL_3_CONDITIONED.value] + level_counts[AdaptationLevel.LEVEL_4_CAUSAL_CHANGE.value]
        l3_l4_pct = (l3_l4_count / total) * 100.0 if total > 0 else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0
        normalized_score = (mean_score / 4.0) * 100.0

        evidence_summary = {
            "mean_score": mean_score,
            "composite_adaptation_pct": normalized_score,
            "level_counts": level_counts,
            "conditioned_or_causal_pct": l3_l4_pct,
        }

        # 2. Failure Family Learning
        tracker = FailureFamilyTracker()
        post_failure_actions = []
        for i, c in enumerate(cycles):
            act = tracker.classify_post_failure_action(c.proposal, cycles[:i])
            post_failure_actions.append(act)
            tracker.update_from_cycle(c)

        fam_dict = {name: f.to_dict() for name, f in tracker.families.items()}
        abandoned_count = sum(1 for f in tracker.families.values() if f.is_exhausted)
        family_summary = {
            "families": fam_dict,
            "exhausted_families_count": abandoned_count,
            "post_failure_actions": post_failure_actions,
        }

        # 3. Shared Failure Inference Detection
        shared_inferences = []
        for i, c in enumerate(cycles):
            s_res = SharedFailureInferenceDetector.detect(c.proposal, c.worker_reasoning, cycles[:i])
            if s_res.identified:
                shared_inferences.append({
                    "cycle": c.iteration,
                    "archetypes": s_res.archetypes_involved,
                    "evidence": s_res.shared_evidence,
                    "explanation": s_res.common_causal_explanation,
                })

        # 4. Temporal Prediction Calibration Epochs
        # Epoch 1: 1-10, Epoch 2: 11-20, Epoch 3: 21+
        epoch_1 = [c for c in cycles if 1 <= c.iteration <= 10]
        epoch_2 = [c for c in cycles if 11 <= c.iteration <= 20]
        epoch_3 = [c for c in cycles if c.iteration >= 21]

        def _calc_epoch_metrics(epoch_cycles: List[IterationCycleRecord]) -> Dict[str, Any]:
            evaluated = [c for c in epoch_cycles if c.claim_empirically_supported is not None]
            if not evaluated:
                return {"count": len(epoch_cycles), "evaluated_claims": 0, "accuracy": 0.0, "mae": 0.0}
            supported = sum(1 for c in evaluated if c.claim_empirically_supported)
            acc = supported / len(evaluated)
            maes = []
            for c in evaluated:
                if c.actual_metric_delta and c.predicted_metric_claim:
                    exp = c.predicted_metric_claim.get("expected_magnitude", 0.0)
                    act = c.actual_metric_delta.get("actual_magnitude", 0.0)
                    maes.append(abs(exp - act))
            mae = sum(maes) / len(maes) if maes else 0.0
            return {
                "count": len(epoch_cycles),
                "evaluated_claims": len(evaluated),
                "accuracy": acc,
                "mae": mae,
            }

        epochs_summary = {
            "epoch_1_early": _calc_epoch_metrics(epoch_1),
            "epoch_2_middle": _calc_epoch_metrics(epoch_2),
            "epoch_3_late": _calc_epoch_metrics(epoch_3),
        }

        # Check if Late MAE improved over Early MAE
        e1_mae = epochs_summary["epoch_1_early"]["mae"]
        e3_mae = epochs_summary["epoch_3_late"]["mae"]
        mae_improved = (e3_mae < e1_mae and e1_mae > 0.0) or (len(epoch_3) == 0 and epochs_summary["epoch_2_middle"]["mae"] < e1_mae)

        # 5. Mistake Recurrence & Correction
        m_tracker = MistakeTracker()
        for i, c in enumerate(cycles):
            m_tracker.check_cycle(c, cycles[:i])
        rec_rate, corr_rate = m_tracker.evaluate_corrections(cycles)

        mistake_summary = {
            "total_mistakes_detected": len(m_tracker.occurrences),
            "error_recurrence_rate": rec_rate,
            "error_correction_rate": corr_rate,
            "mistakes": [asdict(m) for m in m_tracker.occurrences],
        }

        # 6. Attractor Classification
        revisits = []
        for i, c in enumerate(cycles):
            if i > 0 and c.proposal.strategy_type in [prev.proposal.strategy_type for prev in cycles[:i]]:
                aclass = AttractorClassifier.classify_revisit(c.proposal, c.worker_reasoning, cycles[:i])
                revisits.append({"cycle": c.iteration, "class": aclass.value})

        type_counts = {
            "JUSTIFIED_PREMISE_SHIFT": sum(1 for r in revisits if r["class"] == "JUSTIFIED_PREMISE_SHIFT"),
            "REDUNDANT_ATTRACTOR": sum(1 for r in revisits if r["class"] == "REDUNDANT_ATTRACTOR"),
            "DISCRIMINATIVE_REVISIT": sum(1 for r in revisits if r["class"] == "DISCRIMINATIVE_REVISIT"),
        }
        attractor_summary = {
            "total_revisits": len(revisits),
            "class_counts": type_counts,
            "revisits": revisits,
        }

        # 7. Provenance Breakdown
        prov_dist = {}
        for c in cycles:
            meta = c.proposal.metadata or {}
            orig = meta.get("repair_provenance", {}).get("repair_origin") or meta.get("repair_origin", "QWEN_ORIGINAL")
            prov_dist[orig] = prov_dist.get(orig, 0) + 1

        # 8. Promotion Gate Evaluation
        # Tier 1: Required
        r1_holdout = (vault_queries_spent == 0)
        r2_redundancy = (summary.proposals_rejected / total <= 0.10) if total > 0 else True
        max_consec = 0
        cur_streak = 0
        for c in cycles:
            if c.review and c.review.verdict.value == "REJECTED":
                cur_streak += 1
                max_consec = max(max_consec, cur_streak)
            else:
                cur_streak = 0
        r3_safety = (max_consec < 8)
        r4_autonomous = (human_interventions == 0)
        tier1_passed = all([r1_holdout, r2_redundancy, r3_safety, r4_autonomous])

        # Tier 2: Desirable (need >= 4 of 6)
        d1_evidence = (l3_l4_pct >= 60.0)
        d2_pruning = (abandoned_count >= 1)
        d3_calib = mae_improved
        d4_mistake = (corr_rate >= 0.70)
        d5_shared = (len(shared_inferences) >= 1)
        d6_diversity = (len(set(c.proposal.strategy_type for c in cycles)) >= 3)

        tier2_criteria = {
            "d1_evidence_adaptation_ge_60pct": d1_evidence,
            "d2_failure_family_pruned": d2_pruning,
            "d3_temporal_mae_improved": d3_calib,
            "d4_mistake_correction_ge_70pct": d4_mistake,
            "d5_shared_failure_inferred": d5_shared,
            "d6_diversity_ge_3_archetypes": d6_diversity,
        }
        tier2_passed_count = sum(1 for v in tier2_criteria.values() if v)
        tier2_passed = (tier2_passed_count >= 4)

        gate_audit = {
            "tier1_required_invariants": {
                "r1_holdout_zero": r1_holdout,
                "r2_redundancy_le_10pct": r2_redundancy,
                "r3_safety_brake_untripped": r3_safety,
                "r4_autonomous_unattended": r4_autonomous,
                "all_tier1_passed": tier1_passed,
            },
            "tier2_desirable_learning": tier2_criteria,
            "tier2_passed_count": tier2_passed_count,
            "tier2_passed": tier2_passed,
        }

        rationale = []
        if tier1_passed and tier2_passed:
            final_verdict = "LEVEL 2 CONFIRMED — LONGITUDINAL LEARNING DEMONSTRATED"
            rationale.append(f"Passed all Tier 1 required invariants and {tier2_passed_count}/6 Tier 2 desirable learning criteria.")
            rationale.append(f"Evidence-conditioned adaptation reached {l3_l4_pct:.1f}% (Level 3+).")
            rationale.append(f"Error correction rate reached {corr_rate:.1%}.")
        elif tier1_passed and not tier2_passed:
            final_verdict = "LEVEL 2 CONFIRMED — RESEARCH PROCESS IMPROVED BUT LEARNING EVIDENCE LIMITED"
            rationale.append(f"Passed Tier 1 invariants but only met {tier2_passed_count}/6 Tier 2 learning criteria.")
        else:
            final_verdict = "LEVEL 2 NOT CONFIRMED — V1 IS PRIMARILY SEARCH-SPACE SCAFFOLDING"
            rationale.append("Failed one or more Tier 1 required invariants.")

        return Phase8Scorecard(
            timestamp=summary.elapsed_seconds,
            total_cycles=total,
            researcher_id=researcher_id,
            evidence_adaptation_summary=evidence_summary,
            failure_family_summary=family_summary,
            shared_failure_inferences=shared_inferences,
            temporal_calibration_epochs=epochs_summary,
            mistake_recurrence_summary=mistake_summary,
            attractor_summary=attractor_summary,
            provenance_distribution=prov_dist,
            gate_promotion_audit=gate_audit,
            final_verdict=final_verdict,
            verdict_rationale=rationale,
        )