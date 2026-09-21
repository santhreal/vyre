//! Candidate-plan contracts.

use vyre_lower::analyses::candidate_plan::CandidatePlan;

#[test]
fn empty_plan_has_zero_candidates() {
    let plan: CandidatePlan<u32> = CandidatePlan {
        kernel_id: "k".into(),
        candidates: vec![],
    };
    assert_eq!(plan.candidate_count(), 0);
}
