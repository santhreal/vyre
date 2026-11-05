//! Contracts for `vyre_runtime::resident_work_queue::automata_worklist`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_runtime::resident_work_queue::automata_worklist::{
    AutomataStateIndex, AutomataWorklistMode, AutomataWorklistPolicy, AutomataWorklistRequest,
    AUTOMATA_WORKLIST_EVIDENCE_SCHEMA_VERSION,
};
use vyre_runtime::resident_work_queue::task::{
    TaskPriority, TaskState, TASK_FLAG_REQUEUE_REQUESTED,
};

#[test]
fn state_index_pair_uses_shared_task_work_item_abi() {
    let pair = AutomataStateIndex::new(17, 4096);
    let task = pair.to_task_work_item(5, 3, TaskPriority::High, 99, 12, 13);

    assert_eq!(task.state, TaskState::Ready.word());
    assert_eq!(task.task_id, 5);
    assert_eq!(task.tenant_id, 3);
    assert_eq!(task.priority, TaskPriority::High.word());
    assert_eq!(task.op_handle, 99);
    assert_eq!(task.input_handle, 12);
    assert_eq!(task.output_handle, 13);
    assert_eq!(task.param, 17);
    assert_eq!(task.continuation_pc, 4096);
    assert_eq!(task.continuation_data, 17);
    assert_eq!(
        task.flags & TASK_FLAG_REQUEUE_REQUESTED,
        TASK_FLAG_REQUEUE_REQUESTED
    );
}

#[test]
fn policy_emits_nonblocking_worklist_evidence() {
    let policy = AutomataWorklistPolicy::standard();
    let request = AutomataWorklistRequest {
        worklist_depth: policy.nonblocking_depth_threshold,
        state_visit_count: 2048,
        occupancy_proxy_bps: 2_500,
        blocking_active_time_ns: 900,
        nonblocking_active_time_ns: 600,
    };

    let (recommendation, evidence) = policy
        .recommend_with_evidence(request)
        .expect("Fix: valid automata worklist request should emit evidence");

    assert_eq!(recommendation.mode, AutomataWorklistMode::NonBlocking);
    assert_eq!(
        recommendation.state_visit_budget,
        u64::from(policy.nonblocking_depth_threshold * policy.state_visit_budget_multiplier)
    );
    assert_eq!(
        evidence.schema_version,
        AUTOMATA_WORKLIST_EVIDENCE_SCHEMA_VERSION
    );
    assert_eq!(evidence.selected_mode, AutomataWorklistMode::NonBlocking);
    assert_eq!(evidence.worklist_depth, policy.nonblocking_depth_threshold);
    assert_eq!(evidence.state_visit_count, 2048);
    assert_eq!(evidence.occupancy_proxy_bps, 2_500);
    assert_eq!(evidence.blocking_active_time_ns, 900);
    assert_eq!(evidence.nonblocking_active_time_ns, 600);
    assert!(evidence.match_parity_required);
    assert!(evidence.reports_state_index_pairs);
    assert!(evidence.is_complete());
}
