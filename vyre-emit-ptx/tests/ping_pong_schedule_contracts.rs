//! Contract tests for concrete PTX asynchronous copy and ping-pong scheduling.
//!
//! Verifies Section 185.3:
//! - Target-local double buffering, accumulator liveness, wait groups, and epilogue overlap.
//! - Concrete PTX capability gates (sm_80+ for cp.async).
//! - Register layout and instruction scheduling remain concrete inside PTX emitter.

use vyre_emit_ptx::patterns::ldmatrix_cp_async::{
    plan_double_buffer_schedule, AsyncCopyCandidate, AsyncCopyPlan, PipelineStageOp, ScheduleError,
};
use vyre_emit_ptx::ComputeCapability;
use vyre_lower::{BindingLayout, KernelBody, KernelDescriptor};

fn mock_async_copy_desc() -> (KernelDescriptor, AsyncCopyPlan) {
    let desc = KernelDescriptor {
        id: "mock_gemm_tile".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: vyre_lower::Dispatch {
            workgroup_size: [128, 1, 1],
        },
        body: KernelBody {
            ops: vec![],
            literals: vec![],
            child_bodies: vec![],
        },
    };

    let plan = AsyncCopyPlan {
        kernel_id: desc.id.clone(),
        target_supports_cp_async: true,
        target_supports_ldmatrix: true,
        candidates: vec![AsyncCopyCandidate {
            load_op_index: 0,
            store_op_index: 1,
            global_binding_slot: 0,
            shared_binding_slot: 1,
        }],
    };

    (desc, plan)
}

#[test]
fn ping_pong_schedule_requires_sm_80() {
    let (desc, plan) = mock_async_copy_desc();
    let old_target = ComputeCapability { major: 7, minor: 0 }; // Volta (sm_70)

    let err = plan_double_buffer_schedule(&desc, old_target, &plan, &[0, 1], 128)
        .expect_err("Fix: pre-sm_80 target must reject cp.async schedule");

    assert!(
        matches!(err, ScheduleError::UnsupportedTarget { .. }),
        "Fix: expected UnsupportedTarget, got {err:?}"
    );
}

#[test]
fn ping_pong_schedule_generates_prologue_loop_and_epilogue() {
    let (desc, plan) = mock_async_copy_desc();
    let ampere = ComputeCapability { major: 8, minor: 0 }; // Ampere (sm_80)

    let plan = plan_double_buffer_schedule(&desc, ampere, &plan, &[0, 1], 128)
        .expect("Fix: sm_80 target must succeed in double-buffer planning");

    assert_eq!(plan.schedule.stages, 2);
    assert_eq!(plan.schedule.buffer_slots, vec![0, 1]);
    assert!(!plan.schedule.accumulator_liveness.causes_register_spill);
    assert_eq!(plan.schedule.wait_group_policy.max_in_flight_groups, 2);
    assert!(plan.schedule.epilogue_overlap.overlap_with_next_tile);

    // Verify Prologue issues async copy into stage 0
    assert!(plan.prologue.iter().any(|op| matches!(
        op,
        PipelineStageOp::AsyncCopyIssue { stage: 0, .. }
    )));
    assert!(plan.prologue.iter().any(|op| matches!(op, PipelineStageOp::CommitGroup)));

    // Verify Steady state issues stage 1 and waits on group 1
    assert!(plan.steady_state.iter().any(|op| matches!(
        op,
        PipelineStageOp::AsyncCopyIssue { stage: 1, .. }
    )));
    assert!(plan.steady_state.iter().any(|op| matches!(
        op,
        PipelineStageOp::WaitGroup { depth: 1 }
    )));
    assert!(plan.steady_state.iter().any(|op| matches!(op, PipelineStageOp::RotateStage)));

    // Verify Epilogue drains all groups (depth 0)
    assert!(plan.epilogue.iter().any(|op| matches!(
        op,
        PipelineStageOp::WaitGroup { depth: 0 }
    )));
}

#[test]
fn ping_pong_schedule_verifies_buffer_slots_and_register_budget() {
    let (desc, plan) = mock_async_copy_desc();
    let hopper = ComputeCapability { major: 9, minor: 0 }; // Hopper (sm_90)

    // Insufficient buffer slots (< 2) is rejected
    let err_slots = plan_double_buffer_schedule(&desc, hopper, &plan, &[0], 128)
        .expect_err("Fix: single buffer slot cannot double-buffer");
    assert!(matches!(err_slots, ScheduleError::InsufficientBufferSlots { .. }));

    // Insufficient register budget (< 48) is rejected
    let err_regs = plan_double_buffer_schedule(&desc, hopper, &plan, &[0, 1], 16)
        .expect_err("Fix: low register budget must be rejected");
    assert!(matches!(err_regs, ScheduleError::ExcessiveRegisterPressure { .. }));
}
