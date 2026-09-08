//! Contract and regression tests for [`InteractiveSessionStateMachine`].
//!
//! Interactive UI applications require predictable latency, bounded queues,
//! priority inheritance, generation-based supersession, cooperative cancellation,
//! and explicit irreversible-submission boundaries.
//!
//! These tests prove:
//! 1. Measured dispatch ceiling and derived termination contract.
//! 2. Bounded admission and unachievable deadline rejection.
//! 3. Queue saturation backpressure.
//! 4. Automatic generation-based supersession of obsolete frames.
//! 5. Stale generation rejection.
//! 6. Cooperative cancellation before the submission boundary.
//! 7. Cancellation refusal once past the irreversible submission boundary.
//! 8. Dynamic priority inheritance.
//! 9. Device loss fault isolation and coordinated teardown.
//! 10. Stale completion handling.

#![forbid(unsafe_code)]

use vyre_megakernel::Digest;
use vyre_runtime::artifact_admission::{
    CancellationOutcome, DeadlineClass, InteractiveAdmissionError, InteractiveCancellationError,
    InteractiveChannelId, InteractiveCompletion, InteractiveSessionState,
    InteractiveSessionStateMachine, InteractiveSubmissionRequest, PriorityClass,
    INTERACTIVE_DISPATCH_HEADROOM, MAX_INTERACTIVE_STEP_BUDGET_MICROS,
    MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS,
};

fn dummy_digest() -> Digest {
    Digest([0x42; 32])
}

fn make_request(
    channel: u64,
    gen: u64,
    deadline_ns: u64,
    duration_ns: u64,
    priority: PriorityClass,
) -> InteractiveSubmissionRequest {
    InteractiveSubmissionRequest {
        channel_id: InteractiveChannelId(channel),
        frame_generation: gen,
        deadline: DeadlineClass::InteractiveFrame {
            frame_target_ns: deadline_ns,
            target_fps: 60,
        },
        priority,
        estimated_duration_ns: duration_ns,
        artifact: dummy_digest(),
    }
}

/// WHY: Termination and bounded dispatch contracts must be derived from verified
/// measurements, never arbitrary chosen constants.
#[test]
fn the_interactive_dispatch_ceiling_is_derived_from_measurement() {
    assert_eq!(
        MAX_INTERACTIVE_STEP_BUDGET_MICROS,
        MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS * INTERACTIVE_DISPATCH_HEADROOM,
        "Fix: interactive step budget must equal measured ceiling * headroom multiplier"
    );
    assert!(
        MEASURED_INTERACTIVE_DISPATCH_CEILING_MICROS > 0,
        "Fix: measured ceiling must be non-zero"
    );
    assert!(
        INTERACTIVE_DISPATCH_HEADROOM > 1,
        "Fix: headroom multiplier must provide headroom above the heaviest run"
    );
}

#[test]
fn admission_rejects_unachievable_hard_deadline() {
    let sm = InteractiveSessionStateMachine::new();

    // 20ms estimated duration exceeds 16ms target deadline
    let req = make_request(1, 1, 16_000_000, 20_000_000, PriorityClass::Normal);
    let err = sm
        .admit(req, 1_000_000)
        .expect_err("Fix: unachievable deadline must be rejected at admission");

    assert!(matches!(
        err,
        InteractiveAdmissionError::DeadlineUnachievable {
            estimated_ns: 20_000_000,
            remaining_ns: 16_000_000
        }
    ));
}

#[test]
fn admission_queue_saturation_rejects_overflow() {
    let max_depth = 4;
    let sm = InteractiveSessionStateMachine::with_max_queue_depth(max_depth);

    for i in 1..=max_depth {
        let req = make_request(i as u64, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
        sm.admit(req, 1_000_000)
            .expect("Fix: request within queue capacity must be admitted");
    }

    // Next request must be rejected due to queue saturation
    let overflow_req = make_request(99, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let err = sm
        .admit(overflow_req, 1_000_000)
        .expect_err("Fix: queue overflow must be rejected");

    assert!(matches!(
        err,
        InteractiveAdmissionError::QueueSaturated { capacity } if capacity == max_depth
    ));
}

#[test]
fn generation_based_supersession_replaces_stale_frames() {
    let sm = InteractiveSessionStateMachine::new();

    // Admit frame generation 1 on channel 1
    let req1 = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id1 = sm
        .admit(req1, 1_000_000)
        .expect("Fix: frame 1 must be admitted");
    assert_eq!(
        sm.state_of(id1),
        Ok(Some(InteractiveSessionState::Admitted))
    );

    // Prepare frame 1
    sm.prepare(id1).expect("Fix: frame 1 prepare must succeed");
    assert_eq!(
        sm.state_of(id1),
        Ok(Some(InteractiveSessionState::Prepared))
    );

    // A newer frame generation 2 arrives on channel 1 before frame 1 is submitted
    let req2 = make_request(1, 2, 16_000_000, 5_000_000, PriorityClass::High);
    let id2 = sm
        .admit(req2, 2_000_000)
        .expect("Fix: frame 2 must be admitted");

    // Frame 1 must now be superseded automatically
    assert_eq!(
        sm.state_of(id1),
        Ok(Some(InteractiveSessionState::Superseded))
    );
    assert_eq!(
        sm.state_of(id2),
        Ok(Some(InteractiveSessionState::Admitted))
    );

    // Attempting to submit stale frame 1 is refused
    let submit_err = sm
        .submit(id1)
        .expect_err("Fix: submitting superseded frame must fail");
    assert!(matches!(
        submit_err,
        vyre_driver::BackendError::DispatchFailed { .. }
    ));

    // Frame 2 proceeds normally
    sm.prepare(id2).expect("Fix: frame 2 prepare must succeed");
    sm.submit(id2).expect("Fix: frame 2 submit must succeed");
    let completion = sm
        .complete(id2, 7_000_000)
        .expect("Fix: frame 2 complete must succeed");
    assert!(matches!(
        completion,
        InteractiveCompletion::Success {
            frame_generation: 2,
            ..
        }
    ));
}

#[test]
fn stale_generation_admission_is_rejected() {
    let sm = InteractiveSessionStateMachine::new();

    let req_gen5 = make_request(1, 5, 16_000_000, 5_000_000, PriorityClass::Normal);
    sm.admit(req_gen5, 1_000_000)
        .expect("Fix: gen 5 must be admitted");

    // Submitting an older generation 4 on channel 1 must be rejected
    let req_gen4 = make_request(1, 4, 16_000_000, 5_000_000, PriorityClass::Normal);
    let err = sm
        .admit(req_gen4, 2_000_000)
        .expect_err("Fix: stale generation must be rejected");

    assert!(matches!(
        err,
        InteractiveAdmissionError::StaleGeneration {
            provided: 4,
            current: 5,
            ..
        }
    ));
}

#[test]
fn cooperative_cancellation_before_submission_succeeds() {
    let sm = InteractiveSessionStateMachine::new();

    let req = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id = sm.admit(req, 1_000_000).expect("Fix: admit must succeed");

    // Cancel while in Admitted state
    let outcome = sm
        .cancel(id)
        .expect("Fix: cancel in Admitted state must succeed");
    assert_eq!(outcome, CancellationOutcome::Cancelled);
    assert_eq!(
        sm.state_of(id),
        Ok(Some(InteractiveSessionState::Cancelled))
    );

    // Prepare after cancel fails
    assert!(sm.prepare(id).is_err());
}

#[test]
fn cancellation_refused_after_irreversible_submission() {
    let sm = InteractiveSessionStateMachine::new();

    let req = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id = sm.admit(req, 1_000_000).expect("Fix: admit must succeed");
    sm.prepare(id).expect("Fix: prepare must succeed");
    sm.submit(id).expect("Fix: submit must succeed");
    assert_eq!(
        sm.state_of(id),
        Ok(Some(InteractiveSessionState::Submitted))
    );

    // Cancel after crossing the irreversible submission boundary MUST fail
    let err = sm
        .cancel(id)
        .expect_err("Fix: cancellation after GPU submission must fail");

    assert!(matches!(
        err,
        InteractiveCancellationError::IrreversibleSubmission(target_id) if target_id == id
    ));
}

#[test]
fn priority_inheritance_boosts_blocking_request() {
    let sm = InteractiveSessionStateMachine::new();

    let low_req = make_request(1, 1, 50_000_000, 10_000_000, PriorityClass::Low);
    let low_id = sm
        .admit(low_req, 1_000_000)
        .expect("Fix: low req must be admitted");
    assert_eq!(
        sm.effective_priority_of(low_id),
        Ok(Some(PriorityClass::Low))
    );

    // Urgent request waits on resource held by low_id
    let boosted = sm
        .apply_priority_inheritance(low_id, PriorityClass::Urgent)
        .expect("Fix: priority inheritance must apply");

    assert_eq!(boosted, PriorityClass::Urgent);
    assert_eq!(
        sm.effective_priority_of(low_id),
        Ok(Some(PriorityClass::Urgent))
    );
}

#[test]
fn device_loss_faults_active_sessions_and_rejects_new_admissions() {
    let sm = InteractiveSessionStateMachine::new();

    let req = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id = sm.admit(req, 1_000_000).expect("Fix: admit must succeed");

    // Device loss occurs
    sm.fault_all("GPU device lost due to driver reset")
        .expect("Fix: faulting the session must be recorded");

    assert_eq!(sm.state_of(id), Ok(Some(InteractiveSessionState::Faulted)));

    // New admissions must fail immediately
    let new_req = make_request(2, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let err = sm
        .admit(new_req, 2_000_000)
        .expect_err("Fix: admissions after device loss must fail");

    assert!(matches!(err, InteractiveAdmissionError::DeviceLoss));
}

#[test]
fn stale_completion_handling() {
    let sm = InteractiveSessionStateMachine::new();

    let req = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id = sm.admit(req, 1_000_000).expect("Fix: admit must succeed");
    sm.cancel(id).expect("Fix: cancel must succeed");

    // Completing a cancelled request returns Cancelled completion without panicking
    let completion = sm
        .complete(id, 2_000_000)
        .expect("Fix: complete on cancelled request must succeed");
    assert!(matches!(
        completion,
        InteractiveCompletion::Cancelled { request_id } if request_id == id
    ));
}

#[test]
fn a_faulted_completion_reports_the_loss_that_caused_it() {
    // A caller cannot act on "session was faulted". The reason `fault_all`
    // received is the only thing that distinguishes a driver reset from a
    // deliberate teardown, and it was previously dropped.
    let sm = InteractiveSessionStateMachine::new();
    let req = make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal);
    let id = sm.admit(req, 1_000_000).expect("Fix: admit must succeed");

    sm.fault_all("adapter removed while the queue was draining")
        .expect("Fix: faulting the session must be recorded");
    assert_eq!(
        sm.fault_reason(),
        Ok(Some(
            "adapter removed while the queue was draining".to_string()
        ))
    );

    let completion = sm
        .complete(id, 2_000_000)
        .expect("Fix: completing a faulted request must report the fault");
    assert_eq!(
        completion,
        InteractiveCompletion::Faulted {
            request_id: id,
            reason: "adapter removed while the queue was draining".to_string(),
        },
        "a faulted completion must carry the reason the session faulted"
    );
}

#[test]
fn a_healthy_session_reports_no_fault_reason() {
    let sm = InteractiveSessionStateMachine::new();
    assert_eq!(sm.fault_reason(), Ok(None));
}

#[test]
fn a_superseded_completion_names_the_generation_that_superseded_it() {
    // Generations skip: a frontend that drops frames advances by more than one.
    // Reporting `stale + 1` named a generation that was never submitted.
    let sm = InteractiveSessionStateMachine::new();
    let stale = sm
        .admit(
            make_request(1, 1, 16_000_000, 5_000_000, PriorityClass::Normal),
            1_000_000,
        )
        .expect("Fix: the first generation must be admitted");
    sm.admit(
        make_request(1, 7, 16_000_000, 5_000_000, PriorityClass::Normal),
        2_000_000,
    )
    .expect("Fix: a later generation must be admitted");

    assert_eq!(
        sm.state_of(stale),
        Ok(Some(InteractiveSessionState::Superseded))
    );
    let completion = sm
        .complete(stale, 3_000_000)
        .expect("Fix: completing a superseded request must succeed");
    assert_eq!(
        completion,
        InteractiveCompletion::Superseded {
            request_id: stale,
            stale_generation: 1,
            superseded_by_generation: 7,
        },
        "the superseding generation must be the channel's current generation"
    );
}
