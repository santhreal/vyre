//! Contracts for `vyre_runtime::UringCompletionPump`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_runtime::{PipelineError, UringCompletionPump, UringPollState};

#[test]
fn construct_stream_has_no_shutdown() {
    let stream = UringCompletionPump::new();
    assert!(!stream.is_shutdown_requested());
}

#[test]
fn shutdown_is_idempotent() {
    let mut stream = UringCompletionPump::new();
    stream.request_shutdown();
    stream.request_shutdown();
    assert!(stream.is_shutdown_requested());
}

#[test]
fn poll_without_uring_reports_detached_state() {
    let mut stream = UringCompletionPump::new();
    assert_eq!(stream.poll().unwrap(), UringPollState::Detached);
}

#[test]
fn drain_incomplete_is_distinguishable_by_type_not_substring() {
    // Regression: the seg_len calibrator must EXCLUDE a too-fine geometry
    // (drain-incomplete) but PROPAGATE every other backend failure. It used
    // to discriminate by `to_string().contains("drain incomplete")`, which
    // silently turns into "abort the whole calibration" the moment the
    // message wording drifts. The structured variant + predicate is the
    // contract; this test pins it.
    let drain = PipelineError::DrainIncomplete {
        descriptor: "combined megakernel",
        claimed: 3,
        expected: 10,
        unit: "segments",
    };
    assert!(drain.is_drain_incomplete());

    // The Display message stays operator-actionable AND keeps the
    // "drain incomplete" phrase + computed unscanned count, so legacy
    // substring matchers and operator logs do not regress.
    let msg = drain.to_string();
    assert_eq!(
        msg,
        "combined megakernel drain incomplete: only 3 of 10 segments were claimed before the \
         dispatch ended, so 7 segments went unscanned and their matches were dropped. This \
         dispatch's hit set is INCOMPLETE. Fix: raise the dispatch timeout \
         (BatchDispatchConfig.timeout) so the drain loop can exhaust the queue, or shard the \
         batch into smaller queues."
    );

    // The per-rule path uses the same variant with a different descriptor/unit.
    let per_rule = PipelineError::DrainIncomplete {
        descriptor: "megakernel",
        claimed: 0,
        expected: 4,
        unit: "work-items",
    };
    assert!(per_rule.is_drain_incomplete());
    assert!(
        per_rule
            .to_string()
            .starts_with("megakernel drain incomplete: only 0 of 4 work-items were claimed"),
        "msg was: {}",
        per_rule
    );

    // A genuine backend failure is NOT a drain-incomplete: it must surface
    // as a hard error, never be excluded-and-continued by the calibrator.
    let backend = PipelineError::Backend("adapter lost".to_string());
    assert!(!backend.is_drain_incomplete());
}
