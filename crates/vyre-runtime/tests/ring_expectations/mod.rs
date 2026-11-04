//! One owner for the ring and protocol fault expectations these suites state.
//!
//! WHY: `PipelineError::RingEncoding` carries a typed `RingEncodingFault`, and
//! every suite that drives the ring encoder states which class it expects. Each
//! expectation was written out at the point of use, so the same nine-line
//! `assert!(matches!(...))` block stood in thirty-three places across twelve
//! files, and the destructuring of a missing protocol word and of an illegal
//! slot transition stood in several more. A claim written that many times has
//! no owner: a new fault class, or a change to what a rejection must report,
//! is then made in one place and the other copies keep the old claim.
//!
//! What it does not catch: each function checks the class and the fields the
//! variant carries, not the rendered message. A test that pins remediation text
//! reads the `fix` these functions return.

use vyre_runtime::resident_work_queue::protocol::ProtocolError;
use vyre_runtime::resident_work_queue::RingSlotTransition;
use vyre_runtime::{PipelineError, RingEncodingFault};

/// The remediation on a ring-encoding rejection of exactly `expected`.
///
/// `context` names the input under test and is stated in every failure, so a
/// red run says which case produced the wrong class.
#[track_caller]
pub fn ring_fault_fix(
    err: &PipelineError,
    expected: RingEncodingFault,
    context: &str,
) -> &'static str {
    let PipelineError::RingEncoding { fault, fix } = err else {
        panic!("Fix: {context} must be rejected as a ring-encoding fault, got {err:?}")
    };
    assert_eq!(
        *fault, expected,
        "Fix: {context} is a {expected:?} ring-encoding fault, got {fault:?}"
    );
    fix
}

/// Asserts a ring-encoding rejection of exactly `expected`.
#[track_caller]
pub fn assert_ring_fault(err: &PipelineError, expected: RingEncodingFault, context: &str) {
    ring_fault_fix(err, expected, context);
}

/// The buffer, word index and byte length a missing protocol word names.
#[track_caller]
pub fn missing_word(err: &ProtocolError, context: &str) -> (&'static str, usize, usize) {
    let ProtocolError::MissingWord {
        buffer,
        word_idx,
        byte_len,
        ..
    } = err
    else {
        panic!("Fix: {context} must name the word it could not read, got {err:?}")
    };
    (buffer, *word_idx, *byte_len)
}

/// The same fields, for a protocol fault that reached the caller through
/// [`PipelineError::Protocol`].
#[track_caller]
pub fn protocol_missing_word(err: &PipelineError, context: &str) -> (&'static str, usize, usize) {
    let PipelineError::Protocol(protocol) = err else {
        panic!("Fix: {context} must be reported as a protocol fault, got {err:?}")
    };
    missing_word(protocol, context)
}

/// Asserts a publish was rejected because the slot held `current`, and that the
/// rejection reports the transition and the permitted set the legality
/// predicate reads.
#[track_caller]
pub fn assert_publish_rejected_by_status(err: &PipelineError, current: u32, context: &str) {
    let PipelineError::IllegalSlotTransition {
        transition,
        permitted,
        current_status,
        ..
    } = err
    else {
        panic!("Fix: {context} must be rejected as an illegal slot transition, got {err:?}")
    };
    assert_eq!(
        *transition,
        RingSlotTransition::Publish.label(),
        "Fix: the rejection must name the attempted transition"
    );
    assert_eq!(
        *permitted,
        RingSlotTransition::Publish.permitted(),
        "Fix: the rejection must state the same permitted set the predicate enforces"
    );
    assert_eq!(
        *current_status, current,
        "Fix: the rejection must state the status the slot held, not a placeholder"
    );
}
