//! One outcome shape for every lane that uses the reference as a differential
//! oracle.
//!
//! A program whose loop trip count is read from its own input runs for as long
//! as that value says, and a fused program can take that value from an upstream
//! op's output. One `u32` is enough to ask for four billion iterations. The
//! interpreter now bounds the work it will do and refuses such a run with the
//! program and the ceiling it exceeded, so a lane reports the offender from that
//! refusal instead of racing it against a wall clock on a thread it abandons.

use std::panic::AssertUnwindSafe;

use vyre_reference::ReferenceError;

/// What the oracle said about one case.
pub(crate) enum Oracle<T> {
    /// The value to compare the device against.
    Answered(T),
    /// Rejected or panicked, so this case has no oracle. The reason is carried
    /// because a lane that requires an actionable rejection asserts on it.
    Declined(String),
    /// The interpreter refused the run at its work ceiling, which is a defect in
    /// the program under test rather than a property of the input.
    Unbounded {
        /// Program the interpreter refused, as it named it.
        program: String,
        /// Steps the interpreter admitted for the run.
        ceiling: u64,
    },
}

/// The refusal a lane reports for a program the interpreter would not finish.
pub(crate) fn unbounded_reason(program: &str, ceiling: u64, case: &str) -> String {
    format!(
        "Fix: {case}: the reference refused program `{program}` after {ceiling} interpreter \
         steps. The trip count comes from data rather than from a declared extent, so bound it \
         by the extents of the buffer the body indexes; a count taken from data spins on the \
         device as well as here."
    )
}

/// Evaluate one case, classifying a work-ceiling refusal apart from a rejection.
///
/// The evaluation runs on the calling thread: the interpreter's own ceiling ends
/// a runaway program, so there is nothing left for an outer thread to abandon.
/// A panic is still caught, because a lane treats one as a case with no oracle.
pub(crate) fn bounded_oracle<T, F>(evaluate: F) -> Oracle<T>
where
    F: FnOnce() -> Result<T, ReferenceError>,
{
    match std::panic::catch_unwind(AssertUnwindSafe(evaluate)) {
        Ok(Ok(value)) => Oracle::Answered(value),
        Ok(Err(error)) => match error.step_ceiling_source() {
            Some(source) => Oracle::Unbounded {
                program: source.program.clone(),
                ceiling: source.ceiling,
            },
            None => Oracle::Declined(error.to_string()),
        },
        Err(payload) => Oracle::Declined(panic_reason(&payload)),
    }
}

/// Recover a panic message so a declined case still reports what went wrong.
fn panic_reason(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "the oracle panicked with a non-string payload".to_string()
}
