//! One wall-clock ceiling for every lane that uses the reference as a
//! differential oracle.
//!
//! A program whose loop trip count is read from its own input runs for as long
//! as that value says, and a fused program can take that value from an upstream
//! op's output. One `u32` is enough to ask for four billion iterations, which
//! turns a suite into a process that never returns and a job into a cancelled
//! run with no finding. Every oracle call here is bounded, and a call that
//! passes the ceiling is a failure that names what it was evaluating.

use std::panic::AssertUnwindSafe;
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::time::Duration;

/// Wall-clock ceiling on one oracle evaluation, in seconds.
///
/// Far above the milliseconds a bounded program needs on fixture extents.
/// `VYRE_ORACLE_DEADLINE_SECS` raises it, which is how a suspected offender is
/// measured rather than guessed at.
pub(crate) const ORACLE_DEADLINE_SECS: u64 = 20;

/// The ceiling this run enforces.
pub(crate) fn oracle_deadline() -> Duration {
    let seconds = std::env::var("VYRE_ORACLE_DEADLINE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(ORACLE_DEADLINE_SECS);
    Duration::from_secs(seconds)
}

/// What the oracle said about one case.
pub(crate) enum Oracle<T> {
    /// The value to compare the device against.
    Answered(T),
    /// Rejected or panicked, so this case has no oracle. The reason is carried
    /// because a lane that requires an actionable rejection asserts on it.
    Declined(String),
    /// Still running at the ceiling, which is a defect in the program under
    /// test rather than a property of the input.
    TimedOut,
}

/// Evaluate one case on its own thread, bounded by [`oracle_deadline`].
///
/// A thread past the ceiling is left to the process exit: interrupting the
/// interpreter mid-step would need a cancellation path the oracle deliberately
/// does not have. The closure owns its inputs so the caller does not keep a
/// borrow alive across the abandonment.
pub(crate) fn bounded_oracle<T, F>(evaluate: F) -> Oracle<T>
where
    F: FnOnce() -> Result<T, String> + Send + 'static,
    T: Send + 'static,
{
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(evaluate));
        let _ = sender.send(outcome);
    });
    match receiver.recv_timeout(oracle_deadline()) {
        Ok(Ok(Ok(value))) => Oracle::Answered(value),
        Ok(Ok(Err(reason))) => Oracle::Declined(reason),
        Ok(Err(payload)) => Oracle::Declined(panic_reason(&payload)),
        Err(RecvTimeoutError::Disconnected) => {
            Oracle::Declined("the oracle thread ended without answering".to_string())
        }
        Err(RecvTimeoutError::Timeout) => Oracle::TimedOut,
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
