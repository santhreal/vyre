//! The failure domain of a poisoned lock.
//!
//! A `Mutex` or `RwLock` is poisoned when a thread panicked while holding it,
//! and the state behind it is then whatever that panic left. Three answers to
//! that were in this tree at once: report [`BackendError::poisoned_lock`],
//! recover the guard through `PoisonError::into_inner`, and read the state
//! anyway. A caller cannot compose them, because one word meant "the value is
//! stale", "the value is fine" and "the process is unsound" in three
//! subsystems.
//!
//! Two domains cover every lock here, and an owner states which one it is in.
//!
//! A lock over state this process owns is **recoverable**. The owner reports
//! [`BackendError::poisoned_lock`], discards the guarded value, and a caller
//! rebuilds it: a cache, a registry snapshot, a memoized plan. Nothing outside
//! the process saw the half-written state.
//!
//! A lock over state outside the process image is **process fatal**. A graphics
//! loader dispatch table, a device context, a driver-global registry: the panic
//! already left that state half written, no owner in this process can rebuild
//! it, and the next call into it faults inside code that carries no vyre frame.
//! [`process_fatal_poison`](crate::lock_policy::process_fatal_poison) ends the
//! process there, while the reason is still known, instead of surfacing as a
//! SIGSEGV in an ICD an hour later.
//!
//! A `Drop` implementation is why the second domain cannot be an error. The
//! same lock that guards loader startup guards loader teardown, and teardown
//! runs in `Drop`, which has no caller to report to.

use crate::BackendError;

/// End the process, naming the owner and the state its poisoned lock guards.
///
/// `owner` names the subsystem holding the lock and `state` names what the
/// lock excludes concurrent access to, both in the reader's terms rather than
/// as a type name: `"the device factory"` and `"the graphics loader dispatch
/// table"`, not `"LOADER_STARTUP"`.
///
/// Recovering the guard instead hands the next caller exactly the half-written
/// state the lock exists to keep it out of.
pub fn process_fatal_poison(owner: &str, state: &str) -> ! {
    eprintln!(
        "vyre: {owner} holds a poisoned lock over {state}. A thread panicked while that lock \
         was held, so the state behind it is half written and no owner in this process can \
         rebuild it. Fix: report the earlier panic. The process ends here rather than \
         faulting inside the code that state belongs to."
    );
    std::process::abort()
}

/// Take a guard over process-owned state, or report the poison as recoverable.
///
/// The guarded value is discarded with the guard: a caller that receives the
/// error rebuilds it rather than reading what the panic left.
pub fn recoverable_poison<T>(
    result: Result<T, std::sync::PoisonError<T>>,
) -> Result<T, BackendError> {
    result.map_err(BackendError::poisoned_lock)
}
