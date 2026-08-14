//! Decides how `DispatchConfig::timeout` constrains one wgpu dispatch.
//!
//! Three questions, one owner: what absolute instant the budget expires at,
//! whether a requested budget is even serviceable by this backend's queue and
//! readback window, and whether the budget has already been spent at a given
//! phase. Seven copies of the last check lived across the asynchronous, batch
//! and pending-dispatch paths, each with its own diagnostic wording, so a
//! change to the comparison reached only the copies it was applied to.

use std::time::{Duration, Instant};

use vyre_driver::BackendError;

/// Below this the queue submission and readback map cannot complete, so the
/// dispatch would be cancelled after paying for compilation.
const SERVICEABLE_FLOOR: Duration = Duration::from_millis(100);

/// Absolute instant `timeout` expires at, or `None` when no budget was set.
///
/// A budget that overflows `Instant` is treated as no budget: the caller waits
/// rather than failing a dispatch whose deadline is past the representable end
/// of the monotonic clock.
pub(crate) fn deadline(started: Instant, timeout: Option<Duration>) -> Option<Instant> {
    timeout.and_then(|duration| started.checked_add(duration))
}

/// Refuse a budget this backend cannot service before compilation begins.
pub(crate) fn reject_unserviceable(timeout: Option<Duration>) -> Result<(), BackendError> {
    if matches!(timeout, Some(timeout) if timeout <= SERVICEABLE_FLOOR) {
        return Err(BackendError::new(
            "dispatch cancelled before WGPU pipeline compilation because DispatchConfig.timeout is below the backend's serviceable queue/readback window. Fix: raise DispatchConfig.timeout or use an already compiled persistent pipeline.",
        ));
    }
    Ok(())
}

/// Fail when the budget measured from `started` is already spent.
///
/// `phase` names what ran out of budget, so the diagnostic still distinguishes
/// work cancelled before submission from work that overran while waiting.
pub(crate) fn enforce_budget(
    started: Instant,
    timeout: Option<Duration>,
    phase: &str,
) -> Result<(), BackendError> {
    let Some(budget) = timeout else {
        return Ok(());
    };
    let elapsed = started.elapsed();
    if elapsed > budget {
        return Err(BackendError::new(format!(
            "{phase}: took {elapsed:?}, budget {budget:?}. Fix: raise DispatchConfig.timeout or split the program into smaller chunks."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_budget_never_expires_and_never_yields_a_deadline() {
        let started = Instant::now();
        assert!(deadline(started, None).is_none());
        enforce_budget(started, None, "unbudgeted dispatch")
            .expect("Fix: a dispatch without DispatchConfig.timeout must not be cancelled.");
    }

    #[test]
    fn a_spent_budget_names_the_phase_that_overran() {
        let started = Instant::now() - Duration::from_secs(2);
        let error = enforce_budget(
            started,
            Some(Duration::from_millis(1)),
            "batch dispatch phase",
        )
        .expect_err("Fix: an elapsed budget must cancel the dispatch.");
        let rendered = error.to_string();
        assert!(
            rendered.starts_with("batch dispatch phase: took "),
            "Fix: the diagnostic must lead with the phase that overran, got: {rendered}"
        );
        assert!(
            rendered.contains("Fix: raise DispatchConfig.timeout"),
            "Fix: the diagnostic must name the corrective action, got: {rendered}"
        );
    }

    #[test]
    fn an_unspent_budget_passes_and_resolves_to_its_own_instant() {
        let started = Instant::now();
        let budget = Duration::from_secs(30);
        enforce_budget(started, Some(budget), "dispatch")
            .expect("Fix: a budget with time left must not cancel the dispatch.");
        assert_eq!(
            deadline(started, Some(budget)),
            Some(started + budget),
            "Fix: the deadline must be the start instant advanced by the budget."
        );
    }

    #[test]
    fn the_serviceable_floor_is_inclusive_and_rejects_below_it() {
        reject_unserviceable(None).expect("Fix: no timeout is always serviceable.");
        reject_unserviceable(Some(SERVICEABLE_FLOOR + Duration::from_millis(1)))
            .expect("Fix: a budget above the floor must be accepted.");
        assert!(
            reject_unserviceable(Some(SERVICEABLE_FLOOR)).is_err(),
            "Fix: the serviceable floor must itself be rejected, not accepted."
        );
        assert!(
            reject_unserviceable(Some(Duration::ZERO)).is_err(),
            "Fix: a zero budget cannot be serviced."
        );
    }
}
