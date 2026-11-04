//! Thread-local race tracking driven by the executing interpreter.

use std::cell::RefCell;

use vyre_foundation::ir::{ExecutionScope, MemoryScope, StorageDomain};

use super::model::{MemoryAccessKind, RaceFinding};
use super::shadow::ShadowMemory;

/// Shadow memory and findings accumulated for one race exploration.
#[derive(Debug, Default)]
struct RaceTracker {
    shadow: ShadowMemory,
    findings: Vec<RaceFinding>,
}

thread_local! {
    /// Per-thread race tracking state, `None` outside an exploration.
    ///
    /// The interpreter is single-threaded per call and every hook checks this
    /// slot, so an ordinary evaluation pays one thread-local read per memory
    /// access and records nothing. Tracking every access unconditionally would
    /// otherwise allocate a keyed history for every store the oracle performs.
    static RACE_TRACKER: RefCell<Option<RaceTracker>> = const { RefCell::new(None) };
}

/// Restores the tracking state that was in effect before the exploration it
/// brackets, so a nested evaluation cannot leave the thread recording.
pub(crate) struct RaceTrackingGuard {
    previous: Option<RaceTracker>,
}

impl Drop for RaceTrackingGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        RACE_TRACKER.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Enable race tracking on this thread for the length of the returned guard.
pub(crate) fn enter_race_tracking() -> RaceTrackingGuard {
    let previous = RACE_TRACKER.with(|slot| slot.borrow_mut().replace(RaceTracker::default()));
    RaceTrackingGuard { previous }
}

/// Discard the shadow memory of the previous explored order.
///
/// Findings accumulate across orders; the access history does not, because
/// each order is a separate execution of the dispatch.
pub(crate) fn begin_explored_order() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow = ShadowMemory::new();
        }
    });
}

/// Take every finding recorded since tracking was enabled.
pub(crate) fn take_race_findings() -> Vec<RaceFinding> {
    RACE_TRACKER.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .map(|tracker| std::mem::take(&mut tracker.findings))
            .unwrap_or_default()
    })
}

/// Record one memory access made by an executing invocation.
pub(crate) fn note_access(
    buffer: &str,
    index: u64,
    invocation: [u32; 3],
    kind: MemoryAccessKind,
    scope: MemoryScope,
    domain: StorageDomain,
) {
    RACE_TRACKER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(tracker) = slot.as_mut() else {
            return;
        };
        if let Some(finding) = tracker
            .shadow
            .record_access(buffer, index, invocation, kind, scope, domain)
        {
            if !tracker.findings.contains(&finding) {
                tracker.findings.push(finding);
            }
        }
    });
}

/// Record that the lanes of `workgroup` are the ones now executing.
pub(crate) fn note_workgroup(workgroup: [u32; 3]) {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow.enter_workgroup(workgroup);
        }
    });
}

/// Record that a workgroup barrier released every lane holding at it.
pub(crate) fn note_barrier_release() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker
                .shadow
                .advance_barrier_phase(ExecutionScope::Workgroup);
        }
    });
}

/// Record that the whole dispatch passed a grid fence.
pub(crate) fn note_grid_fence() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow.advance_grid_phase();
        }
    });
}
