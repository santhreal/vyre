//! The backend-independent reading of a resident dispatch sequence.
//!
//! A resident sequence is a list of programs launched against resources that
//! stay bound, followed by a list of byte ranges read back from them. Every
//! backend that does not fuse the sequence onto one queue falls back to the
//! same two decisions: what launch configuration each step gets, and how the
//! requested ranges become one readback call. Both the `VyreBackend` defaults
//! and the grid-sync split decorator route through here, so the fallback cannot
//! drift between the trait and a wrapper that overrides it.

use smallvec::SmallVec;

use crate::backend::{
    BackendError, DispatchConfig, ResidentDispatchStep, ResidentReadRange, VyreBackend,
};

/// The launch configuration one resident step is dispatched with.
///
/// A step carries only its own grid override; nothing else from the caller's
/// configuration applies, because a sequence step's shape is decided by the
/// planner that built the step, not by the dispatch that runs the sequence.
pub(crate) fn resident_step_config(step: &ResidentDispatchStep<'_>) -> DispatchConfig {
    DispatchConfig {
        grid_override: step.grid_override,
        ..DispatchConfig::default()
    }
}

/// Dispatch every step in order against the already-bound resident resources.
///
/// # Errors
///
/// Returns the first step's [`BackendError`], leaving later steps undispatched.
pub(crate) fn dispatch_resident_steps<B>(
    backend: &B,
    steps: &[ResidentDispatchStep<'_>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    for step in steps {
        backend.dispatch_resident_timed(
            step.program,
            step.resources,
            &resident_step_config(step),
        )?;
    }
    Ok(())
}

/// Read every requested resident range into `outputs`, in range order.
///
/// # Errors
///
/// Returns [`BackendError`] when a range is invalid or the backend cannot read
/// back resident storage.
pub(crate) fn read_resident_ranges_into<B>(
    backend: &B,
    read_ranges: &[ResidentReadRange<'_>],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError>
where
    B: VyreBackend + ?Sized,
{
    let ranges = read_ranges
        .iter()
        .map(|range| (range.resource, range.byte_offset, range.byte_len))
        .collect::<SmallVec<[_; 8]>>();
    backend.download_resident_ranges_into(&ranges, outputs)
}
