//! Launch-geometry limits for launch preparation and tuning tests.
//!
//! A launch test is about what preparation, validation, or natural-gradient
//! tuning does with a program, not about a backend ceiling, so it states limits
//! wide enough that no per-axis or per-block ceiling binds and varies only the
//! backend name and the per-compute-unit thread budget. Those two are the
//! arguments here; a test that means to hit a ceiling states its own limits.

use crate::validation::LaunchGeometryLimits;

/// Limits under which only `max_threads_per_sm` constrains a launch.
///
/// A zero budget is the unreported case: the backend answers no per-unit
/// residency, which is a distinct contract from a small budget.
#[must_use]
pub const fn wide_limits(backend: &'static str, max_threads_per_sm: u32) -> LaunchGeometryLimits {
    LaunchGeometryLimits {
        backend,
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm,
    }
}
