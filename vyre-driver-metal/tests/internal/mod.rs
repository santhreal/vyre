//! Native Metal backend coverage, grouped by the surface each module exercises.
//!
//! Every module except `backend_registration` covers native device behavior and
//! is therefore Apple-only; `backend_registration` also pins what a non-Apple
//! build must refuse.

mod backend_registration;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod dispatch;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod fixtures;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod pipeline_cache;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod resident_memory;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod telemetry;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod wgpu_differential;
