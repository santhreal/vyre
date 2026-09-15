//! Native Metal backend coverage, grouped by the surface each module exercises.
//!
//! Every module except `backend_registration` covers native device behavior and
//! is therefore Apple-only; `backend_registration` also pins what a non-Apple
//! build must refuse.
//!
//! `resident_table_metrics.rs` sits in this directory and is not declared here.
//! It names a private item of `runtime`, so the library compiles it as one of
//! its own modules and this harness never sees it.

#![cfg(feature = "device-tests")]

// The two driver items every module below names, imported once here. A harness
// root declares modules and exports no driver items, so a glob of it resolves
// nothing; naming them here keeps the glob the submodules use inside this
// subtree instead of the harness root the other test modules share.
use vyre_driver_metal::{acquire, METAL_BACKEND_ID};

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
