//! Shared IR composition infrastructure, child region skeletons, operands, and link anchors.

pub mod builder;
pub mod plumbing;
pub mod prelude;
pub mod fixture_bytes;

pub use builder::*;
pub use plumbing::host::dispatch_buffers::*;
pub use plumbing::host::program_cache::*;
pub use plumbing::host::scratch::*;
#[cfg(feature = "telemetry")]
pub use plumbing::host::telemetry;
pub use plumbing::operand::buffer_names::*;
pub use plumbing::operand::element_zero::*;
pub use plumbing::operand::shape::*;
pub use plumbing::operand::tensor_ref::*;
pub use plumbing::program::attribution::*;
pub use plumbing::program::descriptor::*;
pub use plumbing::program::outputs::*;
pub use plumbing::registration::{contracts, operation_catalog};
pub use plumbing::registration::signatures::*;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    operation_catalog::link_anchor()
}
