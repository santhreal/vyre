//! Compatibility metadata shared with the wgpu trap readback path.

use std::sync::Arc;

/// Reserved trap-sidecar buffer name.
pub const TRAP_SIDECAR_NAME: &str = "__vyre_naga_trap_sidecar";
/// Number of words in the trap sidecar.
pub const TRAP_SIDECAR_WORDS: u32 = 4;

/// Stable numeric code and source tag for one trap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrapTag {
    /// Stable numeric trap code.
    pub code: u32,
    /// Source trap tag.
    pub tag: Arc<str>,
}
