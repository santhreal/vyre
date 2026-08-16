//! Compatibility metadata shared with the secondary-target trap readback path.
//!
//! The trap record's shape is one fact with one owner in `vyre-lower`: the word
//! count and the code-to-tag pair are re-exported here rather than restated, so
//! a layout change cannot leave this path decoding a record nobody writes. Only
//! the reserved buffer name is local, because this pipeline binds the sidecar as
//! a program buffer under its own reserved name while descriptor lowering binds
//! it as a descriptor slot under `vyre_lower::TRAP_SIDECAR_NAME`.

/// Reserved trap-sidecar buffer name for the program lowering path.
pub const TRAP_SIDECAR_NAME: &str = "__vyre_naga_trap_sidecar";

pub use vyre_lower::TRAP_SIDECAR_WORDS;

/// Stable numeric code and source tag for one trap.
pub type TrapTag = vyre_lower::DescriptorTrapTag;
