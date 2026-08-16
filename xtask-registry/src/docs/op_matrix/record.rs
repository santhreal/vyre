//! One row of the op matrix, as every stage of the gate passes it along.

use vyre_foundation::operation::OperationTier as OpTier;

#[derive(Clone)]
pub(super) struct OpRecord {
    pub(super) family: String,
    pub(super) tier: OpTier,
    pub(super) owners: Vec<String>,
    pub(super) ops: Vec<String>,
    pub(super) registry_sources: Vec<String>,
    pub(super) duplicate_ok: bool,
    pub(super) reference: &'static str,
    pub(super) foundation_ir: &'static str,
    pub(super) cuda: &'static str,
    pub(super) wgpu: &'static str,
    pub(super) spirv: &'static str,
    pub(super) release_blocking_notes: String,
    pub(super) tests: Vec<String>,
}
