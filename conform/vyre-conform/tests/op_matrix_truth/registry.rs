//! The live operation registry, as this test reads it.

use vyre_foundation::operation::OperationTier as OpTier;

#[derive(Debug)]
pub(crate) struct RegisteredOp {
    pub(crate) id: String,
    pub(crate) source: &'static str,
    pub(crate) tier: OpTier,
}

pub(crate) fn registered_ops() -> Vec<RegisteredOp> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| RegisteredOp {
            id: entry.id.to_string(),
            source: "vyre-foundation::operation",
            tier: entry.tier,
        })
        .collect()
}
