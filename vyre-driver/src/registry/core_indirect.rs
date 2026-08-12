//! Canonical `core.indirect_dispatch` semantic registration.

use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::operation::{OperationRegistration, OperationRegistry, OperationTier};

const OP_ID: &str = "core.indirect_dispatch";

const SIG: Signature = Signature {
    inputs: &[TypedParam {
        name: "workgroup_count",
        ty: "GpuBufferHandle<[u32;3]>",
    }],
    outputs: &[],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new(OP_ID, OperationTier::Runtime, None, None, None)
        .with_signature(SIG)
        .with_category("core")
}

/// Stable operation id for indirect dispatch.
pub const INDIRECT_DISPATCH_OP_ID: &str = OP_ID;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_dispatch_has_one_signature_only_semantic_owner() {
        let operation = OperationRegistry::global()
            .get(OP_ID)
            .expect("canonical runtime operation");
        assert_eq!(operation.category, Some("core"));
        assert_eq!(operation.signature, Some(&SIG));
        assert!(operation.program().is_none());
    }
}
