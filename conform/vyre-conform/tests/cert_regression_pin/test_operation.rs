//! The registered operation the region-chain bundle calls.
//!
//! It exists only so a canonical bundle can carry a dialect op id in its wire
//! bytes. The reference facet returns its input word unchanged.

use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::operation::{OperationRegistration, OperationTier};
use vyre_reference::ReferenceFacet;

/// Operation id the region-chain bundle calls. It is part of that bundle's
/// wire bytes, so respelling it moves the pinned digest.
pub(crate) const TEST_IDENTITY_U32_OP: &str = "vyre_conform_test::identity_u32";

fn identity_u32_cpu_ref(input: &[u8], output: &mut Vec<u8>) {
    output.clear();
    output.extend_from_slice(input.get(..4).unwrap_or(&[0, 0, 0, 0]));
}

const TEST_IDENTITY_U32_SIGNATURE: Signature = vyre_test_support::u32_signature! {
    inputs: ["value"],
    output: "out",
};

inventory::submit! {
    OperationRegistration::new_unconstrained(
        TEST_IDENTITY_U32_OP,
        OperationTier::External,
        None,
        None,
        None,
    )
    .with_signature(TEST_IDENTITY_U32_SIGNATURE)
    .with_category("vyre-conform-test")
}
inventory::submit! {
    ReferenceFacet::new(TEST_IDENTITY_U32_OP, identity_u32_cpu_ref)
}
