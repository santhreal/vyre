//! The bounded compile policies semantic wrapper tests dispatch with.
//!
//! Each states a budget and an artifact ceiling and no physical geometry, which
//! is what the semantic seam accepts. They differ only in what the target
//! grants, because a kernel that declares workgroup-scoped scratch or subgroup
//! work is refused against a device that reports neither.

use vyre_megakernel::{Digest, SearchBudget, SemanticExecutionPolicy};

/// Returns the bounded compiler policy used by semantic wrapper tests.
#[allow(
    dead_code,
    reason = "this module is included by several test binaries, and each uses the policy its \
              kernels need"
)]
pub(crate) fn policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::unknown_policy(
        Digest([3; 32]),
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}

/// Returns the same bounded policy against a target that grants every
/// capability, for a kernel whose program needs one.
#[allow(
    dead_code,
    reason = "this module is included by several test binaries, and each uses the policy its \
              kernels need"
)]
pub(crate) fn granted_policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::granted_policy(
        Digest([3; 32]),
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}
