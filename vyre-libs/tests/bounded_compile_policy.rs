//! The bounded compile policy semantic wrapper tests dispatch with.
//!
//! It states a budget and an artifact ceiling and no physical geometry, which
//! is what the semantic seam accepts.

use vyre_megakernel::{Digest, SearchBudget, SemanticExecutionPolicy};

/// Returns the bounded compiler policy used by semantic wrapper tests.
pub(crate) fn policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::unknown_policy(
        Digest([3; 32]),
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}
