//! Shared non-geometric semantic execution policy for integration tests.

use vyre_megakernel::{Digest, SearchBudget, SemanticExecutionPolicy};

/// Returns the bounded compiler policy used by semantic wrapper tests.
pub(crate) fn policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::unknown_policy(
        Digest([3; 32]),
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}
