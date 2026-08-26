//! Shared non-geometric semantic execution policy for integration tests.

use std::collections::BTreeMap;

use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionPolicy,
};

/// Returns the bounded compiler policy used by semantic wrapper tests.
pub(crate) fn policy() -> SemanticExecutionPolicy {
    SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([3; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(8, 64, 1, 0, 1_000),
        1_000_000,
    )
}
