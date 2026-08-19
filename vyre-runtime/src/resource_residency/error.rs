use thiserror::Error;

use super::admission::{ResourceSetKey, StateId};

/// Resource residency admission or lifecycle failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResourceResidencyError {
    /// A content or artifact identity is all zeroes.
    #[error("resident resource-set identity is zero. Fix: use verified source and compiler artifact digests")]
    ZeroIdentity,
    /// A request repeats or omits a stable name.
    #[error(
        "{kind} name `{name}` is empty or duplicated. Fix: provide one stable name per binding"
    )]
    DuplicateOrEmptyName {
        /// Binding class.
        kind: &'static str,
        /// Invalid name.
        name: String,
    },
    /// Immutable bytes disagree with their trusted digest.
    #[error("immutable_resource `{name}` does not match its trusted BLAKE3 digest")]
    ImmutableResourceDigestMismatch {
        /// Immutable resource name.
        name: String,
        /// Digest of supplied bytes.
        actual: [u8; 32],
        /// Trusted digest.
        expected: [u8; 32],
    },
    /// Byte arithmetic exceeded the supported domain.
    #[error("residency byte arithmetic overflowed for {context}. Fix: shard the admission")]
    ByteLengthOverflow {
        /// Failed arithmetic context.
        context: String,
    },
    /// Admission exceeds the explicit manager budget.
    #[error("{context} needs {requested} additional bytes with {used} already used, over budget {budget}. Fix: evict idle resource sets or reduce state capacity")]
    OutOfMemory {
        /// Admission class.
        context: &'static str,
        /// Currently accounted bytes.
        used: u64,
        /// Newly requested bytes.
        requested: u64,
        /// Hard budget.
        budget: u64,
    },
    /// Backend resource operation failed.
    #[error("{operation} failed: {detail}")]
    Backend {
        /// Failed operation.
        operation: &'static str,
        /// Backend diagnostic.
        detail: String,
    },
    /// Admission failed and one or more rollback frees also failed.
    #[error("{operation} failed: {detail}; rollback also failed: {cleanup}")]
    Rollback {
        /// Failed operation.
        operation: &'static str,
        /// Primary backend diagnostic.
        detail: String,
        /// Cleanup diagnostic.
        cleanup: String,
    },
    /// Releasing one or more unreachable resources failed.
    #[error("{context} could not release all resident resources: {details}")]
    Release {
        /// Lifecycle operation.
        context: &'static str,
        /// Joined backend diagnostics.
        details: String,
    },
    /// A warm request disagrees with its already resident key.
    #[error("warm resource-set request disagrees with resident immutable resource or artifact bindings. Fix: use a new artifact digest for a changed plan")]
    WarmResourceSetMismatch,
    /// Resource-set key is absent.
    #[error(
        "resource set {key:?} is not resident. Fix: admit the resource_set before starting or binding a state"
    )]
    ResourceSetNotResident {
        /// Missing resource-set key.
        key: ResourceSetKey,
    },
    /// Resource set still owns live states.
    #[error("resource set {key:?} has {active_states} active states. Fix: finish or cancel them before eviction")]
    ResourceSetInUse {
        /// Resource-set key.
        key: ResourceSetKey,
        /// Live state count.
        active_states: u64,
    },
    /// Mutable-state identity space is exhausted.
    #[error("state identity space is exhausted. Fix: restart the residency manager rather than reusing stale identities")]
    StateIdentityOverflow,
    /// Reset generation space is exhausted.
    #[error("state {state:?} generation space is exhausted. Fix: finish it and start a new state")]
    StateGenerationOverflow {
        /// Affected state.
        state: StateId,
    },
    /// State lease is absent, cancelled, or already finished.
    #[error("state {state:?} is not active. Fix: discard stale leases and start a new state")]
    StateLeaseNotFound {
        /// Missing state.
        state: StateId,
    },
    /// Lease predates the latest reset.
    #[error("state {state:?} lease generation {actual_generation} is stale; current generation is {expected_generation}")]
    StaleStateLease {
        /// State identity.
        state: StateId,
        /// Current generation.
        expected_generation: u64,
        /// Supplied generation.
        actual_generation: u64,
    },
    /// Mutable state name is absent.
    #[error("state {state:?} has no state `{name}`")]
    StateNotFound {
        /// State identity.
        state: StateId,
        /// Missing state name.
        name: String,
    },
    /// Mutable state cannot have a zero-byte allocation.
    #[error("mutable state `{name}` is zero bytes. Fix: omit unused state or provide its exact positive size")]
    ZeroStateBytes {
        /// Invalid state name.
        name: String,
    },
    /// Immutable resource name is absent.
    #[error("resident resource set {key:?} has no immutable resource `{name}`")]
    ImmutableResourceNotFound {
        /// Resource-set identity.
        key: ResourceSetKey,
        /// Missing immutable resource name.
        name: String,
    },
    /// Named artifact instance is absent.
    #[error("resident resource set {key:?} has no artifact `{name}`")]
    ArtifactNotFound {
        /// Resource-set identity.
        key: ResourceSetKey,
        /// Missing artifact name.
        name: String,
    },
    /// Internal counters would underflow.
    #[error(
        "residency accounting underflowed. Fix: stop using the manager and rebuild residency state"
    )]
    AccountingUnderflow,
    /// Another thread panicked while holding residency state.
    #[error(
        "residency state lock is poisoned. Fix: rebuild the manager before admitting more work"
    )]
    LockPoisoned,
}
