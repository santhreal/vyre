//! Closed lifetime and epoch definitions.

use crate::memory_model::ExecutionScope;

/// Semantic lifetime epoch bounding value validity.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct LifetimeEpoch {
    /// Monotonic epoch counter or graph schedule phase.
    pub epoch_id: u64,
    /// Execution scope bounding the lifetime.
    pub scope: ExecutionScope,
    /// Optional region token identifying the allocating lexical region.
    pub region_token: Option<String>,
}

impl LifetimeEpoch {
    /// Create a new lifetime epoch with explicit scope.
    #[must_use]
    pub const fn new(epoch_id: u64, scope: ExecutionScope) -> Self {
        Self {
            epoch_id,
            scope,
            region_token: None,
        }
    }

    /// Attach a region token to this lifetime epoch.
    #[must_use]
    pub fn with_region_token(mut self, token: impl Into<String>) -> Self {
        self.region_token = Some(token.into());
        self
    }

    /// Whether this epoch outlives or encloses `other`.
    #[must_use]
    pub fn encloses(&self, other: &Self) -> bool {
        self.epoch_id <= other.epoch_id && self.scope.wire_tag() >= other.scope.wire_tag()
    }
}
