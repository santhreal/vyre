//! The monotonic generation counter a delta stream is keyed on.

use super::*;

/// Thread-safe generation tracker preventing publication of superseded artifacts.
#[derive(Debug, Default, Clone)]
pub struct GenerationTracker {
    generations: std::sync::Arc<std::sync::RwLock<rustc_hash::FxHashMap<String, u64>>>,
}

impl GenerationTracker {
    /// Create an empty generation tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Current published generation for a resource name.
    #[must_use]
    pub fn current_generation(&self, resource_name: &str) -> u64 {
        crate::failure_domain::govern_rwlock_read(
            &self.generations,
            "GenerationTracker",
            "generations",
            crate::failure_domain::RecoveryClass::RestartableFromCanonicalInput,
        )
        .map(|guard| guard.get(resource_name).copied().unwrap_or(0))
        .unwrap_or(0)
    }

    /// Publish a new generation for a resource.
    ///
    /// Rejects superseded generations: `generation` must be strictly greater than `current_generation`.
    pub fn publish_generation(
        &self,
        resource_name: &str,
        generation: u64,
    ) -> Result<(), GraphDeltaError> {
        let mut guard = crate::failure_domain::govern_rwlock_write(
            &self.generations,
            "GenerationTracker",
            "generations",
            crate::failure_domain::RecoveryClass::TransactionallyRecoverable,
        )
        .map_err(|_| GraphDeltaError::LockPoisoned {
            state: "generations".to_string(),
        })?;
        let current = guard.get(resource_name).copied().unwrap_or(0);
        if generation <= current {
            return Err(GraphDeltaError::SupersededGeneration {
                resource_name: resource_name.to_string(),
                current_generation: current,
                attempted_generation: generation,
            });
        }
        guard.insert(resource_name.to_string(), generation);
        Ok(())
    }
}
