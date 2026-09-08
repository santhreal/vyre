//! Generation-scoped namespaces and rolling upgrade coordination for runtime caches and sessions.
//!
//! WHY: closes the class "mixed-version operation in fleets and long-running sessions
//! contaminates cache entries or causes non-atomic rolling upgrade failure".
//! Ensures cache keys and session resources are generation-scoped, preventing cross-version
//! corruption while supporting clean rolling upgrade, rollback, and crash interruption.

use std::collections::BTreeMap;
use std::format;
use std::string::String;

use vyre_foundation::{
    CompatibilityMatrix, ProtocolDomain, ProtocolVersion,
};

/// Generation-scoped cache namespace preventing cross-version contamination.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationScopedNamespace {
    /// Governed protocol domain.
    pub domain: ProtocolDomain,
    /// Exact protocol version of the active session.
    pub version: ProtocolVersion,
    /// Monotonically increasing generation identifier.
    pub generation_id: u64,
}

impl GenerationScopedNamespace {
    /// Create a new generation-scoped namespace.
    #[must_use]
    pub const fn new(domain: ProtocolDomain, version: ProtocolVersion, generation_id: u64) -> Self {
        Self {
            domain,
            version,
            generation_id,
        }
    }

    /// Format a raw key with the generation namespace prefix.
    #[must_use]
    pub fn scoped_key(&self, raw_key: &str) -> String {
        format!(
            "ns:{}:{}:gen_{}:{}",
            self.domain.as_str(),
            self.version,
            self.generation_id,
            raw_key
        )
    }
}

/// Lifecycle state for a rolling upgrade session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpgradePhase {
    /// Operating normally on active generation.
    Active,
    /// Transitioning: both old and new generations are active concurrently.
    RollingUpgrade {
        /// Previous generation being drained.
        draining_generation: u64,
        /// New target generation accepting submissions.
        target_generation: u64,
    },
    /// Upgrade completed; previous generation drained and decommissioned.
    Completed,
    /// Upgrade was interrupted or aborted; rolled back to previous generation.
    RolledBack {
        /// Generation rolled back to.
        restored_generation: u64,
        /// Reason for rollback.
        reason: &'static str,
    },
}

/// Coordinator for rolling upgrade and rollback transitions.
pub struct RollingUpgradeCoordinator {
    domain: ProtocolDomain,
    current_version: ProtocolVersion,
    current_generation: u64,
    phase: UpgradePhase,
    active_resources: BTreeMap<String, u64>, // maps resource_id -> generation_id
}

impl RollingUpgradeCoordinator {
    /// Initialize coordinator with initial version and generation.
    #[must_use]
    pub fn new(domain: ProtocolDomain, initial_version: ProtocolVersion) -> Self {
        Self {
            domain,
            current_version: initial_version,
            current_generation: 1,
            phase: UpgradePhase::Active,
            active_resources: BTreeMap::new(),
        }
    }

    /// Current active namespace.
    #[must_use]
    pub fn current_namespace(&self) -> GenerationScopedNamespace {
        GenerationScopedNamespace::new(self.domain, self.current_version, self.current_generation)
    }

    /// Current upgrade phase.
    #[must_use]
    pub const fn phase(&self) -> UpgradePhase {
        self.phase
    }

    /// Begin rolling upgrade to a target version.
    ///
    /// # Errors
    ///
    /// Returns error if the target version is incompatible or if an upgrade is already in flight.
    pub fn begin_upgrade(
        &mut self,
        target_version: ProtocolVersion,
    ) -> Result<GenerationScopedNamespace, String> {
        if matches!(self.phase, UpgradePhase::RollingUpgrade { .. }) {
            return Err(String::from(
                "Fix: cannot start rolling upgrade while another upgrade is in flight.",
            ));
        }

        let matrix = CompatibilityMatrix::canonical();
        let disp = matrix.check(self.domain, self.current_version, target_version);
        if !disp.is_compatible() {
            return Err(format!(
                "Fix: target version '{target_version}' is incompatible with current version '{}' under domain '{}'.",
                self.current_version,
                self.domain
            ));
        }

        let next_generation = self.current_generation + 1;
        self.phase = UpgradePhase::RollingUpgrade {
            draining_generation: self.current_generation,
            target_generation: next_generation,
        };
        self.current_generation = next_generation;
        self.current_version = target_version;

        Ok(self.current_namespace())
    }

    /// Commit the upgrade after the draining generation completes all pending work.
    pub fn commit_upgrade(&mut self) -> Result<(), String> {
        match self.phase {
            UpgradePhase::RollingUpgrade {
                draining_generation,
                ..
            } => {
                // Clean up resources belonging to drained generation
                self.active_resources
                    .retain(|_, gen| *gen != draining_generation);
                self.phase = UpgradePhase::Completed;
                Ok(())
            }
            _ => Err(String::from(
                "Fix: commit_upgrade called when no rolling upgrade was in progress.",
            )),
        }
    }

    /// Abort and rollback the upgrade atomically.
    pub fn rollback(&mut self, previous_version: ProtocolVersion, reason: &'static str) -> Result<(), String> {
        match self.phase {
            UpgradePhase::RollingUpgrade {
                draining_generation,
                target_generation,
            } => {
                // Evict any partial resources allocated under target_generation
                self.active_resources
                    .retain(|_, gen| *gen != target_generation);
                self.current_generation = draining_generation;
                self.current_version = previous_version;
                self.phase = UpgradePhase::RolledBack {
                    restored_generation: draining_generation,
                    reason,
                };
                Ok(())
            }
            _ => Err(String::from(
                "Fix: rollback called when no rolling upgrade was in progress.",
            )),
        }
    }

    /// Allocate a resource tagged with generation id.
    pub fn register_resource(&mut self, resource_id: impl Into<String>) {
        self.active_resources
            .insert(resource_id.into(), self.current_generation);
    }

    /// Return count of active resources across all generations.
    #[must_use]
    pub fn active_resource_count(&self) -> usize {
        self.active_resources.len()
    }
}
