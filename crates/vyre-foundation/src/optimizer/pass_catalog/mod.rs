//! Optimizer pass catalog backed by live pass registrations.

use super::{
    registered_pass_registrations, CostModelFamily, OptimizerError, OptimizerProfile,
    PassBoundaryClass, PassMetadata, PassPhase,
};

/// Stable catalog entry for one registered optimizer pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptimizationCatalogEntry {
    /// Entry classification.
    pub kind: OptimizationCatalogEntryKind,
    /// Stable pass name.
    pub name: &'static str,
    /// Subsystem owner for the optimization family.
    pub owner: &'static str,
    /// Scheduler phase.
    pub phase: PassPhase,
    /// Boundary class enforced by the scheduler/profile selector.
    pub boundary_class: PassBoundaryClass,
    /// Invariant preserved by this pass.
    pub invariant: &'static str,
    /// Benchmark family covering this optimization.
    pub benchmark: &'static str,
    /// Cost model family declared by the pass.
    pub cost_model_family: CostModelFamily,
    /// Whether this pass preserves public buffer ABI.
    pub preserves_abi: bool,
    /// Backend/runtime capabilities required before the pass may run.
    pub requires_caps: &'static [&'static str],
}

/// Kind of one optimizer catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationCatalogEntryKind {
    /// A runnable [`super::ProgramPassRegistration`] discovered by inventory.
    ExecutablePass,
}

impl OptimizationCatalogEntry {
    /// Build a catalog entry from pass metadata.
    #[must_use]
    pub fn from_metadata(metadata: PassMetadata) -> Self {
        Self {
            kind: OptimizationCatalogEntryKind::ExecutablePass,
            name: metadata.name,
            owner: owner_for(metadata),
            phase: metadata.phase,
            boundary_class: metadata.boundary_class,
            invariant: invariant_for(metadata),
            benchmark: benchmark_for(metadata),
            cost_model_family: metadata.cost_model_family,
            preserves_abi: metadata.preserves_abi,
            requires_caps: metadata.requires_caps,
        }
    }
}

/// Return every registered optimization pass in scheduled order.
///
/// # Errors
/// Returns [`OptimizerError::Scheduling`] when the pass dependency graph is
/// invalid.
pub fn optimization_catalog() -> Result<Vec<OptimizationCatalogEntry>, OptimizerError> {
    Ok(registered_pass_registrations()?
        .iter()
        .map(|registration| OptimizationCatalogEntry::from_metadata(registration.metadata))
        .collect())
}

/// Return registered optimizations accepted by a profile in scheduled order.
///
/// # Errors
/// Returns [`OptimizerError::Scheduling`] when the pass dependency graph is
/// invalid.
pub fn optimization_catalog_for_profile(
    profile: OptimizerProfile,
) -> Result<Vec<OptimizationCatalogEntry>, OptimizerError> {
    Ok(optimization_catalog()?
        .into_iter()
        .filter(|entry| {
            profile.accepts(PassMetadata {
                name: entry.name,
                requires: &[],
                invalidates: &[],
                phase: entry.phase,
                boundary_class: entry.boundary_class,
                requires_caps: entry.requires_caps,
                preserves_abi: entry.preserves_abi,
                cost_model_family: entry.cost_model_family,
            })
        })
        .collect())
}

fn owner_for(metadata: PassMetadata) -> &'static str {
    match metadata.phase {
        PassPhase::Dataflow => "dataflow",
        PassPhase::Megakernel => "vyre-runtime",
        PassPhase::Specialization => "vyre-backend-specialization",
        PassPhase::Loop | PassPhase::ScalarAlgebra | PassPhase::Canonicalization => {
            "vyre-foundation-optimizer"
        }
        PassPhase::Memory | PassPhase::FusionCse | PassPhase::Sync | PassPhase::Cleanup => {
            "vyre-foundation-optimizer"
        }
        PassPhase::Unclassified => match metadata.boundary_class {
            PassBoundaryClass::RuntimeAware => "vyre-runtime",
            PassBoundaryClass::DomainSpecific => "domain-crate",
            _ => "vyre-foundation-optimizer",
        },
    }
}

fn invariant_for(metadata: PassMetadata) -> &'static str {
    match metadata.boundary_class {
        PassBoundaryClass::AbiPreserving => {
            "preserves public buffer ABI and externally visible program semantics"
        }
        PassBoundaryClass::AbiChanging => {
            "requires explicit caller opt-in before changing public buffer ABI"
        }
        PassBoundaryClass::BackendAware => {
            "fires only when declared backend capabilities are present"
        }
        PassBoundaryClass::RuntimeAware => {
            "preserves runtime residency, launch-order, and synchronization contracts"
        }
        PassBoundaryClass::DomainSpecific => {
            "preserves owning domain graph/fact encoding contracts"
        }
        PassBoundaryClass::Unknown => "must not enter release/profile pipelines until classified",
    }
}

fn benchmark_for(metadata: PassMetadata) -> &'static str {
    match metadata.cost_model_family {
        CostModelFamily::Scalar => "foundation.optimizer.scalar",
        CostModelFamily::Loop => "foundation.optimizer.loop",
        CostModelFamily::Memory => "foundation.optimizer.memory",
        CostModelFamily::Fusion => "foundation.optimizer.fusion",
        CostModelFamily::Sync => "foundation.optimizer.sync",
        CostModelFamily::Dataflow => "dataflow.optimizer",
        CostModelFamily::Megakernel => "vyre.megakernel.optimizer",
        CostModelFamily::Unknown => "foundation.optimizer.classification-gap",
    }
}

#[cfg(test)]
#[path = "../../../tests/internal/optimizer/pass_catalog/mod.rs"]
mod tests;
