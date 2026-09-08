//! Typed target property predicates and hardware property rules.
//!
//! Replaces opaque closure predicates with typed, serializable, deterministic
//! target and cost model facts.

use serde::{Deserialize, Serialize};
use super::{EClassId, EGraph, ENodeLang, Rule};
use crate::optimizer::rewrite_contract::RewriteWitness;

/// Typed, serializable target hardware fact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetFact {
    /// Subgroup/warp size (e.g. 32 on NVIDIA, 64 on AMD, 16 on Intel/Apple).
    SubgroupSize(u32),
    /// Maximum workgroup shared memory in bytes.
    SharedMemoryCapacity(u64),
    /// Availability of dedicated matrix multiply-accumulate (MMA / Tensor Core) hardware.
    TensorCoreAvailable,
    /// Availability of asynchronous copy engine (e.g. cp.async).
    AsyncCopySupported,
    /// Subgroup shuffle operations available without shared memory staging.
    SubgroupShuffleAvailable,
    /// Maximum scalar registers per thread before spill risk.
    MaxRegistersPerThread(u32),
    /// FP16 native compute available.
    Float16Supported,
    /// Custom named hardware property.
    Custom(String),
}

/// Cost model fact evaluated during equality saturation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostModelFact {
    /// Maximum allowable register footprint score.
    pub max_register_pressure: f64,
    /// Maximum allowable memory traffic in bytes.
    pub max_memory_traffic: u64,
    /// Minimum required arithmetic intensity (FLOP/Byte).
    pub min_arithmetic_intensity: f64,
}

impl Default for CostModelFact {
    fn default() -> Self {
        Self {
            max_register_pressure: 1.0,
            max_memory_traffic: 1024 * 1024 * 1024,
            min_arithmetic_intensity: 1.0,
        }
    }
}

/// A rewrite rule gated on a set of typed [`TargetFact`] requirements.
///
/// Unlike opaque closure-based gating, `HardwarePropertyRule` has a deterministic
/// identity, serializable structure, and verifiable precondition list.
pub struct HardwarePropertyRule<L: ENodeLang> {
    /// Inner rewrite rule.
    inner: Box<dyn Rule<L>>,
    /// Required target facts that must all hold for the rule to fire.
    required_facts: Vec<TargetFact>,
    /// Active target environment facts.
    active_facts: Vec<TargetFact>,
}

impl<L: ENodeLang> HardwarePropertyRule<L> {
    /// Create a new hardware-gated rule with required facts.
    #[must_use]
    pub fn new(
        inner: Box<dyn Rule<L>>,
        required_facts: Vec<TargetFact>,
        active_facts: Vec<TargetFact>,
    ) -> Self {
        Self {
            inner,
            required_facts,
            active_facts,
        }
    }

    /// Check if all required target facts are satisfied by the active environment.
    #[must_use]
    pub fn is_satisfied(&self) -> bool {
        self.required_facts
            .iter()
            .all(|req| self.active_facts.contains(req))
    }

    /// Return the list of required target facts.
    #[must_use]
    pub fn required_facts(&self) -> &[TargetFact] {
        &self.required_facts
    }
}

impl<L: ENodeLang> Rule<L> for HardwarePropertyRule<L> {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn witness(&self) -> RewriteWitness {
        self.inner.witness()
    }

    fn matches(&self, egraph: &EGraph<L>) -> Vec<(EClassId, EClassId)> {
        if self.is_satisfied() {
            self.inner.matches(egraph)
        } else {
            Vec::new()
        }
    }
}
