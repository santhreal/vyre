//! Typed target property predicates, proof terms, and hardware property rules.
//!
//! Replaces opaque closure predicates with typed, serializable, deterministic
//! target and cost model facts, proof terms, and cache keys.

use serde::{Deserialize, Serialize};
use super::{EClassId, EGraph, ENodeLang, Rule};
use crate::optimizer::rewrite_contract::RewriteWitness;

/// Typed identity of the fact or law authorizing a rewrite rule.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleFactIdentity {
    /// Pure semantic algebraic law from the declared region law registry.
    AlgebraicLaw {
        /// Law name.
        law_name: &'static str,
        /// Law family.
        family: vyre_spec::RegionLawFamily,
    },
    /// Typed hardware capability requirement for schedule space.
    HardwareProperty {
        /// Required target facts.
        required: Vec<TargetFact>,
    },
    /// Custom named typed fact identity.
    TypedFact(&'static str),
}

/// Machine-checkable proof term certifying a rewrite rule.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProofTerm {
    /// Rule identifier.
    pub rule_name: &'static str,
    /// Theorem or formal justification text.
    pub justification: &'static str,
    /// Cryptographic digest of the formal proof obligation.
    pub obligation_digest: [u8; 32],
}

impl ProofTerm {
    /// Create a proof term with an explicit obligation digest.
    #[must_use]
    pub const fn new(
        rule_name: &'static str,
        justification: &'static str,
        obligation_digest: [u8; 32],
    ) -> Self {
        Self {
            rule_name,
            justification,
            obligation_digest,
        }
    }

    /// Construct a proof term by hashing the rule name and justification.
    #[must_use]
    pub fn from_name_and_justification(rule_name: &'static str, justification: &'static str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ProofTerm:v1:");
        hasher.update(rule_name.as_bytes());
        hasher.update(b":");
        hasher.update(justification.as_bytes());
        let obligation_digest = *hasher.finalize().as_bytes();
        Self {
            rule_name,
            justification,
            obligation_digest,
        }
    }
}

/// Deterministic cache key for rule memoization and dependency invalidation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RuleCacheKey(pub [u8; 32]);

impl RuleCacheKey {
    /// Derive a cache key from rule identity components.
    #[must_use]
    pub fn from_components(
        name: &str,
        fact: &RuleFactIdentity,
        obligation_digest: &[u8; 32],
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"RuleCacheKey:v1:");
        hasher.update(name.as_bytes());
        hasher.update(b":");
        match fact {
            RuleFactIdentity::AlgebraicLaw { law_name, family } => {
                hasher.update(b"AlgebraicLaw:");
                hasher.update(law_name.as_bytes());
                hasher.update(b":");
                hasher.update(family.name().as_bytes());
            }
            RuleFactIdentity::HardwareProperty { required } => {
                hasher.update(b"HardwareProperty:");
                for req in required {
                    let debug_str = format!("{req:?}");
                    hasher.update(debug_str.as_bytes());
                }
            }
            RuleFactIdentity::TypedFact(s) => {
                hasher.update(b"TypedFact:");
                hasher.update(s.as_bytes());
            }
        }
        hasher.update(b":");
        hasher.update(obligation_digest);
        Self(*hasher.finalize().as_bytes())
    }
}
/// Typed, serializable target hardware fact.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetFact {
    /// Subgroup width in lanes; shipped devices report 16, 32 or 64.
    SubgroupSize(u32),
    /// Maximum workgroup shared memory in bytes.
    SharedMemoryCapacity(u64),
    /// Availability of a dedicated matrix multiply-accumulate unit.
    TensorCoreAvailable,
    /// Availability of an asynchronous transfer engine.
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

    fn fact_identity(&self) -> RuleFactIdentity {
        RuleFactIdentity::HardwareProperty {
            required: self.required_facts.clone(),
        }
    }

    fn proof_term(&self) -> ProofTerm {
        self.inner.proof_term()
    }

    fn cache_key(&self) -> RuleCacheKey {
        RuleCacheKey::from_components(
            self.name(),
            &self.fact_identity(),
            &self.proof_term().obligation_digest,
        )
    }

    fn matches(&self, egraph: &EGraph<L>) -> Vec<(EClassId, EClassId)> {
        if self.is_satisfied() {
            self.inner.matches(egraph)
        } else {
            Vec::new()
        }
    }
}
