//! Declared contract for every rewriting optimizer pass.
//!
//! A rewrite that ships without a stated level, precondition set, effect set,
//! numerical contract, proof witness, profitability claim, and expansion bound
//! is a rewrite nothing can rank, refuse, or prove. Two consumers make a
//! contract load-bearing rather than descriptive: the scheduler refuses a pass
//! whose result exceeds its declared expansion, and candidate search refuses a
//! pass whose witness records opacity instead of proof.
//!
//! Contracts for this crate's passes are one reviewable table in `shipped`;
//! a pass registered by another crate submits its own. The closure test
//! requires the registered pass set and the contract set to be the same set, so
//! a new pass turns the suite red until its contract is recorded.

use std::fmt;

use vyre_spec::IrLevel;

/// Program property a rewrite requires before it may fire.
///
/// Preconditions are universally quantified over the programs a pass sees: a
/// declared precondition states what the pass checks for itself, not a property
/// a caller is trusted to have arranged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RewritePrecondition {
    /// Every operand the rewrite reads is a literal.
    LiteralOperands,
    /// Integer element types only: the rewrite is not float-safe.
    IntegerElements,
    /// The rewritten region performs no memory effect.
    EffectFreeRegion,
    /// The rewritten region carries no barrier, atomic, or fence.
    SynchronizationFreeRegion,
    /// Loop bounds in the rewritten region are compile-time constants.
    ConstantLoopBounds,
    /// The rewrite moves memory only between buffers it proves disjoint.
    DisjointBuffers,
    /// A dominating guard bounds every index the rewrite moves.
    BoundedIndices,
    /// The rewritten value has exactly one reaching definition.
    SingleReachingDefinition,
    /// The rewrite runs only where the declared buffer ABI stays fixed.
    AbiPreserved,
}

/// Program effect class a rewrite may add, remove, or move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RewriteEffect {
    /// Loads from a declared buffer.
    Reads,
    /// Stores to a declared buffer.
    Writes,
    /// Atomic read-modify-write.
    Atomic,
    /// Barrier, fence, or ordering.
    Synchronization,
    /// Branch or loop structure.
    ControlFlow,
    /// The declared buffer ABI.
    BufferAbi,
    /// Workspace or shared allocation.
    Allocation,
}

/// What a rewrite is allowed to do to computed values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum NumericalContract {
    /// Every result is bit-identical before and after.
    BitExact,
    /// Integer results are identical, wrapping included.
    IntegerWrapping,
    /// Floating-point results may differ by reassociation.
    FloatReassociation,
    /// Floating-point results may differ by contraction into a fused multiply.
    FloatContraction,
    /// Results are computed at a lower declared precision.
    ReducedPrecision,
}

/// Evidence that authorizes a rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RewriteWitness {
    /// Discharged by the solver gate, naming the obligation families whose
    /// rewrite ids this pass fires.
    Obligation(&'static [&'static str]),
    /// Proved by construction, with the argument stated here.
    ///
    /// The argument belongs in the contract rather than in a module docstring,
    /// because a reader checking whether a rewrite is authorized reads the
    /// registry and a docstring is not part of it.
    Structural(&'static str),
    /// No proof is recorded, and the reason is.
    ///
    /// An opaque rewrite may still run where a caller asks for it by name; it
    /// cannot enter candidate search, where the compiler would be choosing an
    /// unproved program on its own.
    Opaque(&'static str),
}

impl RewriteWitness {
    /// Whether a rewrite carrying this witness may enter candidate search.
    #[must_use]
    pub const fn admits_candidate_search(self) -> bool {
        !matches!(self, Self::Opaque(_))
    }

    /// Obligation families this witness discharges; empty unless it names them.
    #[must_use]
    pub const fn obligation_families(self) -> &'static [&'static str] {
        match self {
            Self::Obligation(families) => families,
            Self::Structural(_) | Self::Opaque(_) => &[],
        }
    }

    /// Stable label of the evidence class, for reports and projections.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::Obligation(_) => "obligation",
            Self::Structural(_) => "structural",
            Self::Opaque(_) => "opaque",
        }
    }

    /// Stated argument, empty for a witness that names obligation families
    /// instead of stating one.
    #[must_use]
    pub const fn argument(self) -> &'static str {
        match self {
            Self::Obligation(_) => "",
            Self::Structural(argument) | Self::Opaque(argument) => argument,
        }
    }
}

/// Why running a rewrite is expected to pay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum ProfitabilityFact {
    /// Fewer IR nodes execute.
    RemovesNodes,
    /// Fewer dispatches run.
    RemovesLaunches,
    /// Fewer bytes move.
    RemovesTraffic,
    /// The dependence chain is shorter.
    ShortensDependence,
    /// A wider vector access becomes legal.
    WidensVector,
    /// A later fusion becomes legal.
    EnablesFusion,
    /// More invocations stay resident.
    RaisesOccupancy,
    /// Fewer barriers or atomics execute.
    ReducesSynchronization,
    /// The program is normalized so later rewrites match.
    Canonicalizes,
}

/// Growth a rewrite is allowed to cause.
///
/// Every rewrite that can duplicate code states a bound, because an unbounded
/// rewrite composed with itself is how a bounded compile stops being bounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum BoundedExpansion {
    /// The result never has more nodes than the input.
    NonGrowing,
    /// The result has at most this multiple of the input node count.
    NodeFactor(u32),
    /// The result has at most this many nodes more than the input.
    NodeBudget(u32),
}

impl BoundedExpansion {
    /// Whether a run that grew `before` nodes into `after` stayed in bounds.
    #[must_use]
    pub const fn admits(self, before: usize, after: usize) -> bool {
        match self {
            Self::NonGrowing => after <= before,
            Self::NodeFactor(factor) => match before.checked_mul(factor as usize) {
                Some(limit) => after <= limit,
                None => true,
            },
            Self::NodeBudget(budget) => match before.checked_add(budget as usize) {
                Some(limit) => after <= limit,
                None => true,
            },
        }
    }
}

impl fmt::Display for BoundedExpansion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonGrowing => f.write_str("non-growing"),
            Self::NodeFactor(factor) => write!(f, "node factor {factor}"),
            Self::NodeBudget(budget) => write!(f, "node budget {budget}"),
        }
    }
}

/// The declared contract of one rewriting pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RewriteContract {
    /// Pass name this contract belongs to, matching `PassMetadata::name`.
    pub pass: &'static str,
    /// Compiler level the rewrite owns.
    pub level: IrLevel,
    /// Properties the pass proves before rewriting.
    pub preconditions: &'static [RewritePrecondition],
    /// Effect classes the rewrite adds, removes, or moves.
    pub effects: &'static [RewriteEffect],
    /// What the rewrite may do to computed values.
    pub numerical: NumericalContract,
    /// Evidence authorizing the rewrite.
    pub witness: RewriteWitness,
    /// Why running it is expected to pay.
    pub profitability: &'static [ProfitabilityFact],
    /// Growth it is allowed to cause.
    pub expansion: BoundedExpansion,
}

impl RewriteContract {
    /// Whether this rewrite may be explored by candidate search.
    #[must_use]
    pub const fn admits_candidate_search(&self) -> bool {
        self.witness.admits_candidate_search()
    }

    /// Whether a rewrite at this level may state physical execution policy.
    #[must_use]
    pub const fn admits_physical_policy(&self) -> bool {
        self.level.admits_physical_policy()
    }
}

mod shipped;

/// Link-time registration of one rewrite contract.
///
/// This crate's own passes are recorded in the `shipped` table, which is one
/// reviewable list beside one registry. A pass registered by another crate
/// submits its contract instead:
///
/// ```ignore
/// inventory::submit! {
///     RewriteContractRegistration {
///         contract: RewriteContract { pass: "my_pass", .. },
///     }
/// }
/// ```
#[derive(Debug)]
pub struct RewriteContractRegistration {
    /// The declared contract.
    pub contract: RewriteContract,
}

inventory::collect!(RewriteContractRegistration);

/// Every declared rewrite contract, shipped and externally registered.
#[must_use]
pub fn registered_rewrite_contracts() -> Vec<&'static RewriteContract> {
    let mut contracts: Vec<&'static RewriteContract> = shipped::SHIPPED_CONTRACTS
        .iter()
        .chain(
            inventory::iter::<RewriteContractRegistration>
                .into_iter()
                .map(|registration| &registration.contract),
        )
        .collect();
    contracts.sort_unstable_by_key(|contract| contract.pass);
    contracts
}

/// Contract declared for `pass`, if one is declared.
#[must_use]
pub fn contract_for_pass(pass: &str) -> Option<&'static RewriteContract> {
    registered_rewrite_contracts()
        .into_iter()
        .find(|contract| contract.pass == pass)
}
