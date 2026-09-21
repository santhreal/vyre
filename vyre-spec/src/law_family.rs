//! The closed algebraic-law vocabulary and the proof obligation each member carries.
//!
//! Two hand-written arrays used to state this vocabulary: a name list in
//! `catalog_slices` and a representative list in `all_algebraic_laws`. Both were
//! two members behind [`AlgebraicLaw`], and the test that held them to the enum
//! compared their lengths to each other, so the two categorical variants were
//! invisible to every reader of the catalog and to the check that claimed to
//! close it. Both arrays are derived from [`LawFamily`] now, and the mapping
//! from a law value to its family is an exhaustive match with no catch-all: a
//! variant added to `AlgebraicLaw` fails to compile until it is given a family,
//! a name, a representative and a proof obligation.
//!
//! An obligation states what proves the law rather than asserting the law is
//! proven. A family whose statement needs a payload a bare law name cannot
//! carry, which element is the identity, which operation it distributes over,
//! which order it is monotone in, reports [`LawEvidence::MissingPayload`] and
//! yields [`ProofMethod::None`], which [`GuardedLaw::validate`] rejects.
//! Recording the payload is what turns such a family into a provable one.

use alloc::string::ToString;

use crate::algebraic_law::{
    AlgebraicLaw, CounterexampleGenerator, GuardedLaw, LawDirection, LawGuard, ProofMethod,
};
use crate::ir_level::IrLevel;
use crate::monotonic_direction::MonotonicDirection;
use crate::op_contract::NumericBehavior;

/// One member of the closed algebraic-law vocabulary.
///
/// A family is the law's identity without its payload. `identity` is one family
/// whatever element it names; `distributive` is one family whatever operation it
/// distributes over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LawFamily {
    /// `f(a,b) = f(b,a)`.
    Commutative,
    /// `f(f(a,b),c) = f(a,f(b,c))`.
    Associative,
    /// `f(a,e) = f(e,a) = a`.
    Identity,
    /// `f(e,a) = a`.
    LeftIdentity,
    /// `f(a,e) = a`.
    RightIdentity,
    /// `f(a,a) = e`.
    SelfInverse,
    /// `f(a,a) = a`.
    Idempotent,
    /// `f(a,z) = f(z,a) = z`.
    Absorbing,
    /// `f(z,a) = z`.
    LeftAbsorbing,
    /// `f(a,z) = z`.
    RightAbsorbing,
    /// `f(f(a)) = a`.
    Involution,
    /// `f(g(a,b)) = h(f(a),f(b))`.
    DeMorgan,
    /// `a <= b` implies `f(a) <= f(b)`.
    Monotone,
    /// `a <= b` implies an ordered relation between `f(a)` and `f(b)`.
    Monotonic,
    /// `lo <= f(a,b) <= hi`.
    Bounded,
    /// `f(a,g(a)) = universe`.
    Complement,
    /// `f(a,g(b,c)) = g(f(a,b),f(a,c))`.
    Distributive,
    /// `f(a,g(a,b)) = a`.
    LatticeAbsorption,
    /// `f(g(a,b),b) = a`.
    InverseOf,
    /// Exactly one of `lt`, `eq`, `gt` holds for any operand pair.
    Trichotomy,
    /// `f(a,b) = 0` implies `a = 0` or `b = 0`.
    ZeroProduct,
    /// Composition with the identity arrow leaves a category arrow unchanged.
    CategoricalIdentity,
    /// Category arrows compose associatively.
    CategoricalAssociative,
    /// A predicate the declaration carries itself.
    Custom,
}

/// Number of families in the closed vocabulary.
///
/// Raising this without adding the member to [`LawFamily::ALL`] fails the
/// density assertion below at compile time, and adding a member without raising
/// it fails the length assertion.
const FAMILY_COUNT: usize = 24;

impl LawFamily {
    /// Every family, in vocabulary order.
    pub const ALL: [Self; FAMILY_COUNT] = [
        Self::Commutative,
        Self::Associative,
        Self::Identity,
        Self::LeftIdentity,
        Self::RightIdentity,
        Self::SelfInverse,
        Self::Idempotent,
        Self::Absorbing,
        Self::LeftAbsorbing,
        Self::RightAbsorbing,
        Self::Involution,
        Self::DeMorgan,
        Self::Monotone,
        Self::Monotonic,
        Self::Bounded,
        Self::Complement,
        Self::Distributive,
        Self::LatticeAbsorption,
        Self::InverseOf,
        Self::Trichotomy,
        Self::ZeroProduct,
        Self::CategoricalIdentity,
        Self::CategoricalAssociative,
        Self::Custom,
    ];

    /// Position of this family in the vocabulary.
    ///
    /// Exhaustive with no catch-all, and the const assertions below require the
    /// positions of [`Self::ALL`] to be distinct and to cover `0..FAMILY_COUNT`.
    /// A family added to the enum therefore fails to compile until it has an
    /// index, a slot in `ALL`, and a raised `FAMILY_COUNT`.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Commutative => 0,
            Self::Associative => 1,
            Self::Identity => 2,
            Self::LeftIdentity => 3,
            Self::RightIdentity => 4,
            Self::SelfInverse => 5,
            Self::Idempotent => 6,
            Self::Absorbing => 7,
            Self::LeftAbsorbing => 8,
            Self::RightAbsorbing => 9,
            Self::Involution => 10,
            Self::DeMorgan => 11,
            Self::Monotone => 12,
            Self::Monotonic => 13,
            Self::Bounded => 14,
            Self::Complement => 15,
            Self::Distributive => 16,
            Self::LatticeAbsorption => 17,
            Self::InverseOf => 18,
            Self::Trichotomy => 19,
            Self::ZeroProduct => 20,
            Self::CategoricalIdentity => 21,
            Self::CategoricalAssociative => 22,
            Self::Custom => 23,
        }
    }

    /// Stable catalog name, the string a registration declares.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Commutative => "commutative",
            Self::Associative => "associative",
            Self::Identity => "identity",
            Self::LeftIdentity => "left-identity",
            Self::RightIdentity => "right-identity",
            Self::SelfInverse => "self-inverse",
            Self::Idempotent => "idempotent",
            Self::Absorbing => "absorbing",
            Self::LeftAbsorbing => "left-absorbing",
            Self::RightAbsorbing => "right-absorbing",
            Self::Involution => "involution",
            Self::DeMorgan => "de-morgan",
            Self::Monotone => "monotone",
            Self::Monotonic => "monotonic",
            Self::Bounded => "bounded",
            Self::Complement => "complement",
            Self::Distributive => "distributive",
            Self::LatticeAbsorption => "lattice-absorption",
            Self::InverseOf => "inverse-of",
            Self::Trichotomy => "trichotomy",
            Self::ZeroProduct => "zero-product",
            Self::CategoricalIdentity => "categorical-identity",
            Self::CategoricalAssociative => "categorical-associative",
            Self::Custom => "custom",
        }
    }

    /// The family `law` belongs to.
    ///
    /// Exhaustive over [`AlgebraicLaw`] with no catch-all. A variant added to
    /// that enum fails to compile here.
    #[must_use]
    pub const fn of(law: &AlgebraicLaw) -> Self {
        match law {
            AlgebraicLaw::Commutative => Self::Commutative,
            AlgebraicLaw::Associative => Self::Associative,
            AlgebraicLaw::Identity { .. } => Self::Identity,
            AlgebraicLaw::LeftIdentity { .. } => Self::LeftIdentity,
            AlgebraicLaw::RightIdentity { .. } => Self::RightIdentity,
            AlgebraicLaw::SelfInverse { .. } => Self::SelfInverse,
            AlgebraicLaw::Idempotent => Self::Idempotent,
            AlgebraicLaw::Absorbing { .. } => Self::Absorbing,
            AlgebraicLaw::LeftAbsorbing { .. } => Self::LeftAbsorbing,
            AlgebraicLaw::RightAbsorbing { .. } => Self::RightAbsorbing,
            AlgebraicLaw::Involution => Self::Involution,
            AlgebraicLaw::DeMorgan { .. } => Self::DeMorgan,
            AlgebraicLaw::Monotone => Self::Monotone,
            AlgebraicLaw::Monotonic { .. } => Self::Monotonic,
            AlgebraicLaw::Bounded { .. } => Self::Bounded,
            AlgebraicLaw::Complement { .. } => Self::Complement,
            AlgebraicLaw::DistributiveOver { .. } => Self::Distributive,
            AlgebraicLaw::LatticeAbsorption { .. } => Self::LatticeAbsorption,
            AlgebraicLaw::InverseOf { .. } => Self::InverseOf,
            AlgebraicLaw::Trichotomy { .. } => Self::Trichotomy,
            AlgebraicLaw::ZeroProduct { .. } => Self::ZeroProduct,
            AlgebraicLaw::CategoricalIdentity => Self::CategoricalIdentity,
            AlgebraicLaw::CategoricalAssociative => Self::CategoricalAssociative,
            AlgebraicLaw::Custom { .. } => Self::Custom,
        }
    }

    /// The family `name` spells, or `None` when no family carries that name.
    ///
    /// A registration citing a name no family carries records no law. Mapping
    /// an unrecognized name onto a custom law with a check that returns true is
    /// what let a label assert a property nothing established.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|f| f.name() == name)
    }

    /// One canonical law value for this family.
    ///
    /// The payload is a stable placeholder, not a claim about any operation:
    /// [`Self::obligation`] reports a family whose statement needs a real
    /// payload as carrying no executable evidence.
    #[must_use]
    pub const fn representative(self) -> AlgebraicLaw {
        match self {
            Self::Commutative => AlgebraicLaw::Commutative,
            Self::Associative => AlgebraicLaw::Associative,
            Self::Identity => AlgebraicLaw::Identity { element: 0 },
            Self::LeftIdentity => AlgebraicLaw::LeftIdentity { element: 0 },
            Self::RightIdentity => AlgebraicLaw::RightIdentity { element: 0 },
            Self::SelfInverse => AlgebraicLaw::SelfInverse { result: 0 },
            Self::Idempotent => AlgebraicLaw::Idempotent,
            Self::Absorbing => AlgebraicLaw::Absorbing { element: 0 },
            Self::LeftAbsorbing => AlgebraicLaw::LeftAbsorbing { element: 0 },
            Self::RightAbsorbing => AlgebraicLaw::RightAbsorbing { element: 0 },
            Self::Involution => AlgebraicLaw::Involution,
            Self::DeMorgan => AlgebraicLaw::DeMorgan {
                inner_op: "and",
                dual_op: "or",
            },
            Self::Monotone => AlgebraicLaw::Monotone,
            Self::Monotonic => AlgebraicLaw::Monotonic {
                direction: MonotonicDirection::NonDecreasing,
            },
            Self::Bounded => AlgebraicLaw::Bounded { lo: 0, hi: 32 },
            Self::Complement => AlgebraicLaw::Complement {
                complement_op: "not",
                universe: u32::MAX,
            },
            Self::Distributive => AlgebraicLaw::DistributiveOver { over_op: "add" },
            Self::LatticeAbsorption => AlgebraicLaw::LatticeAbsorption { dual_op: "min" },
            Self::InverseOf => AlgebraicLaw::InverseOf { op: "add" },
            Self::Trichotomy => AlgebraicLaw::Trichotomy {
                less_op: "lt",
                equal_op: "eq",
                greater_op: "gt",
            },
            Self::ZeroProduct => AlgebraicLaw::ZeroProduct { holds: true },
            Self::CategoricalIdentity => AlgebraicLaw::CategoricalIdentity,
            Self::CategoricalAssociative => AlgebraicLaw::CategoricalAssociative,
            Self::Custom => AlgebraicLaw::Custom {
                name: "custom",
                description: "the declaration carries no check to run",
                arity: 0,
                check: check_not_declared,
            },
        }
    }

    /// The proof obligation this family carries.
    ///
    /// Exhaustive with no catch-all.
    #[must_use]
    pub const fn obligation(self) -> LawObligation {
        // Rewrite laws reach the logical level, where the optimizer rewrites
        // `Expr`/`Node`, and the schedule level, which is what a reassociation
        // authorizes. An order or range fact rewrites nothing and reaches the
        // logical level alone. A category law is a whole-graph dispatch fact.
        const REWRITE: &[IrLevel] = &[IrLevel::Logical, IrLevel::Schedule];
        const ANALYSIS: &[IrLevel] = &[IrLevel::Logical];
        const CATEGORY: &[IrLevel] = &[IrLevel::WholeGraph, IrLevel::Logical];

        match self {
            Self::Commutative => LawObligation {
                family: self,
                direction: LawDirection::Bidirectional,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::ReferenceOracleWitness {
                    witness: "exchanged-inputs",
                },
                canonical_form: "f(a,b) = f(b,a)",
                affected_compiler_levels: REWRITE,
            },
            // Reassociation is not valid under rounding, so the law is recorded
            // only where the element type is exact.
            Self::Associative => LawObligation {
                family: self,
                direction: LawDirection::Bidirectional,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::ReferenceOracleWitness { witness: "chained" },
                canonical_form: "f(f(a,b),c) = f(a,f(b,c))",
                affected_compiler_levels: REWRITE,
            },
            Self::Idempotent => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::ReferenceOracleWitness {
                    witness: "repeated-input",
                },
                canonical_form: "f(a,a) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::SelfInverse => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::ReferenceOracleWitness { witness: "chained" },
                canonical_form: "f(f(a,b),b) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::Involution => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::ReferenceOracleWitness {
                    witness: "reapplied",
                },
                canonical_form: "f(f(a)) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::Identity => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,e) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::LeftIdentity => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(e,a) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::RightIdentity => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,e) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::Absorbing => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,z) -> z",
                affected_compiler_levels: REWRITE,
            },
            Self::LeftAbsorbing => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(z,a) -> z",
                affected_compiler_levels: REWRITE,
            },
            Self::RightAbsorbing => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,z) -> z",
                affected_compiler_levels: REWRITE,
            },
            Self::Bounded => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "lo <= f(a,b) <= hi",
                affected_compiler_levels: ANALYSIS,
            },
            Self::Complement => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,g(a)) -> universe",
                affected_compiler_levels: REWRITE,
            },
            Self::ZeroProduct => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Element,
                },
                canonical_form: "f(a,b) = 0 -> a = 0 or b = 0",
                affected_compiler_levels: ANALYSIS,
            },
            Self::DeMorgan => LawObligation {
                family: self,
                direction: LawDirection::Bidirectional,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "f(g(a,b)) = h(f(a),f(b))",
                affected_compiler_levels: REWRITE,
            },
            Self::Distributive => LawObligation {
                family: self,
                direction: LawDirection::Bidirectional,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "f(a,g(b,c)) = g(f(a,b),f(a,c))",
                affected_compiler_levels: REWRITE,
            },
            Self::LatticeAbsorption => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "f(a,g(a,b)) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::InverseOf => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::ExactOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "f(g(a,b),b) -> a",
                affected_compiler_levels: REWRITE,
            },
            Self::Monotone => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Order,
                },
                canonical_form: "a <= b -> f(a) <= f(b)",
                affected_compiler_levels: ANALYSIS,
            },
            Self::Monotonic => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Order,
                },
                canonical_form: "a <= b -> f(a) <= f(b) or f(a) >= f(b)",
                affected_compiler_levels: ANALYSIS,
            },
            Self::Trichotomy => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Order,
                },
                canonical_form: "exactly_one(lt(a,b), eq(a,b), gt(a,b))",
                affected_compiler_levels: ANALYSIS,
            },
            // A category law relates an arrow to the identity arrow and to a
            // composition, and a bare law name states neither partner.
            Self::CategoricalIdentity => LawObligation {
                family: self,
                direction: LawDirection::LeftToRight,
                guard: LawGuard::PureOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "f . id = id . f = f",
                affected_compiler_levels: CATEGORY,
            },
            Self::CategoricalAssociative => LawObligation {
                family: self,
                direction: LawDirection::Bidirectional,
                guard: LawGuard::PureOnly,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Partner,
                },
                canonical_form: "(h . g) . f = h . (g . f)",
                affected_compiler_levels: CATEGORY,
            },
            Self::Custom => LawObligation {
                family: self,
                direction: LawDirection::Canonicalize,
                guard: LawGuard::Unconditional,
                numerical_contract: NumericBehavior::Exact,
                evidence: LawEvidence::MissingPayload {
                    payload: LawPayload::Check,
                },
                canonical_form: "predicate(x0, ..., xn)",
                affected_compiler_levels: ANALYSIS,
            },
        }
    }

    /// Whether a bare declaration of this family carries executable evidence.
    #[must_use]
    pub const fn is_provable_from_name(self) -> bool {
        self.obligation().evidence.is_executable()
    }
}

/// Every family carries a distinct index, and the indices cover the whole
/// vocabulary. A family added without a slot in `ALL`, or given a duplicate or
/// out-of-range index, fails to compile here.
const _: () = {
    assert!(LawFamily::ALL.len() == FAMILY_COUNT);
    let mut seen = [false; FAMILY_COUNT];
    let mut i = 0;
    while i < FAMILY_COUNT {
        let index = LawFamily::ALL[i].index();
        assert!(index < FAMILY_COUNT);
        assert!(!seen[index]);
        seen[index] = true;
        i += 1;
    }
    let mut i = 0;
    while i < FAMILY_COUNT {
        assert!(seen[i]);
        i += 1;
    }
};

/// The payload a law's statement needs and a bare law name cannot carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LawPayload {
    /// An identity, absorbing element, universe, or bound.
    Element,
    /// The second operation the statement relates this one to.
    Partner,
    /// The order the statement compares operands in.
    Order,
    /// The predicate a custom law asserts.
    Check,
}

impl LawPayload {
    /// Stable name recorded in reports and generated projections.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Element => "element",
            Self::Partner => "partner",
            Self::Order => "order",
            Self::Check => "check",
        }
    }
}

/// What discharges a family's obligation, or why nothing can.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LawEvidence {
    /// The named metamorphic witness runs the operation's own program through
    /// the reference oracle over its registered fixture cases.
    ReferenceOracleWitness {
        /// Stable witness name.
        witness: &'static str,
    },
    /// A bare declaration of this family states no payload to substitute, so
    /// nothing executable follows from the label alone.
    MissingPayload {
        /// The payload the statement needs.
        payload: LawPayload,
    },
}

impl LawEvidence {
    /// Whether this evidence is executable.
    #[must_use]
    pub const fn is_executable(self) -> bool {
        matches!(self, Self::ReferenceOracleWitness { .. })
    }

    /// The witness name, when one runs.
    #[must_use]
    pub const fn witness(self) -> Option<&'static str> {
        match self {
            Self::ReferenceOracleWitness { witness } => Some(witness),
            Self::MissingPayload { .. } => None,
        }
    }

    /// The missing payload, when nothing runs.
    #[must_use]
    pub const fn missing_payload(self) -> Option<LawPayload> {
        match self {
            Self::ReferenceOracleWitness { .. } => None,
            Self::MissingPayload { payload } => Some(payload),
        }
    }
}

/// Direction, guard, numerical contract, proof method, counterexample
/// generator, canonical form, and affected compiler levels of one law family.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LawObligation {
    /// The family this obligation belongs to.
    pub family: LawFamily,
    /// Rewrite directionality.
    pub direction: LawDirection,
    /// Precondition under which the law holds.
    pub guard: LawGuard,
    /// Numerical behavior the law is recorded under.
    pub numerical_contract: NumericBehavior,
    /// What discharges the obligation, or why nothing can.
    pub evidence: LawEvidence,
    /// Canonical statement of the law.
    pub canonical_form: &'static str,
    /// Compiler levels the law reaches.
    pub affected_compiler_levels: &'static [IrLevel],
}

impl LawObligation {
    /// The proof method this obligation discharges by.
    ///
    /// A family with no executable evidence yields [`ProofMethod::None`], which
    /// [`GuardedLaw::validate`] rejects. That rejection is the point: a label
    /// carrying no executable evidence is not a recorded law.
    #[must_use]
    pub fn proof_method(&self) -> ProofMethod {
        match self.evidence {
            LawEvidence::ReferenceOracleWitness { witness } => {
                ProofMethod::ReferenceOracleWitness {
                    witness: witness.to_string(),
                }
            }
            LawEvidence::MissingPayload { .. } => ProofMethod::None,
        }
    }

    /// The counterexample generator that searches for a refutation.
    #[must_use]
    pub fn counterexample_generator(&self) -> CounterexampleGenerator {
        CounterexampleGenerator::deterministic(match self.evidence {
            LawEvidence::ReferenceOracleWitness { witness } => witness,
            LawEvidence::MissingPayload { payload } => payload.name(),
        })
    }

    /// Build the fully characterized law record for `law` under this obligation.
    ///
    /// # Panics
    /// Panics when `law` is not a member of [`Self::family`], which is a caller
    /// mixing two families' records.
    #[must_use]
    pub fn guarded_law(&self, law: AlgebraicLaw) -> GuardedLaw {
        assert!(
            LawFamily::of(&law) == self.family,
            "law `{}` is not a member of family `{}`",
            law.name(),
            self.family.name()
        );
        GuardedLaw {
            law,
            direction: self.direction,
            guard: self.guard,
            numerical_contract: self.numerical_contract.clone(),
            proof_method: self.proof_method(),
            counterexample_generator: self.counterexample_generator(),
            canonical_form: Some(self.canonical_form.to_string()),
            affected_compiler_levels: self.affected_compiler_levels.iter().copied().collect(),
        }
    }
}

/// The check a custom law with no declared predicate runs.
///
/// Returns false: nothing was checked, so nothing is established. The former
/// default returned true, which made every unrecognized law name assert a
/// property no code had looked at.
pub fn check_not_declared(_op: fn(&[u8]) -> alloc::vec::Vec<u8>, _args: &[u32]) -> bool {
    false
}

/// Every family's catalog name, in vocabulary order.
const FAMILY_NAMES: [&str; FAMILY_COUNT] = {
    let mut names = [""; FAMILY_COUNT];
    let mut i = 0;
    while i < FAMILY_COUNT {
        names[i] = LawFamily::ALL[i].name();
        i += 1;
    }
    names
};

/// The frozen catalog of algebraic-law names, derived from [`LawFamily`].
#[must_use]
pub fn law_family_names() -> &'static [&'static str] {
    &FAMILY_NAMES
}

/// One canonical representative per family, in vocabulary order.
const FAMILY_REPRESENTATIVES: [AlgebraicLaw; FAMILY_COUNT] = {
    let mut laws = [const { AlgebraicLaw::Commutative }; FAMILY_COUNT];
    let mut i = 0;
    while i < FAMILY_COUNT {
        laws[i] = LawFamily::ALL[i].representative();
        i += 1;
    }
    laws
};

/// One canonical representative for every algebraic-law variant.
#[must_use]
pub fn law_family_representatives() -> &'static [AlgebraicLaw] {
    &FAMILY_REPRESENTATIVES
}
