//! Frozen algebraic-law declarations that conformance engines verify per operation.

use alloc::string::String;
use alloc::vec::Vec;

use crate::float_type::FloatType;
use crate::ir_level::IrLevel;
use crate::monotonic_direction::MonotonicDirection;

/// Function pointer used by custom algebraic law checks.
///
/// The first argument is the operation under test. The second argument is the
/// witness tuple encoded as `u32` values. Returning `true` means the law holds
/// for that witness.
pub type LawCheckFn = fn(fn(&[u8]) -> Vec<u8>, &[u32]) -> bool;

/// An algebraic law that an operation must satisfy in the frozen data contract.
///
/// Laws are declared per-operation in the registry. The algebra checker
/// verifies each law exhaustively on small domains and with witnesses on full
/// domains. Example: `AlgebraicLaw::Commutative` records that `add(a, b)` and
/// `add(b, a)` must produce the same bytes.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum AlgebraicLaw {
    /// Standard notation: `forall a b . f(a,b) = f(b,a)`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Commutative;
    /// ```
    Commutative,
    /// Standard notation: `forall a b c . f(f(a,b),c) = f(a,f(b,c))`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Associative;
    /// ```
    Associative,
    /// Standard notation: `forall a . f(a,e) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Identity { element: 0 };
    /// ```
    Identity {
        /// The identity element as a `u32` value.
        element: u32,
    },
    /// Standard notation: `forall a . f(e,a) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::LeftIdentity { element: 0 };
    /// ```
    LeftIdentity {
        /// The left identity element as a `u32` value.
        element: u32,
    },
    /// Standard notation: `forall a . f(a,e) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::RightIdentity { element: 0 };
    /// ```
    RightIdentity {
        /// The right identity element as a `u32` value.
        element: u32,
    },
    /// Standard notation: `forall a . f(a,a) = e`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::SelfInverse { result: 0 };
    /// ```
    SelfInverse {
        /// The result of `f(a, a)` as a `u32` value.
        result: u32,
    },
    /// Standard notation: `forall a . f(a,a) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Idempotent;
    /// ```
    Idempotent,
    /// Standard notation: `forall a . f(a,z) = z and f(z,a) = z`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Absorbing { element: 0 };
    /// ```
    Absorbing {
        /// The absorbing element.
        element: u32,
    },
    /// Standard notation: `forall a . f(z,a) = z`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::LeftAbsorbing { element: 0 };
    /// ```
    LeftAbsorbing {
        /// The left absorbing argument.
        element: u32,
    },
    /// Standard notation: `forall a . f(a,z) = z`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::RightAbsorbing { element: 0 };
    /// ```
    RightAbsorbing {
        /// The right absorbing argument.
        element: u32,
    },
    /// Standard notation: `forall a . f(f(a)) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Involution;
    /// ```
    Involution,
    /// Standard notation: `forall a b . f(g(a,b)) = h(f(a),f(b))`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::DeMorgan { inner_op: "and", dual_op: "or" };
    /// ```
    DeMorgan {
        /// The operation on the left side.
        inner_op: &'static str,
        /// The dual operation on the right side.
        dual_op: &'static str,
    },
    /// Standard notation: `forall a b . a <= b -> f(a) <= f(b)`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Monotone;
    /// ```
    Monotone,
    /// Standard notation: `forall a b . a <= b -> f(a) <= f(b)` or
    /// `forall a b . a <= b -> f(a) >= f(b)`.
    ///
    /// ```
    /// use vyre_spec::{AlgebraicLaw, MonotonicDirection};
    /// let _law = AlgebraicLaw::Monotonic {
    ///     direction: MonotonicDirection::NonDecreasing,
    /// };
    /// ```
    Monotonic {
        /// Direction of monotonicity.
        direction: MonotonicDirection,
    },
    /// Standard notation: `forall a b . lo <= f(a,b) <= hi`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Bounded { lo: 0, hi: 32 };
    /// ```
    Bounded {
        /// Inclusive lower bound.
        lo: u32,
        /// Inclusive upper bound.
        hi: u32,
    },
    /// Standard notation: `forall a . f(a,g(a)) = universe`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Complement {
    ///     complement_op: "not",
    ///     universe: u32::MAX,
    /// };
    /// ```
    Complement {
        /// The complementary operation.
        complement_op: &'static str,
        /// The constant they sum or combine to.
        universe: u32,
    },
    /// Standard notation: `forall a b c . f(a,g(b,c)) = g(f(a,b),f(a,c))`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::DistributiveOver { over_op: "add" };
    /// ```
    DistributiveOver {
        /// The operation that this law distributes over.
        over_op: &'static str,
    },
    /// Standard notation: `forall a b . f(a,g(a,b)) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::LatticeAbsorption { dual_op: "min" };
    /// ```
    LatticeAbsorption {
        /// The dual lattice operation.
        dual_op: &'static str,
    },
    /// Standard notation: `forall a b . f(g(a,b),b) = a`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::InverseOf { op: "add" };
    /// ```
    InverseOf {
        /// The operation this operation inverts.
        op: &'static str,
    },
    /// Standard notation: `forall a b . exactly_one(lt(a,b), eq(a,b), gt(a,b))`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::Trichotomy {
    ///     less_op: "lt",
    ///     equal_op: "eq",
    ///     greater_op: "gt",
    /// };
    /// ```
    Trichotomy {
        /// Strict less-than operation id.
        less_op: &'static str,
        /// Equality operation id.
        equal_op: &'static str,
        /// Strict greater-than operation id.
        greater_op: &'static str,
    },
    /// Standard notation: `forall a b . f(a,b) = 0 -> a = 0 or b = 0`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// let _law = AlgebraicLaw::ZeroProduct { holds: true };
    /// ```
    ZeroProduct {
        /// Whether this law actually holds.
        holds: bool,
    },
    /// Standard notation: `forall x0 ... xn . predicate(x0, ..., xn)`.
    ///
    /// ```
    /// use vyre_spec::AlgebraicLaw;
    /// fn check(_op: fn(&[u8]) -> Vec<u8>, _args: &[u32]) -> bool { true }
    /// let _law = AlgebraicLaw::Custom {
    ///     name: "custom",
    ///     description: "custom predicate",
    ///     arity: 1,
    ///     check,
    /// };
    /// ```
    Custom {
        /// Human-readable name for this law.
        name: &'static str,
        /// Description of what the law asserts.
        description: &'static str,
        /// Number of `u32` witness values passed to the predicate.
        arity: usize,
        /// Predicate function that returns true when the law holds.
        check: LawCheckFn,
    },
    /// Categorical-IR contract law: the operation participates as an
    /// arrow in the dispatch-graph monoidal category. Composition with
    /// the identity arrow on either side leaves the operation
    /// unchanged (`f ∘ id = id ∘ f = f`). P-SPEC-1: vyre-spec
    /// invariants reflect the categorical-IR contract; conformance
    /// engines verify this law for every op tagged as a category
    /// arrow.
    CategoricalIdentity,
    /// Categorical-IR contract law: the operation composes
    /// associatively as a category arrow (`(h ∘ g) ∘ f = h ∘ (g ∘ f)`).
    /// P-SPEC-1.
    CategoricalAssociative,
}

impl AlgebraicLaw {
    /// Human-readable name for reporting.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Commutative => "commutative",
            Self::Associative => "associative",
            Self::Identity { .. } => "identity",
            Self::LeftIdentity { .. } => "left-identity",
            Self::RightIdentity { .. } => "right-identity",
            Self::SelfInverse { .. } => "self-inverse",
            Self::Idempotent => "idempotent",
            Self::Absorbing { .. } => "absorbing",
            Self::LeftAbsorbing { .. } => "left-absorbing",
            Self::RightAbsorbing { .. } => "right-absorbing",
            Self::Involution => "involution",
            Self::DeMorgan { .. } => "de-morgan",
            Self::Monotone => "monotone",
            Self::Monotonic { .. } => "monotonic",
            Self::Bounded { .. } => "bounded",
            Self::Complement { .. } => "complement",
            Self::DistributiveOver { .. } => "distributive",
            Self::LatticeAbsorption { .. } => "lattice-absorption",
            Self::InverseOf { .. } => "inverse-of",
            Self::Trichotomy { .. } => "trichotomy",
            Self::ZeroProduct { .. } => "zero-product",
            Self::CategoricalIdentity => "categorical-identity",
            Self::CategoricalAssociative => "categorical-associative",
            Self::Custom { name, .. } => name,
        }
    }

    /// Whether this law applies to binary operations.
    #[must_use]
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Self::Commutative
                | Self::Associative
                | Self::Identity { .. }
                | Self::LeftIdentity { .. }
                | Self::RightIdentity { .. }
                | Self::SelfInverse { .. }
                | Self::Idempotent
                | Self::Absorbing { .. }
                | Self::LeftAbsorbing { .. }
                | Self::RightAbsorbing { .. }
                | Self::Bounded { .. }
                | Self::Complement { .. }
                | Self::DistributiveOver { .. }
                | Self::LatticeAbsorption { .. }
                | Self::InverseOf { .. }
                | Self::Trichotomy { .. }
                | Self::ZeroProduct { .. }
                | Self::Custom { .. }
        )
    }

    /// Whether this law applies to unary operations.
    #[must_use]
    pub fn is_unary(&self) -> bool {
        matches!(
            self,
            Self::Involution
                | Self::Monotone
                | Self::Monotonic { .. }
                | Self::Bounded { .. }
                | Self::Complement { .. }
                | Self::DeMorgan { .. }
                | Self::Custom { .. }
        )
    }
}

impl PartialEq for AlgebraicLaw {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Commutative, Self::Commutative)
            | (Self::Associative, Self::Associative)
            | (Self::Idempotent, Self::Idempotent)
            | (Self::Involution, Self::Involution)
            | (Self::Monotone, Self::Monotone)
            | (Self::CategoricalIdentity, Self::CategoricalIdentity)
            | (Self::CategoricalAssociative, Self::CategoricalAssociative) => true,
            (Self::Identity { element: left }, Self::Identity { element: right })
            | (Self::LeftIdentity { element: left }, Self::LeftIdentity { element: right })
            | (Self::RightIdentity { element: left }, Self::RightIdentity { element: right })
            | (Self::Absorbing { element: left }, Self::Absorbing { element: right })
            | (Self::LeftAbsorbing { element: left }, Self::LeftAbsorbing { element: right })
            | (Self::RightAbsorbing { element: left }, Self::RightAbsorbing { element: right })
            | (Self::SelfInverse { result: left }, Self::SelfInverse { result: right }) => {
                left == right
            }
            (
                Self::DeMorgan {
                    inner_op: left_inner,
                    dual_op: left_dual,
                },
                Self::DeMorgan {
                    inner_op: right_inner,
                    dual_op: right_dual,
                },
            ) => left_inner == right_inner && left_dual == right_dual,
            (Self::Monotonic { direction: left }, Self::Monotonic { direction: right }) => {
                left == right
            }
            (
                Self::Bounded {
                    lo: left_lo,
                    hi: left_hi,
                },
                Self::Bounded {
                    lo: right_lo,
                    hi: right_hi,
                },
            ) => left_lo == right_lo && left_hi == right_hi,
            (
                Self::Complement {
                    complement_op: left_op,
                    universe: left_universe,
                },
                Self::Complement {
                    complement_op: right_op,
                    universe: right_universe,
                },
            ) => left_op == right_op && left_universe == right_universe,
            (
                Self::DistributiveOver { over_op: left },
                Self::DistributiveOver { over_op: right },
            )
            | (
                Self::LatticeAbsorption { dual_op: left },
                Self::LatticeAbsorption { dual_op: right },
            )
            | (Self::InverseOf { op: left }, Self::InverseOf { op: right }) => left == right,
            (
                Self::Trichotomy {
                    less_op: left_less,
                    equal_op: left_equal,
                    greater_op: left_greater,
                },
                Self::Trichotomy {
                    less_op: right_less,
                    equal_op: right_equal,
                    greater_op: right_greater,
                },
            ) => {
                left_less == right_less
                    && left_equal == right_equal
                    && left_greater == right_greater
            }
            (Self::ZeroProduct { holds: left }, Self::ZeroProduct { holds: right }) => {
                left == right
            }
            (
                Self::Custom {
                    name: left_name,
                    arity: left_arity,
                    check: left_check,
                    ..
                },
                Self::Custom {
                    name: right_name,
                    arity: right_arity,
                    check: right_check,
                    ..
                },
            ) => {
                left_name == right_name
                    && left_arity == right_arity
                    && core::ptr::fn_addr_eq(*left_check, *right_check)
            }
            _ => false,
        }
    }
}

impl Eq for AlgebraicLaw {}

fn default_custom_check(_: fn(&[u8]) -> Vec<u8>, _: &[u32]) -> bool {
    true
}

impl serde::Serialize for AlgebraicLaw {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.name())
    }
}

impl<'de> serde::Deserialize<'de> for AlgebraicLaw {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = alloc::string::String::deserialize(deserializer)?;
        match s.as_str() {
            "commutative" => Ok(Self::Commutative),
            "associative" => Ok(Self::Associative),
            "identity" => Ok(Self::Identity { element: 0 }),
            "left-identity" => Ok(Self::LeftIdentity { element: 0 }),
            "right-identity" => Ok(Self::RightIdentity { element: 0 }),
            "self-inverse" => Ok(Self::SelfInverse { result: 0 }),
            "idempotent" => Ok(Self::Idempotent),
            "absorbing" => Ok(Self::Absorbing { element: 0 }),
            "left-absorbing" => Ok(Self::LeftAbsorbing { element: 0 }),
            "right-absorbing" => Ok(Self::RightAbsorbing { element: 0 }),
            "involution" => Ok(Self::Involution),
            "de-morgan" => Ok(Self::DeMorgan {
                inner_op: "and",
                dual_op: "or",
            }),
            "monotone" => Ok(Self::Monotone),
            "monotonic" => Ok(Self::Monotonic {
                direction: MonotonicDirection::NonDecreasing,
            }),
            "bounded" => Ok(Self::Bounded {
                lo: 0,
                hi: u32::MAX,
            }),
            "complement" => Ok(Self::Complement {
                complement_op: "not",
                universe: u32::MAX,
            }),
            "distributive" => Ok(Self::DistributiveOver { over_op: "add" }),
            "lattice-absorption" => Ok(Self::LatticeAbsorption { dual_op: "min" }),
            "inverse-of" => Ok(Self::InverseOf { op: "add" }),
            "trichotomy" => Ok(Self::Trichotomy {
                less_op: "lt",
                equal_op: "eq",
                greater_op: "gt",
            }),
            "zero-product" => Ok(Self::ZeroProduct { holds: true }),
            "categorical-identity" => Ok(Self::CategoricalIdentity),
            "categorical-associative" => Ok(Self::CategoricalAssociative),
            _ => Ok(Self::Custom {
                name: "custom",
                description: "deserialized custom law",
                arity: 1,
                check: default_custom_check,
            }),
        }
    }
}
/// Precondition under which an algebraic law holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum LawGuard {
    /// Holds unconditionally across the entire operation domain.
    Unconditional,
    /// Holds only for exact (integer or bitwise) numerical types where rounding drift is zero.
    ExactOnly,
    /// Holds only when the operand is non-zero.
    NonZero,
    /// Holds only for finite numeric values (excluding NaN and infinities).
    FiniteOnly,
    /// Holds only when operands are pure and free of side effects.
    PureOnly,
    /// Holds within an inclusive integer range `[lo, hi]`.
    Range {
        /// Lower inclusive bound.
        lo: i64,
        /// Upper inclusive bound.
        hi: i64,
    },
}

impl LawGuard {
    /// Return the name of the guard for diagnostics and reports.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Unconditional => "unconditional",
            Self::ExactOnly => "exact-only",
            Self::NonZero => "non-zero",
            Self::FiniteOnly => "finite-only",
            Self::PureOnly => "pure-only",
            Self::Range { .. } => "range",
        }
    }

    /// Whether this guard imposes no preconditions.
    #[must_use]
    pub const fn is_unconditional(&self) -> bool {
        matches!(self, Self::Unconditional)
    }
}

impl Default for LawGuard {
    fn default() -> Self {
        Self::Unconditional
    }
}

/// Direction of an algebraic equivalence or rewrite law.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
pub enum LawDirection {
    /// Equivalence holds symmetrically in both directions (`a = b <-> b = a`).
    Bidirectional,
    /// Simplification or reduction oriented left-to-right (`f(a, 0) -> a`).
    LeftToRight,
    /// Expansion or synthesis oriented right-to-left.
    RightToLeft,
    /// Canonicalization ordering rule.
    Canonicalize,
}

impl LawDirection {
    /// Return the name of the law direction.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Bidirectional => "bidirectional",
            Self::LeftToRight => "left-to-right",
            Self::RightToLeft => "right-to-left",
            Self::Canonicalize => "canonicalize",
        }
    }

    /// Whether the law is bidirectional equivalence.
    #[must_use]
    pub const fn is_bidirectional(&self) -> bool {
        matches!(self, Self::Bidirectional)
    }
}

impl Default for LawDirection {
    fn default() -> Self {
        Self::Bidirectional
    }
}

/// Verification or formal proof method attached to an algebraic law.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ProofMethod {
    /// Exhaustively checked on all `u8` inputs.
    ExhaustiveU8,
    /// Exhaustively checked on all `u16` inputs.
    ExhaustiveU16,
    /// Witnessed over `u32` with a deterministic pseudo-random seed and iteration count.
    WitnessedU32 {
        /// Deterministic seed.
        seed: u64,
        /// Witness iteration count.
        count: u64,
    },
    /// Exhaustively verified over a restricted float domain.
    ExhaustiveFloat {
        /// Float type covered.
        typ: FloatType,
    },
    /// Discharged by SMT bit-vector solver logic.
    SmtQfBv {
        /// SMT-LIB logic string (e.g. `"QF_BV"`).
        logic: String,
    },
    /// Discharged by a closed decision procedure.
    DecisionProcedure {
        /// Decision procedure name.
        name: String,
    },
    /// No executable proof method provided (invalid for production contracts).
    None,
}

impl ProofMethod {
    /// Return the name of the proof method.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ExhaustiveU8 => "exhaustive-u8",
            Self::ExhaustiveU16 => "exhaustive-u16",
            Self::WitnessedU32 { .. } => "witnessed-u32",
            Self::ExhaustiveFloat { .. } => "exhaustive-float",
            Self::SmtQfBv { .. } => "smt-qf-bv",
            Self::DecisionProcedure { .. } => "decision-procedure",
            Self::None => "none",
        }
    }

    /// Whether this method provides executable proof evidence.
    #[must_use]
    pub const fn has_executable_proof(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl Default for ProofMethod {
    fn default() -> Self {
        Self::WitnessedU32 {
            seed: 0x5EED_C0DE,
            count: 1024,
        }
    }
}

/// Adversarial counterexample generator for validating and falsifying law hypotheses.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CounterexampleGenerator {
    /// Generator strategy name.
    pub name: String,
    /// Deterministic pseudo-random seed.
    pub seed: u64,
    /// Maximum sample attempts before certifying absence of counterexamples in search space.
    pub max_attempts: usize,
}

impl CounterexampleGenerator {
    /// Construct a counterexample generator with explicit parameters.
    #[must_use]
    pub fn new(name: impl Into<String>, seed: u64, max_attempts: usize) -> Self {
        Self {
            name: name.into(),
            seed,
            max_attempts,
        }
    }

    /// Construct a canonical deterministic generator with default search parameters.
    #[must_use]
    pub fn deterministic(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            seed: 0x5EED_C0DE,
            max_attempts: 1024,
        }
    }

    /// Generate a deterministic stream of adversarial input tuples for an operation of given arity.
    #[must_use]
    pub fn generate_adversarial_inputs(&self, arity: usize) -> Vec<Vec<u64>> {
        let mut results = Vec::with_capacity(self.max_attempts.min(256));
        let corner_cases: &[u64] = &[
            0,
            1,
            2,
            u64::MAX,
            u64::MAX - 1,
            u32::MAX as u64,
            (u32::MAX as u64) + 1,
            0x8000_0000,
            0x7FFF_FFFF,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
        ];
        if arity == 1 {
            for &c in corner_cases {
                results.push(alloc::vec![c]);
            }
        } else if arity == 2 {
            for &c1 in corner_cases {
                for &c2 in corner_cases {
                    results.push(alloc::vec![c1, c2]);
                    if results.len() >= self.max_attempts {
                        return results;
                    }
                }
            }
        } else {
            let mut tuple = alloc::vec![0u64; arity];
            for &c in corner_cases {
                tuple.fill(c);
                results.push(tuple.clone());
            }
        }

        let mut state = self.seed ^ 0x9E37_79B9_7F4A_7C15;
        while results.len() < self.max_attempts {
            let mut tuple = Vec::with_capacity(arity);
            for _ in 0..arity {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                tuple.push(state);
            }
            results.push(tuple);
        }
        results
    }

    /// Search for a counterexample that falsifies the given predicate `predicate(&inputs) -> bool`.
    /// Returns `Some(LawCounterexample)` if a violating input is found.
    pub fn find_counterexample<F>(
        &self,
        arity: usize,
        mut predicate: F,
    ) -> Option<LawCounterexample>
    where
        F: FnMut(&[u64]) -> bool,
    {
        let inputs_stream = self.generate_adversarial_inputs(arity);
        for inputs in inputs_stream {
            if !predicate(&inputs) {
                return Some(LawCounterexample {
                    description: alloc::format!(
                        "counterexample found by generator `{}` at input {:?}",
                        self.name,
                        inputs
                    ),
                    inputs,
                    observed: None,
                    expected: None,
                });
            }
        }
        None
    }
}

impl Default for CounterexampleGenerator {
    fn default() -> Self {
        Self::deterministic("canonical-adversarial-generator")
    }
}

/// Counterexample discovered by validation or metamorphic testing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct LawCounterexample {
    /// Human-readable explanation.
    pub description: String,
    /// Concrete input tuple that falsified the law.
    pub inputs: Vec<u64>,
    /// Observed output value, if applicable.
    pub observed: Option<u64>,
    /// Expected output value, if applicable.
    pub expected: Option<u64>,
}

/// Error encountered during law contract validation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LawValidationError {
    /// Law has no executable proof evidence.
    NoExecutableProofEvidence {
        /// Name of the rejected law.
        law: String,
    },
    /// A counterexample generator refuted the law hypothesis.
    CounterexampleFound {
        /// Name of the refuted law.
        law: String,
        /// Falsifying counterexample.
        counterexample: LawCounterexample,
    },
    /// Guard specification is inconsistent or invalid.
    InvalidGuard {
        /// Name of the law.
        law: String,
        /// Invalidation reason.
        reason: String,
    },
}

impl core::fmt::Display for LawValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoExecutableProofEvidence { law } => {
                write!(
                    f,
                    "law `{law}` rejected: no executable proof evidence provided"
                )
            }
            Self::CounterexampleFound {
                law,
                counterexample,
            } => {
                write!(f, "law `{law}` refuted: {}", counterexample.description)
            }
            Self::InvalidGuard { law, reason } => {
                write!(f, "law `{law}` has invalid guard: {reason}")
            }
        }
    }
}

/// Fully characterized algebraic law carrying guard, direction, numerical contract,
/// proof method, counterexample generator, canonical form, and affected compiler levels.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct GuardedLaw {
    /// Underlying algebraic law pattern.
    pub law: AlgebraicLaw,
    /// Rewrite directionality.
    pub direction: LawDirection,
    /// Precondition guard.
    pub guard: LawGuard,
    /// Numerical contract under which this law holds.
    pub numerical_contract: crate::op_contract::NumericBehavior,
    /// Executable proof method.
    pub proof_method: ProofMethod,
    /// Adversarial counterexample generator.
    pub counterexample_generator: CounterexampleGenerator,
    /// Optional canonical normal form representation.
    pub canonical_form: Option<String>,
    /// Compiler levels affected by this law.
    pub affected_compiler_levels: smallvec::SmallVec<[IrLevel; 4]>,
}

impl GuardedLaw {
    /// Construct a guarded law with unconditional exact semantics.
    #[must_use]
    pub fn unconditional(law: AlgebraicLaw) -> Self {
        let mut affected = smallvec::SmallVec::new();
        affected.push(IrLevel::Logical);
        affected.push(IrLevel::Schedule);
        Self {
            law,
            direction: LawDirection::Bidirectional,
            guard: LawGuard::Unconditional,
            numerical_contract: crate::op_contract::NumericBehavior::Exact,
            proof_method: ProofMethod::WitnessedU32 {
                seed: 0x5EED_C0DE,
                count: 1024,
            },
            counterexample_generator: CounterexampleGenerator::deterministic("default-generator"),
            canonical_form: None,
            affected_compiler_levels: affected,
        }
    }

    /// Attach a proof method.
    #[must_use]
    pub fn with_proof_method(mut self, proof_method: ProofMethod) -> Self {
        self.proof_method = proof_method;
        self
    }

    /// Attach a guard.
    #[must_use]
    pub fn with_guard(mut self, guard: LawGuard) -> Self {
        self.guard = guard;
        self
    }

    /// Attach a direction.
    #[must_use]
    pub fn with_direction(mut self, direction: LawDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Attach a counterexample generator.
    #[must_use]
    pub fn with_counterexample_generator(
        mut self,
        counterexample_generator: CounterexampleGenerator,
    ) -> Self {
        self.counterexample_generator = counterexample_generator;
        self
    }

    /// Attach canonical form.
    #[must_use]
    pub fn with_canonical_form(mut self, canonical_form: impl Into<String>) -> Self {
        self.canonical_form = Some(canonical_form.into());
        self
    }

    /// Validate the law: rejects laws with no executable proof evidence.
    ///
    /// # Errors
    /// Returns [`LawValidationError::NoExecutableProofEvidence`] if `proof_method` is `None`.
    pub fn validate(&self) -> Result<(), LawValidationError> {
        if !self.proof_method.has_executable_proof() {
            return Err(LawValidationError::NoExecutableProofEvidence {
                law: self.law.name().into(),
            });
        }
        Ok(())
    }

    /// Run counterexample verification against an implementation predicate.
    ///
    /// # Errors
    /// Returns [`LawValidationError::CounterexampleFound`] if a counterexample is discovered.
    pub fn verify_with_predicate<F>(
        &self,
        arity: usize,
        predicate: F,
    ) -> Result<(), LawValidationError>
    where
        F: FnMut(&[u64]) -> bool,
    {
        self.validate()?;
        if let Some(counterexample) = self
            .counterexample_generator
            .find_counterexample(arity, predicate)
        {
            return Err(LawValidationError::CounterexampleFound {
                law: self.law.name().into(),
                counterexample,
            });
        }
        Ok(())
    }
}
