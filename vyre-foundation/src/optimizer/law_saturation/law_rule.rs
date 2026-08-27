//! Region rewrites derived from the laws a combine declares.
//!
//! Nothing here names an operator. The rewrite set is read out of
//! [`crate::algebraic_law_registry`] at construction time: for every combine,
//! under the law id its element exactness selects, each declared law that
//! states an equality between two region graphs becomes a rewrite. A combine
//! that registers a law tomorrow contributes its rewrites tomorrow, and an
//! operator whose laws are registered by an out-of-tree extension is answered
//! without appearing in this crate.
//!
//! # Which laws state a region equality
//!
//! [`law_derivation`] is the one answer. A law that constrains the operator's
//! response to its operands without equating two terms (monotonicity, bounds,
//! the zero-product property) derives no rewrite, and neither does a law whose
//! statement names a companion operator by op id, because a scalar operator id
//! is not resolvable from a combine law id. Both are recorded rather than
//! skipped: `AlgebraicLaw` is `#[non_exhaustive]`, so a new law reaches the
//! `Unrecorded` answer and the closure suite turns red until its derivation or
//! its refusal is written down.

use smallvec::{smallvec, SmallVec};
use vyre_spec::{AlgebraicLaw, BinOp, CombineKind, RegionLawFamily};

use crate::algebraic_law_registry::laws_for_op;
use crate::optimizer::rewrite_contract::RewriteWitness;

/// The equality one derived rewrite applies to a matched region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DerivedRewriteKind {
    /// `f(a, b) = f(b, a)`.
    Commute,
    /// `f(f(a, b), c) = f(a, f(b, c))`.
    ReassociateRight,
    /// `f(a, f(b, c)) = f(f(a, b), c)`.
    ReassociateLeft,
    /// `f(a, e) = a`.
    RightIdentity {
        /// Identity element, as the law states it.
        element: u32,
    },
    /// `f(e, a) = a`.
    LeftIdentity {
        /// Identity element, as the law states it.
        element: u32,
    },
    /// `f(a, a) = a`.
    Idempotent,
    /// `f(a, z) = z`.
    RightAbsorbing {
        /// Absorbing element, as the law states it.
        element: u32,
    },
    /// `f(z, a) = z`.
    LeftAbsorbing {
        /// Absorbing element, as the law states it.
        element: u32,
    },
}

/// At most two rewrites come out of one law.
pub type DerivedKinds = SmallVec<[DerivedRewriteKind; 2]>;

/// What one declared law contributes to the derived rewrite set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LawDerivation {
    /// The law equates two region graphs, and these are the rewrites.
    Region(DerivedKinds),
    /// The law constrains the operator's response to its operands rather than
    /// equating two terms. The string states which property.
    Predicate(&'static str),
    /// The law's statement names a companion operator by op id, which a
    /// combine law id does not resolve to a scalar operator.
    CompanionOperator(&'static str),
    /// The law is newer than this derivation. The closure suite fails on it.
    Unrecorded,
}

/// One rewrite the declared laws authorize, with the evidence that authorizes it.
#[derive(Debug, Clone)]
pub struct DerivedRewrite {
    /// Stable rewrite name, for reports.
    pub name: &'static str,
    /// Law family the rewrite cites.
    pub law: RegionLawFamily,
    /// Evidence that the equality the rewrite adds is a true equality.
    pub witness: RewriteWitness,
    /// Operator the rewrite matches.
    pub op: BinOp,
    /// Law id the rewrite was derived from.
    pub law_id: &'static str,
    /// Equality the rewrite applies.
    pub kind: DerivedRewriteKind,
}

/// Whether `law` states an equality between region graphs, and which.
#[must_use]
pub fn law_derivation(law: &AlgebraicLaw) -> LawDerivation {
    match law {
        AlgebraicLaw::Commutative => LawDerivation::Region(smallvec![DerivedRewriteKind::Commute]),
        AlgebraicLaw::Associative | AlgebraicLaw::CategoricalAssociative => {
            LawDerivation::Region(smallvec![
                DerivedRewriteKind::ReassociateRight,
                DerivedRewriteKind::ReassociateLeft
            ])
        }
        AlgebraicLaw::Identity { element } => LawDerivation::Region(smallvec![
            DerivedRewriteKind::LeftIdentity { element: *element },
            DerivedRewriteKind::RightIdentity { element: *element }
        ]),
        AlgebraicLaw::LeftIdentity { element } => {
            LawDerivation::Region(smallvec![DerivedRewriteKind::LeftIdentity {
                element: *element
            }])
        }
        AlgebraicLaw::RightIdentity { element } => {
            LawDerivation::Region(smallvec![DerivedRewriteKind::RightIdentity {
                element: *element
            }])
        }
        AlgebraicLaw::Idempotent => {
            LawDerivation::Region(smallvec![DerivedRewriteKind::Idempotent])
        }
        AlgebraicLaw::Absorbing { element } => LawDerivation::Region(smallvec![
            DerivedRewriteKind::LeftAbsorbing { element: *element },
            DerivedRewriteKind::RightAbsorbing { element: *element }
        ]),
        AlgebraicLaw::LeftAbsorbing { element } => {
            LawDerivation::Region(smallvec![DerivedRewriteKind::LeftAbsorbing {
                element: *element
            }])
        }
        AlgebraicLaw::RightAbsorbing { element } => {
            LawDerivation::Region(smallvec![DerivedRewriteKind::RightAbsorbing {
                element: *element
            }])
        }
        AlgebraicLaw::Monotone | AlgebraicLaw::Monotonic { .. } => LawDerivation::Predicate(
            "monotonicity constrains the operator's response to ordered operands and equates no \
             two terms",
        ),
        AlgebraicLaw::Bounded { .. } => {
            LawDerivation::Predicate("a bound constrains the result range and equates no two terms")
        }
        AlgebraicLaw::ZeroProduct { .. } => LawDerivation::Predicate(
            "the zero-product property constrains which operands can produce zero and equates no \
             two terms",
        ),
        AlgebraicLaw::Trichotomy { .. } => LawDerivation::Predicate(
            "trichotomy states that exactly one of three comparisons holds and equates no two \
             terms",
        ),
        AlgebraicLaw::SelfInverse { .. } => LawDerivation::Predicate(
            "the result of f(a, a) is a constant of the element type, and the mirror recognises a \
             literal only where the program already holds one",
        ),
        AlgebraicLaw::Involution => LawDerivation::CompanionOperator(
            "involution is a law of a unary operator, and the mirror decomposes binary operators \
             only",
        ),
        AlgebraicLaw::DeMorgan { .. }
        | AlgebraicLaw::Complement { .. }
        | AlgebraicLaw::DistributiveOver { .. }
        | AlgebraicLaw::LatticeAbsorption { .. }
        | AlgebraicLaw::InverseOf { .. } => LawDerivation::CompanionOperator(
            "the statement names a companion operator by op id, and a combine law id does not \
             resolve to the scalar operator that carries it",
        ),
        AlgebraicLaw::Custom { .. } => LawDerivation::Predicate(
            "a custom law is a predicate over witness values, which states no rewritable equality",
        ),
        AlgebraicLaw::CategoricalIdentity => LawDerivation::CompanionOperator(
            "composition with the identity arrow is a law of dispatch-graph composition, not of \
             the operator's two operands",
        ),
        _ => LawDerivation::Unrecorded,
    }
}

/// Every rewrite the registry authorizes for scalar combines.
///
/// `exact` states whether the element type the expression combines is exact. A
/// rounding element type selects the rounding law id, under which the registry
/// declares no associativity, so no reassociation is derived for it. A bitwise
/// combine is exact whatever the element type.
#[must_use]
pub fn derived_rewrites(exact: bool) -> Vec<DerivedRewrite> {
    let mut out = Vec::new();
    for combine in CombineKind::ALL {
        let law_id = combine.law_id(exact || combine.is_bitwise());
        let op = combine.scalar_binop();
        for law in laws_for_op(law_id) {
            if let LawDerivation::Region(kinds) = law_derivation(law) {
                out.extend(kinds.into_iter().map(|kind| DerivedRewrite {
                    name: rewrite_name(kind),
                    law: RegionLawFamily::Algebraic,
                    witness: RewriteWitness::Structural(
                        "the equality is the algebraic law the combine registers under the law id \
                         its element exactness selects",
                    ),
                    op,
                    law_id,
                    kind,
                }));
            }
        }
    }
    out
}

const fn rewrite_name(kind: DerivedRewriteKind) -> &'static str {
    match kind {
        DerivedRewriteKind::Commute => "law_commute",
        DerivedRewriteKind::ReassociateRight => "law_reassociate_right",
        DerivedRewriteKind::ReassociateLeft => "law_reassociate_left",
        DerivedRewriteKind::RightIdentity { .. } => "law_right_identity",
        DerivedRewriteKind::LeftIdentity { .. } => "law_left_identity",
        DerivedRewriteKind::Idempotent => "law_idempotent",
        DerivedRewriteKind::RightAbsorbing { .. } => "law_right_absorbing",
        DerivedRewriteKind::LeftAbsorbing { .. } => "law_left_absorbing",
    }
}
