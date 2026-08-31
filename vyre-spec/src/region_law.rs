//! Frozen vocabulary of the law families a rewrite may cite.
//!
//! A rewrite is authorized by a law, not by the pass that happens to implement
//! it. Recording the family a law belongs to is what lets one mechanism derive
//! alternatives for a construct nobody wrote a recipe for: the derivation reads
//! the laws a construct exposes, not a list of kernels somebody anticipated.

/// Family of declarative law a rewrite cites to authorize itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum RegionLawFamily {
    /// Value-level algebra: commutativity, associativity, identity,
    /// distribution, absorption, involution.
    Algebraic,
    /// Recurrence structure: unrolling, peeling, index shifting, scan-recurrence
    /// splitting, blocked recurrence over an associative combine.
    Recurrence,
    /// Reduction structure: reassociation over an associative combine,
    /// tree versus sequential shape, partial-reduction splitting and joining.
    Reduction,
    /// Layout structure: index-map composition, transposition, tiling,
    /// packing, and access-order changes that move no value.
    Layout,
    /// Numerical reformulation: contraction into a fused multiply, reciprocal
    /// substitution, precision selection, range-based simplification.
    Numerical,
}

impl RegionLawFamily {
    /// Stable name used in reports, certificates, and generated projections.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Algebraic => "algebraic",
            Self::Recurrence => "recurrence",
            Self::Reduction => "reduction",
            Self::Layout => "layout",
            Self::Numerical => "numerical",
        }
    }

    /// Whether a law in this family may change a computed value.
    ///
    /// Only the numerical family may, and a law that does states the contract
    /// under which the difference is admitted. A reduction reassociation over
    /// floating point is numerical, not reduction, for exactly this reason.
    #[must_use]
    pub const fn admits_value_difference(self) -> bool {
        matches!(self, Self::Numerical)
    }

    /// Every family, in declaration order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Algebraic,
            Self::Recurrence,
            Self::Reduction,
            Self::Layout,
            Self::Numerical,
        ]
    }

    /// Family whose stable name is `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::all()
            .iter()
            .copied()
            .find(|family| family.name() == name)
    }
}
