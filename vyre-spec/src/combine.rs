use crate::{AtomicOp, CollectiveOp, SubgroupReduceOp};

/// The combine one operator applies, independent of where it appears.
///
/// Atomic read-modify-writes, subgroup reductions and collectives name the same
/// seven combines under three vocabularies. One kind answers all three, so a
/// consumer asking whether an order of application is observable consults one
/// law table instead of three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CombineKind {
    /// Addition, or a sum reduction.
    Add,
    /// Multiplication, or a product reduction.
    Mul,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
}

impl CombineKind {
    /// Every combine kind, in law-id order.
    pub const ALL: [Self; 7] = [
        Self::Add,
        Self::Mul,
        Self::Min,
        Self::Max,
        Self::And,
        Self::Or,
        Self::Xor,
    ];

    /// Op id under which this combine's algebraic laws are registered.
    ///
    /// The same combine over an exact element type and over a rounding one are
    /// separate operators for law purposes: `x + (y + z)` and `(x + y) + z`
    /// agree on integers and disagree on floats.
    #[must_use]
    pub const fn law_id(self, exact: bool) -> &'static str {
        match (self, exact) {
            (Self::Add, true) => "vyre.combine.exact.add",
            (Self::Add, false) => "vyre.combine.rounding.add",
            (Self::Mul, true) => "vyre.combine.exact.mul",
            (Self::Mul, false) => "vyre.combine.rounding.mul",
            (Self::Min, true) => "vyre.combine.exact.min",
            (Self::Min, false) => "vyre.combine.rounding.min",
            (Self::Max, true) => "vyre.combine.exact.max",
            (Self::Max, false) => "vyre.combine.rounding.max",
            (Self::And, true) => "vyre.combine.exact.and",
            (Self::And, false) => "vyre.combine.rounding.and",
            (Self::Or, true) => "vyre.combine.exact.or",
            (Self::Or, false) => "vyre.combine.rounding.or",
            (Self::Xor, true) => "vyre.combine.exact.xor",
            (Self::Xor, false) => "vyre.combine.rounding.xor",
        }
    }

    /// Whether this combine operates on bit patterns rather than numeric value.
    ///
    /// A bitwise combine is exact whatever the element type, because it does not
    /// round.
    #[must_use]
    pub const fn is_bitwise(self) -> bool {
        match self {
            Self::And | Self::Or | Self::Xor => true,
            Self::Add | Self::Mul | Self::Min | Self::Max => false,
        }
    }
}

impl AtomicOp {
    /// The combine this atomic applies, or `None` when the operator's result
    /// depends on the order of application whatever its element type.
    ///
    /// An exchange reads back the value it displaced, a compare-exchange is
    /// conditional on that value, `FetchNand` is non-associative, and an
    /// LRU update is a recency stamp, so none of them is a combine. An
    /// extension atomic states its own laws under its own id.
    #[must_use]
    pub const fn combine(&self) -> Option<CombineKind> {
        match self {
            Self::Add => Some(CombineKind::Add),
            Self::Or => Some(CombineKind::Or),
            Self::And => Some(CombineKind::And),
            Self::Xor => Some(CombineKind::Xor),
            Self::Min => Some(CombineKind::Min),
            Self::Max => Some(CombineKind::Max),
            Self::Exchange
            | Self::CompareExchange
            | Self::CompareExchangeWeak
            | Self::FetchNand
            | Self::LruUpdate
            | Self::Opaque(_) => None,
        }
    }
}

impl SubgroupReduceOp {
    /// The combine this reduction applies.
    #[must_use]
    pub const fn combine(self) -> CombineKind {
        match self {
            Self::Add => CombineKind::Add,
            Self::Mul => CombineKind::Mul,
            Self::Min => CombineKind::Min,
            Self::Max => CombineKind::Max,
            Self::And => CombineKind::And,
            Self::Or => CombineKind::Or,
            Self::Xor => CombineKind::Xor,
        }
    }
}

impl CollectiveOp {
    /// The combine this collective applies.
    #[must_use]
    pub const fn combine(self) -> CombineKind {
        match self {
            Self::Sum => CombineKind::Add,
            Self::Min => CombineKind::Min,
            Self::Max => CombineKind::Max,
            Self::BitAnd => CombineKind::And,
            Self::BitOr => CombineKind::Or,
            Self::BitXor => CombineKind::Xor,
        }
    }
}
