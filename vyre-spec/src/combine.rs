use crate::{AtomicOp, CollectiveOp, SubgroupReduceOp};

/// Every combine, in law-id order, with the lower-case spelling, the frozen
/// subgroup wire tag, and whether it operates on bit patterns.
///
/// Atomic read-modify-writes, subgroup reductions and collectives name the same
/// seven combines under three vocabularies. This table is where they are named:
/// it expands into [`CombineKind`], into [`SubgroupReduceOp`], and into the
/// projections between them, so a combine cannot exist in one vocabulary and be
/// missing from another.
macro_rules! for_each_combine {
    ($emit:ident) => {
        $emit! {
            Add "add" 0x01 false,
            Mul "mul" 0x02 false,
            Min "min" 0x03 false,
            Max "max" 0x04 false,
            And "and" 0x05 true,
            Or "or" 0x06 true,
            Xor "xor" 0x07 true,
        }
    };
}

pub(crate) use for_each_combine;

macro_rules! define_combine_kind {
    ($($variant:ident $spelling:literal $tag:literal $bitwise:literal,)+) => {
        /// The combine one operator applies, independent of where it appears.
        ///
        /// One kind answers for all three vocabularies, so a consumer asking
        /// whether an order of application is observable consults one law table
        /// instead of three.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum CombineKind {
            $(
                #[doc = concat!("The `", $spelling, "` combine.")]
                $variant,
            )+
        }

        impl CombineKind {
            /// Every combine kind, in law-id order.
            pub const ALL: [Self; [$(Self::$variant),+].len()] = [$(Self::$variant),+];

            /// Op id under which this combine's algebraic laws are registered.
            ///
            /// The same combine over an exact element type and over a rounding
            /// one are separate operators for law purposes: `x + (y + z)` and
            /// `(x + y) + z` agree on integers and disagree on floats.
            #[must_use]
            pub const fn law_id(self, exact: bool) -> &'static str {
                match (self, exact) {
                    $(
                        (Self::$variant, true) => concat!("vyre.combine.exact.", $spelling),
                        (Self::$variant, false) => concat!("vyre.combine.rounding.", $spelling),
                    )+
                }
            }

            /// Whether this combine operates on bit patterns rather than
            /// numeric value.
            ///
            /// A bitwise combine is exact whatever the element type, because it
            /// does not round.
            #[must_use]
            pub const fn is_bitwise(self) -> bool {
                match self {
                    $(Self::$variant => $bitwise,)+
                }
            }
        }

        impl SubgroupReduceOp {
            /// The combine this reduction applies.
            #[must_use]
            pub const fn combine(self) -> CombineKind {
                match self {
                    $(Self::$variant => CombineKind::$variant,)+
                }
            }
        }
    };
}

for_each_combine!(define_combine_kind);

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
