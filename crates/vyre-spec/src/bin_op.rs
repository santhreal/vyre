//! Frozen binary-operation discriminants for primitive operation metadata.
// TAG RESERVATIONS: Add=0x01, Sub=0x02, Mul=0x03, Div=0x04, Mod=0x05,
// BitAnd=0x06, BitOr=0x07, BitXor=0x08, Shl=0x09, Shr=0x0A, Eq=0x0B,
// Ne=0x0C, Lt=0x0D, Gt=0x0E, AbsDiff=0x0F, Le=0x10, Ge=0x11,
// And=0x12, Or=0x13, Min=0x14, Max=0x15, SaturatingAdd=0x16,
// SaturatingSub=0x17, SaturatingMul=0x18, Shuffle=0x19, Ballot=0x1A,
// WaveReduce=0x1B, WaveBroadcast=0x1C, RotateLeft=0x1D, WrappingAdd=0x1F, WrappingSub=0x20,
// RotateRight=0x1E, MulHigh=0x21, 0x22..=0x7F reserved, Opaque=0x80.

use crate::extension::ExtensionBinOpId;

/// Computational intensity class for a binary operation.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Deserialize, serde::Serialize,
)]
pub enum OpIntensity {
    /// Zero-cost (bitcasts, aliasing).
    Free,
    /// Single-cycle ALU (Add, Sub, Bitwise).
    Light,
    /// Multi-cycle ALU (Mul, Div, Mod).
    Medium,
    /// High latency / Register heavy (transcendentals, subgroup ops).
    Heavy,
}

/// What a binary operator's result type is, as a function of its operands.
///
/// The type an expression has and the operand discipline a validator enforces
/// are two questions with one answer per operator, and both used to be spelled
/// out as an operator list at the point that asked. Two lists are two chances
/// to forget an operator, and `BinOp` is `#[non_exhaustive]`, so every list
/// downstream ended in a catch-all that classified a new operator without
/// anybody choosing. [`BinOp::result_class`] is the one exhaustive answer, and
/// adding a variant fails to compile there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum BinOpResult {
    /// The operand type, which must be numeric and must match on both sides.
    Numeric,
    /// `Bool`, whatever the operands were.
    Predicate,
    /// An unsigned integer, whatever the operands were.
    Integer,
    /// Declared by the extension rather than by this contract.
    Extension,
}

/// What swapping an operator's operands does to its result.
///
/// Operand order is asked about by the canonical wire form, the canonicalize
/// pass, and common-subexpression keying, and each one used to carry its own
/// operator list. The three lists disagreed: the wire form normalized `min(b, a)`
/// while the pass and the subexpression key did not, and a new operator landed in
/// none of them. [`BinOp::operand_swap`] is the one exhaustive answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum OperandSwap {
    /// Swapping changes the result. The operator is ordered.
    Ordered,
    /// Swapping preserves the computed value, and may change the bits some
    /// element type produces: floating-point `Add` and `Mul` carry the NaN
    /// payload of whichever operand arrived first, and `Min` and `Max` apply the
    /// NaN rule of the device they lower to.
    ValuePreserving,
    /// Swapping preserves the exact bits for every element type the operator
    /// accepts.
    BitPreserving,
}

/// Binary operation kind in the frozen data contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
#[non_exhaustive]
pub enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Remainder.
    Mod,
    /// Wrapping addition.
    WrappingAdd,
    /// Wrapping subtraction.
    WrappingSub,
    /// Bitwise AND.
    BitAnd,
    /// Bitwise OR.
    BitOr,
    /// Bitwise XOR.
    BitXor,
    /// Shift left.
    Shl,
    /// Shift right.
    Shr,
    /// Equality.
    Eq,
    /// Inequality.
    Ne,
    /// Less than.
    Lt,
    /// Greater than.
    Gt,
    /// Less than or equal.
    Le,
    /// Greater than or equal.
    Ge,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Unsigned absolute difference.
    AbsDiff,
    /// Minimum (f32).
    Min,
    /// Maximum (f32).
    Max,
    /// Saturating addition.
    SaturatingAdd,
    /// Saturating subtraction.
    SaturatingSub,
    /// Saturating multiplication.
    SaturatingMul,
    /// GPU subgroup shuffle.
    Shuffle,
    /// GPU subgroup ballot.
    Ballot,
    /// GPU subgroup reduction.
    WaveReduce,
    /// GPU subgroup broadcast.
    WaveBroadcast,
    /// Rotate-left.
    RotateLeft,
    /// Rotate-right.
    RotateRight,
    /// Unsigned multiply-high: upper 32 bits of `(left × right)` treated
    /// as a 64-bit product. Enables Granlund-Montgomery strength reduction
    /// of integer division by constant to 2 instructions.
    MulHigh,
    /// Extension-declared binary operator.
    Opaque(ExtensionBinOpId),
}

impl_builtin_wire_tag!(BinOp, Opaque, {
    Add => 0x01,
    Sub => 0x02,
    Mul => 0x03,
    Div => 0x04,
    Mod => 0x05,
    BitAnd => 0x06,
    BitOr => 0x07,
    BitXor => 0x08,
    Shl => 0x09,
    Shr => 0x0A,
    Eq => 0x0B,
    Ne => 0x0C,
    Lt => 0x0D,
    Gt => 0x0E,
    AbsDiff => 0x0F,
    Le => 0x10,
    Ge => 0x11,
    And => 0x12,
    Or => 0x13,
    Min => 0x14,
    Max => 0x15,
    SaturatingAdd => 0x16,
    SaturatingSub => 0x17,
    SaturatingMul => 0x18,
    Shuffle => 0x19,
    Ballot => 0x1A,
    WaveReduce => 0x1B,
    WaveBroadcast => 0x1C,
    RotateLeft => 0x1D,
    RotateRight => 0x1E,
    WrappingAdd => 0x1F,
    WrappingSub => 0x20,
    MulHigh => 0x21,
});

impl BinOp {
    /// Return the static computational intensity of this operation.
    #[must_use]
    pub fn intensity(&self) -> OpIntensity {
        match self {
            Self::Add
            | Self::Sub
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::Shl
            | Self::Shr
            | Self::WrappingAdd
            | Self::WrappingSub
            | Self::RotateLeft
            | Self::RotateRight
            | Self::SaturatingAdd
            | Self::SaturatingSub
            | Self::SaturatingMul
            | Self::AbsDiff => OpIntensity::Light,
            Self::Ballot | Self::Shuffle | Self::WaveReduce | Self::WaveBroadcast => {
                OpIntensity::Heavy
            }
            _ => OpIntensity::Medium,
        }
    }

    /// The class this operator's result type falls in.
    ///
    /// Exhaustive with no catch-all arm, deliberately. `BinOp` is
    /// `#[non_exhaustive]`, so no crate outside this one can write an
    /// exhaustive match over it, and every consumer that tried ended in a
    /// catch-all that gave a new operator whatever answer the last arm
    /// happened to hold. Adding a variant is a compile error here instead, in
    /// the same patch that adds it.
    #[must_use]
    pub const fn result_class(self) -> BinOpResult {
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::Min
            | Self::Max
            | Self::SaturatingAdd
            | Self::SaturatingSub
            | Self::SaturatingMul => BinOpResult::Numeric,
            Self::Eq
            | Self::Ne
            | Self::Lt
            | Self::Gt
            | Self::Le
            | Self::Ge
            | Self::And
            | Self::Or => BinOpResult::Predicate,
            Self::Mod
            | Self::WrappingAdd
            | Self::WrappingSub
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::Shl
            | Self::Shr
            | Self::RotateLeft
            | Self::RotateRight
            | Self::AbsDiff
            | Self::MulHigh
            | Self::Shuffle
            | Self::Ballot
            | Self::WaveReduce
            | Self::WaveBroadcast => BinOpResult::Integer,
            Self::Opaque(_) => BinOpResult::Extension,
        }
    }

    /// True when both operands must be numeric: `u32`, `i32`, or `f32`.
    ///
    /// Every operator whose result is its operand type, plus `AbsDiff`, whose
    /// operands are numeric even though its result is unsigned. Derived from
    /// [`Self::result_class`] rather than listed again, so the two answers
    /// cannot disagree about an operator.
    #[must_use]
    pub const fn takes_numeric_operands(self) -> bool {
        matches!(self.result_class(), BinOpResult::Numeric) || matches!(self, Self::AbsDiff)
    }

    /// What swapping this operator's operands does to its result.
    ///
    /// Exhaustive with no catch-all arm, for the reason
    /// [`Self::result_class`] states: `BinOp` is `#[non_exhaustive]`, so a
    /// consumer outside this crate cannot write the match, and every consumer
    /// that wrote its own operator list ended up disagreeing with the others.
    /// Adding a variant is a compile error here instead.
    #[must_use]
    pub const fn operand_swap(self) -> OperandSwap {
        match self {
            // Integer and boolean operators whose swap is exact.
            Self::WrappingAdd
            | Self::SaturatingAdd
            | Self::SaturatingMul
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::Eq
            | Self::Ne
            | Self::And
            | Self::Or
            | Self::AbsDiff
            | Self::MulHigh => OperandSwap::BitPreserving,
            // Same value on every input, different bits on some float input.
            Self::Add | Self::Mul | Self::Min | Self::Max => OperandSwap::ValuePreserving,
            Self::Sub
            | Self::Div
            | Self::Mod
            | Self::WrappingSub
            | Self::SaturatingSub
            | Self::Shl
            | Self::Shr
            | Self::Lt
            | Self::Gt
            | Self::Le
            | Self::Ge
            | Self::RotateLeft
            | Self::RotateRight
            | Self::Shuffle
            | Self::Ballot
            | Self::WaveReduce
            | Self::WaveBroadcast => OperandSwap::Ordered,
            // An extension declares its own semantics, so no law is derived.
            Self::Opaque(_) => OperandSwap::Ordered,
        }
    }

    /// True when swapping the operands preserves the computed value.
    #[must_use]
    pub const fn commutes(self) -> bool {
        !matches!(self.operand_swap(), OperandSwap::Ordered)
    }

    /// True when swapping the operands preserves the exact bits for every
    /// element type the operator accepts.
    #[must_use]
    pub const fn commutes_bit_exactly(self) -> bool {
        matches!(self.operand_swap(), OperandSwap::BitPreserving)
    }
}
