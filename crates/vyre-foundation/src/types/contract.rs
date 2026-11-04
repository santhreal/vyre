//! Closed numerical contracts and floating point evaluation rules.

/// IEEE floating-point rounding mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (IEEE default).
    NearestEven,
    /// Round toward zero (truncation).
    TowardZero,
    /// Round toward positive infinity (ceiling).
    TowardPositiveInfinity,
    /// Round toward negative infinity (floor).
    TowardNegativeInfinity,
}

/// Arithmetic overflow / saturation behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SaturationMode {
    /// Two's complement wrapping on overflow.
    Wrap,
    /// Saturate at minimum / maximum representable value.
    Saturate,
    /// Trap / fault on arithmetic overflow.
    TrapOnOverflow,
}

/// Numerical contract governing arithmetic transformations and accuracy guarantees.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Deserialize, serde::Serialize)]
pub struct NumericalContract {
    /// Enable aggressive algebraic rewrites that may alter precision.
    pub fast_math: bool,
    /// Assume arguments and results are finite (no NaNs or infinities).
    pub finite_math_only: bool,
    /// Allow reassociation of associative operators.
    pub allow_reassoc: bool,
    /// Flush subnormal (denormal) values to zero.
    pub flush_subnormals: bool,
    /// Active rounding mode.
    pub rounding: RoundingMode,
    /// Active saturation mode.
    pub saturation: SaturationMode,
}

impl NumericalContract {
    /// Strict IEEE-754 compliant contract with no approximations.
    #[must_use]
    pub const fn strict_ieee() -> Self {
        Self {
            fast_math: false,
            finite_math_only: false,
            allow_reassoc: false,
            flush_subnormals: false,
            rounding: RoundingMode::NearestEven,
            saturation: SaturationMode::Wrap,
        }
    }

    /// Fast-math contract permitting algebraic reassociation and subnormal flushing.
    #[must_use]
    pub const fn fast_math() -> Self {
        Self {
            fast_math: true,
            finite_math_only: true,
            allow_reassoc: true,
            flush_subnormals: true,
            rounding: RoundingMode::NearestEven,
            saturation: SaturationMode::Wrap,
        }
    }
}
