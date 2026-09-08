//! The zero value of a buffer element type.
//!
//! A contraction seeds its accumulator and pads a partially filled tile with a
//! zero of the element type it is contracting. A literal of the wrong width is
//! rejected by the IR validator, so the mapping belongs in one place rather
//! than at every accumulator site.

use vyre_foundation::ir::{DataType, Expr};

/// The additive zero of `dtype`, or `None` when no scalar literal denotes one.
///
/// The float16 types have no literal of their own and are expressed as a cast
/// from `f32`, which is the conversion the backends already lower.
///
/// `DataType` is `#[non_exhaustive]`, so this cannot close over its variants at
/// compile time. It fails closed instead: an element type this function does
/// not name has no zero, and a caller rejects the program rather than seeding
/// an accumulator with a literal of the wrong width. Widening the answer is a
/// deliberate edit here, never a silent consequence of adding a type.
#[must_use]
pub(crate) fn element_zero(dtype: &DataType) -> Option<Expr> {
    match dtype {
        DataType::U8 | DataType::U16 | DataType::U32 => Some(Expr::u32(0)),
        DataType::I8 | DataType::I16 | DataType::I32 => Some(Expr::i32(0)),
        DataType::F32 => Some(Expr::f32(0.0)),
        DataType::F64 => Some(Expr::f64(0.0)),
        DataType::F16 | DataType::BF16 => Some(Expr::cast(dtype.clone(), Expr::f32(0.0))),
        DataType::Bool => Some(Expr::bool(false)),
        // Fail closed. Byte blobs, arrays, vectors, tensors, sparse and
        // quantized layouts, handles and the emulated 64-bit integers hold no
        // scalar accumulator, and an unnamed new type is refused the same way.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zero must be a literal of the element type's own width, not a `u32`
    /// standing in for every type. A wrong-width seed is what the IR validator
    /// rejects at the store that writes it.
    #[test]
    fn each_scalar_element_type_zeroes_at_its_own_width() {
        assert_eq!(element_zero(&DataType::U32), Some(Expr::u32(0)));
        assert_eq!(element_zero(&DataType::I32), Some(Expr::i32(0)));
        assert_eq!(element_zero(&DataType::F32), Some(Expr::f32(0.0)));
        assert_eq!(element_zero(&DataType::Bool), Some(Expr::bool(false)));
        assert_eq!(
            element_zero(&DataType::F16),
            Some(Expr::cast(DataType::F16, Expr::f32(0.0)))
        );
        assert_eq!(
            element_zero(&DataType::BF16),
            Some(Expr::cast(DataType::BF16, Expr::f32(0.0)))
        );
    }

    /// An element type a scalar accumulator cannot hold has no zero, so a
    /// caller rejects the program instead of seeding it with a wrong literal.
    #[test]
    fn an_element_type_no_scalar_accumulator_holds_has_no_zero() {
        assert_eq!(element_zero(&DataType::Bytes), None);
        assert_eq!(element_zero(&DataType::U64), None);
        assert_eq!(element_zero(&DataType::Vec2U32), None);
        assert_eq!(element_zero(&DataType::Tensor), None);
    }
}
