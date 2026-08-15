//! Which flat `DataType` forms a buffer element can be.
//!
//! # Why this is its own module
//!
//! The fixture tables in [`crate::data_type_variants`] need `vyre-foundation`
//! and `smallvec` to build nested payloads, so they sit behind `ir-fixtures`.
//! This list needs neither: a flat form is a bare discriminant plus, for
//! `Array`, an element size. Keeping it out of that feature is what lets
//! `vyre-spec`'s own suites use it, and they are exactly the suites that must
//! not disagree with the fixtures about which flat forms exist.
//!
//! `DataType` is declared in `vyre-spec` and re-exported unchanged as
//! `vyre_foundation::ir::DataType`, so a consumer on either path gets the same
//! type and the same list.

use vyre_spec::DataType;

/// The flat `DataType` forms a buffer declaration can carry as its element.
///
/// `element_size` parameterises `DataType::Array`, the one flat form with a
/// payload. The nested forms, `Handle` and `Opaque` are not here: a buffer
/// element table and a cast-target table are different sets, and a suite that
/// needs the nested forms builds them from these leaves.
#[must_use]
pub fn flat_buffer_element_types(element_size: usize) -> Vec<DataType> {
    vec![
        DataType::U8,
        DataType::U16,
        DataType::U32,
        DataType::I8,
        DataType::I16,
        DataType::I32,
        DataType::I64,
        DataType::U64,
        DataType::Vec2U32,
        DataType::Vec4U32,
        DataType::Bool,
        DataType::Bytes,
        DataType::Array { element_size },
        DataType::F16,
        DataType::BF16,
        DataType::F32,
        DataType::F64,
        DataType::Tensor,
    ]
}
