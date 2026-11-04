//! `DataType` width contracts: bytes, bits, and what has no compile-time width.
//!
//! `IntrinsicTable::missing_backends` is covered by `spec_contract_errors.rs`,
//! which owns the whole missing-backend family including partial population.
use crate::{DataType, OpSignature};

#[test]
fn data_type_min_bytes_is_monotonic_for_integer_scalars() {
    assert_eq!(DataType::U32.min_bytes(), 4);
    assert_eq!(DataType::I32.min_bytes(), 4);
    assert_eq!(DataType::Bool.min_bytes(), 4);
    assert_eq!(DataType::U64.min_bytes(), 8);
    assert_eq!(DataType::Vec2U32.min_bytes(), 8);
    assert_eq!(DataType::Vec4U32.min_bytes(), 16);
    assert_eq!(DataType::Bytes.min_bytes(), 0);
}

#[test]
fn op_signature_min_input_bytes_sums_inputs() {
    let sig = OpSignature {
        inputs: vec![DataType::U32, DataType::U64, DataType::Vec4U32],
        output: DataType::U32,
        input_params: None,
        output_params: None,
        contract: None,
    };
    assert_eq!(sig.min_input_bytes(), 4 + 8 + 16);
}

#[test]
fn sub_byte_types_report_four_bits() {
    assert_eq!(DataType::I4.bit_width(), Some(4));
    assert_eq!(DataType::FP4.bit_width(), Some(4));
    assert_eq!(DataType::NF4.bit_width(), Some(4));
}

#[test]
fn standard_scalars_report_natural_width() {
    assert_eq!(DataType::U8.bit_width(), Some(8));
    assert_eq!(DataType::I8.bit_width(), Some(8));
    assert_eq!(DataType::F8E4M3.bit_width(), Some(8));
    assert_eq!(DataType::F8E5M2.bit_width(), Some(8));
    assert_eq!(DataType::U16.bit_width(), Some(16));
    assert_eq!(DataType::F16.bit_width(), Some(16));
    assert_eq!(DataType::BF16.bit_width(), Some(16));
    assert_eq!(DataType::U32.bit_width(), Some(32));
    assert_eq!(DataType::F32.bit_width(), Some(32));
    assert_eq!(DataType::Bool.bit_width(), Some(32));
    assert_eq!(DataType::U64.bit_width(), Some(64));
    assert_eq!(DataType::F64.bit_width(), Some(64));
    assert_eq!(DataType::Vec2U32.bit_width(), Some(64));
    assert_eq!(DataType::Vec4U32.bit_width(), Some(128));
}

#[test]
fn packed_int4_buffer_math() {
    let count = 1024usize;
    let bits = count * DataType::I4.bit_width().unwrap();
    let bytes = bits.div_ceil(8);
    assert_eq!(bytes, 512);
    assert_eq!(count * DataType::I4.size_bytes().unwrap(), 1024);
}

#[test]
fn vec_scales_inner_bit_width() {
    let v = DataType::Vec {
        element: Box::new(DataType::U32),
        count: 4,
    };
    assert_eq!(v.bit_width(), Some(128));

    let v4 = DataType::Vec {
        element: Box::new(DataType::I4),
        count: 8,
    };
    assert_eq!(v4.bit_width(), Some(32));
}

#[test]
fn variable_and_extension_types_have_no_compile_time_width() {
    assert_eq!(DataType::Tensor.bit_width(), None);
    assert_eq!(
        DataType::TensorShaped {
            element: Box::new(DataType::F32),
            shape: smallvec::smallvec![4, 4],
        }
        .bit_width(),
        None
    );
    assert_eq!(DataType::Array { element_size: 16 }.bit_width(), None);
    assert_eq!(
        DataType::SparseCsr {
            element: Box::new(DataType::F32),
        }
        .bit_width(),
        None
    );
    assert_eq!(
        DataType::Opaque(crate::extension::ExtensionDataTypeId(7)).bit_width(),
        None
    );
}
