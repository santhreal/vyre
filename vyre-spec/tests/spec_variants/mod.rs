//! The hand-written `DataType` and `CollectiveOp` variant tables, and the
//! proptest generators that draw from them.
//!
//! A schema is a wire format, so a sweep that generates only part of one proves
//! the round trip only for that part, and a table copied per target drifts
//! variant by variant. Both copies of the `DataType` generator had already
//! diverged: the signature byte-accounting sweep was missing quantized, opaque,
//! device-mesh and block-sparse types, so nothing checked that an `OpSignature`
//! carrying one of them survives serialization. The tables live here so adding a
//! variant widens every sweep at once.
//!
//! `DataType` and `CollectiveOp` are `#[non_exhaustive]` and expose no variant
//! iterator, so these lists cannot be derived from the type. They are written by
//! hand exactly once instead.
#![allow(dead_code)]

use proptest::prelude::*;
use vyre_spec::extension::ExtensionDataTypeId;
use vyre_spec::{CollectiveOp, DataType, QuantizationScale, QuantizationZeroPoint, TypeId};

/// Storage element types a quantized `DataType` may wrap.
///
/// `DataType::is_quantized_storage` answers for exactly this set;
/// `data_type_surface.rs` holds the contract that the two agree.
pub(crate) const QUANTIZED_STORAGE_TYPES: [DataType; 9] = [
    DataType::I4,
    DataType::I8,
    DataType::I16,
    DataType::U8,
    DataType::U16,
    DataType::F8E4M3,
    DataType::F8E5M2,
    DataType::FP4,
    DataType::NF4,
];

/// Every `DataType` that carries neither an element type nor a payload.
pub(crate) const SCALAR_LEAF_TYPES: [DataType; 22] = [
    DataType::U8,
    DataType::U16,
    DataType::U32,
    DataType::U64,
    DataType::I8,
    DataType::I16,
    DataType::I32,
    DataType::I64,
    DataType::Bool,
    DataType::F16,
    DataType::BF16,
    DataType::F32,
    DataType::F64,
    DataType::F8E4M3,
    DataType::F8E5M2,
    DataType::I4,
    DataType::FP4,
    DataType::NF4,
    DataType::Vec2U32,
    DataType::Vec4U32,
    DataType::Bytes,
    DataType::Tensor,
];

/// Storage types a quantized `DataType` may wrap.
pub(crate) fn quantized_storage_strategy() -> impl Strategy<Value = DataType> {
    prop::sample::select(QUANTIZED_STORAGE_TYPES.to_vec())
}

/// Every `DataType` that carries no element type.
///
/// The scalar leaves keep their combined weight so that adding a parametrized
/// arm below does not quietly starve them.
fn leaf_strategy() -> impl Strategy<Value = DataType> {
    prop_oneof![
        SCALAR_LEAF_TYPES.len() as u32 => prop::sample::select(SCALAR_LEAF_TYPES.to_vec()),
        1 => any::<u32>().prop_map(|raw| DataType::Handle(TypeId(raw))),
        1 => (1usize..=64usize).prop_map(|element_size| DataType::Array { element_size }),
        1 => "[a-z][a-z0-9_.-]{0,48}"
            .prop_map(|name| DataType::Opaque(ExtensionDataTypeId::from_name(&name))),
        1 => prop::collection::vec(1u32..=16, 1..=3).prop_map(|axes| DataType::DeviceMesh {
            axes: axes.as_slice().into()
        }),
    ]
}

/// An arbitrary `DataType`, including the composite variants that nest one.
pub(crate) fn data_type_strategy() -> BoxedStrategy<DataType> {
    leaf_strategy()
        .prop_recursive(3, 64, 4, |inner| {
            let scale = prop_oneof![
                Just(QuantizationScale::PerTensor),
                (0u32..=4u32).prop_map(|axis| QuantizationScale::PerChannel { axis }),
                (1u32..=256u32).prop_map(|group_size| QuantizationScale::PerGroup { group_size }),
            ];
            let zero_point = prop_oneof![
                Just(QuantizationZeroPoint::Absent),
                Just(QuantizationZeroPoint::PerTensor),
                (0u32..=4u32).prop_map(|axis| QuantizationZeroPoint::PerChannel { axis }),
                (1u32..=256u32)
                    .prop_map(|group_size| QuantizationZeroPoint::PerGroup { group_size }),
            ];
            prop_oneof![
                (inner.clone(), 1u8..=16u8).prop_map(|(element, count)| DataType::Vec {
                    element: Box::new(element),
                    count,
                }),
                (inner.clone(), prop::collection::vec(1u32..=16, 0..=4)).prop_map(
                    |(element, shape)| DataType::TensorShaped {
                        element: Box::new(element),
                        shape: shape.as_slice().into(),
                    },
                ),
                inner.clone().prop_map(|element| DataType::SparseCsr {
                    element: Box::new(element),
                }),
                inner.clone().prop_map(|element| DataType::SparseCoo {
                    element: Box::new(element),
                }),
                (inner, 1u32..=16u32, 1u32..=16u32).prop_map(
                    |(element, block_rows, block_cols)| DataType::SparseBsr {
                        element: Box::new(element),
                        block_rows,
                        block_cols,
                    },
                ),
                (quantized_storage_strategy(), scale, zero_point).prop_map(
                    |(storage, scale, zero_point)| DataType::Quantized {
                        storage: Box::new(storage),
                        scale,
                        zero_point,
                    }
                ),
            ]
        })
        .boxed()
}

/// Every `CollectiveOp` a program may carry.
///
/// `CollectiveOp` is `#[non_exhaustive]` with no iterator, so this list is
/// written by hand exactly once. A second copy would let a new reduction
/// operator reach one wire sweep and not the other.
pub(crate) fn collective_op_strategy() -> impl Strategy<Value = CollectiveOp> {
    prop_oneof![
        Just(CollectiveOp::Sum),
        Just(CollectiveOp::Min),
        Just(CollectiveOp::Max),
        Just(CollectiveOp::BitAnd),
        Just(CollectiveOp::BitOr),
        Just(CollectiveOp::BitXor),
    ]
}
