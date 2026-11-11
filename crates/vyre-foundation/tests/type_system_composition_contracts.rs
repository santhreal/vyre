//! How the semantic type system composes, and what each layer states about a
//! value.
//!
//! Scalars carry a width, vectors carry a lane count over a scalar, tensors
//! carry a shape and a quantization, and records and variants carry named
//! members. Each case pins the arithmetic one layer performs over the layer
//! below it, so a width, lane count, or member offset that stops composing
//! fails here rather than at a lowering boundary.

use vyre_foundation::ir::ExecutionScope;
use vyre_foundation::types::*;

#[test]
fn scalar_and_vector_orthogonal_types() {
    let u32_t = ScalarType::u32();
    assert_eq!(u32_t.bit_width(), 32);
    assert_eq!(u32_t.byte_width(), 4);
    assert!(u32_t.is_integer());
    assert!(!u32_t.is_float());

    let f16_t = ScalarType::f16();
    assert_eq!(f16_t.bit_width(), 16);
    assert_eq!(f16_t.byte_width(), 2);
    assert!(f16_t.is_float());

    let vec4_f32 = VectorType::vec4(ScalarType::f32());
    assert_eq!(vec4_f32.lanes, 4);
    assert_eq!(vec4_f32.total_bits(), 128);
    assert_eq!(vec4_f32.total_bytes(), 16);
}

#[test]
fn tensor_and_quantization_orthogonal_types() {
    let interner = ShapeInterner::new();
    let d0 = interner.symbol("batch");
    let d1 = interner.constant(512);
    let shape = interner.intern_shape(&[d0, d1]);

    let tensor_dense = TensorType::dense_row_major(ScalarType::f32(), shape);
    assert!(tensor_dense.sparsity.is_dense());
    assert_eq!(tensor_dense.quantization, None);

    let quant = QuantizationMeaning::Symmetric {
        scale_symbol: "tensor_scale".into(),
        bits: 8,
    };
    let tensor_quant = tensor_dense.clone().with_quantization(quant);
    assert_eq!(tensor_quant.quantization.as_ref().unwrap().bits(), 8);

    let tensor_sparse = TensorType::sparse(ScalarType::f32(), shape, Sparsity::Csr);
    assert!(!tensor_sparse.sparsity.is_dense());
    assert_eq!(tensor_sparse.sparsity.name(), "csr");
}

#[test]
fn capability_and_ownership_orthogonal_types() {
    let ro = ResourceCapability::ReadOnly;
    let rw = ResourceCapability::ReadWrite;
    let atomic = ResourceCapability::Atomic;

    assert!(ro.can_read());
    assert!(!ro.can_write());
    assert!(!ro.can_atomic());

    assert!(rw.can_read());
    assert!(rw.can_write());
    assert!(!rw.can_atomic());

    assert!(atomic.can_atomic());

    assert_eq!(ro.join(rw), ResourceCapability::ReadWrite);
    assert_eq!(rw.join(atomic), ResourceCapability::Atomic);

    let imm = OwnershipMutability::Immutable;
    let mut_excl = OwnershipMutability::ExclusiveOwned;
    let linear = OwnershipMutability::LinearConsumed;

    assert!(!imm.is_mutable());
    assert!(imm.can_alias());

    assert!(mut_excl.is_mutable());
    assert!(!mut_excl.can_alias());

    assert!(linear.is_linear());
    assert!(!linear.can_alias());
}

#[test]
fn record_and_variant_orthogonal_types() {
    let f1 = RecordField {
        name: "index".into(),
        ty: Box::new(SemanticType::scalar(ScalarType::u32())),
        offset_bytes: Some(0),
    };
    let f2 = RecordField {
        name: "weight".into(),
        ty: Box::new(SemanticType::scalar(ScalarType::f32())),
        offset_bytes: Some(4),
    };
    let rec = RecordType::new(vec![f1, f2]);
    assert_eq!(rec.len(), 2);
    assert_eq!(rec.field("index").unwrap().name, "index");
    assert_eq!(rec.field("missing"), None);

    let v1 = VariantCase {
        name: "Empty".into(),
        tag: 0,
        payload: None,
    };
    let v2 = VariantCase {
        name: "Value".into(),
        tag: 1,
        payload: Some(Box::new(SemanticType::scalar(ScalarType::f64()))),
    };
    let var = VariantType::new(ScalarType::u8(), vec![v1, v2]);
    assert_eq!(var.case_by_name("Empty").unwrap().tag, 0);
    assert_eq!(var.case_by_tag(1).unwrap().name, "Value");
}

#[test]
fn lifetime_and_numerical_contract_types() {
    let epoch1 = LifetimeEpoch::new(1, ExecutionScope::Workgroup).with_region_token("region_0");
    let epoch2 = LifetimeEpoch::new(2, ExecutionScope::Workgroup);

    assert!(epoch1.encloses(&epoch2));
    assert!(!epoch2.encloses(&epoch1));

    let strict = NumericalContract::strict_ieee();
    assert!(!strict.fast_math);
    assert!(!strict.flush_subnormals);

    let fast = NumericalContract::fast_math();
    assert!(fast.fast_math);
    assert!(fast.flush_subnormals);
}
