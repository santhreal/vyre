//! Runtime value truthiness and element-decode contracts.

use proptest::prelude::*;
use vyre_reference::value::Value;

#[test]
fn neg_zero_truthiness_is_false() {
    assert!(!Value::Float(-0.0).truthy());
}

#[test]
fn pos_zero_truthiness_is_false() {
    assert!(!Value::Float(0.0).truthy());
}

#[test]
fn nonzero_float_truthiness_is_true() {
    assert!(Value::Float(1.0).truthy());
    assert!(Value::Float(-1.0).truthy());
    assert!(Value::Float(f64::INFINITY).truthy());
    assert!(Value::Float(f64::NEG_INFINITY).truthy());
}

#[test]
fn f32_element_decode_canonicalizes_subnormal_and_nan_payload_bits() {
    let positive_subnormal =
        Value::from_element_bytes(vyre_foundation::ir::DataType::F32, &1u32.to_le_bytes())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - f32 positive subnormal decode must succeed");
    assert_eq!(
        positive_subnormal.try_as_f32().unwrap().to_bits(),
        0x0000_0000
    );

    let negative_subnormal =
        Value::from_element_bytes(vyre_foundation::ir::DataType::F32, &0x8000_0001u32.to_le_bytes())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - f32 negative subnormal decode must succeed");
    assert_eq!(
        negative_subnormal.try_as_f32().unwrap().to_bits(),
        0x8000_0000
    );

    let payload_nan =
        Value::from_element_bytes(vyre_foundation::ir::DataType::F32, &0x7fa0_0001u32.to_le_bytes())
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - f32 payload NaN decode must succeed");
    assert_eq!(payload_nan.try_as_f32().unwrap().to_bits(), 0x7fc0_0000);
}

proptest! {
    #[test]
    fn neg_zero_select_branches_to_false(
        positive_sign in proptest::bool::ANY,
    ) {
        let zero = if positive_sign { 0.0_f64 } else { -0.0_f64 };
        prop_assert!(!Value::Float(zero).truthy(),
            "Value::Float({zero}).truthy() must be false to match backend bool(0.0)/bool(-0.0) semantics");
    }
}
