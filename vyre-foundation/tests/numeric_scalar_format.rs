//! Which data types a numeric contract can be stated over.
//!
//! WHY: [`ScalarFormat`] decides whether a value has an error bound at all. A
//! type it does not classify is a value no contract covers, and the composition
//! that proves a graph's output budget silently skips it. The variant space is
//! read out of the `DataType` declaration at run time, so adding a data type
//! turns this red until someone records whether it is scalar arithmetic.
//!
//! What it does not catch: whether the classification is the right one. A new
//! float mapped to an integer format passes here and fails the semantics case
//! below only if its exactness disagrees.

use std::collections::BTreeSet;

use vyre_foundation::numeric::ScalarFormat;
use vyre_spec::DataType;
use vyre_test_support::data_type_variants::{
    data_type_variant_samples, declared_data_type_variants, variant_name,
};

/// Data types that are not scalar arithmetic, and why.
///
/// A variant listed here has no error bound because it is not a number: a
/// handle, a byte buffer, a mesh descriptor, an opaque extension type, or a
/// truth value. A variant that carries an element type is classified through
/// that element and is not listed.
const NOT_SCALAR: &[&str] = &[
    "Array",
    "Bool",
    "Bytes",
    "DeviceMesh",
    "Handle",
    "Opaque",
    "Tensor",
];

#[test]
fn every_declared_data_type_is_classified_or_named_unclassifiable() {
    let declared = declared_data_type_variants();
    let samples = data_type_variant_samples();
    let sampled: BTreeSet<String> = samples.iter().map(variant_name).collect();
    assert!(
        declared.difference(&sampled).next().is_none(),
        "Fix: the shared fixtures no longer cover every declared DataType variant, so this \
         classification is judged against a partial enum"
    );

    let mut unclassified = Vec::new();
    for sample in &samples {
        let name = variant_name(sample);
        if ScalarFormat::of(sample).is_none() && !NOT_SCALAR.contains(&name.as_str()) {
            unclassified.push(name);
        }
    }
    assert!(
        unclassified.is_empty(),
        "Fix: {unclassified:?} carry no scalar format, so no numeric contract covers a value of \
         that type. Map each one in ScalarFormat::of, or add it to NOT_SCALAR with the reason it \
         is not arithmetic"
    );

    let stale: Vec<&&str> = NOT_SCALAR
        .iter()
        .filter(|name| !declared.contains(**name))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: {stale:?} are named unclassifiable and vyre-spec no longer declares them"
    );
}

#[test]
fn a_composite_type_is_classified_through_its_element() {
    let vector = DataType::Vec {
        element: Box::new(DataType::F16),
        count: 4,
    };
    assert_eq!(ScalarFormat::of(&vector), Some(ScalarFormat::F16));

    let quantized = DataType::Quantized {
        storage: Box::new(DataType::I4),
        scale: vyre_spec::QuantizationScale::PerTensor,
        zero_point: vyre_spec::QuantizationZeroPoint::Absent,
    };
    assert_eq!(ScalarFormat::of(&quantized), Some(ScalarFormat::I4));
}

#[test]
fn every_scalar_format_agrees_with_the_semantics_table() {
    for format in ScalarFormat::ALL {
        let semantics = format.semantics();
        assert_eq!(
            semantics.datatype,
            format.data_type(),
            "{format} reads the semantics of another type"
        );
        assert!(
            !(format.is_exact() && semantics.mantissa_bits.is_some()),
            "{format} states exact arithmetic and the semantics table gives it a mantissa"
        );
        assert_eq!(
            format.ulp_fraction().is_some(),
            semantics.mantissa_bits.is_some(),
            "{format} states a rounding step it has no mantissa for"
        );
        assert_eq!(
            ScalarFormat::of(&format.data_type()),
            Some(format),
            "{format} does not round-trip through its own data type"
        );
    }
}

#[test]
fn a_narrower_float_rounds_more_coarsely() {
    let f32_step = ScalarFormat::F32
        .ulp_fraction()
        .expect("binary32 rounds at a fraction");
    let f16_step = ScalarFormat::F16
        .ulp_fraction()
        .expect("binary16 rounds at a fraction");
    let f64_step = ScalarFormat::F64
        .ulp_fraction()
        .expect("binary64 rounds at a fraction");
    assert!(f64_step < f32_step, "binary64 is finer than binary32");
    assert!(f32_step < f16_step, "binary32 is finer than binary16");
}
