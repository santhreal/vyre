//! Test crate.

#![cfg(all(feature = "nn-attention", feature = "nn-norm"))]
#![allow(deprecated)]
use proptest::prelude::*;
use vyre::ir::Program;
use vyre_foundation::operation::SemanticOperation;
use vyre_foundation::optimizer::optimize;
use vyre_libs::operation_catalog::all_entries;

fn entry(id: &'static str) -> SemanticOperation {
    all_entries()
        .find(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("Fix: missing canonical operation registration for {id}"))
}

fn bytes_from_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn output_bytes(program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let values = vyre_reference::reference_inputs(program, inputs.to_vec());
    vyre_reference::reference_eval(program, &values)
        .unwrap_or_else(|error| panic!("Fix: reference execution failed: {error}"))
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

fn harness_path_outputs(entry: &SemanticOperation, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let program = entry
        .program()
        .expect("Fix: registered library operation must provide a neutral builder");
    let errors = vyre::validate(&program);
    assert!(
        errors.is_empty(),
        "Fix: {} failed validation on adversarial f32 input: {:?}",
        entry.id,
        errors
            .into_iter()
            .map(|error| error.message().to_string())
            .collect::<Vec<_>>()
    );
    let wire = program
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: {} failed wire encode: {error}", entry.id));
    let decoded = Program::from_wire(&wire)
        .unwrap_or_else(|error| panic!("Fix: {} failed wire decode: {error}", entry.id));
    let optimized_once = optimize(decoded).expect("registered optimizer must converge");
    let optimized_twice =
        optimize(optimized_once.clone()).expect("registered optimizer must converge");
    assert_eq!(
        optimized_once, optimized_twice,
        "Fix: {} optimize() must be idempotent on adversarial f32 input",
        entry.id
    );
    output_bytes(&optimized_once, inputs)
}

fn special_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        Just(f32::NAN),
        Just(f32::from_bits(0x7fc0_0001)),
        Just(f32::INFINITY),
        Just(f32::NEG_INFINITY),
        Just(0.0f32),
        Just(-0.0f32),
        Just(f32::MAX),
        Just(f32::MIN_POSITIVE),
        Just(f32::from_bits(1)),
        any::<u32>().prop_map(f32::from_bits),
    ]
}

fn softmax_case() -> impl Strategy<Value = [f32; 4]> {
    prop::array::uniform4(special_f32())
}

fn attention_case() -> impl Strategy<Value = [f32; 8]> {
    prop::array::uniform8(special_f32())
}

fn bounded_layer_norm_case() -> impl Strategy<Value = [f32; 4]> {
    prop::array::uniform4(-1.0e3f32..1.0e3f32)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, .. ProptestConfig::default() })]

    #[test]
    fn softmax_special_values_match_harness(input in softmax_case()) {
        let inputs = vec![
            bytes_from_f32(&input),
            vec![0u8; input.len() * core::mem::size_of::<f32>()],
        ];
        let direct = std::panic::catch_unwind(|| output_bytes(&entry("vyre-libs::nn::softmax").program().expect("Fix: registered library operation must provide a neutral builder"), &inputs))
            .expect("Fix: softmax reference path must not panic on NaN/Inf/subnormal inputs");
        let harness = std::panic::catch_unwind(|| harness_path_outputs(&entry("vyre-libs::nn::softmax"), &inputs))
            .expect("Fix: softmax universal harness path must not panic on NaN/Inf/subnormal inputs");
        prop_assert_eq!(direct, harness);
    }

    #[test]
    fn layer_norm_special_values_survive_harness(input in softmax_case()) {
        let inputs = vec![
            bytes_from_f32(&input),
            vec![0u8; input.len() * core::mem::size_of::<f32>()],
        ];
        let direct = std::panic::catch_unwind(|| output_bytes(&entry("vyre-libs::nn::layer_norm").program().expect("Fix: registered library operation must provide a neutral builder"), &inputs))
            .expect("Fix: layer_norm reference path must not panic on NaN/Inf/subnormal inputs");
        let harness = std::panic::catch_unwind(|| harness_path_outputs(&entry("vyre-libs::nn::layer_norm"), &inputs))
            .expect("Fix: layer_norm universal harness path must not panic on NaN/Inf/subnormal inputs");
        prop_assert_eq!(
            direct.len(),
            harness.len(),
            "optimized and direct layer_norm paths must preserve output arity"
        );
        prop_assert!(harness.iter().all(|output| output.len() == input.len() * 4));
    }

    #[test]
    fn layer_norm_finite_values_match_harness(input in bounded_layer_norm_case()) {
        use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};

        let operation = entry("vyre-libs::nn::layer_norm");
        let program = operation.program().expect("Fix: registered library operation must provide a neutral builder");
        let inputs = vec![
            bytes_from_f32(&input),
            vec![0u8; input.len() * core::mem::size_of::<f32>()],
        ];
        let direct = output_bytes(&program, &inputs);
        let harness = harness_path_outputs(&operation, &inputs);
        prop_assert!(
            matches!(compare_output_buffers(&program, &direct, &harness), BufferParity::Ok),
            "optimized layer_norm exceeded the operation's finite-value parity contract"
        );
    }

    #[test]
    fn attention_special_values_match_harness(q in attention_case(), k in attention_case(), v in attention_case()) {
        let inputs = vec![
            bytes_from_f32(&q),
            bytes_from_f32(&k),
            bytes_from_f32(&v),
            vec![0u8; q.len() * core::mem::size_of::<f32>()],
        ];
        let direct = std::panic::catch_unwind(|| output_bytes(&entry("vyre-libs::nn::attention").program().expect("Fix: registered library operation must provide a neutral builder"), &inputs))
            .expect("Fix: attention reference path must not panic on NaN/Inf/subnormal inputs");
        let harness = std::panic::catch_unwind(|| harness_path_outputs(&entry("vyre-libs::nn::attention"), &inputs))
            .expect("Fix: attention universal harness path must not panic on NaN/Inf/subnormal inputs");
        prop_assert_eq!(direct, harness);
    }
}
