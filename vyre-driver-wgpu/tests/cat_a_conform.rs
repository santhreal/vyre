//! Focused GPU conform checks for Cat-A fixture-bearing ops.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::cat_a_dispatch_config;

use std::sync::OnceLock;

use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::fp_parity;
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::library_entries;
fn backend() -> &'static WgpuBackend {
    static BACKEND: OnceLock<WgpuBackend> = OnceLock::new();
    BACKEND.get_or_init(|| {
        let adapters = vyre_driver_wgpu::runtime::enumerate_adapters();
        assert!(
            !adapters.is_empty(),
            "Fix: cat_a_conform requires a live GPU adapter."
        );
        WgpuBackend::acquire().expect("Fix: cat_a_conform must acquire the live GPU backend")
    })
}

fn entry(id: &'static str) -> SemanticOperation {
    library_entries()
        .find(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("Fix: missing canonical operation registration for {id}"))
}

fn assert_gpu_matches_fixture(id: &'static str) {
    let entry = entry(id);
    let program = entry
        .program()
        .unwrap_or_else(|| panic!("Fix: fixture-bearing operation `{id}` must provide a program"));
    let config = cat_a_dispatch_config(&program);
    let inputs = (entry.test_inputs.expect("Fix: test_inputs required"))();
    let expected = (entry
        .expected_output
        .expect("Fix: expected_output required"))();
    assert_eq!(
        inputs.len(),
        expected.len(),
        "Fix: fixture case count mismatch for {id}"
    );
    assert!(
        !inputs.is_empty(),
        "Fix: {id} has empty test_inputs; GPU conform fixtures must execute at least one case."
    );
    assert!(
        !expected.is_empty(),
        "Fix: {id} has empty expected_output; GPU conform fixtures must provide an oracle."
    );

    for (case_index, (input_set, expected_outputs)) in
        inputs.iter().zip(expected.iter()).enumerate()
    {
        let outputs = backend()
            .dispatch(&program, input_set, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: GPU dispatch failed for {id} case {case_index}: {error}")
            });
        let parity =
            fp_parity::compare_operation_outputs(entry.id, &program, &outputs, expected_outputs);
        assert!(
            matches!(&parity, fp_parity::BufferParity::Ok),
            "GPU fixture drift for {} case {}: {:?}",
            entry.id,
            case_index,
            parity
        );
    }
}

#[test]
fn matmul_tiled_matches_fixture_on_gpu() {
    assert_gpu_matches_fixture("vyre-libs::math::matmul_tiled");
}

#[test]
fn softmax_matches_fixture_on_gpu() {
    assert_gpu_matches_fixture("vyre-libs::nn::softmax");
}

#[test]
fn layer_norm_matches_fixture_on_gpu() {
    assert_gpu_matches_fixture("vyre-libs::nn::layer_norm");
}

#[test]
fn attention_matches_fixture_on_gpu() {
    assert_gpu_matches_fixture("vyre-libs::nn::attention");
}
