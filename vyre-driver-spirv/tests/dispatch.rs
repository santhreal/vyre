//! Runtime dispatch through the Vulkan-backed SPIR-V backend, against the CPU
//! reference.
//!
//! Every case dispatches a program on a live Vulkan compute device and compares
//! every output lane to `vyre-reference`. A missing Vulkan device is a probe or
//! driver configuration failure, so these tests fail rather than skip.

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_spirv::SpirvBackendRegistration;
use vyre_foundation::ir::Program;
use vyre_reference::value::Value;

#[path = "target_artifacts/elementwise.rs"]
mod elementwise;
use elementwise::{
    bytes_to_u32_values, elementwise_add_program, elementwise_fma_program,
    output_first_elementwise_add_program, u32_values_to_bytes,
};

fn require_vulkan_backend() -> SpirvBackendRegistration {
    SpirvBackendRegistration::acquire().unwrap_or_else(|error| {
        panic!(
            "Fix: SPIR-V dispatch tests require a live Vulkan compute GPU. \
             Missing Vulkan is a driver/probe configuration failure, not a skipped test. \
             Probe error: {error}"
        )
    })
}

/// Run the CPU reference interpreter over `lanes` and return its output bytes.
fn reference_outputs(program: &Program, lanes: &[&[u32]]) -> Vec<Vec<u8>> {
    let inputs = lanes
        .iter()
        .map(|lane| Value::Bytes(u32_values_to_bytes(lane).into()))
        .collect::<Vec<_>>();
    vyre_reference::reference_eval(program, &inputs)
        .expect("Fix: reference evaluation must succeed for valid test programs.")
        .iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Compare every output buffer lane for lane.
///
/// Buffer count is asserted first: a backend that returns one buffer where the
/// reference returns two otherwise passes a per-buffer zip silently.
fn assert_lanes_match_reference(context: &str, device: &[Vec<u8>], reference: &[Vec<u8>]) {
    assert_eq!(
        device.len(),
        reference.len(),
        "Fix: SPIR-V {context} must produce the same number of output buffers as the reference."
    );
    for (index, (device_bytes, reference_bytes)) in device.iter().zip(reference).enumerate() {
        assert_eq!(
            bytes_to_u32_values(device_bytes),
            bytes_to_u32_values(reference_bytes),
            "Fix: SPIR-V {context} output buffer {index} does not match the reference."
        );
    }
}

/// An output at binding 0 must not be fed a host input buffer.
///
/// Host inputs are bound through the binding plan, not by raw binding order. A
/// backend that walks bindings in order consumes the first input into the output
/// slot here, which is the only program shape where that bug is visible.
#[test]
fn spirv_output_first_binding_matches_reference() {
    let backend = require_vulkan_backend();

    let count = 128u32;
    let program = output_first_elementwise_add_program(count);
    let a = (0..count).map(|i| i.wrapping_mul(5)).collect::<Vec<u32>>();
    let b = (0..count).map(|i| i.wrapping_mul(7)).collect::<Vec<u32>>();

    let outputs = backend
        .dispatch(
            &program,
            &[u32_values_to_bytes(&a), u32_values_to_bytes(&b)],
            &DispatchConfig::default(),
        )
        .expect("Fix: SPIR-V dispatch must bind output-first programs through the binding plan.");

    assert_lanes_match_reference(
        "output-first binding",
        &outputs,
        &reference_outputs(&program, &[&a, &b]),
    );
}

#[test]
fn spirv_elementwise_add_matches_reference() {
    let backend = require_vulkan_backend();

    let count = 256u32;
    let program = elementwise_add_program(count);
    let a = (0..count).collect::<Vec<u32>>();
    let b = (0..count).map(|i| i.wrapping_mul(3)).collect::<Vec<u32>>();

    let outputs = backend
        .dispatch(
            &program,
            &[u32_values_to_bytes(&a), u32_values_to_bytes(&b)],
            &DispatchConfig::default(),
        )
        .expect("Fix: SPIR-V dispatch of an element-wise add must succeed.");

    assert_lanes_match_reference(
        "element-wise add",
        &outputs,
        &reference_outputs(&program, &[&a, &b]),
    );
}

#[test]
fn spirv_elementwise_fma_matches_reference() {
    let backend = require_vulkan_backend();

    let count = 128u32;
    let program = elementwise_fma_program(count);
    let a = (1..=count).collect::<Vec<u32>>();

    let outputs = backend
        .dispatch(
            &program,
            &[u32_values_to_bytes(&a)],
            &DispatchConfig::default(),
        )
        .expect("Fix: SPIR-V dispatch of a multiply-add must succeed.");

    assert_lanes_match_reference(
        "element-wise multiply-add",
        &outputs,
        &reference_outputs(&program, &[&a]),
    );
}

#[test]
fn spirv_backend_factory_reports_backend_identity() {
    let backend = vyre_driver_spirv::spirv_factory()
        .expect("Fix: SPIR-V factory must return a backend handle");
    assert_eq!(backend.id(), vyre_driver_spirv::SPIRV_BACKEND_ID);
}

#[test]
fn spirv_device_buffer_api_rejects_host_shim_fallback() {
    let backend = require_vulkan_backend();
    let err = backend
        .allocate_device_buffer(16)
        .expect_err("Fix: SPIR-V must not allocate HostShimBuffer as a fake resident buffer");
    let msg = format!("{err}");
    assert!(
        msg.contains("DeviceBuffer") && msg.contains("HostShimBuffer dispatch is forbidden"),
        "Fix: SPIR-V DeviceBuffer rejection must name the forbidden host-shim fallback: {msg}"
    );
}

#[test]
fn spirv_backend_id_is_stable() {
    assert_eq!(vyre_driver_spirv::SPIRV_BACKEND_ID, "spirv");
}
