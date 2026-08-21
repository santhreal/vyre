//! WGPU's arguments to the shared registered-target-compiler contract.
//!
//! The contract itself is `tests/support/target_compiler_contract.rs`. Only what
//! is genuinely WGPU-native stays here: the payload format this backend
//! registers, the shape of a WGSL module, and the target operation facets this
//! backend publishes.

#![cfg(feature = "device-tests")]

use vyre_foundation::ir::BufferAccess;

#[path = "../../tests/support/target_compiler_contract.rs"]
mod target_compiler_contract;
use target_compiler_contract::{
    assert_materializer_executes_payload, assert_materializer_executes_resident_binding,
    assert_target_compiler_emits_bundle, TargetExpectation,
};

fn wgpu() -> TargetExpectation<'static> {
    TargetExpectation {
        backend_id: vyre_driver_wgpu::WGPU_BACKEND_ID,
        format_identity: "wgsl",
        format_version: 2,
        entry_point: "main",
        output_access: BufferAccess::WriteOnly,
    }
}

/// WHY: WGPU payload production is a pure registered compiler operation with no
/// device probe, and the WGSL it emits must declare a compute entry point.
#[test]
fn registered_target_compiler_emits_selected_wgsl_bundle() {
    assert_target_compiler_emits_bundle(&wgpu(), |bundle| {
        let source = std::str::from_utf8(&bundle.modules[0].bytes)
            .expect("Fix: WGPU target module bytes must be UTF-8");
        assert!(
            source.contains("@compute"),
            "Fix: WGPU target module must declare a `@compute` entry point"
        );
    });
}

/// WHY: target support is a facet of the canonical semantic identity, not a
/// second backend-owned operation catalog.
#[test]
fn registered_target_facets_resolve_canonical_operations() {
    let facets = vyre_driver::registered_target_operation_facets()
        .expect("valid target facet registry")
        .iter()
        .filter(|facet| facet.target_id == vyre_driver_wgpu::WGPU_BACKEND_ID)
        .collect::<Vec<_>>();
    assert!(
        !facets.is_empty(),
        "WGPU target compiler must expose at least one supported canonical operation"
    );
    for facet in facets {
        let operation = vyre_foundation::operation::OperationRegistry::global()
            .get(facet.operation_id)
            .expect("target facet must resolve one canonical semantic operation");
        assert!(
            operation.build.is_some(),
            "{} target facet must reference a neutral program",
            facet.operation_id
        );
    }
}

/// WHY: WGPU materialization must execute authenticated WGSL instead of
/// re-emitting a Program.
#[test]
fn registered_materializer_executes_authenticated_wgsl() {
    assert_materializer_executes_payload(&wgpu());
}

/// WHY: resident benchmark hot loops must submit authenticated artifact
/// instances, not bypass materialization through raw `Program` dispatch.
#[test]
fn registered_materializer_executes_resident_artifact_bindings() {
    assert_materializer_executes_resident_binding(&wgpu());
}
