//! CUDA's arguments to the shared registered-target-compiler contract.
//!
//! The contract itself is `tests/support/target_compiler_contract.rs`. Only what
//! is genuinely CUDA-native stays here: the payload format this backend
//! registers, and the shape of a PTX module.

use vyre_foundation::ir::BufferAccess;

#[path = "../../tests/support/target_compiler_contract.rs"]
mod target_compiler_contract;
use target_compiler_contract::{
    assert_materializer_executes_payload, assert_materializer_executes_resident_binding,
    assert_target_compiler_emits_bundle, TargetExpectation,
};

fn cuda() -> TargetExpectation<'static> {
    TargetExpectation {
        backend_id: vyre_driver_cuda::CUDA_BACKEND_ID,
        format_identity: "ptx",
        format_version: 1,
        entry_point: "main",
        output_access: BufferAccess::ReadWrite,
    }
}

/// WHY: CUDA payload production must not acquire a GPU or compile a caller-owned
/// Program, and the PTX it emits must define the entry point its own
/// materializer admits.
#[test]
fn registered_target_compiler_emits_selected_ptx_bundle() {
    assert_target_compiler_emits_bundle(&cuda(), |bundle| {
        let ptx = std::str::from_utf8(&bundle.modules[0].bytes)
            .expect("Fix: CUDA target module bytes must be UTF-8 PTX");
        assert!(
            ptx.contains(".visible .entry main("),
            "Fix: CUDA target module must define `.visible .entry main`"
        );
    });
}

/// WHY: CUDA materialization must load authenticated PTX and execute it without
/// re-emitting it.
#[test]
fn registered_materializer_executes_authenticated_ptx() {
    assert_materializer_executes_payload(&cuda());
}

/// WHY: resident resources must remain inside the authenticated artifact route.
#[test]
fn registered_materializer_executes_authenticated_ptx_with_resident_bindings() {
    assert_materializer_executes_resident_binding(&cuda());
}
