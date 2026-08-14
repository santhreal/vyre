//! Integration test for the CUDA backend: resident dispatch contracts.
//!
//! This parent was deleted whole by f04f5a0644 because two of its tests used
//! `vyre::scan`, which that commit removed. The four chunk files under
//! `resident_dispatch_contracts/` survived with no parent, so cargo compiled
//! none of them and 15 CUDA resident contracts stopped running while
//! docs/optimization/OP_MATRIX.toml still cited this file as the proving test
//! for `elementwise_add`. The chunks and the two tests below are restored; the
//! two `vyre::scan` tests are not, because the product they exercised is gone.

#[path = "resident_dispatch_contracts/basic_resident_contracts.rs"]
mod basic_resident_contracts;
mod common;
#[path = "resident_dispatch_contracts/optimizer_combined_contracts.rs"]
mod optimizer_combined_contracts;
#[path = "resident_dispatch_contracts/repeated_sequence_contracts.rs"]
mod repeated_sequence_contracts;
#[path = "resident_dispatch_contracts/sequence_readback_contracts.rs"]
mod sequence_readback_contracts;

use common::{bytes_u32, u32_bytes};

use vyre_driver::{DispatchConfig, Resource, VyreBackend};
use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration, CudaOptimizerDispatcher};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_self_substrate::optimizer::dispatcher::{
    OptimizerDispatcher, ResidentDispatchStep, ResidentReadRange,
};

fn cuda_resident_borrowed_fallback_active() -> bool {
    if std::env::var_os("VYRE_CUDA_RESIDENT_BORROWED_FALLBACK").is_none() {
        return false;
    }
    #[cfg(debug_assertions)]
    {
        true
    }
    #[cfg(not(debug_assertions))]
    {
        std::env::var("VYRE_CUDA_ALLOW_BORROWED_FALLBACK")
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON"))
            .unwrap_or(false)
    }
}

fn expected_readback_bytes(native_resident: u64, fallback_resident: u64) -> u64 {
    if cuda_resident_borrowed_fallback_active() {
        fallback_resident
    } else {
        native_resident
    }
}

/// Law-10 release-path contract: a NATIVE resident dispatch must never
/// silently escape to the borrowed host-buffer fallback. After a clean
/// native dispatch on a GPU host, the resident borrowed-fallback
/// telemetry counter must read exactly zero. A nonzero value means the
/// resident fast path quietly degraded to the slower borrowed path and
/// the operator was never told -- exactly the kind of invisible recall/
/// perf regression the release perf gate exists to catch, since a
/// borrowed-fallback dispatch can masquerade as native megakernel
/// speedup in the evidence CSV.
#[test]
fn release_path_resident_dispatch_keeps_borrowed_fallback_counter_at_zero() {
    // The borrowed fallback is only *taken* when explicitly opted in
    // (debug builds, or release builds with VYRE_CUDA_ALLOW_BORROWED_FALLBACK).
    // The release gate runs without that env, where the zero-counter
    // invariant below is the contract. When a developer deliberately
    // enables the fallback for debugging, the invariant intentionally
    // does not hold; we surface that loudly rather than asserting a
    // contradiction. This mirrors `expected_readback_bytes`'s handling
    // of the same env toggle.
    if cuda_resident_borrowed_fallback_active() {
        eprintln!(
            "release_path_resident_dispatch_keeps_borrowed_fallback_counter_at_zero: \
             VYRE_CUDA_RESIDENT_BORROWED_FALLBACK is active; the native zero-counter \
             invariant is intentionally not asserted on this opt-in debugging run."
        );
        return;
    }
    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(7)),
        )],
    );

    let input = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident input allocation failed.");
    let output = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident output allocation failed.");
    backend
        .upload_resident(input, &u32_bytes(&[1, 2, 3, 4]))
        .expect("Fix: CUDA resident input upload failed.");

    backend.reset_telemetry();
    backend
        .dispatch_resident(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: CUDA native resident dispatch must execute without the borrowed fallback.");

    let output_bytes = backend
        .download_resident(output)
        .expect("Fix: CUDA resident output download failed.");
    assert_eq!(
        bytes_u32(&output_bytes),
        vec![8, 9, 10, 11],
        "Fix: native resident dispatch produced wrong results; the kernel did not run on the resident buffers."
    );

    let telemetry = backend.telemetry_snapshot();
    assert_eq!(
        telemetry.resident_borrowed_fallback_dispatches, 0,
        "Fix: a native resident dispatch silently escaped to the borrowed host-buffer fallback \
         ({} dispatch(es)); the resident fast path must stay native so the release perf gate \
         cannot pass on a degraded path.",
        telemetry.resident_borrowed_fallback_dispatches
    );

    backend
        .free_resident(input)
        .expect("Fix: CUDA resident input free failed.");
    backend
        .free_resident(output)
        .expect("Fix: CUDA resident output free failed.");
}

/// WHY: the object-safe resident async seam must use CUDA's native pending
/// dispatch rather than the synchronous trait fallback, while preserving exact
/// output and immediate resident-handle validation.
#[test]
fn trait_resident_async_dispatch_preserves_output_and_rejects_nonresident_bindings() {
    let backend = CudaBackendRegistration::new(
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host."),
    );
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(9)),
        )],
    );
    let input = backend
        .allocate_resident(16)
        .expect("Fix: trait resident input allocation failed.");
    let output = backend
        .allocate_resident(16)
        .expect("Fix: trait resident output allocation failed.");
    backend
        .upload_resident(&input, &u32_bytes(&[1, 2, 3, 4]))
        .expect("Fix: trait resident input upload failed.");

    let pending = backend
        .dispatch_resident_async(
            &program,
            &[input.clone(), output.clone()],
            &DispatchConfig::default(),
        )
        .expect("Fix: native CUDA resident async submission failed.");
    let in_flight_error = backend
        .free_resident(output.clone())
        .expect_err("Fix: pending CUDA work must retain resident output ownership until await.");
    assert!(
        in_flight_error.to_string().contains("in-flight"),
        "Fix: pending resident ownership error must identify the active dispatch: {in_flight_error}"
    );
    let outputs = pending
        .await_result()
        .expect("Fix: native CUDA resident async readback failed.");
    assert_eq!(
        outputs.len(),
        1,
        "Fix: one output binding must produce one async readback slot."
    );
    assert_eq!(
        bytes_u32(&outputs[0]),
        vec![10, 11, 12, 13],
        "Fix: resident async dispatch diverged from the exact resident program result."
    );

    let nonresident = Resource::Borrowed(vec![0; 16]);
    let error = match backend.dispatch_resident_async(
        &program,
        &[nonresident, output.clone()],
        &DispatchConfig::default(),
    ) {
        Ok(_) => {
            panic!("Fix: a borrowed binding must fail before returning resident pending work.")
        }
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("resident"),
        "Fix: nonresident-binding error must name the corrective boundary: {error}"
    );

    backend
        .free_resident(input)
        .expect("Fix: trait resident input free failed.");
    backend
        .free_resident(output)
        .expect("Fix: trait resident output free failed.");
}
