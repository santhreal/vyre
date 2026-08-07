//! Resident timed-dispatch output contract tests for the WGPU backend.

use std::sync::Arc;

use vyre_driver::Resource;
use vyre_driver::VyreBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[test]
fn resident_timed_dispatch_returns_public_readwrite_outputs() {
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU resident-output regression test requires a live GPU backend.");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("input", Expr::u32(0)), Expr::u32(5)),
        )],
    );

    let out = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must support resident output allocation.");
    let input = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must support resident input allocation.");
    let result = (|| {
        backend.upload_resident(&out, &[0, 0, 0, 0])?;
        backend.upload_resident(&input, &37u32.to_le_bytes())?;
        let timed = backend.dispatch_resident_timed(
            &program,
            &[out.clone(), input.clone()],
            &vyre_driver::DispatchConfig::default(),
        )?;
        assert_eq!(
            timed.outputs.len(),
            1,
            "resident timed dispatch must return public ReadWrite outputs"
        );
        assert_eq!(timed.outputs[0], 42u32.to_le_bytes());
        assert!(
            timed.device_ns.unwrap_or_default() > 0,
            "Fix: WGPU resident timed dispatch must report GPU timestamp device_ns so release benchmarks do not fall back to readback wall time."
        );
        Ok::<(), vyre_driver::BackendError>(())
    })();
    let free_out = backend.free_resident(out);
    let free_input = backend.free_resident(input);
    result.expect("Fix: resident timed dispatch must execute and read back outputs.");
    free_out.expect("Fix: WGPU resident output cleanup must succeed.");
    free_input.expect("Fix: WGPU resident input cleanup must succeed.");
}

/// WHY: resident callers need independent mutable slots to overlap queue work.
/// Both submissions must stay live before either result is retired, and timed
/// retirement must preserve device-owned attribution.
#[test]
fn resident_async_dispatch_retires_two_independent_slots() {
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU resident-async regression test requires a live GPU backend.");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("input", Expr::u32(0)), Expr::u32(5)),
        )],
    );
    let out_a = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must allocate the first resident output.");
    let input_a = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must allocate the first resident input.");
    let out_b = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must allocate the second resident output.");
    let input_b = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must allocate the second resident input.");

    let result = (|| {
        backend.upload_resident(&out_a, &[0; 4])?;
        backend.upload_resident(&input_a, &37u32.to_le_bytes())?;
        backend.upload_resident(&out_b, &[0; 4])?;
        backend.upload_resident(&input_b, &91u32.to_le_bytes())?;
        let pending_a = backend.dispatch_resident_async(
            &program,
            &[out_a.clone(), input_a.clone()],
            &vyre_driver::DispatchConfig::default(),
        )?;
        let pending_b = backend.dispatch_resident_async(
            &program,
            &[out_b.clone(), input_b.clone()],
            &vyre_driver::DispatchConfig::default(),
        )?;

        let timed_a = pending_a.await_timed_result()?;
        let timed_b = pending_b.await_timed_result()?;
        assert_eq!(timed_a.outputs, vec![42u32.to_le_bytes()]);
        assert_eq!(timed_b.outputs, vec![96u32.to_le_bytes()]);
        for timed in [timed_a, timed_b] {
            assert!(
                timed.device_ns.unwrap_or_default() > 0,
                "native asynchronous retirement must preserve WGPU timestamp attribution"
            );
            assert!(
                timed.enqueue_ns.is_some() && timed.wait_ns.is_some(),
                "native asynchronous retirement must separate enqueue and wait time"
            );
        }
        Ok::<(), vyre_driver::BackendError>(())
    })();

    let free_out_a = backend.free_resident(out_a);
    let free_input_a = backend.free_resident(input_a);
    let free_out_b = backend.free_resident(out_b);
    let free_input_b = backend.free_resident(input_b);
    result.expect("Fix: both resident async slots must retire with exact outputs.");
    free_out_a.expect("Fix: first WGPU resident output cleanup must succeed.");
    free_input_a.expect("Fix: first WGPU resident input cleanup must succeed.");
    free_out_b.expect("Fix: second WGPU resident output cleanup must succeed.");
    free_input_b.expect("Fix: second WGPU resident input cleanup must succeed.");
}

#[test]
fn persistent_resource_output_dispatch_rejects_borrowed_outputs_before_launch() {
    let backend = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU resident-output regression test requires a live GPU backend.");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("input", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("input", Expr::u32(0)), Expr::u32(5)),
        )],
    );
    let config = vyre_driver::DispatchConfig::default();
    let pipeline = vyre_driver::pipeline::compile(Arc::new(backend.clone()), &program, &config)
        .expect("Fix: WGPU compiled pipeline creation must succeed for resident outputs.");
    let input = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must support resident input allocation.");
    let result = (|| {
        backend.upload_resident(&input, &37u32.to_le_bytes())?;
        let err = pipeline
            .dispatch_persistent_resource_outputs(
                &[Resource::Borrowed(vec![0; 4]), input.clone()],
                &config,
            )
            .expect_err("Fix: resident-output mode must reject borrowed output resources");
        assert!(
            err.to_string().contains("cannot return borrowed output binding"),
            "borrowed output resource error must explain zero-copy resident-output requirements, got: {err}"
        );
        Ok::<(), vyre_driver::BackendError>(())
    })();
    let free_input = backend.free_resident(input);
    result.expect("Fix: borrowed output rejection must happen before launch.");
    free_input.expect("Fix: WGPU resident input cleanup must succeed.");
}
