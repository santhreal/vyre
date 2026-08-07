//! Native Metal resident asynchronous ownership and overlap contracts.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

fn add_five_program() -> Program {
    Program::wrapped(
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
    )
}

/// WHY: independent mutable resident slots must overlap without recycling any
/// command buffer, pipeline, sizes buffer, or resident resource before retirement.
#[test]
fn resident_async_dispatch_retires_two_independent_slots() {
    let backend = vyre_driver_metal::acquire()
        .expect("Fix: Metal resident-async regression test requires a live Metal device.");
    let program = add_five_program();
    let out_a = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the first resident output.");
    let input_a = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the first resident input.");
    let out_b = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the second resident output.");
    let input_b = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the second resident input.");

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
                timed.enqueue_ns.is_some() && timed.wait_ns.is_some(),
                "native Metal asynchronous retirement must separate enqueue and wait time"
            );
        }
        Ok::<(), vyre_driver::BackendError>(())
    })();

    let free_out_a = backend.free_resident(out_a);
    let free_input_a = backend.free_resident(input_a);
    let free_out_b = backend.free_resident(out_b);
    let free_input_b = backend.free_resident(input_b);
    result.expect("Fix: both Metal resident async slots must retire with exact outputs.");
    free_out_a.expect("Fix: first Metal resident output cleanup must succeed.");
    free_input_a.expect("Fix: first Metal resident input cleanup must succeed.");
    free_out_b.expect("Fix: second Metal resident output cleanup must succeed.");
    free_input_b.expect("Fix: second Metal resident input cleanup must succeed.");
}

/// WHY: abandoning a pending handle must synchronize before its retained Metal
/// objects are released, otherwise a later read can race freed in-flight state.
#[test]
fn dropping_pending_retires_owned_resources() {
    let backend = vyre_driver_metal::acquire()
        .expect("Fix: Metal pending-drop regression test requires a live Metal device.");
    let program = add_five_program();
    let out = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the resident output.");
    let input = backend
        .allocate_resident(4)
        .expect("Fix: Metal must allocate the resident input.");

    let result = (|| {
        backend.upload_resident(&out, &[0; 4])?;
        backend.upload_resident(&input, &37u32.to_le_bytes())?;
        let pending = backend.dispatch_resident_async(
            &program,
            &[out.clone(), input.clone()],
            &vyre_driver::DispatchConfig::default(),
        )?;
        drop(pending);
        let bytes = backend.download_resident(&out)?;
        assert_eq!(bytes, 42u32.to_le_bytes());
        Ok::<(), vyre_driver::BackendError>(())
    })();

    let free_out = backend.free_resident(out);
    let free_input = backend.free_resident(input);
    result.expect("Fix: dropping a Metal pending handle must retire its command safely.");
    free_out.expect("Fix: Metal resident output cleanup must succeed.");
    free_input.expect("Fix: Metal resident input cleanup must succeed.");
}
