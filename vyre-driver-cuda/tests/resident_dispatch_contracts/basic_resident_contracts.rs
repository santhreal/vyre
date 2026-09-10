use super::*;

#[test]
fn resident_dispatch_runs_without_host_buffer_arguments() {
    let (lanes, _) = dispatch_resident_lanes(&mul_program("input", "out", 3), &SEED);
    assert_eq!(lanes, vec![3, 6, 9, 12]);
}

#[test]
fn resident_dispatch_preserves_plain_read_write_state() {
    let backend = acquire();
    // A single read-write binding, not the two-binding shape the rest of this
    // family uses: the contract is that in-place state survives the dispatch.
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(LANES),
        ],
        [1, 1, 1],
        vec![Node::store(
            "state",
            Expr::gid_x(),
            Expr::add(Expr::load("state", Expr::gid_x()), Expr::u32(7)),
        )],
    );

    let state = seeded_handle_lane(&backend, "state", &SEED);
    backend
        .dispatch_resident(&program, &[state], &DispatchConfig::default())
        .expect("Fix: CUDA resident dispatch must update plain read-write state in place.");

    assert_eq!(download_lanes(&backend, state, "state"), vec![8, 9, 10, 11]);

    free_handle_lanes(&backend, &[(state, "state")]);
}

#[test]
fn async_resident_dispatch_holds_handles_until_awaited() {
    let backend = acquire();
    let program = add_program("input", "out", 5);
    let input = seeded_handle_lane(&backend, "input", &[10, 20, 30, 40]);
    let output = handle_lane(&backend, "output");

    let pending = backend
        .dispatch_resident_async(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: CUDA resident async dispatch must enqueue without host buffer arguments.");
    pending
        .await_result()
        .expect("Fix: CUDA resident async dispatch must complete successfully.");

    assert_eq!(
        download_lanes(&backend, output, "output"),
        vec![15, 25, 35, 45]
    );

    free_handle_lanes(&backend, &[(input, "input"), (output, "output")]);
}

#[test]
fn timed_resident_dispatch_reports_device_time_and_outputs() {
    let backend = acquire();
    let program = mul_program("input", "out", 2);
    let input = seeded_handle_lane(&backend, "input", &[2, 4, 6, 8]);
    let output = handle_lane(&backend, "output");

    let timed = backend
        .dispatch_resident_timed(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: timed CUDA resident dispatch must complete successfully.");
    assert_eq!(bytes_u32(&timed.outputs[0]), vec![4, 8, 12, 16]);
    assert!(
        timed.wall_ns > 0,
        "Fix: CUDA resident timing fallback must return wall-clock timing."
    );

    free_handle_lanes(&backend, &[(input, "input"), (output, "output")]);
}

/// WHY: `BufferAccess::ReadOnly` is what the PTX emitter compiles into
/// `ld.global.nc`, a load served by a cache that is not coherent with this
/// kernel's stores. A resident handle is `Copy`, so the same allocation can be
/// presented at a read-only slot and at a written one, which makes that load
/// return stale bytes with no diagnostic. The refusal is at the dispatch
/// boundary, before the launch.
///
/// It does not catch two distinct resident allocations a driver later maps onto
/// one device address range.
#[test]
fn a_resident_dispatch_aliasing_a_read_only_slot_onto_a_written_slot_is_refused() {
    let backend = acquire();
    let program = mul_program("input", "out", 2);
    let shared = seeded_handle_lane(&backend, "input", &SEED);

    let error = backend
        .dispatch_resident_timed(&program, &[shared, shared], &DispatchConfig::default())
        .expect_err(
            "Fix: one allocation at a read-only slot and a written slot must be refused.",
        );
    let text = error.to_string();
    assert!(
        text.contains("read-only binding `input`") && text.contains("writable binding `out`"),
        "Fix: the refusal must name both bindings so the caller knows which pair to split. Got: {text}"
    );
    assert_eq!(
        download_lanes(&backend, shared, "input"),
        SEED.to_vec(),
        "Fix: a refused dispatch must not have launched."
    );

    free_handle_lanes(&backend, &[(shared, "input")]);
}
