#![cfg(feature = "device-tests")]

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
