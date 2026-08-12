use super::*;

#[test]
fn resident_dispatch_runs_without_host_buffer_arguments() {
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
            Expr::mul(Expr::load("input", Expr::gid_x()), Expr::u32(3)),
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

    backend
        .dispatch_resident(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: CUDA resident dispatch must execute the scalar trainer-safe subset.");

    let output_bytes = backend
        .download_resident(output)
        .expect("Fix: CUDA resident output download failed.");
    assert_eq!(bytes_u32(&output_bytes), vec![3, 6, 9, 12]);

    backend
        .free_resident(input)
        .expect("Fix: CUDA resident input free failed.");
    backend
        .free_resident(output)
        .expect("Fix: CUDA resident output free failed.");
}

#[test]
fn resident_dispatch_preserves_plain_read_write_state() {
    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let program = Program::wrapped(
        vec![BufferDecl::storage("state", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![Node::store(
            "state",
            Expr::gid_x(),
            Expr::add(Expr::load("state", Expr::gid_x()), Expr::u32(7)),
        )],
    );

    let state = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident state allocation failed.");
    backend
        .upload_resident(state, &u32_bytes(&[1, 2, 3, 4]))
        .expect("Fix: CUDA resident state upload failed.");

    backend
        .dispatch_resident(&program, &[state], &DispatchConfig::default())
        .expect("Fix: CUDA resident dispatch must update plain read-write state in place.");

    let output_bytes = backend
        .download_resident(state)
        .expect("Fix: CUDA resident state download failed.");
    assert_eq!(bytes_u32(&output_bytes), vec![8, 9, 10, 11]);

    backend
        .free_resident(state)
        .expect("Fix: CUDA resident state free failed.");
}

#[test]
fn async_resident_dispatch_holds_handles_until_awaited() {
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
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(5)),
        )],
    );

    let input = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident input allocation failed.");
    let output = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident output allocation failed.");
    backend
        .upload_resident(input, &u32_bytes(&[10, 20, 30, 40]))
        .expect("Fix: CUDA resident input upload failed.");

    let pending = backend
        .dispatch_resident_async(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: CUDA resident async dispatch must enqueue without host buffer arguments.");
    pending
        .await_result()
        .expect("Fix: CUDA resident async dispatch must complete successfully.");

    let output_bytes = backend
        .download_resident(output)
        .expect("Fix: CUDA resident output download failed.");
    assert_eq!(bytes_u32(&output_bytes), vec![15, 25, 35, 45]);

    backend
        .free_resident(input)
        .expect("Fix: CUDA resident input free failed after await.");
    backend
        .free_resident(output)
        .expect("Fix: CUDA resident output free failed after await.");
}

#[test]
fn timed_resident_dispatch_reports_device_time_and_outputs() {
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
            Expr::mul(Expr::load("input", Expr::gid_x()), Expr::u32(2)),
        )],
    );

    let input = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident input allocation failed.");
    let output = backend
        .allocate_resident(16)
        .expect("Fix: CUDA resident output allocation failed.");
    backend
        .upload_resident(input, &u32_bytes(&[2, 4, 6, 8]))
        .expect("Fix: CUDA resident input upload failed.");

    let timed = backend
        .dispatch_resident_timed(&program, &[input, output], &DispatchConfig::default())
        .expect("Fix: timed CUDA resident dispatch must complete successfully.");
    assert_eq!(bytes_u32(&timed.outputs[0]), vec![4, 8, 12, 16]);
    assert!(
        timed.wall_ns > 0,
        "Fix: CUDA resident timing fallback must return wall-clock timing."
    );

    backend
        .free_resident(input)
        .expect("Fix: CUDA resident input free failed after timed dispatch.");
    backend
        .free_resident(output)
        .expect("Fix: CUDA resident output free failed after timed dispatch.");
}
