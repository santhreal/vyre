//! CUDA module-cache performance contracts.

mod common;
use common::u32_bytes;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn repeated_dispatch_reuses_loaded_cuda_module() {
    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(2),
            BufferDecl::output("out", 1, DataType::U32).with_count(2),
        ],
        [64, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(7)),
        )],
    );

    assert_eq!(
        backend
            .cached_module_count()
            .expect("Fix: CUDA module cache lock failed."),
        0
    );
    backend
        .dispatch(&program, &[u32_bytes(&[1, 2])], &DispatchConfig::default())
        .expect("Fix: first CUDA dispatch should load one module.");
    assert_eq!(
        backend
            .cached_module_count()
            .expect("Fix: CUDA module cache lock failed."),
        1
    );
    backend
        .dispatch(&program, &[u32_bytes(&[3, 4])], &DispatchConfig::default())
        .expect("Fix: second CUDA dispatch should reuse cached module.");
    assert_eq!(
        backend
            .cached_module_count()
            .expect("Fix: CUDA module cache lock failed."),
        1,
        "Fix: repeated CUDA dispatches of the same program must not load duplicate modules."
    );
}

#[test]
fn repeated_dispatch_reuses_transient_device_allocations() {
    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1024),
            BufferDecl::output("out", 1, DataType::U32).with_count(1024),
        ],
        [256, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(1)),
        )],
    );
    let input = u32_bytes(&(0..1024).collect::<Vec<_>>());

    assert_eq!(
        backend
            .cached_transient_allocation_bytes()
            .expect("Fix: CUDA transient allocation pool lock failed."),
        0
    );
    backend
        .dispatch(
            &program,
            std::slice::from_ref(&input),
            &DispatchConfig::default(),
        )
        .expect("Fix: first CUDA dispatch should complete and return transient allocations.");
    let after_first = backend
        .cached_transient_allocation_bytes()
        .expect("Fix: CUDA transient allocation pool lock failed.");
    assert!(
        after_first > 0,
        "Fix: CUDA dispatch must retain transient device allocations for reuse instead of freeing every sample."
    );

    backend
        .dispatch(&program, &[input], &DispatchConfig::default())
        .expect("Fix: second CUDA dispatch should reuse transient allocations.");
    let after_second = backend
        .cached_transient_allocation_bytes()
        .expect("Fix: CUDA transient allocation pool lock failed.");
    assert_eq!(
        after_second, after_first,
        "Fix: repeated same-shape CUDA dispatches must reuse the transient allocation pool without unbounded growth."
    );

    backend
        .cleanup()
        .expect("Fix: CUDA cleanup must clear transient allocation pool.");
    assert_eq!(
        backend
            .cached_transient_allocation_bytes()
            .expect("Fix: CUDA transient allocation pool lock failed."),
        0
    );
}

#[test]
fn repeated_dispatch_reuses_cuda_launch_resources() {
    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(256),
            BufferDecl::output("out", 1, DataType::U32).with_count(256),
        ],
        [256, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(Expr::load("input", Expr::gid_x()), Expr::u32(9)),
        )],
    );
    let input = u32_bytes(&(0..256).collect::<Vec<_>>());

    assert_eq!(
        backend
            .cached_launch_resource_counts()
            .expect("Fix: CUDA launch-resource pool lock failed."),
        (0, 0)
    );
    backend
        .dispatch(
            &program,
            std::slice::from_ref(&input),
            &DispatchConfig::default(),
        )
        .expect("Fix: first CUDA dispatch should complete and return launch resources.");
    let after_first = backend
        .cached_launch_resource_counts()
        .expect("Fix: CUDA launch-resource pool lock failed.");
    assert_eq!(
        after_first,
        (1, 1),
        "Fix: CUDA dispatch must retain the stream and completion event for reuse."
    );

    backend
        .dispatch(&program, &[input], &DispatchConfig::default())
        .expect("Fix: second CUDA dispatch should reuse launch resources.");
    assert_eq!(
        backend
            .cached_launch_resource_counts()
            .expect("Fix: CUDA launch-resource pool lock failed."),
        after_first,
        "Fix: repeated same-shape CUDA dispatches must reuse launch resources without growth."
    );

    backend
        .cleanup()
        .expect("Fix: CUDA cleanup must clear cached launch resources.");
    assert_eq!(
        backend
            .cached_launch_resource_counts()
            .expect("Fix: CUDA launch-resource pool lock failed."),
        (0, 0)
    );
}
