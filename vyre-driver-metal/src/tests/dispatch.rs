//! One-shot and borrowed dispatch: program lowering, output ranges, config
//! rejection, grid sizing, threadgroup and trap sidecar allocation.

use crate::*;

#[test]
fn apple_dispatches_store_literal_program() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before dispatch.",
    );
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must execute a one-store u32 Program end to end.");
    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}

#[test]
fn apple_native_metal_matches_wgpu_on_same_program_bytes() {
    use vyre_driver::{DispatchConfig, VyreBackend as _};
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let idx = Expr::var("idx");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(8),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::U32).with_count(8),
            BufferDecl::storage("out", 2, BufferAccess::WriteOnly, DataType::U32)
                .with_count(8)
                .with_output_byte_range(0..32),
        ],
        [8, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(8)),
                vec![Node::store(
                    "out",
                    idx.clone(),
                    Expr::add(
                        Expr::load("a", idx.clone()),
                        Expr::mul(Expr::load("b", idx), Expr::u32(3)),
                    ),
                )],
            ),
        ],
    );
    let a = [1u32, 2, 3, 4, 5, 6, 7, 8]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = [10u32, 11, 12, 13, 14, 15, 16, 17]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let expected = [31u32, 35, 39, 43, 47, 51, 55, 59]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();

    let metal = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before differential dispatch.",
    );
    let wgpu = vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU-on-Metal must acquire on the Apple GPU differential lane.");
    let config = DispatchConfig::default();
    let metal_outputs = metal
        .dispatch(&program, &[a.clone(), b.clone()], &config)
        .expect("Fix: native Metal must dispatch the differential Program.");
    let wgpu_outputs = wgpu
        .dispatch(&program, &[a, b], &config)
        .expect("Fix: WGPU-on-Metal must dispatch the same differential Program.");

    assert_eq!(
        metal_outputs,
        vec![expected.clone()],
        "Fix: native Metal output must match the explicit byte oracle before comparing backends."
    );
    assert_eq!(
        wgpu_outputs,
        vec![expected],
        "Fix: WGPU-on-Metal output must match the explicit byte oracle before comparing backends."
    );
    assert_eq!(
        metal_outputs, wgpu_outputs,
        "Fix: native Metal and WGPU-on-Metal must produce byte-identical outputs for the same Program."
    );
}

#[test]
fn apple_dispatch_handles_empty_and_unaligned_output_ranges() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("empty", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(0)
                .with_output_byte_range(0..0),
            BufferDecl::storage("word", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(1..2),
        ],
        [1, 1, 1],
        vec![Node::store("word", Expr::u32(0), Expr::u32(0x1122_3344))],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before boundary dispatch.",
    );
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must honor shared empty and unaligned output layout planning.");
    assert_eq!(
        outputs,
        vec![Vec::new(), vec![0x33]],
        "Fix: Metal output collection must preserve empty outputs and trim unaligned one-byte ranges from the stored word."
    );
}

#[test]
fn apple_dispatch_config_errors_are_actionable() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before negative dispatch tests.",
    );
    let mut cooperative = DispatchConfig::default();
    cooperative.cooperative = true;
    let cooperative_error = backend
        .dispatch(&program, &[], &cooperative)
        .expect_err("Fix: native Metal must reject cooperative dispatch until implemented.")
        .to_string();
    assert!(
        cooperative_error.contains("Metal cooperative grid dispatch")
            && cooperative_error.contains("metal"),
        "Fix: cooperative dispatch rejection must name the unsupported Metal feature and backend: {cooperative_error}"
    );

    let mut zero_iterations = DispatchConfig::default();
    zero_iterations.fixpoint_iterations = Some(0);
    let zero_error = backend
        .dispatch(&program, &[], &zero_iterations)
        .expect_err("Fix: native Metal must reject explicit zero fixpoint iterations.")
        .to_string();
    assert!(
        zero_error.contains("fixpoint_iterations=0") && zero_error.contains("Fix:"),
        "Fix: zero-iteration dispatch rejection must include actionable fix text: {zero_error}"
    );
}

#[test]
fn apple_dispatch_grid_uses_declared_output_count_not_trimmed_readback() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let local = Expr::var("local");
    let token = Expr::var("token");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(512)
                .with_output_byte_range(0..8),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("local", Expr::LocalId { axis: 0 }),
            Node::let_bind("token", Expr::WorkgroupId { axis: 0 }),
            Node::if_then(
                Expr::and(
                    Expr::eq(local, Expr::u32(0)),
                    Expr::lt(token.clone(), Expr::u32(2)),
                ),
                vec![Node::store(
                    "out",
                    token.clone(),
                    Expr::add(token, Expr::u32(11)),
                )],
            ),
        ],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before dispatch.",
    );
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must infer grid from declared dispatch domain.");
    assert_eq!(
        outputs,
        vec![[11u32.to_le_bytes(), 12u32.to_le_bytes()].concat()]
    );
}

#[test]
fn apple_dispatch_allocates_threadgroup_memory() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let local = Expr::var("local");
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::workgroup("scratch", 4, DataType::U32),
            BufferDecl::storage("out", 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [4, 1, 1],
        vec![
            Node::let_bind("local", Expr::LocalId { axis: 0 }),
            Node::if_then(
                Expr::lt(local.clone(), Expr::u32(4)),
                vec![Node::store(
                    "scratch",
                    local.clone(),
                    Expr::load("values", local.clone()),
                )],
            ),
            Node::barrier(),
            Node::if_then(
                Expr::eq(local, Expr::u32(0)),
                vec![Node::store(
                    "out",
                    Expr::u32(0),
                    Expr::add(
                        Expr::add(
                            Expr::load("scratch", Expr::u32(0)),
                            Expr::load("scratch", Expr::u32(1)),
                        ),
                        Expr::add(
                            Expr::load("scratch", Expr::u32(2)),
                            Expr::load("scratch", Expr::u32(3)),
                        ),
                    ),
                )],
            ),
        ],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before dispatch.",
    );
    let input = [1u32, 2, 3, 4]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let outputs = backend
        .dispatch(&program, &[input], &DispatchConfig::default())
        .expect("Fix: native Metal must allocate threadgroup memory before dispatch.");
    assert_eq!(outputs, vec![10u32.to_le_bytes().to_vec()]);
}

#[test]
fn apple_dispatch_allocates_internal_trap_sidecar() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::u32(42)),
            Node::trap(Expr::u32(7), "fault"),
        ],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before dispatch.",
    );
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must allocate backend-owned trap sidecar storage.");
    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}

#[test]
fn apple_dispatches_subgroup_size_builtin() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::subgroup_size())],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before dispatch.",
    );
    assert!(
        backend.supports_subgroup_ops(),
        "Fix: native Metal must advertise subgroup ops only while its MSL path executes subgroup builtins."
    );
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: native Metal must dispatch subgroup-size builtin programs.");
    let observed = u32::from_le_bytes(
        outputs[0]
            .as_slice()
            .try_into()
            .expect("Fix: subgroup-size smoke output must be one u32."),
    );
    assert_eq!(
        Some(observed),
        backend.subgroup_size(),
        "Fix: Metal-reported subgroup size must match the executed subgroup builtin."
    );
}

#[test]
fn apple_borrowed_dispatch_into_reuses_caller_output_slots() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(123))],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before borrowed-into dispatch.",
    );
    let mut outputs = vec![Vec::with_capacity(64)];
    let original_capacity = outputs[0].capacity();

    backend
        .dispatch_borrowed_into(&program, &[], &DispatchConfig::default(), &mut outputs)
        .expect(
            "Fix: native Metal borrowed-into dispatch must execute through the public backend API.",
        );

    assert_eq!(
        outputs,
        vec![123u32.to_le_bytes().to_vec()],
        "Fix: borrowed-into Metal dispatch must write real kernel output into caller-owned slots."
    );
    assert!(
        outputs[0].capacity() >= original_capacity,
        "Fix: borrowed-into Metal dispatch must preserve reusable caller output capacity."
    );
}

#[test]
fn apple_borrowed_timed_dispatch_reports_enqueue_and_wait() {
    use vyre_driver::DispatchConfig;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(77))],
    );

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before timed dispatch.",
    );
    let timed = backend
        .dispatch_borrowed_timed(&program, &[], &DispatchConfig::default())
        .expect(
            "Fix: native Metal borrowed timed dispatch must execute through the real command path.",
        );

    assert_eq!(timed.outputs, vec![77u32.to_le_bytes().to_vec()]);
    assert!(
        timed.wall_ns > 0,
        "Fix: Metal borrowed timed dispatch must report nonzero wall time."
    );
    assert!(
        timed.enqueue_ns.is_some() && timed.wait_ns.is_some(),
        "Fix: Metal borrowed timed dispatch must expose native enqueue and wait timing."
    );
    assert_eq!(
        timed.device_ns, None,
        "Fix: Metal must not fake device timing until counter/timestamp support is implemented."
    );
}
