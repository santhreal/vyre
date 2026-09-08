//! Registration contract for the pure Rust reference backend adapter.

use vyre_driver::{acquire, backend_dispatches};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};

use crate::dispatch_fixtures;
use dispatch_fixtures::u32_out_buffer;

/// The backend id comes from [`vyre_driver_reference::registered_backend_id`]
/// and not from the `const`, because calling it is what keeps this crate's
/// object file, and its registration, in the linked test binary. Reading the
/// `const` inlines at the use site and links nothing, which left both tests
/// reporting an unregistered backend on the Mach-O leg of the matrix while the
/// ELF legs passed.
fn registered_id() -> &'static str {
    vyre_driver_reference::registered_backend_id()
        .expect("Fix: this build must compile the cpu-ref registration.")
}

#[test]
fn cpu_ref_registers_as_dispatch_backend() {
    let id = registered_id();
    assert!(
        backend_dispatches(id).expect("valid backend registry"),
        "Fix: vyre-driver-reference must register cpu-ref as a dispatch-capable backend."
    );

    let backend = acquire(id)
        .expect("Fix: cpu-ref backend registration must construct without host hardware.");
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let outputs = backend
        .dispatch(&program, &[], &vyre_driver::DispatchConfig::default())
        .expect("Fix: cpu-ref backend must dispatch a minimal Program.");

    assert_eq!(
        outputs,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: cpu-ref backend output must match reference interpreter bytes."
    );
}

/// WHY: the interpreter partitions a body at each whole-grid fence and runs the
/// whole grid through one segment before the next, over one memory. That is what
/// a cooperative launch buys, so a cooperative request is executable here and
/// answers the same as a plain one.
///
/// This asserted the opposite for as long as the backend reported no grid sync.
/// The report sent every fenced program through the launch-boundary cut, whose
/// segments hand state to each other through device-resident storage that a
/// one-shot host submission does not have, and the later segment read a zeroed
/// carrier. The refusal was the defect and this test was its pin.
#[test]
fn cpu_ref_executes_a_fenced_program_under_a_cooperative_launch() {
    let id = registered_id();
    let backend = acquire(id)
        .expect("Fix: cpu-ref backend registration must construct without host hardware.");
    let program = Program::wrapped(
        vec![
            BufferDecl::read("seed", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::load("seed", Expr::u32(0))),
            Node::logical_barrier(MemoryOrdering::GridSync),
            Node::store(
                "out",
                Expr::u32(0),
                Expr::add(Expr::load("out", Expr::u32(0)), Expr::u32(1)),
            ),
        ],
    );
    let seed = 41u32.to_le_bytes().to_vec();
    let mut config = vyre_driver::DispatchConfig::default();
    config.cooperative = true;
    let cooperative = backend
        .dispatch(&program, std::slice::from_ref(&seed), &config)
        .expect("Fix: cpu-ref must execute a whole-grid fence under a cooperative launch.");
    assert_eq!(
        cooperative,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: the segment after the fence must read what the segment before it wrote."
    );
    let plain = backend
        .dispatch(
            &program,
            std::slice::from_ref(&seed),
            &vyre_driver::DispatchConfig::default(),
        )
        .expect("Fix: cpu-ref must execute a whole-grid fence without a cooperative request.");
    assert_eq!(
        cooperative, plain,
        "Fix: a cooperative request selects a launch, not a semantics; both must answer alike."
    );
}
