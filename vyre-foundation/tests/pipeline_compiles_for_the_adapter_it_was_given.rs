//! The pass pipeline compiles for the adapter it was handed, not a profile
//! the passes picked for themselves.
//!
//! WHY: `DecodeScanFuse` reads device facts and had a `ProgramPass::transform`
//! that hardcoded `AdapterCaps::conservative()`. The caps-aware entry existed
//! the whole time and only a caller who already knew to ask could reach it, so
//! every program that went through the scheduler was compiled against a device
//! with no optional features and a 16 KiB shared-memory budget, whatever the
//! real adapter reported. Nothing failed: the pipeline produced a valid program
//! and a slower one.
//!
//! These tests fail if a scheduler built for a known adapter produces the
//! conservative program. They do not check a particular promoted buffer: the
//! point is that the adapter reached the pass at all, so each asserts the two
//! profiles disagree and that the scheduler's output follows the profile it was
//! given.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::memory::decode_scan_fuse::DecodeScanFuse;
use vyre_foundation::optimizer::{AdapterCaps, PassScheduler, ProgramPassKind};

/// A profile the pass must respect and cannot reach by accident: more shared
/// memory per workgroup than the conservative fallback admits.
fn roomy_adapter() -> AdapterCaps {
    AdapterCaps {
        max_shared_memory_bytes: 128 * 1024,
        ..AdapterCaps::conservative()
    }
}

/// Two handoff buffers: one inside the conservative budget, one outside it and
/// inside the roomy one. The small buffer is what makes the pass analyze as
/// runnable at all, since analysis judges opportunity against the conservative
/// profile; the large buffer is promoted on exactly one of the two profiles, so
/// the two pipelines can only agree if the adapter never reached the pass.
fn handoff_kernel() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("small_handoff", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(512),
            BufferDecl::storage("wide_handoff", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(8_192),
            BufferDecl::output("out", 2, DataType::U32).with_count(8_192),
        ],
        [64, 1, 1],
        vec![
            Node::store("small_handoff", Expr::gid_x(), Expr::u32(1)),
            Node::store(
                "wide_handoff",
                Expr::gid_x(),
                Expr::load("small_handoff", Expr::gid_x()),
            ),
            Node::store(
                "out",
                Expr::gid_x(),
                Expr::load("wide_handoff", Expr::gid_x()),
            ),
        ],
    )
}

#[test]
fn a_scheduler_built_for_an_adapter_does_not_produce_the_conservative_program() {
    let roomy = roomy_adapter();
    let floor = AdapterCaps::conservative();
    assert!(
        roomy.max_shared_memory_bytes > floor.max_shared_memory_bytes,
        "Fix: the two profiles must disagree or this test proves nothing."
    );

    let program = handoff_kernel();
    let fallback = PassScheduler::with_passes(vec![ProgramPassKind::new(DecodeScanFuse)])
        .run(program.clone())
        .expect("Fix: the fallback pipeline must converge.");
    let promoted = PassScheduler::for_adapter(vec![ProgramPassKind::new(DecodeScanFuse)], roomy)
        .run(program)
        .expect("Fix: the adapter pipeline must converge.");

    assert_ne!(
        promoted, fallback,
        "Fix: the scheduler was given an adapter reporting {} shared bytes per workgroup and \
         still produced the program the conservative fallback produces. The adapter is not \
         reaching the pass: check that the pass declares `adapter_dependent = true` and that \
         the scheduler calls `batch_apply` with its own adapter.",
        roomy.max_shared_memory_bytes
    );
}

#[test]
fn the_scheduler_reports_the_adapter_it_was_built_for() {
    let roomy = roomy_adapter();
    let scheduler = PassScheduler::for_adapter(vec![ProgramPassKind::new(DecodeScanFuse)], roomy);
    assert_eq!(
        scheduler.adapter().max_shared_memory_bytes,
        roomy.max_shared_memory_bytes,
        "Fix: a scheduler must carry the adapter it was constructed with."
    );

    let fallback = PassScheduler::with_passes(vec![ProgramPassKind::new(DecodeScanFuse)]);
    assert_eq!(
        fallback.adapter().backend,
        AdapterCaps::conservative().backend,
        "Fix: a scheduler built without an adapter must state the conservative fallback, so the \
         profile a program was compiled against is readable from the scheduler rather than \
         hidden inside whichever pass reached for one first."
    );
}

#[test]
fn an_ir_only_pass_ignores_the_adapter() {
    use vyre_foundation::optimizer::passes::cleanup::empty_block_collapse::EmptyBlockCollapsePass;

    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![
            Node::Block(Vec::new()),
            Node::store("out", Expr::u32(0), Expr::u32(1)),
        ],
    );
    let floor = PassScheduler::with_passes(vec![ProgramPassKind::new(EmptyBlockCollapsePass)])
        .run(program.clone())
        .expect("Fix: the fallback pipeline must converge.");
    let roomy = PassScheduler::for_adapter(
        vec![ProgramPassKind::new(EmptyBlockCollapsePass)],
        roomy_adapter(),
    )
    .run(program)
    .expect("Fix: the adapter pipeline must converge.");

    assert_eq!(
        floor, roomy,
        "Fix: an IR-only rewrite must produce the same program on every device. A pass whose \
         output moves with the adapter has to say so with `adapter_dependent = true`."
    );
}
