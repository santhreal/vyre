//! `atomic_minimize` must reach an identity atomic in every operand position a
//! node carries, not only the four the pass happened to name.
//!
//! The pass runs only when its own analysis reports a candidate, and that
//! analysis re-derived "which node variants carry expressions" with a match
//! ending in a catch-all. `Trap::address` and the `offset`/`size` of an async
//! copy fell into that arm, so an identity atomic reachable only through one of
//! them made the analysis report SKIP: the pass was never invoked and the
//! atomic survived. A missed rewrite leaves no trace, which is what makes this
//! worth pinning rather than noticing later.
//!
//! Both the analysis and the rewrite now enumerate node operands through
//! `visit::node_operands`, so a `Node` variant that gains an
//! expression position fails to compile there instead of being skipped here.

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{
    AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program,
};
use vyre_foundation::optimizer::passes::algebraic::atomic_minimize::AtomicMinimizePass;
use vyre_foundation::optimizer::{PassAnalysis, ProgramPass};
use vyre_foundation::visit::walk_exprs;

fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        entry,
    )
}

/// `atomic_add(buf, 0, value)` under Relaxed. With `value == 0` the
/// read-modify-write leaves memory as it found it, so it means exactly `buf[0]`
/// and the pass may collapse it; with any other value it must not.
fn relaxed_atomic_add(value: Expr) -> Expr {
    Expr::Atomic {
        op: AtomicOp::Add,
        buffer: Ident::from("buf"),
        index: Box::new(Expr::u32(0)),
        expected: None,
        value: Box::new(value),
        ordering: MemoryOrdering::Relaxed,
    }
}

fn count_atomics(program: &Program) -> usize {
    let mut count = 0;
    walk_exprs(program, |expr| {
        if matches!(expr, Expr::Atomic { .. }) {
            count += 1;
        }
    });
    count
}

/// One entry per node operand position, each holding `atomic` in that position.
/// Named by the position rather than by the variant, so a reader can see which
/// one regressed from the failure message alone.
fn cases(atomic: &Expr) -> Vec<(&'static str, Vec<Node>)> {
    vec![
        ("Trap::address", vec![Node::trap(atomic.clone(), "oob")]),
        (
            "AsyncLoad::offset",
            vec![Node::async_load_gpu_driven(
                Ident::from("buf"),
                Ident::from("out"),
                atomic.clone(),
                Expr::u32(1),
                Ident::from("t0"),
            )],
        ),
        (
            "AsyncLoad::size",
            vec![Node::async_load_gpu_driven(
                Ident::from("buf"),
                Ident::from("out"),
                Expr::u32(0),
                atomic.clone(),
                Ident::from("t0"),
            )],
        ),
        (
            "AsyncStore::offset",
            vec![Node::async_store(
                Ident::from("buf"),
                Ident::from("out"),
                atomic.clone(),
                Expr::u32(1),
                Ident::from("t0"),
            )],
        ),
        (
            "AsyncStore::size",
            vec![Node::async_store(
                Ident::from("buf"),
                Ident::from("out"),
                Expr::u32(0),
                atomic.clone(),
                Ident::from("t0"),
            )],
        ),
        (
            "Store::index",
            vec![Node::store("out", atomic.clone(), Expr::u32(1))],
        ),
        (
            "Store::value",
            vec![Node::store("out", Expr::u32(0), atomic.clone())],
        ),
        (
            "If::cond",
            vec![Node::if_then(
                Expr::eq(atomic.clone(), Expr::u32(0)),
                vec![Node::Return],
            )],
        ),
        (
            "Loop::to",
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                atomic.clone(),
                vec![Node::Return],
            )],
        ),
        ("Let::value", vec![Node::let_bind("x", atomic.clone())]),
        (
            "Assign::value",
            vec![
                Node::let_bind("x", Expr::u32(0)),
                Node::assign("x", atomic.clone()),
            ],
        ),
        (
            "nested in a Loop body",
            vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::store("out", Expr::var("i"), atomic.clone())],
            )],
        ),
    ]
}

/// The analysis must report a candidate for every operand position.
#[test]
fn analysis_finds_a_candidate_in_every_node_operand_position() {
    for (position, entry) in cases(&relaxed_atomic_add(Expr::u32(0))) {
        let analysis = ProgramPass::analyze(&AtomicMinimizePass, &program(entry));
        assert!(
            matches!(analysis, PassAnalysis::RUN),
            "Fix: atomic_minimize must run for an identity atomic in {position}, got {analysis:?}",
        );
    }
}

/// And the rewrite must actually collapse it, so the analysis is not merely
/// optimistic about a position the transform still skips.
#[test]
fn the_rewrite_collapses_the_atomic_in_every_node_operand_position() {
    for (position, entry) in cases(&relaxed_atomic_add(Expr::u32(0))) {
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(
            result.changed,
            "Fix: atomic_minimize must rewrite an identity atomic in {position}",
        );
        assert_eq!(
            count_atomics(&result.program),
            0,
            "Fix: an identity atomic survived in {position}",
        );
    }
}

/// Negative control: `atomic_add` of 1 is not the identity, so it must survive
/// in every one of those positions. Without this the two tests above would also
/// pass if the pass collapsed atomics indiscriminately.
#[test]
fn a_non_identity_atomic_survives_in_every_node_operand_position() {
    for (position, entry) in cases(&relaxed_atomic_add(Expr::u32(1))) {
        let result = AtomicMinimizePass::transform(program(entry));
        assert!(
            !result.changed,
            "Fix: atomic_add of 1 is not the identity and must survive in {position}",
        );
        assert_eq!(
            count_atomics(&result.program),
            1,
            "Fix: a non-identity atomic was rewritten in {position}",
        );
    }
}

/// A program with nothing to do must come back as the same allocation, not an
/// equal rebuild. The pass used to rebuild the entry unconditionally, so this is
/// the borrow-preservation contract `rewrite_program` brought with it.
#[test]
fn a_program_with_no_candidate_is_returned_unchanged() {
    let entry = vec![Node::store(
        "out",
        Expr::u32(0),
        relaxed_atomic_add(Expr::u32(1)),
    )];
    let before = program(entry);
    let before_entry_ptr = before.entry().as_ptr();

    let result = AtomicMinimizePass::transform(before);

    assert!(!result.changed);
    assert_eq!(
        result.program.entry().as_ptr(),
        before_entry_ptr,
        "Fix: an unchanged entry must be the same allocation, not an equal copy.",
    );
}
