//! V055 workgroup-uniform exit contracts.
//!
//! A synchronizing loop may exit after its final barrier only when every return
//! path is proven collective. Barrier-settled loads at a uniform index are the
//! motivating positive case. Lane-dependent guards, intervening writes, and
//! release-only barriers are negative twins.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{
    validate, BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program,
};

fn program(body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::storage("flag", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)],
        [4, 1, 1],
        vec![Node::loop_for("iter", Expr::u32(0), Expr::u32(8), body)],
    )
}

fn barrier(ordering: MemoryOrdering) -> Node {
    Node::Barrier { ordering }
}

fn guarded_return(cond: Expr) -> Node {
    Node::if_then(cond, vec![Node::Return])
}

fn v055_messages(program: &Program) -> Vec<String> {
    validate(program)
        .iter()
        .filter(|error| error.message().contains("V055"))
        .map(|error| error.message().to_string())
        .collect()
}

/// An acquiring barrier makes a same-address scalar load collective.
///
/// This is the exact proof that removes the DCE loop's redundant trailing
/// barrier: all lanes read `flag[0]` after prior writes are settled.
#[test]
fn settled_scalar_load_exit_is_accepted() {
    let program = program(vec![
        barrier(MemoryOrdering::SeqCst),
        guarded_return(Expr::eq(Expr::load("flag", Expr::u32(0)), Expr::u32(0))),
    ]);

    assert_eq!(v055_messages(&program), Vec::<String>::new());
}

/// A lane-dependent guard remains illegal after the same barrier.
///
/// This negative twin proves the carve-out derives agreement instead of
/// treating every post-barrier return as safe.
#[test]
fn lane_dependent_exit_is_rejected() {
    let program = program(vec![
        barrier(MemoryOrdering::SeqCst),
        guarded_return(Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0))),
    ]);

    assert_eq!(v055_messages(&program).len(), 1);
}

/// Writing the loaded buffer after the barrier invalidates its settled value.
///
/// Without another acquiring barrier, lanes can observe the write at different
/// times even though they load the same address.
#[test]
fn intervening_write_rejects_scalar_load_exit() {
    let program = program(vec![
        barrier(MemoryOrdering::SeqCst),
        Node::store("flag", Expr::u32(0), Expr::u32(0)),
        guarded_return(Expr::eq(Expr::load("flag", Expr::u32(0)), Expr::u32(0))),
    ]);

    assert_eq!(v055_messages(&program).len(), 1);
}

/// A settled buffer does not make divergent addresses uniform.
///
/// Every lane must read the same address. A lane-indexed load can produce a
/// different exit decision even when no writes follow the barrier.
#[test]
fn divergent_load_index_is_rejected() {
    let program = program(vec![
        barrier(MemoryOrdering::SeqCst),
        guarded_return(Expr::eq(
            Expr::load("flag", Expr::LocalId { axis: 0 }),
            Expr::u32(0),
        )),
    ]);

    assert_eq!(v055_messages(&program).len(), 1);
}

/// A release-only barrier cannot prove a following load sees settled writes.
///
/// Release publishes earlier writes but does not acquire the workgroup's
/// published values for the exit read.
#[test]
fn release_only_barrier_does_not_settle_exit_load() {
    let program = program(vec![
        barrier(MemoryOrdering::Release),
        guarded_return(Expr::eq(Expr::load("flag", Expr::u32(0)), Expr::u32(0))),
    ]);

    assert_eq!(v055_messages(&program).len(), 1);
}

/// An unconditional post-barrier return is collective by construction.
///
/// V055 protects agreement between lanes. It must not require a redundant
/// barrier when every invocation reaches the return on the same path.
#[test]
fn unconditional_exit_is_accepted() {
    let program = program(vec![barrier(MemoryOrdering::SeqCst), Node::Return]);

    assert_eq!(v055_messages(&program), Vec::<String>::new());
}

/// Uniformity derived from a settled load must propagate through a local.
///
/// Real builders name predicates before branching. Losing the proof at a
/// `Let` would reintroduce the same false rejection under harmless refactoring.
#[test]
fn settled_load_uniformity_propagates_through_let_binding() {
    let program = program(vec![
        barrier(MemoryOrdering::Acquire),
        Node::let_bind(
            "done",
            Expr::eq(Expr::load("flag", Expr::u32(0)), Expr::u32(0)),
        ),
        guarded_return(Expr::var("done")),
    ]);

    assert_eq!(v055_messages(&program), Vec::<String>::new());
}

/// An atomic read-modify-write condition is never a collective exit proof.
///
/// Each lane receives a different prior value, and the atomic also dirties the
/// buffer for every load that follows it before another barrier.
#[test]
fn atomic_exit_condition_is_rejected() {
    let program = program(vec![
        barrier(MemoryOrdering::SeqCst),
        guarded_return(Expr::eq(
            Expr::atomic_add("flag", Expr::u32(0), Expr::u32(1)),
            Expr::u32(0),
        )),
    ]);

    assert_eq!(v055_messages(&program).len(), 1);
}
