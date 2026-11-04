//! The class closed here: two implementations of one rewrite disagreeing about
//! what is legal.
//!
//! Loop-invariant hoisting existed twice. The pass-engine copy refused to hoist
//! a binding whose name is bound elsewhere in the enclosing scope, because
//! hoisting flat-splices the binding into that scope and two of them are a
//! duplicate sibling binding the validator rejects. The resident pipeline's
//! copy had no such guard and admitted loads the pass engine refused, so the
//! same program came out of the two pipelines rewritten differently, and out of
//! the resident one invalid.
//!
//! The two properties below are what the surviving owner has to hold at once:
//! a load from a read-only buffer leaves the loop, and no hoist produces a
//! program that fails validation.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_licm::LoopLicm;
use vyre_foundation::transform::licm::apply_licm;
use vyre_foundation::visit::child_bodies;

/// `count` sibling loops, each binding `name` to the same read-only load and
/// storing it. With one loop the binding is unique in the enclosing scope and
/// hoisting it is legal; with two, hoisting both would bind `name` twice as
/// siblings.
fn sibling_loops(count: usize, name: &str) -> Program {
    let entry = (0..count)
        .map(|index| {
            Node::loop_for(
                format!("i{index}"),
                Expr::u32(0),
                Expr::u32(4),
                vec![
                    Node::let_bind(name, Expr::load("src", Expr::u32(0))),
                    Node::store("out", Expr::var(format!("i{index}")), Expr::var(name)),
                ],
            )
        })
        .collect();
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        entry,
    )
}

/// Bindings of `name` that sit outside every loop.
fn bindings_outside_loops(nodes: &[Node], name: &str) -> usize {
    nodes
        .iter()
        .map(|node| {
            if matches!(node, Node::Loop { .. }) {
                return 0;
            }
            let here = usize::from(
                matches!(node, Node::Let { name: bound, .. } if bound.as_str() == name),
            );
            here + child_bodies(node)
                .into_iter()
                .map(|body| bindings_outside_loops(body, name))
                .sum::<usize>()
        })
        .sum()
}

/// WHY: the pass engine refused every `Load`, so a read-only load stayed in the
/// loop and was re-issued on every iteration. This goes red against that
/// refusal, and it is the capability the deleted second implementation carried.
#[test]
fn a_read_only_load_leaves_the_loop() {
    let result = LoopLicm::transform(sibling_loops(1, "base"));
    assert!(
        result.changed,
        "a load from a buffer nothing writes is invariant across the iteration space"
    );
    assert_eq!(
        bindings_outside_loops(result.program.entry(), "base"),
        1,
        "the binding must sit above the loop, not inside it"
    );
    result
        .program
        .validate()
        .expect("the hoisted program must still validate");
}

/// WHY: the resident pipeline's implementation hoisted both bindings and
/// produced two sibling `Let base` in one scope. This goes red against it at
/// the validator, which is where that program would have been rejected.
#[test]
fn a_name_bound_by_two_sibling_loops_stays_put() {
    let before = sibling_loops(2, "base");
    let after = apply_licm(&before);

    assert_eq!(
        bindings_outside_loops(after.entry(), "base"),
        0,
        "hoisting either binding would bind the same name twice in one scope"
    );
    after
        .validate()
        .expect("loop-invariant hoisting must not produce a program the validator rejects");
}

/// WHY: the read-only fact is the whole licence. A buffer the program may write
/// can be stored to inside or after the loop, so its load is not invariant.
#[test]
fn a_load_from_a_writable_buffer_stays_put() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![
                Node::let_bind("base", Expr::load("scratch", Expr::u32(0))),
                Node::store("out", Expr::var("i"), Expr::var("base")),
            ],
        )],
    );

    let result = LoopLicm::transform(program);
    assert!(
        !result.changed,
        "a load from a buffer the program may write cannot leave the loop"
    );
}
