//! The class closed here: a greedy sibling walk that consumes BOTH members of
//! a pair it refused to rewrite.
//!
//! `fuse_in_body` used to advance two nodes past a non-fusable pair, and the
//! comment on that line said the second loop "gets its chance against its own
//! successor on the scheduler's next iteration". It never did. Refusing every
//! pair leaves the body byte-identical, so the pass reports no change, the next
//! iteration walks the same nodes and reaches the same decision, and the pair
//! `(L1, L2)` is skipped for the life of the compile.
//!
//! The property is stated over the position of the fusable pair rather than
//! over one arrangement: for a body of N sibling loops in which exactly one
//! adjacent pair fuses, that pair must fuse wherever it sits. A walk that skips
//! by two is right only for an even offset, so a test pinned to a single layout
//! passes against the defect half the time.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_fusion::LoopFusion;
use vyre_foundation::visit::child_bodies;

/// A loop over `0..4` writing `value` into `buffer`.
///
/// Two of these fuse exactly when their buffers differ: bounds match, the loop
/// variables differ, and the touched-buffer sets are disjoint.
fn write_loop(var: &str, buffer: &str, value: u32) -> Node {
    Node::loop_for(
        var,
        Expr::u32(0),
        Expr::u32(4),
        vec![Node::store(buffer, Expr::var(var), Expr::u32(value))],
    )
}

/// `count` sibling loops, all but the pair at `pair_at` writing the SAME
/// buffer so no other adjacency can fuse.
///
/// Loop `pair_at + 1` is the only one writing `other`, so `(pair_at,
/// pair_at + 1)` is the sole fusable pair in the body.
fn program_with_one_fusable_pair(count: usize, pair_at: usize) -> Program {
    let entry = (0..count)
        .map(|index| {
            let buffer = if index == pair_at + 1 {
                "other"
            } else {
                "same"
            };
            write_loop(&format!("v{index}"), buffer, index as u32)
        })
        .collect();
    Program::wrapped(
        vec![
            BufferDecl::output("same", 0, DataType::U32).with_count(4),
            BufferDecl::output("other", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        entry,
    )
}

/// Sibling `Node::Loop` count directly under the program's entry region.
///
/// Descent is [`child_bodies`], the one owner of which variants nest, so this
/// counts the loops under whatever wrapper `Program::wrapped` puts them in
/// without naming that wrapper. `Node::Loop` stops the descent, which is what
/// makes the answer "sibling loops" rather than "every loop anywhere".
fn top_level_loops(program: &Program) -> usize {
    fn loops_in(nodes: &[Node]) -> usize {
        nodes
            .iter()
            .map(|node| match node {
                Node::Loop { .. } => 1,
                other => child_bodies(other).into_iter().map(loops_in).sum(),
            })
            .sum()
    }
    loops_in(program.entry())
}

/// WHY: the fusable pair must be found at every offset, not only at an even
/// one. Against the skip-by-two walk this goes red for `pair_at = 1` and
/// `pair_at = 3`, where the refused pair `(0, 1)` consumed loop 1 and the walk
/// resumed at loop 2 with the opportunity already behind it.
#[test]
fn a_fusable_pair_fuses_at_every_offset() {
    for pair_at in 0..4usize {
        let program = program_with_one_fusable_pair(5, pair_at);
        let before = top_level_loops(&program);
        assert_eq!(before, 5, "fixture must start with five sibling loops");

        let result = LoopFusion::transform(program);
        assert!(
            result.changed,
            "pair_at={pair_at}: a body holding a fusable pair must report a change"
        );
        assert_eq!(
            top_level_loops(&result.program),
            4,
            "pair_at={pair_at}: the one fusable pair must become one loop"
        );
    }
}

/// WHY: the refusal path must still terminate and must still report no change.
/// A walk that advances by one on a refusal repeats the left operand as the
/// next right operand, and an off-by-one there either loops forever or emits a
/// duplicate node.
#[test]
fn a_body_with_no_fusable_pair_is_left_alone() {
    let program = Program::wrapped(
        vec![BufferDecl::output("same", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        (0..5)
            .map(|index| write_loop(&format!("v{index}"), "same", index))
            .collect(),
    );
    let before = program.clone();

    let result = LoopFusion::transform(program);
    assert!(
        !result.changed,
        "no adjacent pair touches disjoint buffers, so nothing may fuse"
    );
    assert_eq!(
        result.program, before,
        "a refused body must come back byte-identical"
    );
}
