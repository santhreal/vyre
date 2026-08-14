use super::*;
use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node};
use crate::test_util::count_nodes;
use crate::transform::visit::for_each_node;

fn buf() -> BufferDecl {
    BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
}

fn program_with_entry(entry: Vec<Node>) -> Program {
    Program::wrapped(vec![buf()], [1, 1, 1], entry)
}

/// Every `If` in `nodes`, at any nesting depth.
///
/// Counting comes from `test_util::count_nodes`, which descends through
/// `transform::visit::for_each_node`, the one owner of which node variants
/// nest. The hand-written match this replaces re-listed the four body-bearing
/// variants and ended in `_ => 0`, so an `If` inside a fifth nesting variant
/// went uncounted and the collapse assertions below read a smaller program than
/// the pass had produced.
fn count_ifs(nodes: &[Node]) -> usize {
    count_nodes(nodes, |node| matches!(node, Node::If { .. }))
}

/// The condition of the first `If` in document order, at any nesting depth.
///
/// Descent comes from `transform::visit::for_each_node`, the same owner
/// `count_ifs` uses, so the node this returns is always one of the nodes that
/// function counted. The hand-written match this replaces ended in `_ => {}`.
fn first_if_cond(entry: &[Node]) -> Option<&Expr> {
    let mut found = None;
    for_each_node(entry, |node| {
        if found.is_none() {
            if let Node::If { cond, .. } = node {
                found = Some(cond);
            }
        }
    });
    found
}

#[test]
fn coalesces_nested_if_with_two_pure_conds() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::var("c2"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(result.changed);
    let total = count_ifs(result.program.entry());
    assert_eq!(total, 1, "two nested Ifs collapse into one");
    let cond = first_if_cond(result.program.entry()).expect("Fix: must have an If");
    assert_eq!(cond, &Expr::and(Expr::var("c1"), Expr::var("c2")));
}

#[test]
fn does_not_coalesce_when_outer_has_sibling() {
    // Outer If body has an extra Store sibling alongside the inner
    // If  -  coalescing would change observable order.
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![
            Node::store("buf", Expr::u32(0), Expr::u32(7)),
            Node::if_then(
                Expr::var("c2"),
                vec![Node::store("buf", Expr::u32(1), Expr::u32(8))],
            ),
        ],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(
        !result.changed,
        "must not hoist sibling Store into combined branch"
    );
}

#[test]
fn does_not_coalesce_when_outer_has_otherwise() {
    let entry = vec![Node::if_then_else(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::var("c2"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
        vec![Node::store("buf", Expr::u32(0), Expr::u32(9))],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(!result.changed, "outer else-arm must be preserved");
}

#[test]
fn does_not_coalesce_when_inner_has_otherwise() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then_else(
            Expr::var("c2"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
            vec![Node::store("buf", Expr::u32(0), Expr::u32(9))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(!result.changed, "inner else-arm must be preserved");
}

#[test]
fn does_not_coalesce_when_outer_cond_loads_memory() {
    let entry = vec![Node::if_then(
        Expr::eq(Expr::load("buf", Expr::u32(0)), Expr::u32(0)),
        vec![Node::if_then(
            Expr::var("c2"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(
        !result.changed,
        "outer cond reads memory; conjoining could change ordering"
    );
}

#[test]
fn does_not_coalesce_when_inner_cond_loads_memory() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::eq(Expr::load("buf", Expr::u32(0)), Expr::u32(0)),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(
        !result.changed,
        "inner cond reads memory; conjoining could change ordering"
    );
}

#[test]
fn coalesces_three_level_nesting_in_one_pass() {
    // If(c1) { If(c2) { If(c3) { body } } } → If(And(And(c1,c2),c3)) { body }
    // bottom-up rewrite: inner two coalesce first, then outer
    // joins.
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::var("c2"),
            vec![Node::if_then(
                Expr::var("c3"),
                vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
            )],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(result.changed);
    let total = count_ifs(result.program.entry());
    assert_eq!(total, 1, "three nested Ifs collapse into one");
    let cond = first_if_cond(result.program.entry()).expect("Fix: must have an If");
    // Order: c2 and c3 join first, then c1 ANDed with that.
    let expected = Expr::and(Expr::var("c1"), Expr::and(Expr::var("c2"), Expr::var("c3")));
    assert_eq!(cond, &expected);
}

#[test]
fn analyze_skips_program_with_no_coalesceable_pair() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
    )];
    let program = program_with_entry(entry);
    assert_eq!(
        crate::optimizer::ProgramPass::analyze(&BranchCoalesce, &program),
        PassAnalysis::SKIP
    );
}

#[test]
fn analyze_runs_when_coalesceable_pair_present() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::var("c2"),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    assert_eq!(
        crate::optimizer::ProgramPass::analyze(&BranchCoalesce, &program),
        PassAnalysis::RUN
    );
}

#[test]
fn coalesces_inside_loop_body() {
    // Nested If inside a Loop body still coalesces  -  the Loop
    // itself is not the trigger; the rule fires on the inner pair.
    let loop_var = Ident::from("i");
    let entry = vec![Node::loop_for(
        loop_var.as_str(),
        Expr::u32(0),
        Expr::u32(8),
        vec![Node::if_then(
            Expr::var("c1"),
            vec![Node::if_then(
                Expr::var("c2"),
                vec![Node::store("buf", Expr::var("i"), Expr::u32(7))],
            )],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(result.changed);
    let total = count_ifs(result.program.entry());
    assert_eq!(total, 1, "nested If inside Loop coalesces");
}

/// `BufLen` is a dispatch-time constant (no observable side effect),
/// so `i < buf_len(buf)` is a perfectly valid coalesce operand.
/// Previously rejected as impure; this regression test pins the fix.
#[test]
fn coalesces_when_inner_cond_uses_buflen() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::lt(
                Expr::var("i"),
                Expr::BufLen {
                    buffer: Ident::from("buf"),
                },
            ),
            vec![Node::store("buf", Expr::var("i"), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(result.changed, "BufLen-bearing inner cond must coalesce");
    let total = count_ifs(result.program.entry());
    assert_eq!(total, 1);
}

/// `Fma` is pure arithmetic (a*b + c) when its operands are pure;
/// no reason to block coalescing on it.
#[test]
fn coalesces_when_inner_cond_uses_fma() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::lt(
                Expr::Fma {
                    a: Box::new(Expr::var("x")),
                    b: Box::new(Expr::var("y")),
                    c: Box::new(Expr::var("z")),
                },
                Expr::f32(1.0),
            ),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(result.changed, "pure-Fma inner cond must coalesce");
    let total = count_ifs(result.program.entry());
    assert_eq!(total, 1);
}

/// Negative twin: an Fma whose operand reads memory still blocks
/// coalesce. The recursive Fma check pins this.
#[test]
fn does_not_coalesce_fma_operand_loads_memory() {
    let entry = vec![Node::if_then(
        Expr::var("c1"),
        vec![Node::if_then(
            Expr::lt(
                Expr::Fma {
                    a: Box::new(Expr::load("buf", Expr::u32(0))),
                    b: Box::new(Expr::var("y")),
                    c: Box::new(Expr::var("z")),
                },
                Expr::f32(1.0),
            ),
            vec![Node::store("buf", Expr::u32(0), Expr::u32(7))],
        )],
    )];
    let program = program_with_entry(entry);
    let result = BranchCoalesce::transform(program);
    assert!(
        !result.changed,
        "Fma carrying a Load still reads memory; refuse"
    );
}
