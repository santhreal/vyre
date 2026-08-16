//! The class closed here: a variable rename that reaches only the operand
//! slots someone remembered.
//!
//! Fusing two loops keeps the first loop variable and renames the second body
//! onto it. The rename walked `Let`, `Assign`, `Store`, `If`, `Loop`, `Block`
//! and `Region` and returned every other node whole. `AsyncLoad`, `AsyncStore`
//! and `Trap` carry `Expr` operands too, so a body whose async copy offset read
//! the second induction variable came out of fusion still reading a name that
//! nothing binds any more. The fusion gate deliberately admits async nodes,
//! which is what made the hole reachable rather than theoretical.
//!
//! The property is stated over the operand slots the workspace declares, not
//! over the ones this file thought of: `node_operand_samples` plants a marker
//! expression in every operand-carrying variant, and a new variant with an
//! operand joins that set and is fused here without anyone editing this test.
//! The surviving assertion is that the fused program reads no variable it does
//! not bind.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_fusion::LoopFusion;
use vyre_foundation::visit::{child_bodies, walk_exprs};
use vyre_test_support::ir_variants::node_operand_samples;

/// Induction variable of the second loop. Fusion retires it onto [`SURVIVING`].
const RETIRED: &str = "fixture_j";
/// Induction variable of the first loop, which fusion keeps.
const SURVIVING: &str = "fixture_k";

/// Two adjacent loops over the same literal range whose buffers are disjoint,
/// so the only thing that can refuse fusion is the content of `second`.
///
/// The declared buffers cover the names the shared operand fixtures use, so a
/// fused program still names only declared buffers.
fn two_loops(second: Node) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("fixture_out", 0, DataType::U32).with_count(4),
            BufferDecl::output("fixture_buffer", 1, DataType::U32).with_count(4),
            BufferDecl::output("fixture_src", 2, DataType::U32).with_count(4),
            BufferDecl::output("fixture_dst", 3, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::loop_for(
                SURVIVING,
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::store(
                    "fixture_out",
                    Expr::var(SURVIVING),
                    Expr::u32(1),
                )],
            ),
            Node::loop_for(RETIRED, Expr::u32(0), Expr::u32(4), vec![second]),
        ],
    )
}

/// Whether any scope in `nodes` binds `name`.
///
/// Descent is [`child_bodies`], the one owner of which variants nest, so a new
/// body-carrying variant is searched without this file naming it.
fn binds(nodes: &[Node], name: &str) -> bool {
    nodes.iter().any(|node| {
        let bound_here = matches!(node, Node::Loop { var, .. } if var.as_str() == name)
            || matches!(
                node,
                Node::Let { name: bound, .. } | Node::Assign { name: bound, .. }
                    if bound.as_str() == name
            );
        bound_here || child_bodies(node).into_iter().any(|body| binds(body, name))
    })
}

/// Whether any expression anywhere in `program` reads `name`.
fn reads(program: &Program, name: &str) -> bool {
    let target = Expr::var(name);
    let mut seen = false;
    walk_exprs(program, |expr| {
        if *expr == target {
            seen = true;
        }
    });
    seen
}

/// WHY: against the pre-fix rename this goes red for the `AsyncLoad.offset`,
/// `AsyncStore.offset` and `Trap.address` samples, whose operand kept the
/// retired name while the loop that bound it was gone. The refusal branch is
/// asserted too, so a pass that stopped fusing anything cannot make the
/// dangling-read assertion vacuous.
#[test]
fn a_fused_body_reads_no_variable_it_does_not_bind() {
    let samples = node_operand_samples(&Expr::var(RETIRED));
    assert!(
        !samples.is_empty(),
        "the shared operand fixture set must not be empty"
    );

    let mut fused = 0usize;
    for sample in &samples {
        let result = LoopFusion::transform(two_loops(sample.node.clone()));
        // A trap's effect cannot be summarised, so the legality gate refuses the
        // pair rather than interleaving iterations around it. Every other
        // operand-carrying variant is admitted, and admitting them is what makes
        // the rename load-bearing.
        let expected = !matches!(sample.node, Node::Trap { .. });
        assert_eq!(
            result.changed,
            expected,
            "{}: fusion decision changed, so the oracle below no longer describes \
             which bodies reach the rename",
            sample.label()
        );

        if !result.changed {
            assert!(
                binds(result.program.entry(), RETIRED),
                "{}: a refused fusion must leave the second loop and its binding in place",
                sample.label()
            );
            continue;
        }

        fused += 1;
        assert!(
            !binds(result.program.entry(), RETIRED),
            "{}: fusion keeps one induction variable, so nothing may still bind the other",
            sample.label()
        );
        assert!(
            !reads(&result.program, RETIRED),
            "{}: the fused body reads {RETIRED}, which no scope binds any more",
            sample.label()
        );
    }

    assert!(
        fused > 0,
        "no sample fused, so nothing exercised the rename"
    );
}

/// WHY: one marker per variant proves the variant is handled, not that every
/// one of its operand slots is. A rename arm written for `offset` alone leaves
/// `size` reading the retired name, and the fused program is accepted by the
/// same tests that pass for the offset fix.
#[test]
fn every_operand_slot_of_an_async_copy_is_renamed() {
    let both_slots = [
        Node::AsyncLoad {
            source: Ident::from("fixture_src"),
            destination: Ident::from("fixture_dst"),
            offset: Box::new(Expr::var(RETIRED)),
            size: Box::new(Expr::var(RETIRED)),
            tag: Ident::from("fixture_tag"),
        },
        Node::AsyncStore {
            source: Ident::from("fixture_src"),
            destination: Ident::from("fixture_dst"),
            offset: Box::new(Expr::var(RETIRED)),
            size: Box::new(Expr::var(RETIRED)),
            tag: Ident::from("fixture_tag"),
        },
    ];

    for node in both_slots {
        let result = LoopFusion::transform(two_loops(node));
        assert!(
            result.changed,
            "an async copy over disjoint buffers must fuse"
        );
        assert!(
            !reads(&result.program, RETIRED),
            "both operand slots of the copy must carry the surviving name"
        );
    }
}
