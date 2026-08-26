//! Observable contract of the encoded-order rewrite walk shared by semantic
//! optimizer stages and combined delta decoding.
//!
//! The walk has one owner, `optimizer::rewrite_walk`, and each pass supplies
//! only the decision taken at an Expr id. These tests pin what a caller can see:
//! the exact rewritten entry for every decision branch, the post-order id the
//! decision is keyed on, and the Node variants the walk descends into. A pass
//! that stops descending into `Region` bodies, mis-numbers an id, or applies a
//! decision to an Expr class the kernel never marks turns one of them red.
//!
//! Not covered here: what the GPU kernels compute. These tests feed the
//! decisions in directly, which is the only way to reach every branch without a
//! device.

use std::sync::Arc;

use vyre_foundation::ir::{BinOp, Expr, Node, Program, UnOp};
use vyre_pass_engine::optimizer::combined_decode::apply_combined_arena_deltas;
use vyre_pass_engine::optimizer::pattern_match_via_encoded::rewrite_action as ra;

/// `x + y` at ids 0, 1, 2.
fn add_program(left: Expr, right: Expr) -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::let_bind(
            "v",
            Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(left),
                right: Box::new(right),
            },
        )],
    )
}

fn only_expr(program: &Program) -> Expr {
    match program.entry() {
        [Node::Region { body, .. }] => match body.as_slice() {
            [Node::Let { value, .. }] => value.clone(),
            other => panic!("expected one Let, got {other:?}"),
        },
        [Node::Let { value, .. }] => value.clone(),
        other => panic!("expected a wrapped Let, got {other:?}"),
    }
}

fn deltas(len: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    (
        vec![0; len],
        vec![0; len],
        vec![0; len],
        vec![ra::NONE; len],
    )
}

#[test]
fn resident_decode_folds_arithmetic_binop_to_u32_literal() {
    let program = add_program(Expr::u32(2), Expr::u32(3));
    let (swap, mut foldable, mut value, action) = deltas(3);
    foldable[2] = 1;
    value[2] = 5;
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    assert_eq!(only_expr(&out), Expr::LitU32(5));
}

#[test]
fn resident_decode_folds_comparison_binop_to_bool_literal() {
    for (raw, expected) in [(0u32, false), (1, true), (7, true)] {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::let_bind(
                "v",
                Expr::BinOp {
                    op: BinOp::Lt,
                    left: Box::new(Expr::u32(1)),
                    right: Box::new(Expr::u32(2)),
                },
            )],
        );
        let (swap, mut foldable, mut value, action) = deltas(3);
        foldable[2] = 1;
        value[2] = raw;
        let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
        assert_eq!(
            only_expr(&out),
            Expr::LitBool(expected),
            "a folded comparison must stay Bool-shaped for value {raw}"
        );
    }
}

#[test]
fn resident_decode_applies_each_rewrite_action() {
    let inner = Expr::BinOp {
        op: BinOp::Mul,
        left: Box::new(Expr::u32(8)),
        right: Box::new(Expr::u32(9)),
    };
    // ids: 0 = 8, 1 = 9, 2 = inner Mul, 3 = 4, 4 = outer Add.
    let cases: [(u32, Expr); 7] = [
        (ra::REPLACE_WITH_LEFT, inner.clone()),
        (ra::REPLACE_WITH_RIGHT, Expr::u32(4)),
        (ra::REPLACE_WITH_LIT_ZERO, Expr::LitU32(0)),
        (ra::REPLACE_WITH_LIT_TRUE, Expr::LitBool(true)),
        (ra::REPLACE_WITH_LIT_FALSE, Expr::LitBool(false)),
        (ra::REPLACE_WITH_LEFT_INNER_LEFT, Expr::u32(8)),
        (ra::REPLACE_WITH_LEFT_INNER_RIGHT, Expr::u32(9)),
    ];
    for (tag, expected) in cases {
        let program = add_program(inner.clone(), Expr::u32(4));
        let (swap, foldable, value, mut action) = deltas(5);
        action[4] = tag;
        let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
        assert_eq!(only_expr(&out), expected, "rewrite action {tag} misapplied");
    }
}

#[test]
fn resident_decode_collapses_involutive_unop_pair() {
    let program = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::let_bind(
            "v",
            Expr::UnOp {
                op: UnOp::BitNot,
                operand: Box::new(Expr::UnOp {
                    op: UnOp::BitNot,
                    operand: Box::new(Expr::var("x")),
                }),
            },
        )],
    );
    // ids: 0 = Var, 1 = inner UnOp, 2 = outer UnOp.
    let (swap, foldable, value, mut action) = deltas(3);
    action[2] = ra::REPLACE_WITH_GRAND_OPERAND;
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    assert_eq!(only_expr(&out), Expr::var("x"));
}

#[test]
fn resident_decode_swaps_operands_only_when_no_higher_priority_action_fires() {
    let swapped = Expr::BinOp {
        op: BinOp::Add,
        left: Box::new(Expr::u32(3)),
        right: Box::new(Expr::u32(2)),
    };
    let program = add_program(Expr::u32(2), Expr::u32(3));
    let (mut swap, foldable, value, action) = deltas(3);
    swap[2] = 1;
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    assert_eq!(only_expr(&out), swapped);

    // A fold at the same id wins: nothing is left to swap.
    let (mut swap, mut foldable, mut value, action) = deltas(3);
    swap[2] = 1;
    foldable[2] = 1;
    value[2] = 5;
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    assert_eq!(only_expr(&out), Expr::LitU32(5));

    // So does a pattern-match action.
    let (mut swap, foldable, value, mut action) = deltas(3);
    swap[2] = 1;
    action[2] = ra::REPLACE_WITH_RIGHT;
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    assert_eq!(only_expr(&out), Expr::u32(3));
}

#[test]
fn a_folded_leaf_keeps_a_typed_literal_and_replaces_anything_else() {
    for (leaf, expected) in [
        (Expr::u32(1), Expr::u32(1)),
        (Expr::LitI32(-4), Expr::LitI32(-4)),
        (Expr::LitF32(2.5), Expr::LitF32(2.5)),
        (Expr::LitBool(false), Expr::LitBool(false)),
        (Expr::var("x"), Expr::LitU32(42)),
        (Expr::gid_x(), Expr::LitU32(42)),
        (Expr::SubgroupSize, Expr::LitU32(42)),
    ] {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::let_bind("v", leaf.clone())],
        );
        let (swap, mut foldable, mut value, action) = deltas(1);
        foldable[0] = 1;
        value[0] = 42;
        let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
        assert_eq!(
            only_expr(&out),
            expected,
            "leaf {leaf:?} folded to the wrong Expr"
        );
    }
}

#[test]
fn load_select_and_fma_keep_their_rewritten_children_even_when_marked_foldable() {
    let cases: [Expr; 3] = [
        Expr::Load {
            buffer: "buf".into(),
            index: Box::new(Expr::var("x")),
        },
        Expr::Select {
            cond: Box::new(Expr::var("x")),
            true_val: Box::new(Expr::var("x")),
            false_val: Box::new(Expr::var("x")),
        },
        Expr::Fma {
            a: Box::new(Expr::var("x")),
            b: Box::new(Expr::var("x")),
            c: Box::new(Expr::var("x")),
        },
    ];
    for root in cases {
        let child_count = match &root {
            Expr::Load { .. } => 1,
            Expr::Select { .. } | Expr::Fma { .. } => 3,
            other => panic!("unexpected case {other:?}"),
        };
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::let_bind("v", root.clone())],
        );
        // Every id is foldable, including the root. Children collapse, the root
        // does not.
        let (swap, foldable, value, action) = (
            vec![0; child_count + 1],
            vec![1; child_count + 1],
            vec![7; child_count + 1],
            vec![ra::NONE; child_count + 1],
        );
        let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
        let rewritten = only_expr(&out);
        assert_eq!(
            std::mem::discriminant(&rewritten),
            std::mem::discriminant(&root),
            "{root:?} must survive as its own Expr kind"
        );
        let folded_children = format!("{rewritten:?}").matches("LitU32(7)").count();
        assert_eq!(
            folded_children, child_count,
            "every child of {root:?} should have folded: {rewritten:?}"
        );
    }
}

#[test]
fn the_walk_descends_into_every_expr_bearing_node_variant() {
    // One `Var("x")` per Expr slot, every slot foldable, so any slot the walk
    // fails to visit shows up as a surviving `Var`.
    let x = || Expr::var("x");
    let body = vec![
        Node::let_bind("a", x()),
        Node::assign("a", x()),
        Node::store("out", x(), x()),
        Node::if_then_else(
            x(),
            vec![Node::let_bind("b", x())],
            vec![Node::assign("a", x())],
        ),
        Node::loop_for("i", x(), x(), vec![Node::store("out", x(), x())]),
        Node::Block(vec![Node::let_bind("c", x())]),
        Node::Region {
            generator: "g".into(),
            source_region: None,
            body: Arc::new(vec![Node::let_bind("d", x())]),
        },
        Node::AsyncLoad {
            source: "s".into(),
            destination: "d".into(),
            offset: Box::new(x()),
            size: Box::new(x()),
            tag: "t".into(),
        },
        Node::AsyncStore {
            source: "s".into(),
            destination: "d".into(),
            offset: Box::new(x()),
            size: Box::new(x()),
            tag: "t".into(),
        },
        Node::Trap {
            address: Box::new(x()),
            tag: "t".into(),
        },
    ];
    // Comfortably more slots than the body has Exprs; the lookup clamps.
    let slots = 64;
    let program = Program::wrapped(Vec::new(), [1, 1, 1], body);
    let out = apply_combined_arena_deltas(
        &program,
        &vec![0; slots],
        &vec![1; slots],
        &vec![9; slots],
        &vec![ra::NONE; slots],
    );
    let rendered = format!("{:?}", out.entry());
    assert!(
        !rendered.contains("Var("),
        "the walk skipped an Expr slot: {rendered}"
    );
}

#[test]
fn the_walk_stops_at_the_reachable_prefix() {
    let program = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::var("x")),
            Node::Return,
            Node::let_bind("dead", Expr::var("y")),
        ],
    );
    let (swap, foldable, value, action) = (vec![0; 4], vec![1; 4], vec![3; 4], vec![ra::NONE; 4]);
    let out = apply_combined_arena_deltas(&program, &swap, &foldable, &value, &action);
    let body = match out.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    };
    assert_eq!(
        body,
        vec![Node::let_bind("a", Expr::LitU32(3)), Node::Return],
        "Nodes after Return are unreachable and must not survive the rewrite"
    );
}
