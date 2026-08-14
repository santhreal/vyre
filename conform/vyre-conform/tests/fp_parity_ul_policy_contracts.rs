//! Contracts for [`vyre_conform::fp_parity`]  -  ULP budgets were spelled out in
//! module docs (`REFERENCE_TRANSCENDENTAL_ULP_BUDGET`, elementary vs transcendental
//! backend envelopes) but had no direct regression tests in this crate.

#![forbid(unsafe_code)]

use std::sync::Arc;

use vyre_conform::fp_parity::{
    f32_ulp_tolerance, BACKEND_ELEMENTARY_F32_ULP_BUDGET, BACKEND_TRANSCENDENTAL_ULP_BUDGET,
    REFERENCE_TRANSCENDENTAL_ULP_BUDGET,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program, UnOp};

fn minimal_elementary_f32_copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::F32),
            BufferDecl::output("out", 1, DataType::F32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len("out")),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::load("in", Expr::var("idx")),
                )],
            ),
        ],
    )
}

fn minimal_tanh_f32_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::F32),
            BufferDecl::output("out", 1, DataType::F32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len("out")),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::UnOp {
                        op: UnOp::Tanh,
                        operand: Box::new(Expr::load("in", Expr::var("idx"))),
                    },
                )],
            ),
        ],
    )
}

#[test]
fn reference_transcendental_budget_matches_documented_audit_anchor() {
    assert_eq!(
        REFERENCE_TRANSCENDENTAL_ULP_BUDGET, 4,
        "documented ceiling for the deterministic reference vs correctly-rounded transcendentals"
    );
}

#[test]
fn elementary_f32_program_uses_elementary_backend_budget() {
    let program = minimal_elementary_f32_copy_program();
    assert_eq!(
        f32_ulp_tolerance(&program),
        BACKEND_ELEMENTARY_F32_ULP_BUDGET,
        "non-transcendental F32 programs follow the elementary contraction budget under every \
         feature combination: a backend may fold a*b+c into one FMA, so no feature can bind the \
         backend-vs-reference window to zero without an emitter that forbids contraction"
    );
}

#[test]
fn transcendental_f32_program_uses_transcendental_backend_budget() {
    let program = minimal_tanh_f32_program();
    assert_eq!(
        f32_ulp_tolerance(&program),
        BACKEND_TRANSCENDENTAL_ULP_BUDGET,
        "documented native-transcendental envelope must apply whenever IR contains UnOp::Tanh"
    );
}

#[test]
fn transcendental_inside_nested_region_is_detected() {
    let inner = vec![
        Node::let_bind("idx", Expr::gid_x()),
        Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::buf_len("out")),
            vec![Node::store(
                "out",
                Expr::var("idx"),
                Expr::UnOp {
                    op: UnOp::Sqrt,
                    operand: Box::new(Expr::load("in", Expr::var("idx"))),
                },
            )],
        ),
    ];
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [64, 1, 1],
        vec![Node::Region {
            generator: Ident::from("vyre-conform::fixture.transcendental_region"),
            source_region: None,
            body: Arc::new(inner),
        }],
    );
    assert_eq!(
        f32_ulp_tolerance(&program),
        BACKEND_TRANSCENDENTAL_ULP_BUDGET,
        "policy scan must recurse through Region bodies so nested sqrt cannot hide behind a wrapper"
    );
}

#[path = "../../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use spec_variant_tables::builtin_un_ops;

/// `UnOp` variants a backend is allowed to lower to an approximate native
/// instruction, so a program containing one gets the wide backend window.
const APPROXIMATE_UN_OPS: &[&str] = &[
    "Cos",
    "Sin",
    "Sqrt",
    "Exp",
    "Log",
    "Log2",
    "Exp2",
    "Tan",
    "Acos",
    "Asin",
    "Atan",
    "Tanh",
    "Sinh",
    "Cosh",
    "InverseSqrt",
];

/// `UnOp` variants every backend must match the reference on within the
/// elementary window. `Reciprocal` sits here because cuda and wgpu both lower
/// it to a division rather than to an approximate reciprocal instruction.
const EXACT_UN_OPS: &[&str] = &[
    "Negate",
    "BitNot",
    "LogicalNot",
    "Popcount",
    "Clz",
    "Ctz",
    "ReverseBits",
    "Abs",
    "Floor",
    "Ceil",
    "Round",
    "Trunc",
    "Sign",
    "IsNan",
    "IsInf",
    "IsFinite",
    "Unpack4Low",
    "Unpack4High",
    "Unpack8Low",
    "Unpack8High",
    "Reciprocal",
];

/// A program that applies `op` to an f32 load, so the policy scan sees exactly
/// one unary operator. Never validated: the scan is syntactic, and several of
/// these operators are integer-only.
fn single_un_op_f32_program(op: UnOp) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::buf_len("out")),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::UnOp {
                        op,
                        operand: Box::new(Expr::load("in", Expr::var("idx"))),
                    },
                )],
            ),
        ],
    )
}

#[test]
fn every_frozen_un_op_carries_exactly_one_ulp_classification() {
    let mut unclassified = Vec::new();
    let mut double_classified = Vec::new();
    for op in builtin_un_ops() {
        let name = format!("{op:?}");
        let approximate = APPROXIMATE_UN_OPS.contains(&name.as_str());
        let exact = EXACT_UN_OPS.contains(&name.as_str());
        match (approximate, exact) {
            (false, false) => unclassified.push(name),
            (true, true) => double_classified.push(name),
            _ => {}
        }
    }
    assert!(
        unclassified.is_empty(),
        "Fix: decide the f32 ULP window for UnOp variant(s) {unclassified:?}. Put each in \
         APPROXIMATE_UN_OPS if a backend may lower it to an approximate native instruction, \
         and in vyre-foundation/src/fp_parity.rs expr_has_transcendental as well; otherwise \
         put it in EXACT_UN_OPS."
    );
    assert!(
        double_classified.is_empty(),
        "UnOp variant(s) {double_classified:?} are in both classification tables; a variant \
         gets one window"
    );
}

#[test]
fn no_ulp_classification_entry_names_a_variant_that_no_longer_exists() {
    let frozen: Vec<String> = builtin_un_ops()
        .iter()
        .map(|op| format!("{op:?}"))
        .collect();
    let stale: Vec<&&str> = APPROXIMATE_UN_OPS
        .iter()
        .chain(EXACT_UN_OPS)
        .filter(|name| !frozen.iter().any(|frozen| frozen == **name))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: drop {stale:?} from the ULP classification tables; \
         tests/support/spec_variant_tables.rs no longer lists those UnOp variants"
    );
}

#[test]
fn each_frozen_un_op_gets_the_window_its_classification_declares() {
    for op in builtin_un_ops() {
        let name = format!("{op:?}");
        let expected = if APPROXIMATE_UN_OPS.contains(&name.as_str()) {
            BACKEND_TRANSCENDENTAL_ULP_BUDGET
        } else {
            BACKEND_ELEMENTARY_F32_ULP_BUDGET
        };
        assert_eq!(
            f32_ulp_tolerance(&single_un_op_f32_program(op)),
            expected,
            "UnOp::{name} is classified in this suite but \
             vyre-foundation/src/fp_parity.rs expr_has_transcendental disagrees"
        );
    }
}
