//! CUDA harness for the self-hosted optimizer, and the IR shapes its
//! end-to-end pass suites assert against.
//!
//! `vyre_pass_engine::optimizer` drives every GPU pass through
//! `gpu_pipeline_resident` against a `ProgramDispatcher`. Running that on a live
//! `CudaBackend` and reading the post-pipeline node shape is one
//! implementation, not one per pass suite. Each suite keeps only the programs
//! and expected literals that are its reason to exist.

#![cfg(feature = "device-tests")]

use vyre::ir::UnOp;
use vyre::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_driver_cuda::CudaProgramDispatcher;
use vyre_pass_engine::optimizer::pipeline_resident::gpu_pipeline_resident;

use super::live_backend;

/// Run `p` through the full persistent-resident GPU optimizer pipeline on the
/// live device.
///
/// # Panics
///
/// Panics when the pipeline itself fails, which is a backend or pass defect
/// rather than an unmet expectation about the optimized shape.
pub(crate) fn run_pipeline(p: Program) -> Program {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    gpu_pipeline_resident(p, &dispatcher).expect("pipeline must succeed")
}

/// Peel the `Region` wrapper `Program::wrapped` adds and return the top-level
/// node list.
pub(crate) fn body_of(out: &Program) -> Vec<Node> {
    match out.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    }
}

/// The stored value of the single surviving `Store`.
///
/// # Panics
///
/// Panics when no `Store` survived, which means the pass under test deleted the
/// program's only observable effect rather than rewriting its value.
pub(crate) fn store_value(out: &Program) -> Expr {
    let body = body_of(out);
    let store = body
        .iter()
        .find(|n| matches!(n, Node::Store { .. }))
        .unwrap_or_else(|| panic!("store survives; body={body:?}"));
    match store {
        Node::Store { value, .. } => value.clone(),
        other => unreachable!("find matched a non-Store node: {other:?}"),
    }
}

/// The condition of the single surviving `If`.
///
/// # Panics
///
/// Panics when no `If` survived, which means the pass folded the branch away
/// instead of simplifying its condition.
pub(crate) fn if_cond(out: &Program) -> Expr {
    let body = body_of(out);
    let branch = body
        .iter()
        .find(|n| matches!(n, Node::If { .. }))
        .unwrap_or_else(|| panic!("if survives; body={body:?}"));
    match branch {
        Node::If { cond, .. } => cond.clone(),
        other => unreachable!("find matched a non-If node: {other:?}"),
    }
}

/// Whether any `Let` binding named `name` survived the pipeline.
pub(crate) fn binds_let(out: &Program, name: &str) -> bool {
    body_of(out)
        .iter()
        .any(|n| matches!(n, Node::Let { name: bound, .. } if bound.as_str() == name))
}

/// Whether any `Let` binding at all survived the pipeline.
pub(crate) fn binds_any_let(out: &Program) -> bool {
    body_of(out).iter().any(|n| matches!(n, Node::Let { .. }))
}

/// Run `p` and read the surviving store's value: the shape every fold and
/// identity case asserts on.
pub(crate) fn optimized_store_value(p: Program) -> Expr {
    store_value(&run_pipeline(p))
}

/// A buffer-free program storing `value` into `buf[0]`.
///
/// This is the shape every literal-fold case uses: no operand reaches outside
/// the expression, so the whole store must collapse to one literal.
pub(crate) fn store_program(value: Expr) -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::store("buf", Expr::u32(0), value)],
    )
}

/// A program binding `x` to a non-literal value (`Load(input, 0)`) before
/// storing `value`.
///
/// The load is what makes the case test an identity rule rather than a fold:
/// const-prop at the end of the pipeline cannot turn `Var(x)` into a literal,
/// so a collapse to `Var(x)` proves the pattern matcher fired. The `input`
/// buffer is declared so the IR stays well-typed.
pub(crate) fn x_load_program(value: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("buf", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::load("input", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), value),
        ],
    )
}

/// A program binding both `x` and `y` to distinct non-literal loads before
/// storing `value`.
///
/// Cancellation rules such as `(x + y) - y` need two independently opaque
/// operands, so neither side can be const-folded into the other.
pub(crate) fn xy_load_program(value: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("inx", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("iny", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("buf", 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::load("inx", Expr::u32(0))),
            Node::let_bind("y", Expr::load("iny", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), value),
        ],
    )
}

/// A program whose `If` on `cond` stores a distinct marker word per arm.
///
/// Which marker survives names the arm the pipeline selected, so a branch-fold
/// case reads the taken arm off the store instead of inspecting the branch.
pub(crate) fn branch_marker_program(cond: Expr, then_word: u32, else_word: u32) -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::if_then_else(
            cond,
            vec![Node::store("buf", Expr::u32(0), Expr::u32(then_word))],
            vec![Node::store("buf", Expr::u32(0), Expr::u32(else_word))],
        )],
    )
}

/// A program binding `b` to a non-literal Bool before branching on `cond` with
/// a distinct marker word per arm.
///
/// `b` is `Load(input, 0) == 7`, which is Bool-typed and opaque to const-prop,
/// so a boolean identity rule over `Var("b")` is the only thing that can
/// collapse `cond`. Build `cond` from `Expr::var("b")`.
pub(crate) fn b_load_branch_program(cond: Expr, then_word: u32, else_word: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("buf", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind(
                "b",
                Expr::eq(Expr::load("input", Expr::u32(0)), Expr::u32(7)),
            ),
            Node::if_then_else(
                cond,
                vec![Node::store("buf", Expr::u32(0), Expr::u32(then_word))],
                vec![Node::store("buf", Expr::u32(0), Expr::u32(else_word))],
            ),
        ],
    )
}

/// Assert the pipeline decided `cond` statically: the `If` is gone and the one
/// surviving store holds `expected`.
///
/// # Panics
///
/// Panics when an `If` survived, which means the condition did not fold to a
/// literal, or when the wrong arm's marker survived.
#[track_caller]
pub(crate) fn assert_branch_folded_to(out: &Program, expected: u32) {
    let body = body_of(out);
    assert!(
        body.iter().all(|n| !matches!(n, Node::If { .. })),
        "cond must fold to a literal and drop the If; body={body:?}"
    );
    assert_lit_u32(&store_value(out), expected);
}

/// Assert the surviving `If` condition is no longer headed by `op`.
///
/// A boolean identity rule that fires rewrites the condition to a simpler
/// expression while leaving the runtime branch in place, so the contract is the
/// absence of the original operator rather than one specific replacement.
///
/// # Panics
///
/// Panics when no `If` survived, or when its condition still has `op` on top.
#[track_caller]
pub(crate) fn assert_cond_not_headed_by(out: &Program, op: BinOp) {
    let cond = if_cond(out);
    assert!(
        !matches!(&cond, Expr::BinOp { op: actual, .. } if *actual == op),
        "{op:?} must collapse out of the cond; got cond={cond:?}"
    );
}

/// `store_program(value)` run through the pipeline, returning the folded store
/// value.
pub(crate) fn folded_store_value(value: Expr) -> Expr {
    optimized_store_value(store_program(value))
}

/// `x_load_program(value)` run through the pipeline, returning the collapsed
/// store value.
pub(crate) fn folded_x_store_value(value: Expr) -> Expr {
    optimized_store_value(x_load_program(value))
}

/// `xy_load_program(value)` run through the pipeline, returning the collapsed
/// store value.
pub(crate) fn folded_xy_store_value(value: Expr) -> Expr {
    optimized_store_value(xy_load_program(value))
}

/// `branch_marker_program` run through the pipeline, returning the marker word
/// of the arm that survived.
pub(crate) fn taken_branch_marker(cond: Expr, then_word: u32, else_word: u32) -> Expr {
    optimized_store_value(branch_marker_program(cond, then_word, else_word))
}

/// Build a `BinOp` for the variants without a named `Expr` constructor.
pub(crate) fn binop(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr::BinOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Build a `UnOp` for the variants without a named `Expr` constructor.
pub(crate) fn unop(op: UnOp, operand: Expr) -> Expr {
    Expr::UnOp {
        op,
        operand: Box::new(operand),
    }
}

/// Whether `value` is exactly `LitU32(expected)`.
///
/// The predicate form exists for cases parameterized over an op, where the
/// caller's assertion message has to name which op failed.
pub(crate) fn is_lit_u32(value: &Expr, expected: u32) -> bool {
    matches!(value, Expr::LitU32(actual) if *actual == expected)
}

/// Whether `value` is a boolean-valued literal equal to `expected`.
///
/// A comparison can fold either to `LitBool` or to the `LitU32` 0/1 word the
/// GPU fold kernel writes, and both satisfy the same contract.
pub(crate) fn is_bool_word(value: &Expr, expected: u32) -> bool {
    match value {
        Expr::LitU32(actual) => *actual == expected,
        Expr::LitBool(actual) => u32::from(*actual) == expected,
        _ => false,
    }
}

/// Assert `value` is exactly `LitU32(expected)`.
///
/// # Panics
///
/// Panics with the actual expression when the pass produced a different
/// literal, or did not fold to a literal at all.
#[track_caller]
pub(crate) fn assert_lit_u32(value: &Expr, expected: u32) {
    assert!(
        is_lit_u32(value, expected),
        "expected LitU32({expected}); got {value:?}"
    );
}

/// Assert `value` is exactly `LitI32(expected)`.
///
/// # Panics
///
/// Panics with the actual expression on any other shape.
#[track_caller]
pub(crate) fn assert_lit_i32(value: &Expr, expected: i32) {
    assert!(
        matches!(value, Expr::LitI32(actual) if *actual == expected),
        "expected LitI32({expected}); got {value:?}"
    );
}

/// Assert `value` is exactly `LitBool(expected)`.
///
/// This is stricter than [`is_bool_word`] on purpose: a self-comparison rule
/// that fires must yield a Bool-typed literal, not a numeric word.
///
/// # Panics
///
/// Panics with the actual expression on any other shape.
#[track_caller]
pub(crate) fn assert_lit_bool(value: &Expr, expected: bool) {
    assert!(
        matches!(value, Expr::LitBool(actual) if *actual == expected),
        "expected LitBool({expected}); got {value:?}"
    );
}

/// Assert `value` collapsed to exactly `Var(name)`.
///
/// # Panics
///
/// Panics with the actual expression when the identity rule did not fire, or
/// selected the wrong operand.
#[track_caller]
pub(crate) fn assert_var(value: &Expr, name: &str) {
    match value {
        Expr::Var(bound) if bound.as_str() == name => {}
        other => panic!("expected Var({name}); got {other:?}"),
    }
}

/// Assert `value` is still a `BinOp` over the two given `u32` literals, meaning
/// the fold was correctly skipped.
///
/// # Panics
///
/// Panics when the operand shape changed, which is how a guard that stopped
/// firing shows up: folding these operands would be the defect.
#[track_caller]
pub(crate) fn assert_unfolded_u32_binop(value: &Expr, left_lit: u32, right_lit: u32) {
    match value {
        Expr::BinOp { left, right, .. } => {
            assert_lit_u32(left.as_ref(), left_lit);
            assert_lit_u32(right.as_ref(), right_lit);
        }
        other => panic!("expected unfolded BinOp({left_lit}, {right_lit}); got {other:?}"),
    }
}
