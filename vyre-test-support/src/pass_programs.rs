//! Input programs and expected shapes for the self-hosted optimizer pass suites.
//!
//! # Why this has one owner
//!
//! `gpu_canonicalize`, `gpu_const_fold` and `gpu_dce` run as vyre Programs on
//! whichever backend a suite dispatches through, and every backend's end-to-end
//! suite feeds them the same inputs and expects the same rewrite. Written per
//! backend, the input and the expectation drift: the wgpu suite and the CUDA
//! suite can end up asserting different rewrites of `1 + a` while both stay
//! green, and neither file says which one the pass owes.
//!
//! The case tables here are that input and that expectation, once. What stays in
//! each suite is the dispatcher and the run: a pass proven on naga's WGSL output
//! is not proven on PTX, so each backend still runs every case on its own live
//! device.
//!
//! # What the expectation is
//!
//! An expected `Expr` or node body compared for equality, not a partial `matches!`
//! over the operator and one operand. A partial match cannot see a pass that
//! rewrote the operator it was not asked about, and the rewrites here are exactly
//! specified: `1 + a` becomes `a + 1` and nothing else.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// A program that copies one element from `input` to `output`.
#[must_use]
pub fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read_write(input, 0, DataType::U32),
            BufferDecl::read_write(output, 1, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::LocalId { axis: 0 },
            Expr::load(input, Expr::LocalId { axis: 0 }),
        )],
    )
}

/// One `u32` copied from a read buffer to a backend-allocated output buffer.
///
/// The semantic seam addresses graph values rather than invocations, so a
/// fixture proving that seam declares a read input and a real output and
/// indexes logically. `copy_program` predates it and states two read-write
/// buffers indexed by local invocation, which no semantic request produces.
#[must_use]
pub fn logical_copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::logical_index(0),
            Expr::load("src", Expr::logical_index(0)),
        )],
    )
}

/// A program that adds single-element u32 values from `left` and `right` into `output`.
#[must_use]
pub fn add_program(left: &str, right: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(left, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(right, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(output, 2, BufferAccess::ReadWrite, DataType::U32),
        ],
        [32, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::add(
                Expr::load(left, Expr::u32(0)),
                Expr::load(right, Expr::u32(0)),
            ),
        )],
    )
}

/// The one-workgroup over-fire dispatch floor shared by every over-fire gate: the
/// largest declared buffer element count plus one whole workgroup of lanes.
#[must_use]
pub fn overfire_grid(program: &Program) -> u32 {
    let workgroup_lanes = program.workgroup_size()[0].max(1);
    let max_count = program
        .buffers()
        .iter()
        .map(BufferDecl::count)
        .max()
        .unwrap_or(0);
    max_count.saturating_add(workgroup_lanes)
}

/// A buffer-free program holding just `entry`, the shape every optimizer pass
/// suite feeds in.
#[must_use]
pub fn wrapped(entry: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

/// Peel the `Region` wrapper [`wrapped`] adds and read the single let-bound value.
///
/// # Panics
/// Panics when the program is not a single `Region` holding a single `Let`,
/// which means the pass under test changed the node shape rather than the value.
#[must_use]
pub fn first_let_value(program: &Program) -> Expr {
    match program.entry() {
        [Node::Region { body, .. }] => match body.as_slice() {
            [Node::Let { value, .. }] => value.clone(),
            body => panic!("expected single Let in body, got {body:?}"),
        },
        entry => panic!("expected wrapped Program with single Region, got {entry:?}"),
    }
}

/// The top-level node list, with the `Region` wrapper peeled if there is one.
#[must_use]
pub fn region_body(program: &Program) -> Vec<Node> {
    match program.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    }
}

/// One canonicalize case: a single-`Let` program and the value the pass owes.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalizeCase {
    /// Case name, as it appears in a failure message and in a lookup.
    pub label: &'static str,
    build: fn() -> Program,
    expected: fn() -> Expr,
}

impl CanonicalizeCase {
    /// The input program.
    #[must_use]
    pub fn input(&self) -> Program {
        (self.build)()
    }

    /// The let-bound value canonicalize owes for [`Self::input`].
    #[must_use]
    pub fn expected_first_let(&self) -> Expr {
        (self.expected)()
    }
}

/// Every canonicalize rewrite a backend suite asserts.
///
/// The commutative cases move the literal to the right; the two-literal,
/// two-variable and non-commutative rows are the cases the pass must leave
/// alone, which is the half a rewrite pass gets wrong by firing too eagerly.
pub const CANONICALIZE_CASES: &[CanonicalizeCase] = &[
    CanonicalizeCase {
        label: "lit_plus_var",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::add(Expr::u32(1), Expr::var("a")),
            )])
        },
        expected: || Expr::add(Expr::var("a"), Expr::u32(1)),
    },
    CanonicalizeCase {
        label: "var_plus_lit",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::add(Expr::var("a"), Expr::u32(1)),
            )])
        },
        expected: || Expr::add(Expr::var("a"), Expr::u32(1)),
    },
    CanonicalizeCase {
        label: "two_lits",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::add(Expr::u32(2), Expr::u32(3)),
            )])
        },
        expected: || Expr::add(Expr::u32(2), Expr::u32(3)),
    },
    CanonicalizeCase {
        label: "two_vars",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::add(Expr::var("a"), Expr::var("b")),
            )])
        },
        expected: || Expr::add(Expr::var("a"), Expr::var("b")),
    },
    CanonicalizeCase {
        label: "lit_times_var",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::mul(Expr::u32(5), Expr::var("a")),
            )])
        },
        expected: || Expr::mul(Expr::var("a"), Expr::u32(5)),
    },
    CanonicalizeCase {
        label: "non_commutative_div",
        build: || {
            wrapped(vec![Node::let_bind(
                "x",
                Expr::div(Expr::u32(10), Expr::var("a")),
            )])
        },
        expected: || Expr::div(Expr::u32(10), Expr::var("a")),
    },
];

/// One multi-pass case: an input program and the node body that must survive
/// `canonicalize -> const_fold -> dce`.
#[derive(Debug, Clone, Copy)]
pub struct PipelineCase {
    /// Case name, as it appears in a failure message and in a lookup.
    pub label: &'static str,
    build: fn() -> Program,
    expected: fn() -> Vec<Node>,
}

impl PipelineCase {
    /// The input program.
    #[must_use]
    pub fn input(&self) -> Program {
        (self.build)()
    }

    /// The node body the three passes owe for [`Self::input`].
    #[must_use]
    pub fn expected_body(&self) -> Vec<Node> {
        (self.expected)()
    }
}

/// Every `canonicalize -> const_fold -> dce` case a backend suite asserts.
pub const PIPELINE_CASES: &[PipelineCase] = &[
    // let dead = 99;                  dropped by DCE, nothing reads it
    // let live = 1 + 2;               two literals, canonical already, folds to 3
    // store buf 0 (3 + live);         canonicalize moves the literal right; the
    //                                 sum cannot fold because `live` is a Var
    PipelineCase {
        label: "dead_let_and_unfoldable_store",
        build: || {
            wrapped(vec![
                Node::let_bind("dead", Expr::u32(99)),
                Node::let_bind("live", Expr::add(Expr::u32(1), Expr::u32(2))),
                Node::store(
                    "buf",
                    Expr::u32(0),
                    Expr::add(Expr::u32(3), Expr::var("live")),
                ),
            ])
        },
        expected: || {
            vec![
                Node::let_bind("live", Expr::u32(3)),
                Node::store(
                    "buf",
                    Expr::u32(0),
                    Expr::add(Expr::var("live"), Expr::u32(3)),
                ),
            ]
        },
    },
    // let a = 5 + 7;                  folds to 12, kept: the store reads it
    // let b = a * 2;                  dropped by DCE, nothing reads b
    // let c = b - 4;                  dropped by DCE, nothing reads c
    // store buf 0 (a + 1);            already canonical, `a` is a Var so no fold
    PipelineCase {
        label: "unused_compute_chain",
        build: || {
            wrapped(vec![
                Node::let_bind("a", Expr::add(Expr::u32(5), Expr::u32(7))),
                Node::let_bind("b", Expr::mul(Expr::var("a"), Expr::u32(2))),
                Node::let_bind("c", Expr::sub(Expr::var("b"), Expr::u32(4))),
                Node::store("buf", Expr::u32(0), Expr::add(Expr::var("a"), Expr::u32(1))),
            ])
        },
        expected: || {
            vec![
                Node::let_bind("a", Expr::u32(12)),
                Node::store("buf", Expr::u32(0), Expr::add(Expr::var("a"), Expr::u32(1))),
            ]
        },
    },
];

/// The canonicalize case named `label`.
///
/// # Panics
/// Panics naming every available case when `label` is not in the table.
#[must_use]
pub fn canonicalize_case(label: &str) -> &'static CanonicalizeCase {
    CANONICALIZE_CASES
        .iter()
        .find(|case| case.label == label)
        .unwrap_or_else(|| {
            let available: Vec<&str> = CANONICALIZE_CASES.iter().map(|c| c.label).collect();
            panic!(
                "no canonicalize case named {label:?}. Fix: use one of {available:?} or add the \
                 case to CANONICALIZE_CASES."
            )
        })
}

/// The pipeline case named `label`.
///
/// # Panics
/// Panics naming every available case when `label` is not in the table.
#[must_use]
pub fn pipeline_case(label: &str) -> &'static PipelineCase {
    PIPELINE_CASES
        .iter()
        .find(|case| case.label == label)
        .unwrap_or_else(|| {
            let available: Vec<&str> = PIPELINE_CASES.iter().map(|c| c.label).collect();
            panic!(
                "no pipeline case named {label:?}. Fix: use one of {available:?} or add the case \
                 to PIPELINE_CASES."
            )
        })
}

/// Assert `optimized` binds exactly the value the canonicalize case owes.
///
/// `backend` names the arm that produced `optimized`, because the useful part of
/// the failure is which backend's dispatch of the pass disagreed.
///
/// # Panics
/// Panics with both values when the let-bound value is not the expected one.
#[track_caller]
pub fn assert_canonicalized(backend: &str, case: &CanonicalizeCase, optimized: &Program) {
    let got = first_let_value(optimized);
    let expected = case.expected_first_let();
    assert_eq!(
        got, expected,
        "{backend} canonicalize case `{}` produced the wrong let-bound value",
        case.label
    );
}

/// Assert `optimized` holds exactly the node body the pipeline case owes.
///
/// # Panics
/// Panics with both bodies when they differ.
#[track_caller]
pub fn assert_pipeline_body(backend: &str, case: &PipelineCase, optimized: &Program) {
    let got = region_body(optimized);
    let expected = case.expected_body();
    assert_eq!(
        got, expected,
        "{backend} pipeline case `{}` produced the wrong surviving body",
        case.label
    );
}
