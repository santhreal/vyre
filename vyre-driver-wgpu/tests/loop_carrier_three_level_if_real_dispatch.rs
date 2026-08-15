//! Real-GPU regression test for the c-parser scope-walker bug shape:
//! a loop body whose only outer-var Assign is nested 3-deep inside
//! `if_then(cond_a) { if_then(cond_b) { if_then_else(cond_c) { assign(out_var, ..) } } }`.
//!
//! `vyre-emit-naga/tests/carrier_scope_regression.rs` only checks WGSL
//! validation; it cannot catch behavioral divergence. This test
//! actually dispatches the program on the live wgpu backend and asserts
//! the carrier value escapes the loop with the expected post-iteration
//! value.

mod common;
use common::acquire_live_backend as live_backend;

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_spec::c11_token::{TOK_LBRACE, TOK_RBRACE};

const SENTINEL: u32 = u32::MAX;

/// The two sequential conditionals the c-parser scope walker emits, reading
/// `scope_kind` and latching `scope_open` to `latch`.
///
/// This is the shape under test, and it is the same shape
/// `vyre_libs::parsing::c::parse::vast` emits in
/// `c11_typedef_scope_open_for_row`: an unconditional depth bump on `}`, then a
/// three-level nest whose innermost arm is the only writer of `scope_open`.
/// Both gates read `scope_open`, so the merge between the two conditionals is
/// the chokepoint a carrier lowering has to get right. Three of the programs
/// below differ only in how they reach `scope_kind`, so restating the nest per
/// program let them drift apart from the production shape they exist to pin.
fn scope_latch(latch: Expr) -> Vec<Node> {
    vec![
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("scope_open"), Expr::u32(SENTINEL)),
                Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_RBRACE)),
            ),
            vec![Node::assign(
                "scope_depth",
                Expr::add(Expr::var("scope_depth"), Expr::u32(1)),
            )],
        ),
        Node::if_then(
            Expr::eq(Expr::var("scope_open"), Expr::u32(SENTINEL)),
            vec![Node::if_then(
                Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_LBRACE)),
                vec![Node::if_then_else(
                    Expr::eq(Expr::var("scope_depth"), Expr::u32(0)),
                    vec![Node::assign("scope_open", latch)],
                    vec![Node::assign(
                        "scope_depth",
                        Expr::sub(Expr::var("scope_depth"), Expr::u32(1)),
                    )],
                )],
            )],
        ),
    ]
}

/// The full backward scope walk for one row: fresh carriers, then a reverse
/// scan from `row` down to zero running [`scope_latch`] on each token.
fn backward_scope_walk(row: Expr) -> Vec<Node> {
    let mut body = vec![
        Node::let_bind(
            "scope_rev",
            Expr::sub(
                Expr::sub(row.clone(), Expr::u32(1)),
                Expr::var("scope_scan"),
            ),
        ),
        Node::let_bind("scope_kind", Expr::load("kinds", Expr::var("scope_rev"))),
    ];
    body.extend(scope_latch(Expr::var("scope_rev")));
    vec![
        Node::let_bind("scope_open", Expr::u32(SENTINEL)),
        Node::let_bind("scope_depth", Expr::u32(0)),
        Node::loop_for("scope_scan", Expr::u32(0), row, body),
    ]
}

fn kind_bytes(kinds: [u32; 4]) -> Vec<u8> {
    kinds.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn dispatch_words(prog: &Program, kinds: [u32; 4]) -> Vec<u32> {
    let outputs = live_backend()
        .dispatch(prog, &[kind_bytes(kinds)], &DispatchConfig::default())
        .expect("dispatch succeeds");
    assert_eq!(outputs.len(), 1);
    outputs[0]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Mirrors the c-parser `c11_annotate_typedef_names` scope walker shape:
/// outer `let scope_open = SENTINEL`, then a loop iterating `i = 0..N`,
/// where the assign to `scope_open` lives 3 levels deep:
/// `if (scope_open == SENTINEL) { if (kind == LBRACE) { if_then_else (depth == 0) { assign scope_open = i } { assign depth-=1 } } }`.
///
/// Inputs: a `kinds` buffer with one byte per token; we set kinds[0]=LBRACE so
/// iteration 0 should latch `scope_open` to 0 and leave it pinned for the rest.
#[test]
fn three_level_if_assign_in_loop_propagates_via_carrier() {
    const N: u32 = 4;

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("kinds", 0, BufferAccess::ReadOnly, DataType::U32).with_count(N),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind("scope_open", Expr::u32(SENTINEL)),
                Node::let_bind("scope_depth", Expr::u32(0)),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::u32(N),
                    vec![
                        Node::let_bind("scope_kind", Expr::load("kinds", Expr::var("i"))),
                        Node::if_then(
                            Expr::eq(Expr::var("scope_open"), Expr::u32(SENTINEL)),
                            vec![Node::if_then(
                                Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_LBRACE)),
                                vec![Node::if_then_else(
                                    Expr::eq(Expr::var("scope_depth"), Expr::u32(0)),
                                    vec![Node::assign("scope_open", Expr::var("i"))],
                                    vec![Node::assign(
                                        "scope_depth",
                                        Expr::sub(Expr::var("scope_depth"), Expr::u32(1)),
                                    )],
                                )],
                            )],
                        ),
                    ],
                ),
                Node::store("out", Expr::u32(0), Expr::var("scope_open")),
            ],
        )],
    );

    // 4 tokens: kinds = [LBRACE, 0, 0, 0]
    let words = dispatch_words(&prog, [TOK_LBRACE, 0, 0, 0]);
    assert_eq!(
        words,
        vec![0],
        "scope_open must latch to 0 in iteration 0 and propagate through to post-loop store; got {words:?}",
    );
}

/// EXACT match for c11_annotate_typedef_names_impl shape: parallel
/// per-invocation work (no outer loop), each invocation has its own
/// `scope_open` / `scope_depth` let-bindings, then runs the scope_scan walker.
/// No `if gid==0` gate; multiple invocations execute concurrently.
#[test]
fn parallel_per_row_scope_walker_via_invocation_id() {
    const N: u32 = 4;

    let row = Expr::InvocationId { axis: 0 };
    let mut gated = backward_scope_walk(row.clone());
    gated.push(Node::store("out", row.clone(), Expr::var("scope_open")));

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("kinds", 0, BufferAccess::ReadOnly, DataType::U32).with_count(N),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [N, 1, 1],
        vec![Node::if_then(Expr::lt(row, Expr::u32(N)), gated)],
    );

    let words = dispatch_words(&prog, [TOK_LBRACE, TOK_LBRACE, 0, 0]);
    assert_eq!(
        words,
        vec![SENTINEL, 0, 1, 1],
        "parallel per-invocation scope walker must produce one scope per row; got {words:?}",
    );
}

/// Real repro: the scope walker `scope_scan` loop lives INSIDE an outer per-row
/// `for t in 0..N` loop. Each outer iteration starts with a fresh `let
/// scope_open = SENTINEL`, runs the inner walker, then writes to a different
/// row of the output. The inner walker's carrier mechanism must NOT bleed
/// state across outer iterations: every outer iter starts clean.
#[test]
fn nested_outer_loop_with_inner_scope_walker_per_row() {
    const N: u32 = 4;

    let mut outer_body = backward_scope_walk(Expr::var("t"));
    outer_body.push(Node::store("out", Expr::var("t"), Expr::var("scope_open")));

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("kinds", 0, BufferAccess::ReadOnly, DataType::U32).with_count(N),
            BufferDecl::output("out", 1, DataType::U32).with_count(N),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![Node::loop_for("t", Expr::u32(0), Expr::u32(N), outer_body)],
        )],
    );

    // kinds = [LBRACE, LBRACE, anything, anything]
    // For t=0: walker has 0 iters, scope_open stays SENTINEL, store SENTINEL
    // For t=1: walker iterates scope_scan=0 (rev=0): kind=LBRACE, depth=0, scope_open=0, store 0
    // For t=2: walker iterates scope_scan=0 (rev=1, kind=LBRACE, depth=0, scope_open=1), scan=1 skipped, store 1
    // For t=3: walker scope_scan=0..3, rev=2,1,0. rev=2 kind=anything, rev=1 LBRACE depth=0, scope_open=1
    let words = dispatch_words(&prog, [TOK_LBRACE, TOK_LBRACE, 0, 0]);
    assert_eq!(
        words,
        vec![SENTINEL, 0, 1, 1],
        "outer-per-row scope walker must produce one scope per row; got {words:?}",
    );
}

/// Closer to the actual c-parser scope walker: TWO sequential conditionals in
/// the loop body, both writing outer-scope vars. The first writes `scope_depth`
/// only; the second is the 3-level nest that writes `scope_open` OR
/// `scope_depth`. Both conditional gates read `scope_open`, so the merge
/// between the two conditionals is the chokepoint.
#[test]
fn two_sequential_conditionals_with_shared_carrier_propagate() {
    let mut loop_body = vec![Node::let_bind(
        "scope_kind",
        Expr::load("kinds", Expr::var("i")),
    )];
    loop_body.extend(scope_latch(Expr::var("i")));

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("kinds", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
            vec![
                Node::let_bind("scope_open", Expr::u32(SENTINEL)),
                Node::let_bind("scope_depth", Expr::u32(0)),
                Node::loop_for("i", Expr::u32(0), Expr::u32(4), loop_body),
                Node::store("out", Expr::u32(0), Expr::var("scope_open")),
                Node::store("out", Expr::u32(1), Expr::var("scope_depth")),
            ],
        )],
    );

    // Test fixture mirrors c-parser scope_open_before(idx=2) where tokens
    // are [RBRACE, LBRACE, LBRACE, ...]. Walking i=0..3:
    //   i=0 (RBRACE): cond1 fires, scope_depth=1
    //   i=1 (LBRACE): cond2 fires, kind==LBRACE, scope_depth(1)==0 FALSE, scope_depth=0
    //   i=2 (LBRACE): cond2 fires, kind==LBRACE, scope_depth(0)==0 TRUE, scope_open=2
    //   i=3 (?): scope_open != SENTINEL, both skip
    // Expected: scope_open=2, scope_depth=0
    let words = dispatch_words(&prog, [TOK_RBRACE, TOK_LBRACE, TOK_LBRACE, 0]);
    assert_eq!(
        words,
        vec![2, 0],
        "two sequential conditionals must propagate carrier values across iterations; got {words:?}",
    );
}
