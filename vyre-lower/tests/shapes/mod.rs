//! Mutation shapes shared by the `branch_collapse` descriptor tests and the
//! backend differential tests, each paired with an independent host oracle.
//!
//! Every program here has the same skeleton: a `p < N` thread guard, a
//! variable bound by `Node::let_bind` to a LITERAL, a `Node::assign` to that
//! variable somewhere inside a nested body, and then a guard that READS the
//! variable and decides what gets stored. The guard is the thing under test:
//! if `branch_collapse` folds it using the pre-mutation literal, the stored
//! value is wrong and the oracle catches it.
//!
//! The shapes differ only in which body-carrying construct holds the
//! assignment, because that is the axis the diagnosis had to cover:
//! `If` then-arm, `If` otherwise-arm, `Loop` body, `Region` body, and a
//! read-after-join.
//!
//! Oracles are written as plain Rust over the same input array. They are
//! deliberately NOT derived from the IR, so they cannot inherit an IR-level
//! mistake.

// Each integration test target compiles its own copy of this module and uses a
// different subset: the descriptor tests need the programs, the differential
// tests need the programs AND the oracles. Unused-in-this-target is therefore
// the normal case here, not a defect.
#![allow(dead_code)]

use std::sync::Arc;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// Thread count and buffer length for every shape.
pub const REPRO_N: u32 = 8;
/// Comparand `k` in the repro's `n == k` guards.
pub const REPRO_K: u32 = 0;
/// Sentinel standing in for the tokenizer's `Rank::ABSENT`.
pub const ABSENT: u32 = u32::MAX;
/// Loop trip count for the loop-bodied shapes. Kept below `REPRO_N` so a
/// `src[i]` read inside the loop is always in bounds.
pub const TRIP: u32 = 4;

fn tid() -> Expr {
    Expr::InvocationId { axis: 0 }
}

fn buffers(n: u32) -> Vec<BufferDecl> {
    vec![
        BufferDecl::read("src", 0, DataType::U32).with_count(n),
        BufferDecl::output("out", 1, DataType::U32).with_count(n),
    ]
}

/// Wrap `body` in the standard `if (p < n) { .. }` thread guard.
fn guarded(n: u32, body: Vec<Node>) -> Program {
    Program::wrapped(
        buffers(n),
        [n, 1, 1],
        vec![Node::if_then(Expr::lt(tid(), Expr::u32(n)), body)],
    )
}

/// Store `out[p] = <var>` after selecting through two complementary guards on
/// `probe`, so BOTH directions of the comparison are exercised. A pass that
/// wrongly folds either guard changes the stored value.
fn select_on(probe: &str, sentinel: u32, when_equal: u32, when_differs: Expr) -> Vec<Node> {
    vec![
        Node::let_bind("outv", Expr::u32(0)),
        Node::if_then(
            Expr::eq(Expr::var(probe), Expr::u32(sentinel)),
            vec![Node::assign("outv", Expr::u32(when_equal))],
        ),
        Node::if_then(
            Expr::ne(Expr::var(probe), Expr::u32(sentinel)),
            vec![Node::assign("outv", when_differs)],
        ),
        Node::store("out", tid(), Expr::var("outv")),
    ]
}

// ---------------------------------------------------------------------------
// 1. The reported repro shape.
// ---------------------------------------------------------------------------

/// ```text
/// if (p < n) {
///   let end = 0;
///   if (end == 0) {
///     let a = p;
///     let n = 0;
///     3x: if (n == k) { let l = src[a]; if (l != 0) { a = a + 1; n = k + 1 } }
///     if (n != 0) { end = a }
///   }
///   store out[p] = end
/// }
/// ```
pub fn repro_program(n: u32) -> Program {
    let mut inner = vec![
        Node::let_bind("a", tid()),
        Node::let_bind("n", Expr::u32(REPRO_K)),
    ];
    for _ in 0..3 {
        inner.push(Node::if_then(
            Expr::eq(Expr::var("n"), Expr::u32(REPRO_K)),
            vec![
                Node::let_bind("l", Expr::load("src", Expr::var("a"))),
                Node::if_then(
                    Expr::ne(Expr::var("l"), Expr::u32(0)),
                    vec![
                        Node::assign("a", Expr::add(Expr::var("a"), Expr::u32(1))),
                        Node::assign("n", Expr::add(Expr::u32(REPRO_K), Expr::u32(1))),
                    ],
                ),
            ],
        ));
    }
    inner.push(Node::if_then(
        Expr::ne(Expr::var("n"), Expr::u32(0)),
        vec![Node::assign("end", Expr::var("a"))],
    ));

    guarded(
        n,
        vec![
            Node::let_bind("end", Expr::u32(0)),
            Node::if_then(Expr::eq(Expr::var("end"), Expr::u32(0)), inner),
            Node::store("out", tid(), Expr::var("end")),
        ],
    )
}

/// `end` starts 0. The first `n == k` guard fires because `n` is 0 there. If
/// `src[p]` is nonzero the inner body advances `a` to `p + 1` and sets `n` to
/// 1, which closes the remaining two guards, and the trailing `n != 0` guard
/// publishes `a`. Otherwise nothing moves and `end` stays 0.
pub fn repro_oracle(src: &[u32]) -> Vec<u32> {
    src.iter()
        .enumerate()
        .map(|(p, &v)| if v != 0 { p as u32 + 1 } else { 0 })
        .collect()
}

// ---------------------------------------------------------------------------
// 2. Assignment inside a Loop body.
// ---------------------------------------------------------------------------

/// `acc` is let-bound to the literal 0 and incremented by a `Node::assign`
/// inside `if_then` inside `loop_for`. The guards afterwards read `acc`.
pub fn loop_assign_program(n: u32) -> Program {
    let mut body = vec![
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(TRIP),
            vec![Node::if_then(
                Expr::ne(Expr::load("src", Expr::var("i")), Expr::u32(0)),
                vec![Node::assign(
                    "acc",
                    Expr::add(Expr::var("acc"), Expr::u32(1)),
                )],
            )],
        ),
    ];
    body.extend(select_on("acc", 0, 111, Expr::var("acc")));
    guarded(n, body)
}

/// Count of nonzero entries in the first `TRIP` slots; 111 when that count is
/// zero. Uniform across threads because the loop does not read `p`.
pub fn loop_assign_oracle(src: &[u32]) -> Vec<u32> {
    let count = src.iter().take(TRIP as usize).filter(|&&v| v != 0).count() as u32;
    let value = if count == 0 { 111 } else { count };
    vec![value; src.len()]
}

// ---------------------------------------------------------------------------
// 3. Assignment inside an else (otherwise) arm only.
// ---------------------------------------------------------------------------

/// `v` is let-bound to the literal 5 and reassigned ONLY in the `otherwise`
/// arm. The then-arm binds an unrelated local so the arm is non-empty without
/// touching `v`.
pub fn else_assign_program(n: u32) -> Program {
    let mut body = vec![
        Node::let_bind("v", Expr::u32(5)),
        Node::if_then_else(
            Expr::ne(Expr::load("src", tid()), Expr::u32(0)),
            vec![Node::let_bind("untouched", Expr::u32(0))],
            vec![Node::assign("v", Expr::u32(9))],
        ),
    ];
    body.extend(select_on("v", 5, 50, Expr::var("v")));
    guarded(n, body)
}

/// Nonzero `src[p]` leaves `v` at 5 and selects 50; zero takes the else arm,
/// sets `v` to 9, and selects 9.
pub fn else_assign_oracle(src: &[u32]) -> Vec<u32> {
    src.iter().map(|&v| if v != 0 { 50 } else { 9 }).collect()
}

// ---------------------------------------------------------------------------
// 4. Assignment inside a nested Region.
// ---------------------------------------------------------------------------

/// `v` is let-bound to the literal 3 and reassigned by a `Node::assign` inside
/// an `if_then` inside a `Node::Region`. `Region` carries no execution
/// semantics but does carry a body, so the region-exit merge has to publish
/// the in-region value back to the parent.
pub fn region_assign_program(n: u32) -> Program {
    let mut body = vec![
        Node::let_bind("v", Expr::u32(3)),
        Node::Region {
            generator: "test.region.assign".into(),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::ne(Expr::load("src", tid()), Expr::u32(0)),
                vec![Node::assign("v", Expr::u32(11))],
            )]),
        },
    ];
    body.extend(select_on("v", 3, 30, Expr::var("v")));
    guarded(n, body)
}

/// Nonzero `src[p]` publishes 11 out of the region; zero leaves `v` at 3 and
/// selects 30.
pub fn region_assign_oracle(src: &[u32]) -> Vec<u32> {
    src.iter().map(|&v| if v != 0 { 11 } else { 30 }).collect()
}

// ---------------------------------------------------------------------------
// 5. Assigned in one branch, read after the join.
// ---------------------------------------------------------------------------

/// `flag` is let-bound to the literal 0 and reassigned in exactly ONE arm of
/// an if/else. Every read of it happens after the join, so it must observe the
/// merged value rather than the pre-branch literal.
pub fn join_program(n: u32) -> Program {
    let mut body = vec![
        Node::let_bind("flag", Expr::u32(0)),
        Node::if_then_else(
            Expr::ne(Expr::load("src", tid()), Expr::u32(0)),
            vec![Node::assign("flag", Expr::u32(1))],
            vec![Node::let_bind("untouched", Expr::u32(0))],
        ),
    ];
    body.extend(select_on("flag", 0, 77, Expr::u32(88)));
    guarded(n, body)
}

/// Nonzero `src[p]` sets `flag` to 1 and selects 88; zero leaves it 0 and
/// selects 77.
pub fn join_oracle(src: &[u32]) -> Vec<u32> {
    src.iter().map(|&v| if v != 0 { 88 } else { 77 }).collect()
}

// ---------------------------------------------------------------------------
// 6. The tokenizer's sentinel shape (self-referencing min).
// ---------------------------------------------------------------------------

/// The shape reported from `exatok/src/gpu_select.rs`: `rank_acc` let-bound to
/// a literal sentinel, mutated by a SELF-REFERENCING `min` inside `if_then`
/// inside `loop_for`, with the enclosing guard comparing against `rank_acc`
/// itself. `index_acc` tracks the winning slot the same way.
///
/// Stored value is `rank_acc * 10 + index_acc + p`, so a wrong minimum, a
/// wrong index, and a per-thread fault are all separately visible.
pub fn sentinel_min_program(n: u32) -> Program {
    let mut body = vec![
        Node::let_bind("rank_acc", Expr::u32(ABSENT)),
        Node::let_bind("index_acc", Expr::u32(ABSENT)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(TRIP),
            vec![
                Node::let_bind("r", Expr::load("src", Expr::var("i"))),
                Node::if_then(
                    Expr::lt(Expr::var("r"), Expr::var("rank_acc")),
                    vec![
                        Node::assign("rank_acc", Expr::min(Expr::var("rank_acc"), Expr::var("r"))),
                        Node::assign("index_acc", Expr::var("i")),
                    ],
                ),
            ],
        ),
    ];
    body.extend(select_on(
        "rank_acc",
        ABSENT,
        999,
        Expr::add(
            Expr::add(
                Expr::mul(Expr::var("rank_acc"), Expr::u32(10)),
                Expr::var("index_acc"),
            ),
            tid(),
        ),
    ));
    guarded(n, body)
}

/// Running minimum over the first `TRIP` slots with the index of the last
/// strict improvement; 999 when nothing beat the sentinel.
pub fn sentinel_min_oracle(src: &[u32]) -> Vec<u32> {
    let mut rank = ABSENT;
    let mut index = ABSENT;
    for (i, &r) in src.iter().take(TRIP as usize).enumerate() {
        if r < rank {
            rank = rank.min(r);
            index = i as u32;
        }
    }
    if rank == ABSENT {
        vec![999; src.len()]
    } else {
        (0..src.len())
            .map(|p| {
                rank.wrapping_mul(10)
                    .wrapping_add(index)
                    .wrapping_add(p as u32)
            })
            .collect()
    }
}
