//! Oracle-differential regression: loop fusion must not fuse two adjacent loops
//! when the second body's atomic compare-exchange `expected` operand reads a
//! scalar the first body mutates.
//!
//! `fusion_has_scalar_dependency` blocks fusion when a scalar written by one
//! loop body is read by the other (the interleaving would reorder the
//! dependency). It gathers each body's variable reads with
//! `collect_vars_in_expr`, whose `Expr::Atomic { index, value, .. }` arm walked
//! only `index` and `value` — it DROPPED the compare-exchange `expected`
//! operand. So a CAS whose `expected` reads the cross-loop scalar `s` was
//! invisible to the dependency check, and the loops fused.
//!
//! Concretely, with `s` mutated by loop A and `buf = [0, 10]`:
//!   * Unfused: loop A runs to completion (`s == 1`), then loop B's CAS reads
//!     `s == 1` at every iteration — `buf[0]==0 != 1` and `buf[1]==10 != 1`,
//!     both compares fail, `buf` stays `[0, 10]`.
//!   * Fused: each iteration sets `s = i` then runs the CAS, so iteration 0 sees
//!     `s == 0`, `buf[0]==0 == 0` SUCCEEDS and writes 77 — `buf` becomes
//!     `[77, 10]`.
//! The reference oracle returns `buf`, so the divergence is observable. The fix
//! walks `expected` in `collect_vars_in_expr`, so the scalar dependency is seen
//! and fusion is (correctly) refused.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::optimizer::passes::loops::loop_fusion::LoopFusion;
use vyre_reference::value::Value;

/// ```text
/// let s = 99;
/// loop i in 0..2 { s = i; }                       // body A mutates outer scalar s
/// loop j in 0..2 {                                // body B reads s in CAS expected
///     let x = compare_exchange(buf, j, expected=s, new=77);
///     out[j] = x;
/// }
/// ```
fn program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(2),
            BufferDecl::storage("buf", 1, BufferAccess::ReadWrite, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("s", Expr::u32(99)),
            Node::Loop {
                var: Ident::from("i"),
                from: Expr::u32(0),
                to: Expr::u32(2),
                body: vec![Node::assign("s", Expr::var("i"))],
            },
            Node::Loop {
                var: Ident::from("j"),
                from: Expr::u32(0),
                to: Expr::u32(2),
                body: vec![
                    Node::let_bind(
                        "x",
                        Expr::atomic_compare_exchange(
                            "buf",
                            Expr::var("j"),
                            Expr::var("s"),
                            Expr::u32(77),
                        ),
                    ),
                    Node::store("out", Expr::var("j"), Expr::var("x")),
                ],
            },
        ],
    )
}

#[test]
fn loop_fusion_declines_when_atomic_expected_reads_a_cross_loop_scalar() {
    let program = program();
    // buf = [0, 10] — chosen so the CAS outcome depends on WHEN `s` is read:
    // s==0 (per-iteration, fused) makes buf[0]'s compare succeed; s==1 (final,
    // unfused) makes every compare fail.
    let inputs = [Value::from(
        [0u32, 10].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )];

    let base = vyre_reference::reference_eval(&program, &inputs)
        .expect("base program is well-formed and must run on the reference oracle");

    let fused = LoopFusion::transform(program).program;
    let after = vyre_reference::reference_eval(&fused, &inputs)
        .expect("fused program must still run on the reference oracle");

    assert_eq!(
        base, after,
        "loop fusion changed observable semantics: it fused across an atomic \
         compare-exchange whose `expected` reads the cross-loop scalar `s`. \
         Unfused reads s after loop A (s==1, both compares fail, buf stays \
         [0,10]); fused reads s per-iteration (s==0 makes buf[0]'s compare \
         succeed, buf becomes [77,10])."
    );
}
