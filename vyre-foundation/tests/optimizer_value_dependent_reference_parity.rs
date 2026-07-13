//! Two-dimensional integer oracle differential: (expression shape × runtime
//! input value). `optimizer::pre_lowering::optimize` must preserve the
//! reference-interpreter result of every integer program for EVERY input value,
//! not merely for one.
//!
//! Why this exists beyond `optimizer_idempotence_proptest`: that harness's
//! reference-parity check feeds programs whose only runtime value is `gid_x`,
//! and it runs `reference_eval(.., &[Value::U32(0)])` on a single invocation 
//! so `gid_x` is pinned to `0`. A rewrite that is value-correct at `0` but wrong
//! for other inputs slips through. That is exactly how the `(x << 16) << 16`
//! shift-fusion miscompile evaded it: at `x == 0` the buggy `x << 31` and the
//! correct `0` agree, so the bug only shows for odd `x`
//! (see `const_fold_shift_fusion_amount_overflow`). It also never exercised
//! `Mod`/`rem` at all.
//!
//! This harness loads a single runtime `u32` from an input buffer, forces the
//! stored value to depend on it (so the input is live through dead-buffer
//! elimination and every probe value is observable), and checks
//! `reference_eval(base) == reference_eval(optimize(base))` across a curated set
//! of adversarial input values (odd/even, low/high halves, sign bit, all-ones,
//! alternating bits) plus a proptest-random value. Any value-dependent integer
//! miscompile in const-fold / strength-reduce / canonicalize / fusion surfaces
//! as a byte-level divergence, and proptest shrinks it to a minimal
//! `(expression, input value)` reproducer.

use proptest::prelude::*;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::pre_lowering as optimize;
use vyre_reference::value::Value;

/// The single runtime scalar every generated program reads. Keeping the index a
/// literal `0` over a count-1 buffer keeps the program oracle-valid.
fn load_in() -> Expr {
    Expr::load("in", Expr::u32(0))
}

/// Non-zero literal divisor, drives the `Div`/`Mod` strength reductions
/// (power-of-two → `Shr`/`BitAnd`, other → Granlund-Montgomery / `x-(x/d)*d`)
/// while never tripping the oracle's defined-but-divergent zero-divisor path.
fn nonzero_div_lit() -> impl Strategy<Value = Expr> {
    (1u32..=255).prop_map(Expr::u32)
}

/// Shift / rotate amount. `0..=31` lets nested shifts compose past the 32-bit
/// width (e.g. `(x << 16) << 16`), exercising the shift-fusion overflow lane.
fn shift_lit() -> impl Strategy<Value = Expr> {
    (0u32..=31).prop_map(Expr::u32)
}

/// Leaf: the runtime value (weighted heavily so trees actually depend on input)
/// or a small / boundary constant.
fn leaf() -> impl Strategy<Value = Expr> {
    prop_oneof![
        5 => Just(load_in()),
        2 => (0u32..=64).prop_map(Expr::u32),
        1 => prop_oneof![
            Just(1u32), Just(2), Just(3), Just(7), Just(8), Just(255), Just(256),
            Just(0x8000_0000u32), Just(0x7FFF_FFFFu32), Just(0xFFFF_FFFFu32),
        ]
        .prop_map(Expr::u32),
    ]
}

/// Bounded pure-`u32` expression over `load_in()` and constants, covering every
/// reference-supported integer `BinOp` (`MulHigh`, `Mod`, `AbsDiff`, `Min`,
/// `Max`, and the rotates included, none of which the existing differential
/// grammar exercises).
fn expr_strategy() -> impl Strategy<Value = Expr> {
    leaf().prop_recursive(5, 64, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::add(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::sub(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::mul(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::mulhi(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitand(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitor(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::bitxor(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::min(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::max(l, r)),
            (inner.clone(), inner.clone()).prop_map(|(l, r)| Expr::abs_diff(l, r)),
            (inner.clone(), nonzero_div_lit()).prop_map(|(l, r)| Expr::div(l, r)),
            (inner.clone(), nonzero_div_lit()).prop_map(|(l, r)| Expr::rem(l, r)),
            (inner.clone(), shift_lit()).prop_map(|(l, r)| Expr::shl(l, r)),
            (inner.clone(), shift_lit()).prop_map(|(l, r)| Expr::shr(l, r)),
            (inner.clone(), shift_lit()).prop_map(|(l, r)| Expr::rotate_left(l, r)),
            (inner, shift_lit()).prop_map(|(l, r)| Expr::rotate_right(l, r)),
        ]
    })
}

/// `out[0] = in[0] + <expr>`. The outer `+ in[0]` guarantees the input buffer
/// stays live (so reference inputs map 1:1 and every probe value is observable)
/// while the wrapped subtree carries the full operator surface. The outer add is
/// injective in its second argument for a fixed `in[0]`, so any divergence the
/// subtree produces still reaches the output.
fn program_for(expr: Expr) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::add(load_in(), expr))],
    )
}

/// Adversarial input values: odd/even, the two 16-bit halves, the sign bit,
/// `i32::MAX` pattern, all-ones, and the two alternating-bit patterns, the
/// classes that distinguish shift/rotate/mask/division rewrites from a value
/// that happens to agree at `0`.
const PROBES: &[u32] = &[
    0,
    1,
    2,
    3,
    7,
    8,
    15,
    16,
    255,
    256,
    0x0000_FFFF,
    0xFFFF_0000,
    0x5555_5555,
    0xAAAA_AAAA,
    0x8000_0000,
    0x7FFF_FFFF,
    0xFFFF_FFFF,
];

fn assert_parity_at(program: &Program, optimized: &Program, v: u32) -> Result<(), TestCaseError> {
    let inputs = [Value::U32(v)];
    let base = vyre_reference::reference_eval(program, &inputs)
        .expect("base integer program must run on the reference interpreter");
    let opt = vyre_reference::reference_eval(optimized, &inputs)
        .expect("optimized integer program must run on the reference interpreter");
    prop_assert_eq!(
        &base,
        &opt,
        "optimize::optimize changed the observable reference result at in[0]={:#010x}",
        v
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 384, ..ProptestConfig::default() })]

    /// For a random integer expression, the optimizer must preserve the
    /// reference result at every adversarial input value and one random value.
    #[test]
    fn optimize_preserves_reference_result_for_every_input_value(
        expr in expr_strategy(),
        rand_val in any::<u32>(),
    ) {
        let program = program_for(expr);
        let optimized = optimize::optimize(program.clone());
        for &v in PROBES {
            assert_parity_at(&program, &optimized, v)?;
        }
        assert_parity_at(&program, &optimized, rand_val)?;
    }
}

/// Deterministic anchor: a hand-built nest of the rewrite-prone shapes
/// (overflowing double shift, shift-then-mask, division and modulo by a
/// non-power-of-two constant, rotate) over an odd loaded value, where the
/// `gid_x == 0` harness is blind. Locks the invariant independently of the
/// proptest case budget.
#[test]
fn optimize_preserves_nested_shift_div_mod_rotate_for_odd_inputs() {
    // out[0] = in + ((((in << 16) << 16) ^ (in >> 3)) + (in / 6) + (in % 6) + rotl(in,7))
    let inb = load_in();
    let nest = Expr::add(
        Expr::add(
            Expr::bitxor(
                Expr::shl(Expr::shl(inb.clone(), Expr::u32(16)), Expr::u32(16)),
                Expr::shr(inb.clone(), Expr::u32(3)),
            ),
            Expr::add(
                Expr::div(inb.clone(), Expr::u32(6)),
                Expr::rem(inb.clone(), Expr::u32(6)),
            ),
        ),
        Expr::rotate_left(inb, Expr::u32(7)),
    );
    let program = program_for(nest);
    let optimized = optimize::optimize(program.clone());

    for &v in PROBES {
        let inputs = [Value::U32(v)];
        let base = vyre_reference::reference_eval(&program, &inputs)
            .expect("base program must run on the reference interpreter");
        let opt = vyre_reference::reference_eval(&optimized, &inputs)
            .expect("optimized program must run on the reference interpreter");
        assert_eq!(
            base, opt,
            "optimize::optimize diverged from the reference result at in[0]={v:#010x}"
        );
    }
}
