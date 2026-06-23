//! Oracle-differential regression: const_fold's shift-fusion peephole
//! `(x << a) << b -> x << (a + b)` clamped the fused amount with `.min(31)`,
//! which is WRONG when `a + b >= 32`.
//!
//! vyre's shift semantics mask the amount mod 32 and shift bits off the top
//! (the reference oracle evaluates `left << (right & 31)` / `left >> (right &
//! 31)` on `u32`). So `(x << 16) << 16 == 0` for EVERY 32-bit `x` -- both
//! 16-bit shifts compose to a 32-bit shift that clears the word. But the buggy
//! fuse produced `x << min(16 + 16, 31) == x << 31 == (x & 1) << 31`, which is
//! `0x8000_0000` for odd `x`, not `0`. A single mod-32 shift cannot represent a
//! `>= 32` shift, so the fuse must decline (or fold to 0); it must never invent
//! `x << 31`.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::pre_lowering as optimize;
use vyre_reference::value::Value;

/// `out[0] = (in[0] << 16) << 16`. `in[0]` is loaded at runtime so const_fold
/// cannot collapse it through the typed literal evaluator -- the structural
/// shift-fusion peephole is what fires.
fn double_shift_left_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::storage("in", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::shl(
                Expr::shl(Expr::load("in", Expr::u32(0)), Expr::u32(16)),
                Expr::u32(16),
            ),
        )],
    )
}

#[test]
fn const_fold_preserves_double_shift_that_overflows_the_width() {
    let program = double_shift_left_program();
    // in[0] = 1 (odd, so bit 0 is set): (1 << 16) << 16 == 0, while the buggy
    // fuse to (1 << 31) == 0x8000_0000.
    let inputs = [Value::U32(1)];

    let base = vyre_reference::reference_eval(&program, &inputs)
        .expect("unoptimized program must run on the reference interpreter");

    let optimized = optimize::optimize(program.clone());
    let opt = vyre_reference::reference_eval(&optimized, &inputs)
        .expect("optimized program must run on the reference interpreter");

    assert_eq!(
        base, opt,
        "const_fold shift fusion changed observable semantics: (x << 16) << 16 \
         must stay 0 for every x, not become x << 31"
    );
}
