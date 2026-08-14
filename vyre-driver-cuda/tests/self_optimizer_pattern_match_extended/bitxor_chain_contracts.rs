use super::*;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

use crate::common::self_optimizer::optimized_store_value;

#[test]
fn cuda_bitxor_chain_cancels_right_via_cse() {
    // `let x = Load(input, 0); let y = Load(input, 0); store buf 0 ((x ^ y) ^ y)`
    //
    // Unlike `xy_load_program`, both operands read the SAME slot, so nothing but
    // CSE proving `x` and `y` alias can let the outer BitXor cancel the inner
    // pair. That aliasing is the contract, which is why this shape is local to
    // this case.
    let p = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("buf", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("x", Expr::load("input", Expr::u32(0))),
            Node::let_bind("y", Expr::load("input", Expr::u32(0))),
            Node::store(
                "buf",
                Expr::u32(0),
                binop(
                    BinOp::BitXor,
                    binop(BinOp::BitXor, Expr::var("x"), Expr::var("y")),
                    Expr::var("y"),
                ),
            ),
        ],
    );
    // After CSE proves x and y both alias Load(input,0) and the outer BitXor
    // folds, what remains is `Var(x)`, or potentially a const-prop'd single Load
    // reference. Both forms pass; a surviving BitXor does not.
    let value = optimized_store_value(p);
    assert!(
        !matches!(
            &value,
            Expr::BinOp {
                op: BinOp::BitXor,
                ..
            }
        ),
        "BitXor chain must collapse; got {value:?}"
    );
}
