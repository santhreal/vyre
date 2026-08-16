//! `semiring_gemm_wide`  -  the Lineage semiring GEMM over `W`-word cells.
//!
//! The dense [`super::semiring_gemm`] carries one `u32` per cell, which caps a
//! Lineage clause set at 32 rules. This form carries `w` contiguous `u32` words
//! per cell, so a clause set holds `32 * w` rules. Combine treats a cell as one
//! value: an all-zero cell on either side absorbs to all-zero, otherwise the
//! result is the word-wise bitwise OR. Accumulate is the word-wise bitwise OR.
//!
//! Only the Lineage semiring is defined for wide cells. The other six semirings
//! in [`super::Semiring`] are scalar, so a multi-word cell has no meaning for
//! them.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::OP_ID;

/// One lane per output cell, each lane walking the cell's `w` words.
pub const SEMIRING_GEMM_WIDE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Emits a generic `M x K . K x N -> M x N` matmul Program for `W`-wide lineage cells.
///
/// A cell has `w` contiguous `u32` words. Under wide lineage, the combine
/// operation is:
///   If ALL words of A are 0 OR ALL words of B are 0 -> result is all 0s.
///   Otherwise -> bitwise OR of A and B words.
/// Accumulate is bitwise OR.
///
/// `seed` names a buffer whose cell values initialize the accumulators, which is
/// how a Datalog fixpoint keeps its seed facts across a round.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn semiring_gemm_wide(
    a: &str,
    b: &str,
    c: &str,
    seed: Option<&str>,
    m: u32,
    n: u32,
    k: u32,
    w: u32,
) -> Program {
    let cells = m * n;
    let t = Expr::InvocationId { axis: 0 };

    let i_expr = Expr::div(t.clone(), Expr::u32(n));
    let j_expr = Expr::rem(t.clone(), Expr::u32(n));

    let mut body = vec![Node::let_bind("i", i_expr), Node::let_bind("j", j_expr)];

    // Initialize W accumulators. For Datalog fixpoint, we initialize
    // from the seed facts so the state grows monotonically.
    for word_idx in 0..w {
        if let Some(seed_name) = seed {
            let seed_idx = Expr::add(Expr::mul(t.clone(), Expr::u32(w)), Expr::u32(word_idx));
            body.push(Node::let_bind(
                format!("acc_{word_idx}"),
                Expr::load(seed_name, seed_idx),
            ));
        } else {
            body.push(Node::let_bind(format!("acc_{word_idx}"), Expr::u32(0)));
        }
    }

    // Inner loop kk from 0 to k
    let mut inner_loop_body = Vec::new();

    // Check if A cell is zero and B cell is zero (boolean logic)
    let mut a_is_zero = Expr::bool(true);
    let mut b_is_zero = Expr::bool(true);

    for word_idx in 0..w {
        let a_idx = Expr::add(
            Expr::mul(
                Expr::add(Expr::mul(Expr::var("i"), Expr::u32(k)), Expr::var("kk")),
                Expr::u32(w),
            ),
            Expr::u32(word_idx),
        );
        let b_idx = Expr::add(
            Expr::mul(
                Expr::add(Expr::mul(Expr::var("kk"), Expr::u32(n)), Expr::var("j")),
                Expr::u32(w),
            ),
            Expr::u32(word_idx),
        );

        inner_loop_body.push(Node::let_bind(
            format!("a_{word_idx}"),
            Expr::load(a, a_idx),
        ));
        inner_loop_body.push(Node::let_bind(
            format!("b_{word_idx}"),
            Expr::load(b, b_idx),
        ));

        a_is_zero = Expr::and(
            a_is_zero,
            Expr::eq(Expr::var(format!("a_{word_idx}")), Expr::u32(0)),
        );
        b_is_zero = Expr::and(
            b_is_zero,
            Expr::eq(Expr::var(format!("b_{word_idx}")), Expr::u32(0)),
        );
    }

    let either_zero = Expr::or(a_is_zero, b_is_zero);

    let mut combine_and_accumulate = Vec::new();
    for word_idx in 0..w {
        let combined = Expr::select(
            either_zero.clone(),
            Expr::u32(0),
            Expr::bitor(
                Expr::var(format!("a_{word_idx}")),
                Expr::var(format!("b_{word_idx}")),
            ),
        );
        combine_and_accumulate.push(Node::assign(
            format!("acc_{word_idx}"),
            Expr::bitor(Expr::var(format!("acc_{word_idx}")), combined),
        ));
    }

    inner_loop_body.extend(combine_and_accumulate);

    body.push(Node::loop_for(
        "kk",
        Expr::u32(0),
        Expr::u32(k),
        inner_loop_body,
    ));

    for word_idx in 0..w {
        let c_idx = Expr::add(Expr::mul(t.clone(), Expr::u32(w)), Expr::u32(word_idx));
        body.push(Node::store(c, c_idx, Expr::var(format!("acc_{word_idx}"))));
    }

    let if_block = vec![Node::if_then(Expr::lt(t.clone(), Expr::u32(cells)), body)];

    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadOnly, DataType::U32).with_count(m * k * w),
        BufferDecl::storage(b, 1, BufferAccess::ReadOnly, DataType::U32).with_count(k * n * w),
        BufferDecl::storage(c, 2, BufferAccess::ReadWrite, DataType::U32).with_count(cells * w),
    ];
    if let Some(seed_name) = seed {
        if seed_name != a && seed_name != b && seed_name != c {
            buffers.push(
                BufferDecl::storage(seed_name, 3, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(cells * w),
            );
        }
    }

    Program::wrapped(
        buffers,
        SEMIRING_GEMM_WIDE_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            &format!("anonymous::{OP_ID}::semiring_gemm_wide"),
            if_block,
        )],
    )
}
