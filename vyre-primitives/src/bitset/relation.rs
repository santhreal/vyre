//! Shared packed-bitset scalar relation reductions.
//!
//! Relation ops all scan `lhs` and `rhs` word-wise, reduce each per-word
//! predicate into `out_scalar[0]` with atomic AND, and differ only in the
//! predicate they apply per word. That is the grid-stride atomic-scalar shape
//! `reduce::atomic_scalar` owns, over two read-only inputs instead of one, so
//! the relation supplies the predicate and the shape stays in one place. It used
//! to be a second copy, which is how it came to be missing the first-workgroup
//! gate that keeps the reduction correct under any dispatch grid.

use vyre_foundation::ir::{Expr, Program};

use crate::reduce::atomic_scalar::atomic_grid_stride_u32;

/// Supported bitset-wide scalar relations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BitsetRelation {
    /// Every word must match exactly.
    Equal,
    /// Every `lhs` bit must also be present in `rhs`.
    SubsetOf,
}

impl BitsetRelation {
    fn predicate(self, lhs_word: Expr, rhs_word: Expr) -> Expr {
        match self {
            Self::Equal => Expr::eq(lhs_word, rhs_word),
            Self::SubsetOf => {
                Expr::eq(Expr::bitand(lhs_word, Expr::bitnot(rhs_word)), Expr::u32(0))
            }
        }
    }
}

/// Build `out_scalar[0] = forall w: relation(lhs[w], rhs[w])`.
#[must_use]
pub(crate) fn bitset_relation_program(
    op_id: &'static str,
    lhs: &str,
    rhs: &str,
    out_scalar: &str,
    words: u32,
    relation: BitsetRelation,
) -> Program {
    atomic_grid_stride_u32(
        &[lhs, rhs],
        out_scalar,
        words,
        1,
        |index| {
            Expr::select(
                relation.predicate(Expr::load(lhs, index.clone()), Expr::load(rhs, index)),
                Expr::u32(1),
                Expr::u32(0),
            )
        },
        |out, value| Expr::atomic_and(out, Expr::u32(0), value),
        op_id,
    )
}
