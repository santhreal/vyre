//! Shared guarded indexed move kernels for gather/scatter.
//!
//! Both operations read `indices[i]`, guard it against the logical
//! element count, and move one u32 between `src` and `dst`. The mode
//! only decides which side is indexed indirectly.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Guarded indexed move direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndexedMoveKind {
    /// `dst[i] = src[indices[i]]`.
    Gather,
    /// `dst[indices[i]] = src[i]`.
    Scatter,
}

impl IndexedMoveKind {
    fn name(self) -> &'static str {
        match self {
            Self::Gather => "gather",
            Self::Scatter => "scatter",
        }
    }

    fn store_node(self, src: &str, dst: &str, lane: Expr) -> Node {
        match self {
            Self::Gather => Node::store(dst, lane, Expr::load(src, Expr::var("idx"))),
            Self::Scatter => Node::store(dst, Expr::var("idx"), Expr::load(src, lane)),
        }
    }
}

/// Build guarded gather/scatter over `count` u32 lanes.
#[must_use]
pub(crate) fn indexed_move_program(
    op_id: &'static str,
    src: &str,
    indices: &str,
    dst: &str,
    count: u32,
    kind: IndexedMoveKind,
) -> Program {
    if count == 0 {
        return trap_program(
            op_id,
            Some((dst, DataType::U32)),
            format!("Fix: {} requires count > 0, got {count}.", kind.name()),
        );
    }

    let t = Expr::InvocationId { axis: 0 };
    // Out-of-range index handling must MATCH the CPU reference explicitly, not rely
    // on an implicit "output buffer is zero-initialized" runtime contract (that is
    // the bitset_test_bit divergence class, a skipped lane silently reads dst's
    // prior contents when dst is reused).
    let guarded = match kind {
        // Gather emits exactly one output per lane; the CPU ref writes 0 for an
        // out-of-range index (`src.get(idx).unwrap_or(0)`), so the GPU must too.
        // Without the else branch, an out-of-range lane is left UNWRITTEN and its
        // value depends on dst's prior contents (a GPU/CPU parity divergence).
        IndexedMoveKind::Gather => Node::if_then_else(
            Expr::lt(Expr::var("idx"), Expr::u32(count)),
            vec![kind.store_node(src, dst, t.clone())],
            vec![Node::store(dst, t.clone(), Expr::u32(0))],
        ),
        // Scatter writes dst[idx] = src[lane]; an out-of-range idx has no valid
        // destination slot, so it is correctly SKIPPED, matching the CPU ref,
        // which `continue`s past `dst_index >= dst.len()` (no default write).
        IndexedMoveKind::Scatter => Node::if_then(
            Expr::lt(Expr::var("idx"), Expr::u32(count)),
            vec![kind.store_node(src, dst, t.clone())],
        ),
    };
    let body = vec![
        Node::let_bind("idx", Expr::load(indices, t.clone())),
        guarded,
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(src, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(indices, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::storage(dst, 2, BufferAccess::ReadWrite, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            op_id,
            vec![Node::if_then(Expr::lt(t.clone(), Expr::u32(count)), body)],
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_ref(kind: IndexedMoveKind, src: &[u32], indices: &[u32], dst_len: usize) -> Vec<u32> {
        match kind {
            IndexedMoveKind::Gather => indices
                .iter()
                .map(|&idx| src.get(idx as usize).copied().unwrap_or(0))
                .collect(),
            IndexedMoveKind::Scatter => {
                let mut dst = vec![0_u32; dst_len];
                for (src_index, &dst_index) in indices.iter().enumerate() {
                    if let Some(slot) = dst.get_mut(dst_index as usize) {
                        if let Some(&value) = src.get(src_index) {
                            *slot = value;
                        }
                    }
                }
                dst
            }
        }
    }

    fn reference_indexed_move_into(
        kind: IndexedMoveKind,
        src: &[u32],
        indices: &[u32],
        dst_len: usize,
        out: &mut Vec<u32>,
    ) {
        let res = scalar_ref(kind, src, indices, dst_len);
        out.clear();
        out.extend_from_slice(&res);
    }

    #[test]
    fn generated_indexed_moves_match_scalar_reference() {
        let mut state = 0x1D15_EA5E_u32;
        for case in 0..4096_u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let src_len = (state as usize % 97) + 1;
            let index_len = ((state >> 8) as usize % 101) + 1;
            let dst_len = ((state >> 16) as usize % 103) + 1;
            let mut src = Vec::with_capacity(src_len);
            for src_index in 0..src_len {
                state = state.rotate_left(7) ^ (src_index as u32).wrapping_mul(0x9E37_79B9);
                src.push(state);
            }
            let mut indices = Vec::with_capacity(index_len);
            for index in 0..index_len {
                state = state.rotate_left(11) ^ (index as u32).wrapping_mul(0x85EB_CA6B);
                let value = match index % 7 {
                    0 => 0,
                    1 => (src_len - 1) as u32,
                    2 => dst_len.saturating_sub(1) as u32,
                    3 => src_len as u32,
                    4 => dst_len as u32,
                    5 => u32::MAX,
                    _ => state % (src_len.max(dst_len) as u32 + 3),
                };
                indices.push(value);
            }

            for kind in [IndexedMoveKind::Gather, IndexedMoveKind::Scatter] {
                let mut got = Vec::new();
                reference_indexed_move_into(kind, &src, &indices, dst_len, &mut got);
                assert_eq!(
                    got,
                    scalar_ref(kind, &src, &indices, dst_len),
                    "case {case} kind {kind:?}"
                );
            }
        }
    }

    #[test]
    fn indexed_moves_clear_stale_tail_without_reallocating() {
        let src = [10_u32, 20, 30, 40];
        let indices = [3_u32, 0, 99, 1];
        for kind in [IndexedMoveKind::Gather, IndexedMoveKind::Scatter] {
            let mut out = Vec::with_capacity(16);
            out.extend_from_slice(&[u32::MAX; 16]);
            let ptr = out.as_ptr();

            reference_indexed_move_into(kind, &src, &indices, 4, &mut out);

            assert_eq!(out, scalar_ref(kind, &src, &indices, 4));
            assert_eq!(out.as_ptr(), ptr);
        }
    }

    #[test]
    fn compatibility_wrapper_matches_reference() {
        let src = [10_u32, 20, 30, 40];
        let indices = [3_u32, 0, 99, 1];

        for kind in [IndexedMoveKind::Gather, IndexedMoveKind::Scatter] {
            let mut compat = Vec::with_capacity(16);
            let mut reference = Vec::with_capacity(16);

            reference_indexed_move_into(kind, &src, &indices, 4, &mut compat);
            reference_indexed_move_into(kind, &src, &indices, 4, &mut reference);

            assert_eq!(compat, reference);
        }
    }
}
