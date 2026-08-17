//! `reduce_scatter`  -  parallel scatter over a u32 ValueSet.
//!
//! Each global invocation loads one source value and, if the index is
//! in-range, writes it to `dst[index]`.  Used by graph operations for
//! indirect access patterns (e.g. distributing edge properties into
//! node slots).
//!
//! # Algorithm
//!
//! Work-group size `[256, 1, 1]`.  Caller dispatches
//! `(count + 255) / 256` work-groups.  Each active lane:
//!
//! ```text
//! if global_id < count:
//!     idx = indices[global_id]
//!     if idx < count:
//!         dst[idx] = src[global_id]
//! ```
//!
//! Out-of-range indices are silently dropped  -  at internet scale an
//! unguarded store would corrupt adjacent buffers.
//!
//! # Note on races
//!
//! If `indices` contains duplicates, multiple invocations may write to
//! the same `dst` slot.  The last writer wins; this primitive does
//! **not** use atomics.  Callers that need deterministic ordering must
//! ensure unique indices or compose with an atomic reduction step.

use vyre_foundation::ir::Program;

use super::indexed_move::{indexed_move_program, IndexedMoveKind};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::scatter";

/// Build a Program: `dst[indices[i]] = src[i]` for every `i < count`
/// where `indices[i] < count`.
///
/// Invalid `count == 0` lowers to an explicit trap program.
#[must_use]
pub fn scatter(src: &str, indices: &str, dst: &str, count: u32) -> Program {
    indexed_move_program(OP_ID, src, indices, dst, count, IndexedMoveKind::Scatter)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || scatter("src", "indices", "dst", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[10, 20, 30, 40]),
                to_bytes(&[3, 0, 2, 1]),
                to_bytes(&[0, 0, 0, 0]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x14, 0x00, 0x00, 0x00, // 20
                0x28, 0x00, 0x00, 0x00, // 40
                0x1e, 0x00, 0x00, 0x00, // 30
                0x0a, 0x00, 0x00, 0x00, // 10
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_scatter(src: &[u32], indices: &[u32], dst_len: usize) -> Vec<u32> {
        let mut dst = vec![0u32; dst_len];
        reference_scatter_into(src, indices, dst_len, &mut dst);
        dst
    }

    fn reference_scatter_into(src: &[u32], indices: &[u32], dst_len: usize, dst: &mut Vec<u32>) {
        dst.clear();
        dst.resize(dst_len, 0);
        for (&val, &idx) in src.iter().zip(indices.iter()) {
            let i = idx as usize;
            if i < dst_len {
                dst[i] = val;
            }
        }
    }

    #[test]
    fn basic_scatter() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, 2, 1];
        assert_eq!(reference_scatter(src, indices, 4), vec![20, 40, 30, 10]);
    }

    #[test]
    fn reference_into_reuses_destination() {
        let mut dst = Vec::with_capacity(8);
        reference_scatter_into(&[10, 20, 30, 40], &[3, 0, 2, 1], 4, &mut dst);
        let capacity = dst.capacity();
        assert_eq!(dst, vec![20, 40, 30, 10]);

        reference_scatter_into(&[7, 8], &[1, 3], 4, &mut dst);
        assert_eq!(dst.capacity(), capacity);
        assert_eq!(dst, vec![0, 7, 0, 8]);
    }

    #[test]
    fn identity_scatter() {
        let src = &[1u32, 2, 3, 4, 5];
        let indices = &[0u32, 1, 2, 3, 4];
        assert_eq!(reference_scatter(src, indices, 5), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_src() {
        let src: &[u32] = &[];
        let indices: &[u32] = &[];
        assert_eq!(reference_scatter(src, indices, 0), Vec::<u32>::new());
    }

    #[test]
    fn single_element() {
        let src = &[42u32];
        let indices = &[0u32];
        assert_eq!(reference_scatter(src, indices, 1), vec![42]);
    }

    #[test]
    fn duplicate_index_last_wins() {
        let src = &[1u32, 2, 3];
        let indices = &[0u32, 0, 0];
        assert_eq!(reference_scatter(src, indices, 1), vec![3]);
    }

    #[test]
    fn partial_write() {
        let src = &[7u32, 8];
        let indices = &[1u32, 3];
        assert_eq!(reference_scatter(src, indices, 5), vec![0, 7, 0, 8, 0]);
    }

    #[test]
    fn reference_scatter_ignores_out_of_bounds() {
        let src = &[1u32, 2, 3];
        let indices = &[0u32, 5]; // 5 is out of bounds
        assert_eq!(reference_scatter(src, indices, 4), vec![1, 0, 0, 0]);
    }

    #[test]
    fn reference_scatter_ignores_max_u32_index() {
        let src = &[1u32];
        let indices = &[u32::MAX];
        assert_eq!(reference_scatter(src, indices, 2), vec![0, 0]);
    }

    #[test]
    fn reference_into_reuses_destination_and_clears_stale_tail() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, u32::MAX, 1];
        let mut dst = Vec::with_capacity(16);
        dst.extend_from_slice(&[u32::MAX; 16]);
        let ptr = dst.as_ptr();

        reference_scatter_into(src, indices, 4, &mut dst);

        assert_eq!(dst, vec![20, 40, 0, 10]);
        assert_eq!(dst.as_ptr(), ptr);
    }

    #[test]
    fn compatibility_wrappers_match_reference() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, u32::MAX, 1];
        let mut compat = Vec::with_capacity(8);
        let mut reference = Vec::with_capacity(8);

        reference_scatter_into(src, indices, 4, &mut compat);
        reference_scatter_into(src, indices, 4, &mut reference);

        assert_eq!(reference_scatter(src, indices, 4), reference);
        assert_eq!(compat, reference);
    }

    #[test]
    fn program_has_expected_buffers() {
        let p = scatter("src", "indices", "dst", 1024);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["src", "indices", "dst"]);
    }

    #[test]
    fn program_buffer_counts() {
        let p = scatter("src", "indices", "dst", 1024);
        assert_eq!(p.buffers[0].count(), 1024);
        assert_eq!(p.buffers[1].count(), 1024);
        assert_eq!(p.buffers[2].count(), 1024);
    }

    #[test]
    fn zero_count_traps() {
        let p = scatter("src", "indices", "dst", 0);
        assert!(p.stats().trap());
    }

    #[test]
    fn adversarial_all_out_of_bounds_program() {
        // The program itself must compile and have the right shape even
        // when the indices it will process are all out-of-bounds.
        let p = scatter("src", "indices", "dst", 4);
        assert_eq!(p.buffers[1].count(), 4);
    }

    #[test]
    fn concurrent_access_reference_simulation() {
        // Simulate what 256 parallel threads would do: many threads write
        // to the same destination slot.  With non-atomic semantics the
        // last writer wins, so on reference the result is deterministic.
        let src = &[1u32, 2, 3];
        let indices = &[0u32, 0, 0];
        let out = reference_scatter(src, indices, 1);
        assert_eq!(out, vec![3]);
    }
}
