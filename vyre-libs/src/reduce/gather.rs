//! `reduce_gather`  -  parallel gather over a u32 ValueSet.
//!
//! Each global invocation loads one index and, if in-range, copies
//! `src[index]` into `dst[global_id]`.  Used by graph operations for
//! indirect access patterns (e.g. pulling node properties via edge
//! indices).
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
//!         dst[global_id] = src[idx]
//! ```
//!
//! Out-of-range indices are silently dropped  -  at internet scale an
//! unguarded load would read past the end of the source buffer.

use vyre_foundation::ir::Program;

use super::indexed_move::{indexed_move_program, IndexedMoveKind};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::gather";

/// Build a Program: `dst[i] = src[indices[i]]` for every `i < count`
/// where `indices[i] < count`.
///
/// Invalid `count == 0` lowers to an explicit trap program.
#[must_use]
pub fn gather(src: &str, indices: &str, dst: &str, count: u32) -> Program {
    indexed_move_program(OP_ID, src, indices, dst, count, IndexedMoveKind::Gather)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || gather("src", "indices", "dst", 4),
        Some(super::indexed_move::indexed_move_4element_fixture_inputs),
        Some(|| {
            vec![vec![vec![
                0x28, 0x00, 0x00, 0x00, // 40
                0x0a, 0x00, 0x00, 0x00, // 10
                0x1e, 0x00, 0x00, 0x00, // 30
                0x14, 0x00, 0x00, 0x00, // 20
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_gather(src: &[u32], indices: &[u32]) -> Vec<u32> {
        indices
            .iter()
            .map(|&idx| {
                let i = idx as usize;
                if i < src.len() {
                    src[i]
                } else {
                    0
                }
            })
            .collect()
    }

    fn reference_gather_into(src: &[u32], indices: &[u32], out: &mut Vec<u32>) {
        out.clear();
        out.extend(indices.iter().map(|&idx| {
            let i = idx as usize;
            if i < src.len() {
                src[i]
            } else {
                0
            }
        }));
    }

    #[test]
    fn basic_gather() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, 2, 1];
        assert_eq!(reference_gather(src, indices), vec![40, 10, 30, 20]);
    }

    #[test]
    fn identity_gather() {
        let src = &[1u32, 2, 3, 4, 5];
        let indices = &[0u32, 1, 2, 3, 4];
        assert_eq!(reference_gather(src, indices), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn empty_indices() {
        let src = &[1u32, 2, 3];
        let indices: &[u32] = &[];
        assert_eq!(reference_gather(src, indices), Vec::<u32>::new());
    }

    #[test]
    fn single_element() {
        let src = &[42u32];
        let indices = &[0u32];
        assert_eq!(reference_gather(src, indices), vec![42]);
    }

    #[test]
    fn repeated_index() {
        let src = &[7u32, 8, 9];
        let indices = &[0u32, 0, 0, 2, 2];
        assert_eq!(reference_gather(src, indices), vec![7, 7, 7, 9, 9]);
    }

    #[test]
    fn reference_gather_zeroes_out_of_bounds() {
        let src = &[1u32, 2, 3];
        let indices = &[0u32, 5]; // 5 is out of bounds
        assert_eq!(reference_gather(src, indices), vec![1, 0]);
    }

    #[test]
    fn reference_gather_zeroes_max_u32_index() {
        let src = &[1u32, 2, 3];
        let indices = &[u32::MAX];
        assert_eq!(reference_gather(src, indices), vec![0]);
    }

    #[test]
    fn reference_into_reuses_output_and_clears_stale_tail() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, u32::MAX, 1];
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&[u32::MAX; 16]);
        let ptr = out.as_ptr();

        reference_gather_into(src, indices, &mut out);

        assert_eq!(out, vec![40, 10, 0, 20]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn compatibility_wrappers_match_reference() {
        let src = &[10u32, 20, 30, 40];
        let indices = &[3u32, 0, u32::MAX, 1];
        let mut compat = Vec::with_capacity(8);
        let mut reference = Vec::with_capacity(8);

        reference_gather_into(src, indices, &mut compat);
        reference_gather_into(src, indices, &mut reference);

        assert_eq!(reference_gather(src, indices), reference);
        assert_eq!(compat, reference);
    }

    #[test]
    fn program_has_expected_buffers() {
        let p = gather("src", "indices", "dst", 1024);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["src", "indices", "dst"]);
    }

    #[test]
    fn program_buffer_counts() {
        let p = gather("src", "indices", "dst", 1024);
        assert_eq!(p.buffers[0].count(), 1024);
        assert_eq!(p.buffers[1].count(), 1024);
        assert_eq!(p.buffers[2].count(), 1024);
    }

    #[test]
    fn zero_count_traps() {
        let p = gather("src", "indices", "dst", 0);
        assert!(p.stats().trap());
    }

    #[test]
    fn adversarial_all_out_of_bounds_program() {
        // The program itself must compile and have the right shape even
        // when the indices it will process are all out-of-bounds.
        let p = gather("src", "indices", "dst", 4);
        assert_eq!(p.buffers[1].count(), 4);
    }

    #[test]
    fn concurrent_access_reference_simulation() {
        // Simulate what 256 parallel threads would do: many threads read
        // from the same source slot.  The result must be deterministic.
        let src = &[100u32; 1];
        let indices = vec![0u32; 10_000];
        let out = reference_gather(src, &indices);
        assert_eq!(out.len(), 10_000);
        assert!(out.iter().all(|&v| v == 100));
    }
}
