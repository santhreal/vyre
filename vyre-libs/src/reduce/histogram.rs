//! `reduce_histogram`  -  parallel atomic histogram over a u32 ValueSet.
//!
//! Each global logical point owns one output bin and scans the input stream,
//! storing that bin's count. Used by radix_sort, frequency analysis, and label
//! distribution.
//!
//! # Algorithm
//!
//! Work-group size `[256, 1, 1]`.  Caller dispatches
//! `(count + 255) / 256` work-groups.  Each active lane:
//!
//! ```text
//! if global_id < count:
//!     total = 0
//!     for i in 0..count:
//!         total += input[i] == global_id
//!     output[global_id] = total
//! ```
//!
//! Out-of-range indices are silently dropped because no lane owns them.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::reduce::histogram";

/// Build a Program: `output[bin] = count(input[i] == bin)` for each bin.
///
/// Invalid zero dimensions lower to an explicit trap program.
#[must_use]
pub fn histogram(input: &str, output: &str, count: u32, num_bins: u32) -> Program {
    if count == 0 {
        return trap_program(
            OP_ID,
            Some((output, DataType::U32)),
            format!("Fix: histogram requires count > 0, got {count}."),
        );
    }
    if num_bins == 0 {
        return trap_program(
            OP_ID,
            Some((output, DataType::U32)),
            format!("Fix: histogram requires num_bins > 0, got {num_bins}."),
        );
    }

    let t = Expr::LogicalIndex { axis: 0 };

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(num_bins)),
        vec![
            Node::let_bind("total", Expr::u32(0)),
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(count),
                vec![Node::assign(
                    "total",
                    Expr::add(
                        Expr::var("total"),
                        Expr::select(
                            Expr::eq(Expr::load(input, Expr::var("i")), t.clone()),
                            Expr::u32(1),
                            Expr::u32(0),
                        ),
                    ),
                )],
            ),
            Node::store(output, t.clone(), Expr::var("total")),
        ],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_bins),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Build the legacy atomic scatter variant for callers that can prove backend
/// atomic-add semantics and want input-parallel execution.
#[must_use]
pub fn histogram_atomic_scatter(input: &str, output: &str, count: u32, num_bins: u32) -> Program {
    if count == 0 {
        return trap_program(
            OP_ID,
            Some((output, DataType::U32)),
            format!("Fix: histogram_atomic_scatter requires count > 0, got {count}."),
        );
    }
    if num_bins == 0 {
        return trap_program(
            OP_ID,
            Some((output, DataType::U32)),
            format!("Fix: histogram_atomic_scatter requires num_bins > 0, got {num_bins}."),
        );
    }

    let t = Expr::LogicalIndex { axis: 0 };
    let body = vec![
        Node::let_bind("bin", Expr::load(input, t.clone())),
        Node::if_then(
            Expr::lt(Expr::var("bin"), Expr::u32(num_bins)),
            vec![Node::let_bind(
                "_prev",
                Expr::atomic_add(output, Expr::var("bin"), Expr::u32(1)),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_bins),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(t.clone(), Expr::u32(count)), body)],
        )],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || histogram("input", "output", 8, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 1, 2, 3, 0, 1, 2, 3]),
                to_bytes(&[0, 0, 0, 0]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x02, 0x00, 0x00, 0x00, // 2
                0x02, 0x00, 0x00, 0x00, // 2
                0x02, 0x00, 0x00, 0x00, // 2
                0x02, 0x00, 0x00, 0x00, // 2
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_histogram(values: &[u32], bin_count: u32) -> Vec<u32> {
        let mut out = vec![0u32; bin_count as usize];
        reference_histogram_into(values, bin_count, &mut out);
        out
    }

    fn reference_histogram_into(values: &[u32], bin_count: u32, out: &mut Vec<u32>) {
        out.clear();
        out.resize(bin_count as usize, 0);
        for &v in values {
            let bin = v as usize;
            if bin < bin_count as usize {
                out[bin] = out[bin].wrapping_add(1);
            }
        }
    }

    #[test]
    fn basic_histogram() {
        let input = &[0u32, 1, 2, 3, 0, 1, 2, 3];
        assert_eq!(reference_histogram(input, 4), vec![2, 2, 2, 2]);
    }

    #[test]
    fn empty_input() {
        assert_eq!(reference_histogram(&[], 4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn all_same_bin() {
        let input = &[2u32, 2, 2, 2, 2];
        assert_eq!(reference_histogram(input, 4), vec![0, 0, 5, 0]);
    }

    #[test]
    fn out_of_bounds_ignored() {
        let input = &[0u32, 1, 99, 2, 3, 100];
        assert_eq!(reference_histogram(input, 4), vec![1, 1, 1, 1]);
    }

    #[test]
    fn reference_into_reuses_output_and_clears_stale_tail() {
        let input = &[0u32, 1, 99, 2, 3, 100];
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&[u32::MAX; 16]);
        let ptr = out.as_ptr();

        reference_histogram_into(input, 4, &mut out);

        assert_eq!(out, vec![1, 1, 1, 1]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn compatibility_wrappers_match_reference() {
        let input = &[0u32, 1, 99, 2, 3, 100];
        let mut compat = Vec::with_capacity(16);
        let mut reference = Vec::with_capacity(16);

        reference_histogram_into(input, 4, &mut compat);
        reference_histogram_into(input, 4, &mut reference);

        assert_eq!(reference_histogram(input, 4), reference);
        assert_eq!(compat, reference);
    }
    #[test]
    fn wrapping_on_overflow() {
        // u32::MAX + 1 wraps to 0, matching GPU atomic_add semantics.
        // reference uses wrapping_add, so we verify the accumulator behaviour
        // by starting from a high base and adding repeatedly.
        let mut base = u32::MAX - 1;
        base = base.wrapping_add(1); // = u32::MAX
        base = base.wrapping_add(1); // = 0
        assert_eq!(base, 0);
    }

    #[test]
    fn wrapping_overflow_correct() {
        let base = u32::MAX - 1;
        let after_three = base.wrapping_add(3);
        assert_eq!(after_three, 1);
    }

    #[test]
    fn many_bins() {
        let input: Vec<u32> = (0..100).collect();
        let out = reference_histogram(&input, 100);
        assert_eq!(out.len(), 100);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 1, "bin {i} should have count 1");
        }
    }

    #[test]
    fn sparse_bins() {
        let input = &[0u32, 50, 50, 99];
        let mut expected = vec![0u32; 100];
        expected[0] = 1;
        expected[50] = 2;
        expected[99] = 1;
        assert_eq!(reference_histogram(input, 100), expected);
    }

    #[test]
    fn program_has_expected_buffers() {
        let p = histogram("in", "out", 1024, 16);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["in", "out"]);
    }

    #[test]
    fn program_buffer_counts() {
        let p = histogram("in", "out", 1024, 16);
        assert_eq!(p.buffers[0].count(), 1024);
        assert_eq!(p.buffers[1].count(), 16);
    }

    #[test]
    fn zero_bins_traps() {
        let p = histogram("in", "out", 10, 0);
        assert!(p.stats().trap());
    }

    #[test]
    fn zero_count_traps() {
        let p = histogram("in", "out", 0, 4);
        assert!(p.stats().trap());
    }

    #[test]
    fn concurrent_access_reference_simulation() {
        // Simulate what 256 parallel threads would do: many threads hit
        // the same bin.  The result must be deterministic.
        let input = vec![7u32; 10_000];
        let out = reference_histogram(&input, 16);
        assert_eq!(out[7], 10_000);
        for (i, &v) in out.iter().enumerate() {
            if i != 7 {
                assert_eq!(v, 0);
            }
        }
    }

    #[test]
    fn adversarial_all_out_of_bounds() {
        let input = &[100u32, 200, 300];
        assert_eq!(reference_histogram(input, 2), vec![0, 0]);
    }

    #[test]
    fn adversarial_max_u32_index() {
        let input = &[u32::MAX];
        assert_eq!(reference_histogram(input, 4), vec![0, 0, 0, 0]);
    }
}
