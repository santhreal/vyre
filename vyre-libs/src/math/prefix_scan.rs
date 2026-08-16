//! Subgroup prefix-sum (inclusive / exclusive scan)  -  core 1000×
//! primitive for variable-length compaction.
//!
//! # Use cases
//!
//! * **Hit-buffer compaction:** each lane produces 0 or 1 live
//!   flag; an exclusive scan over the flag vector gives the
//!   destination slot for each live hit. One dispatch provides the
//!   parallel compaction primitive used by PHASE9_EMIT.
//! * **Histogram prefix:** turn a bin-count vector into the CDF
//!   lookup used by the radix-sort primitive.
//! * **Segmented-reduce baseline:** classical parallel-scan is
//!   the inner kernel of a `(segment_offsets, values)` pair.
//!
//! # Algorithm
//!
//! Work-efficient Blelloch scan over `N` elements in one workgroup of at most
//! 256 lanes. A lane owns a contiguous run of
//! `ceil(N / lanes)` elements: it sums its run, the workgroup scans the run
//! sums with [`reduce::workgroup_tree`](crate::reduce::workgroup_tree), and the
//! lane replays its run from the resulting offset.
//!
//! ```text
//!   stage:   scratch[lane] = sum(in[lane*r .. lane*r+r])
//!   sweep:   scratch      = exclusive scan of the run sums
//!   replay:  out[lane*r+k] = scratch[lane] + sum(in[lane*r ..= lane*r+k])
//! ```
//!
//! Total work is `2N` element reads plus the `2*lanes-2` additions of the
//! sweep. The workgroup is never inflated past 256 lanes, so
//! `N = 1024` dispatches 256 lanes of four elements rather than 1024 lanes of
//! one, and `N = 513` dispatches 256 rather than the 1024 a
//! next-power-of-two lane count would ask for.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use vyre_primitives::ir_safe::clamped_load_to;
use crate::reduce::workgroup_tree::blelloch_inclusive_sum_nodes;

/// Canonical op id for inclusive sum-scan.
pub const OP_ID_INCLUSIVE_SUM: &str = "vyre-primitives::math::prefix_scan_inclusive_sum";
/// Canonical op id for exclusive sum-scan.
pub const OP_ID_EXCLUSIVE_SUM: &str = "vyre-primitives::math::prefix_scan_exclusive_sum";

/// Which scan variant to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanKind {
    /// `out[i] = sum(in[0..=i])`.
    InclusiveSum,
    /// `out[i] = sum(in[0..i])`  -  identity element (`0`) at slot 0.
    ExclusiveSum,
}

/// Lanes a single-workgroup scan dispatches at most.
///
/// A scan reaches [`MAX_SINGLE_BLOCK_SCAN`] elements by giving each lane a run
/// of elements, not by adding lanes. Two facts set this width, and it is the
/// smaller of them: 256 is the workgroup size the fleet schedules at full
/// occupancy, and past it a scan spends lanes on a sweep whose active fraction
/// halves every round; `PORTABLE_WORKGROUP_INVOCATIONS` is the extent every
/// registered target profile admits, and an op declares its geometry with no
/// device in hand. Writing the occupancy choice alone would leave a second
/// copy of the portable ceiling here to drift.
pub const SCAN_WORKGROUP_LANES: u32 = if SCAN_OCCUPANCY_LANES
    < vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS
{
    SCAN_OCCUPANCY_LANES
} else {
    vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS
};

/// Workgroup width the fleet schedules at full occupancy for a sweep scan.
const SCAN_OCCUPANCY_LANES: u32 = 256;
/// Largest element count one workgroup scans.
///
/// Above this the scan is a multi-block chain, which
/// `reduce::multi_block_prefix_scan` owns and `vyre-libs::math::scan_prefix_sum`
/// selects. This builder traps instead of silently scanning a prefix.
pub const MAX_SINGLE_BLOCK_SCAN: u32 = 1024;

/// Emit a single-workgroup prefix-sum Program.
///
/// `n` is the number of input slots, in `1..=`[`MAX_SINGLE_BLOCK_SCAN`]. The
/// emitted workgroup holds `min(n.next_power_of_two(), 256)`
/// lanes and each lane walks `ceil(n / lanes)` elements.
#[must_use]
pub fn prefix_scan(in_buf: &str, out_buf: &str, n: u32, kind: ScanKind) -> Program {
    let op_id = match kind {
        ScanKind::InclusiveSum => OP_ID_INCLUSIVE_SUM,
        ScanKind::ExclusiveSum => OP_ID_EXCLUSIVE_SUM,
    };
    prefix_scan_with_op_id(in_buf, out_buf, n, kind, op_id)
}

/// Emit a single-workgroup prefix-sum Program with an explicit region generator
/// id, so a composition can carry its own op id over the shared body.
#[must_use]
pub fn prefix_scan_with_op_id(
    in_buf: &str,
    out_buf: &str,
    n: u32,
    kind: ScanKind,
    op_id: &'static str,
) -> Program {
    if n == 0 || n > MAX_SINGLE_BLOCK_SCAN {
        return trap_program(
            op_id,
            Some((out_buf, DataType::U32)),
            format!(
                "Fix: prefix_scan scans one workgroup and requires n in 1..={MAX_SINGLE_BLOCK_SCAN}, got {n}. Build larger scans with vyre-libs::math::scan_prefix_sum, which selects the multi-block chain."
            ),
        );
    }

    let lanes = n.next_power_of_two().min(256);
    let run = n.div_ceil(lanes);
    let lane = Expr::InvocationId { axis: 0 };
    let scratch_a = format!("__{out_buf}_scan_a");
    let scratch_b = format!("__{out_buf}_scan_b");
    let run_base = Expr::mul(lane.clone(), Expr::u32(run));

    // Stage: one lane, one run sum. Every lane writes, so the sweep reads a
    // fully initialized buffer without a separate zero-fill pass.
    let mut staged = Expr::u32(0);
    for step in 0..run {
        staged = Expr::add(staged, run_element(in_buf, &run_base, step, n));
    }
    let mut body = vec![
        Node::store(&scratch_a, lane.clone(), staged),
        Node::barrier(),
    ];

    body.extend(blelloch_inclusive_sum_nodes(
        &scratch_a, &scratch_b, &lane, lanes,
    ));

    // Replay: the sweep leaves the INCLUSIVE run-sum prefix in `scratch_a` and
    // this lane's own run sum in `scratch_b`, so their difference is the
    // exclusive offset the run starts from.
    let offset = format!("__{out_buf}_scan_offset");
    body.push(Node::let_bind(
        offset.as_str(),
        Expr::load(&scratch_a, lane.clone()).wrapping_sub(Expr::load(&scratch_b, lane.clone())),
    ));
    let mut running = Expr::var(offset.as_str());
    for step in 0..run {
        let element = run_element(in_buf, &run_base, step, n);
        let inclusive = format!("__{out_buf}_scan_run_{step}");
        body.push(Node::let_bind(
            inclusive.as_str(),
            Expr::add(running, element.clone()),
        ));
        let index = Expr::add(run_base.clone(), Expr::u32(step));
        let value = match kind {
            ScanKind::InclusiveSum => Expr::var(inclusive.as_str()),
            ScanKind::ExclusiveSum => Expr::var(inclusive.as_str()).wrapping_sub(element),
        };
        body.push(Node::if_then(
            Expr::lt(index.clone(), Expr::u32(n)),
            vec![Node::store(out_buf, index, value)],
        ));
        running = Expr::var(inclusive.as_str());
    }

    let output_bytes = usize::try_from(n).unwrap_or(usize::MAX).saturating_mul(4);
    let buffers = vec![
        BufferDecl::storage(in_buf, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
        BufferDecl::output(out_buf, 1, DataType::U32)
            .with_count(n)
            .with_output_byte_range(0..output_bytes),
        BufferDecl::workgroup(&scratch_a, lanes, DataType::U32),
        BufferDecl::workgroup(&scratch_b, lanes, DataType::U32),
    ];

    Program::wrapped(
        buffers,
        [lanes, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}

/// Element `step` of the run based at `run_base`, or zero when the run overruns
/// `n`.
///
/// The load is clamped as well as selected: a lane whose run overruns the input
/// still issues the load on every backend that evaluates both arms of a select,
/// and an unclamped index would read past the buffer there.
fn run_element(in_buf: &str, run_base: &Expr, step: u32, n: u32) -> Expr {
    let index = Expr::add(run_base.clone(), Expr::u32(step));
    Expr::select(
        Expr::lt(index.clone(), Expr::u32(n)),
        clamped_load_to(in_buf, index, Expr::u32(n)),
        Expr::u32(0),
    )
}

/// CPU-reference prefix scan. Conformance tests verify the GPU
/// Program produces the same output for every input.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(input: &[u32], kind: ScanKind) -> Vec<u32> {
    let mut out = Vec::new();
    try_cpu_ref_into(input, kind, &mut out)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - prefix_scan cpu_ref failed: output allocation failed");
    out
}

/// Fallible CPU-reference prefix scan.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(input: &[u32], kind: ScanKind) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    try_cpu_ref_into(input, kind, &mut out)?;
    Ok(out)
}

/// CPU-reference prefix scan using a caller-owned output buffer.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_into(input: &[u32], kind: ScanKind, out: &mut Vec<u32>) {
    try_cpu_ref_into(input, kind, out)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - prefix_scan cpu_ref_into failed: output allocation failed");
}

/// Fallible CPU-reference prefix scan using a caller-owned output buffer.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into(input: &[u32], kind: ScanKind, out: &mut Vec<u32>) -> Result<(), String> {
    if input.len() > out.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            out,
            input.len() - out.len(),
            "prefix scan CPU oracle",
            "scan output",
        )?;
    }
    out.clear();
    let mut acc = 0_u32;
    match kind {
        ScanKind::InclusiveSum => {
            for &x in input {
                acc = acc.wrapping_add(x);
                out.push(acc);
            }
        }
        ScanKind::ExclusiveSum => {
            for &x in input {
                out.push(acc);
                acc = acc.wrapping_add(x);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inclusive_cpu_ref_matches_textbook() {
        assert_eq!(
            cpu_ref(&[1, 2, 3, 4], ScanKind::InclusiveSum),
            vec![1, 3, 6, 10],
        );
    }

    #[test]
    fn exclusive_cpu_ref_matches_textbook() {
        assert_eq!(
            cpu_ref(&[1, 2, 3, 4], ScanKind::ExclusiveSum),
            vec![0, 1, 3, 6],
        );
    }

    #[test]
    fn empty_cpu_ref_returns_empty() {
        assert_eq!(cpu_ref(&[], ScanKind::InclusiveSum), Vec::<u32>::new());
        assert_eq!(cpu_ref(&[], ScanKind::ExclusiveSum), Vec::<u32>::new());
    }

    #[test]
    fn wrap_on_overflow() {
        // Overflow check: wrapping_add semantics.
        assert_eq!(
            cpu_ref(&[u32::MAX, 1], ScanKind::InclusiveSum),
            vec![u32::MAX, 0],
        );
    }

    #[test]
    fn cpu_ref_into_reuses_output_buffer() {
        let mut out = Vec::with_capacity(16);
        let ptr = out.as_ptr();
        cpu_ref_into(&[1, 2, 3, 4], ScanKind::ExclusiveSum, &mut out);
        assert_eq!(out, vec![0, 1, 3, 6]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn cpu_ref_into_truncates_stale_tail_without_reallocating() {
        let mut out = Vec::with_capacity(16);
        out.extend([99u32; 16]);
        let ptr = out.as_ptr();

        try_cpu_ref_into(&[1, 2, 3, 4], ScanKind::InclusiveSum, &mut out).unwrap();

        assert_eq!(out, vec![1, 3, 6, 10]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn generated_cpu_ref_matches_independent_wrapping_scan() {
        for len in 0..128usize {
            let input: Vec<u32> = (0..len)
                .map(|idx| {
                    (idx as u32)
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(len as u32)
                })
                .collect();
            for kind in [ScanKind::InclusiveSum, ScanKind::ExclusiveSum] {
                let mut out = Vec::with_capacity(len + 3);
                try_cpu_ref_into(&input, kind, &mut out).unwrap();
                let mut expected = Vec::with_capacity(len);
                let mut acc = 0u32;
                for &value in &input {
                    match kind {
                        ScanKind::InclusiveSum => {
                            acc = acc.wrapping_add(value);
                            expected.push(acc);
                        }
                        ScanKind::ExclusiveSum => {
                            expected.push(acc);
                            acc = acc.wrapping_add(value);
                        }
                    }
                }
                assert_eq!(
                    out, expected,
                    "generated prefix scan len={len} kind={kind:?}"
                );
            }
        }
    }

    #[test]
    fn emitted_inclusive_program_has_expected_buffers() {
        let p = prefix_scan("in", "out", 32, ScanKind::InclusiveSum);
        assert_eq!(p.workgroup_size, [32, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["in", "out", "__out_scan_a", "__out_scan_b"]);
    }

    #[test]
    fn emitted_exclusive_program_has_expected_buffers() {
        let p = prefix_scan("in", "out", 64, ScanKind::ExclusiveSum);
        assert_eq!(p.workgroup_size, [64, 1, 1]);
    }

    #[test]
    fn non_power_of_two_n_pads_to_next_power_of_two() {
        let p = prefix_scan("in", "out", 5, ScanKind::InclusiveSum);
        assert_eq!(p.workgroup_size, [8, 1, 1]);
    }

    #[test]
    fn zero_n_traps() {
        let p = prefix_scan("in", "out", 0, ScanKind::InclusiveSum);
        assert!(p.stats().trap());
    }

    #[test]
    fn over_limit_n_traps() {
        let p = prefix_scan("in", "out", 2048, ScanKind::InclusiveSum);
        assert!(p.stats().trap());
    }

    #[test]
    fn binary_power_of_two_sizes_accepted() {
        for n in &[1_u32, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let program = prefix_scan("in", "out", *n, ScanKind::InclusiveSum);
            let names: Vec<&str> = program.buffers().iter().map(|b| b.name()).collect();
            assert!(
                names.contains(&"in"),
                "prefix_scan must declare in for n={n}"
            );
            assert!(
                names.contains(&"out"),
                "prefix_scan must declare out for n={n}"
            );
        }
    }
}
