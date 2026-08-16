//! Generated truth and structure checks for the arbitrary-length prefix scan.

#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

mod ir_shape;
use ir_shape::{contains_invocation_id, contains_loop, grid_sync_barrier_count};

use proptest::prelude::*;
use vyre_libs::reduce::multi_block_prefix_scan::{cpu_ref, multi_block_prefix_scan_sum_u32};

const BLOCK_LANES: u32 = 1024;

fn independent_wrapping_prefix(values: &[u32]) -> Vec<u32> {
    let mut acc = 0_u32;
    values
        .iter()
        .map(|value| {
            acc = acc.wrapping_add(*value);
            acc
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4_096))]

    #[test]
    fn cpu_ref_matches_independent_wrapping_prefix_for_generated_inputs(
        values in proptest::collection::vec(any::<u32>(), 0..=2048),
    ) {
        prop_assert_eq!(cpu_ref(&values), independent_wrapping_prefix(&values));
    }

    #[test]
    fn large_builder_uses_parallel_multi_block_chain_for_generated_sizes(
        n in (BLOCK_LANES + 1)..=(BLOCK_LANES * 4),
    ) {
        let program = multi_block_prefix_scan_sum_u32("input", "output", n);
        let num_blocks = n.div_ceil(BLOCK_LANES);

        prop_assert_eq!(program.workgroup_size(), [BLOCK_LANES, 1, 1]);
        prop_assert!(
            !contains_loop(&program),
            "multi-block scan must not regress to a serial per-element loop for n={n}"
        );
        prop_assert!(
            !contains_invocation_id(&program),
            "large multi-block scan must use local/workgroup ids so fused overdispatch cannot address per-block scratch with a global lane for n={n}"
        );
        prop_assert_eq!(
            grid_sync_barrier_count(&program),
            2,
            "three-pass multi-block scan must split Pass A, Pass B, and Pass C with grid-level barriers for n={}",
            n
        );
        let has_partials = program.buffers().iter().any(|buffer| {
            buffer.name() == "__output_mbps_partials"
                && buffer.count() == num_blocks * BLOCK_LANES
                && !buffer.is_output()
                && buffer.is_pipeline_live_out()
        });
        let guarded_scratch_words = program
            .buffers()
            .iter()
            .filter(|buffer| buffer.name().contains("_guarded_scan_"))
            .map(|buffer| buffer.count())
            .collect::<Vec<_>>();
        let has_block_totals = program.buffers().iter().any(|buffer| {
            buffer.name() == "__output_mbps_block_totals"
                && buffer.count() == num_blocks
                && !buffer.is_output()
                && buffer.is_pipeline_live_out()
        });
        let has_output = program.buffers().iter().any(|buffer| {
            buffer.name() == "output" && buffer.count() == n && buffer.is_output()
        });
        let output_markers = program
            .buffers()
            .iter()
            .filter(|buffer| buffer.is_output())
            .count();

        prop_assert_eq!(output_markers, 1);
        prop_assert_eq!(
            guarded_scratch_words,
            vec![BLOCK_LANES, BLOCK_LANES],
            "guarded internal block-total scan must allocate full-block scratch for fused 1024-lane launches"
        );
        prop_assert!(has_partials);
        prop_assert!(has_block_totals);
        prop_assert!(has_output);
    }
}
