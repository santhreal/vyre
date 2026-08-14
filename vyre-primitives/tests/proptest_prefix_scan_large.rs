//! Generated structure checks for the public large prefix-scan builder.

#![cfg(all(feature = "math", feature = "cpu-parity"))]

mod ir_shape;
use ir_shape::{contains_invocation_zero_gate, contains_loop, grid_sync_barrier_count};

use vyre_foundation::ir::{Node, Program};
use vyre_primitives::math::prefix_scan::{
    cpu_ref, prefix_scan_large, prefix_scan_large_with_op_id, ScanKind,
};
use vyre_primitives::reduce::multi_block_prefix_scan::BLOCK_LANES;

const CUSTOM_OP_ID: &str = "vyre-primitives::math::prefix_scan_large::generated_shape";

#[test]
fn generated_large_prefix_scan_uses_multi_block_chain() {
    const CASES: u32 = 10_000;
    const WINDOW: u32 = BLOCK_LANES * 4;

    for case in 0..CASES {
        let n = BLOCK_LANES + 1 + case.wrapping_mul(97) % WINDOW;
        let program = if case % 2 == 0 {
            prefix_scan_large("input", "output", n)
        } else {
            prefix_scan_large_with_op_id("input", "output", n, CUSTOM_OP_ID)
        };
        let expected_generator = if case % 2 == 0 {
            "vyre-primitives::math::prefix_scan_inclusive_sum"
        } else {
            CUSTOM_OP_ID
        };

        assert_top_region_generator(&program, expected_generator);
        assert_eq!(program.workgroup_size(), [BLOCK_LANES, 1, 1], "case {case}");
        assert!(
            !contains_loop(&program),
            "large prefix_scan case {case} n={n} must use the multi-block scan chain"
        );
        assert!(
            !contains_invocation_zero_gate(&program),
            "large prefix_scan case {case} n={n} must expose parallel work"
        );
        assert_eq!(
            grid_sync_barrier_count(&program),
            2,
            "large prefix_scan case {case} n={n} must contain the Pass-A/Pass-B/Pass-C split"
        );
        assert!(program.buffers().iter().any(|buffer| {
            buffer.name() == "output" && buffer.count() == n && buffer.is_output()
        }));
        assert_eq!(
            program
                .buffers()
                .iter()
                .filter(|buffer| buffer.is_output())
                .count(),
            1,
            "large prefix_scan case {case} n={n} must expose only the final output"
        );
    }
}

#[test]
fn generated_large_prefix_scan_cpu_reference_wraps() {
    const CASES: u32 = 10_000;

    for case in 0..CASES {
        let len = (case as usize % 257) + 1;
        let input: Vec<u32> = (0..len)
            .map(|idx| {
                (idx as u32)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(case)
                    .rotate_left((case % 31) + 1)
            })
            .collect();
        let mut acc = 0_u32;
        let expected: Vec<u32> = input
            .iter()
            .map(|value| {
                acc = acc.wrapping_add(*value);
                acc
            })
            .collect();

        assert_eq!(
            cpu_ref(&input, ScanKind::InclusiveSum),
            expected,
            "case {case}"
        );
    }
}

#[test]
fn small_large_prefix_scan_uses_workgroup_scan() {
    let program = prefix_scan_large_with_op_id("input", "output", 17, CUSTOM_OP_ID);

    assert_top_region_generator(&program, CUSTOM_OP_ID);
    assert_eq!(program.workgroup_size(), [32, 1, 1]);
    assert!(!contains_loop(&program));
    assert_eq!(grid_sync_barrier_count(&program), 0);
    assert!(program
        .buffers()
        .iter()
        .any(|buffer| buffer.name() == "output" && buffer.count() == 17 && buffer.is_output()));
}

#[test]
fn empty_large_prefix_scan_keeps_zero_byte_output_surface() {
    let program = prefix_scan_large("input", "output", 0);

    assert_top_region_generator(&program, "vyre-primitives::math::prefix_scan_inclusive_sum");
    assert_eq!(program.workgroup_size(), [1, 1, 1]);
    assert!(program
        .entry()
        .iter()
        .all(|node| { matches!(node, Node::Region { body, .. } if body.is_empty()) }));
    assert!(program.buffers().iter().any(|buffer| {
        buffer.name() == "output" && buffer.is_output() && buffer.output_byte_range() == Some(0..0)
    }));
}

fn assert_top_region_generator(program: &Program, expected: &str) {
    match program.entry() {
        [Node::Region { generator, .. }] => assert_eq!(generator.as_str(), expected),
        other => {
            panic!("Fix: prefix_scan_large should wrap the chain in one Region, got {other:?}")
        }
    }
}
