//! Contracts for `vyre_driver::megakernel_execution`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::megakernel_execution::{
    plan_megakernel_execution, plan_megakernel_memory_budget, FrontierTopology,
    MegakernelByteLayout, MegakernelDeviceCapabilities, MegakernelExecutionSample,
    MegakernelGraphShape, MegakernelMemoryError,
};

#[test]
fn memory_planner_bounds_peak_bytes_by_topology() {
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let plan = plan_megakernel_memory_budget(
        FrontierTopology::FusedWave,
        graph,
        MegakernelByteLayout {
            bytes_per_node: 16,
            bytes_per_edge: 8,
            frontier_bytes: 4_096,
            scratch_bytes: 2_048,
            output_bytes: 512,
            budget_bytes: 128 * 1024,
        },
    )
    .expect("Fix: valid fused plan should fit the explicit device-memory budget");

    assert_eq!(plan.graph_bytes, 48_000);
    assert_eq!(plan.scratch_bytes, 8_192);
    assert_eq!(plan.required_bytes, 60_800);
    assert!(plan.memory_pressure_bps > 0);
}

#[test]
fn memory_planner_rejects_budget_and_overflow_failures() {
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let err = plan_megakernel_memory_budget(
        FrontierTopology::DenseFrontier,
        graph,
        MegakernelByteLayout {
            bytes_per_node: 16,
            bytes_per_edge: 8,
            frontier_bytes: 4_096,
            scratch_bytes: 2_048,
            output_bytes: 512,
            budget_bytes: 32 * 1024,
        },
    )
    .expect_err("over-budget dense plan must fail before allocation");
    assert!(matches!(
        err,
        MegakernelMemoryError::OverBudget {
            topology: FrontierTopology::DenseFrontier,
            ..
        }
    ));
    assert!(err.to_string().contains("Fix: choose a sparse topology"));

    let overflow = plan_megakernel_memory_budget(
        FrontierTopology::SparseFrontier,
        MegakernelGraphShape {
            node_count: u64::MAX,
            edge_count: 0,
        },
        MegakernelByteLayout {
            bytes_per_node: 2,
            budget_bytes: u64::MAX,
            ..MegakernelByteLayout::default()
        },
    )
    .expect_err("overflowing graph byte count must be rejected");
    assert!(matches!(
        overflow,
        MegakernelMemoryError::ByteCountOverflow {
            field: "node layout bytes"
        }
    ));
}

#[test]
fn generated_execution_plans_never_exceed_budget_or_hide_overflow() {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for case_index in 0..1024usize {
        let node_count = 1 + next_u64(&mut state) % 8_192;
        let edge_count = node_count + next_u64(&mut state) % 65_536;
        let bytes_per_node = 1 + next_u64(&mut state) % 64;
        let bytes_per_edge = 1 + next_u64(&mut state) % 32;
        let frontier_bytes = next_u64(&mut state) % 65_536;
        let scratch_bytes = next_u64(&mut state) % 16_384;
        let output_bytes = next_u64(&mut state) % 8_192;
        let budget_bytes = 64 * 1024 + next_u64(&mut state) % (4 * 1024 * 1024);
        let sample = MegakernelExecutionSample {
            dispatch_cost_ns: 100.0 + (next_u64(&mut state) % 10_000) as f64,
            frontier_density: (next_u64(&mut state) % 10_001) as f64 / 10_000.0,
            readback_bytes: next_u64(&mut state) % (1 << 20),
        };

        let result = plan_megakernel_execution(
            sample,
            MegakernelGraphShape {
                node_count,
                edge_count,
            },
            MegakernelByteLayout {
                bytes_per_node,
                bytes_per_edge,
                frontier_bytes,
                scratch_bytes,
                output_bytes,
                budget_bytes,
            },
            250.0,
            0.85,
            MegakernelDeviceCapabilities::FUSION_CAPABLE,
        );
        match result {
            Ok(plan) => {
                assert!(
                    plan.memory.required_bytes <= plan.memory.budget_bytes,
                    "case {case_index}"
                );
                assert!(plan.memory.memory_pressure_bps <= 10_000);
            }
            Err(MegakernelMemoryError::OverBudget {
                required_bytes,
                budget_bytes,
                ..
            }) => assert!(required_bytes > budget_bytes, "case {case_index}"),
            Err(MegakernelMemoryError::ByteCountOverflow { .. }) => {}
            Err(MegakernelMemoryError::InvalidSample { field }) => {
                panic!("case {case_index} generated an unrepresentable {field}")
            }
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
