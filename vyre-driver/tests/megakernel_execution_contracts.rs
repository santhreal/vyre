//! Contracts for `vyre_driver::megakernel_execution`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::megakernel_execution::{
    plan_megakernel_execution, plan_megakernel_memory_budget, select_megakernel_topology,
    select_megakernel_topology_stable, MegakernelByteLayout, MegakernelDeviceCapabilities,
    MegakernelExecutionSample, MegakernelExecutionTopology, MegakernelGraphShape,
    MegakernelMemoryBudget, MegakernelMemoryError,
};

#[test]
fn topology_selector_uses_sparse_dense_hybrid_and_fused_bands() {
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = MegakernelMemoryBudget {
        required_bytes: 1_000,
        budget_bytes: 10_000,
    };
    let warp_sparse = select_megakernel_topology(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.01,
            readback_bytes: 256,
        },
        graph,
        memory,
        100.0,
        0.0,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(
        warp_sparse.topology,
        MegakernelExecutionTopology::WarpSparseFrontier
    );
    assert_eq!(
        warp_sparse.stable_explanation(),
        "megakernel-topology-v1|topology=WarpSparseFrontier|memory_pressure_bps=1000|average_degree_bps=40000|launch_pressure_bps=1000|reason=ultra_sparse_warp_specialized"
    );

    let block_dense = select_megakernel_topology(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.90,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(
        block_dense.topology,
        MegakernelExecutionTopology::BlockDenseFrontier
    );

    let hybrid = select_megakernel_topology(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.35,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(hybrid.topology, MegakernelExecutionTopology::HybridFrontier);

    let fused = select_megakernel_topology(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.90,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(fused.topology, MegakernelExecutionTopology::FusedWave);
    assert_eq!(fused.launch_pressure_bps, 2_500);

    let unfusable = select_megakernel_topology(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.90,
        MegakernelDeviceCapabilities::FUSION_INCAPABLE,
    );
    assert_eq!(
        unfusable.topology,
        MegakernelExecutionTopology::HybridFrontier,
        "Fix: a fused wave crosses wave boundaries inside one launch, so a device without a \
         device-wide barrier cannot run it however high the measured fusion pressure is."
    );
}

#[test]
fn stable_topology_selector_prevents_variant_flapping_near_thresholds() {
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = MegakernelMemoryBudget {
        required_bytes: 1_000,
        budget_bytes: 10_000,
    };
    let sparse_to_hybrid = select_megakernel_topology_stable(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.14,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        MegakernelExecutionTopology::SparseFrontier,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(
        sparse_to_hybrid.topology,
        MegakernelExecutionTopology::SparseFrontier
    );

    let held_fusion = select_megakernel_topology_stable(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.65,
        MegakernelExecutionTopology::FusedWave,
        MegakernelDeviceCapabilities::FUSION_CAPABLE,
    );
    assert_eq!(held_fusion.topology, MegakernelExecutionTopology::FusedWave);

    let released_fusion = select_megakernel_topology_stable(
        MegakernelExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.65,
        MegakernelExecutionTopology::FusedWave,
        MegakernelDeviceCapabilities::FUSION_INCAPABLE,
    );
    assert_ne!(
        released_fusion.topology,
        MegakernelExecutionTopology::FusedWave,
        "Fix: hysteresis must not hold a fused wave on a device that cannot run one."
    );
}

#[test]
fn memory_planner_bounds_peak_bytes_by_topology() {
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let plan = plan_megakernel_memory_budget(
        MegakernelExecutionTopology::FusedWave,
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
        MegakernelExecutionTopology::DenseFrontier,
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
            topology: MegakernelExecutionTopology::DenseFrontier,
            ..
        }
    ));
    assert!(err.to_string().contains("Fix: choose a sparse topology"));

    let overflow = plan_megakernel_memory_budget(
        MegakernelExecutionTopology::SparseFrontier,
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
        }
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
