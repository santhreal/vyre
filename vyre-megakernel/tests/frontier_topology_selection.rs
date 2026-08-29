//! Integration tests and contracts for the frontier-density traversal topology dimension.

use vyre_megakernel::{
    select_frontier_topology, select_frontier_topology_stable, FrontierExecutionSample,
    FrontierGraphShape, FrontierMemoryBudget, FrontierTopology,
};

#[test]
fn frontier_topology_selector_uses_sparse_dense_hybrid_and_fused_bands() {
    let graph = FrontierGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = FrontierMemoryBudget {
        required_bytes: 1_000,
        budget_bytes: 10_000,
    };

    let warp_sparse = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.01,
            readback_bytes: 256,
        },
        graph,
        memory,
        100.0,
        0.0,
        true,
    );
    assert_eq!(warp_sparse.topology, FrontierTopology::WarpSparseFrontier);
    assert_eq!(
        warp_sparse.stable_explanation(),
        "megakernel-topology-v1|topology=WarpSparseFrontier|memory_pressure_bps=1000|average_degree_bps=40000|launch_pressure_bps=1000|reason=ultra_sparse_warp_specialized"
    );

    let block_dense = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.90,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        true,
    );
    assert_eq!(block_dense.topology, FrontierTopology::BlockDenseFrontier);

    let dense = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.75,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        true,
    );
    assert_eq!(dense.topology, FrontierTopology::DenseFrontier);

    let sparse = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.10,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        true,
    );
    assert_eq!(sparse.topology, FrontierTopology::SparseFrontier);

    let hybrid = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.35,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        true,
    );
    assert_eq!(hybrid.topology, FrontierTopology::HybridFrontier);

    let fused = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.90,
        true,
    );
    assert_eq!(fused.topology, FrontierTopology::FusedWave);
    assert_eq!(fused.launch_pressure_bps, 2_500);

    let unfusable = select_frontier_topology(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.90,
        false,
    );
    assert_eq!(
        unfusable.topology,
        FrontierTopology::HybridFrontier,
        "Fix: without device-wide barrier capability, fused wave must fall back to hybrid."
    );
}

#[test]
fn stable_frontier_topology_selector_prevents_variant_flapping_near_thresholds() {
    let graph = FrontierGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = FrontierMemoryBudget {
        required_bytes: 1_000,
        budget_bytes: 10_000,
    };

    let sparse_to_hybrid = select_frontier_topology_stable(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.14,
            readback_bytes: 512,
        },
        graph,
        memory,
        100.0,
        0.0,
        FrontierTopology::SparseFrontier,
        true,
    );
    assert_eq!(
        sparse_to_hybrid.topology,
        FrontierTopology::SparseFrontier,
        "Hysteresis must hold sparse frontier in transition band."
    );

    let held_fusion = select_frontier_topology_stable(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.65,
        FrontierTopology::FusedWave,
        true,
    );
    assert_eq!(
        held_fusion.topology,
        FrontierTopology::FusedWave,
        "Hysteresis must hold fused wave near boundary."
    );

    let released_fusion = select_frontier_topology_stable(
        FrontierExecutionSample {
            dispatch_cost_ns: 1_000.0,
            frontier_density: 0.50,
            readback_bytes: 1 << 20,
        },
        graph,
        memory,
        250.0,
        0.65,
        FrontierTopology::FusedWave,
        false,
    );
    assert_ne!(
        released_fusion.topology,
        FrontierTopology::FusedWave,
        "Fix: hysteresis must not hold a fused wave on a device that lacks barriers."
    );
}

