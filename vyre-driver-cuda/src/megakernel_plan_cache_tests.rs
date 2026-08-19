use super::megakernel_plan_cache_records::{
    CudaMegakernelAnalysisKind, CudaMegakernelDeviceKey, CudaMegakernelPlanCacheKey,
};
use super::CudaMegakernelPlanCache;
use crate::megakernel_scheduler::CudaMegakernelScheduleSample;
use crate::synthetic_device_caps::synthetic_sm120_envelope_default;
use vyre_driver::megakernel_execution::{
    MegakernelByteLayout, MegakernelExecutionTopology, MegakernelGraphShape,
    MegakernelTopologyDecision,
};

/// The byte layout every execution-plan case in this module shares, with the
/// two counts the cases actually vary left to the caller.
fn byte_layout(scratch_bytes: u64, budget_bytes: u64) -> MegakernelByteLayout {
    MegakernelByteLayout {
        bytes_per_node: 16,
        bytes_per_edge: 8,
        frontier_bytes: 4_096,
        scratch_bytes,
        output_bytes: 512,
        budget_bytes,
    }
}

fn device() -> CudaMegakernelDeviceKey {
    CudaMegakernelDeviceKey {
        sm_major: 12,
        sm_minor: 0,
        warp_size: 32,
        supports_grid_sync: true,
        supports_tensor_cores: true,
        max_workgroup_size: 1024,
    }
}

fn key(
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    frontier_density: f64,
    memory_pressure_bps: u32,
) -> CudaMegakernelPlanCacheKey {
    CudaMegakernelPlanCacheKey::new(
        graph_layout_hash,
        analysis_kind,
        device(),
        frontier_density,
        memory_pressure_bps,
        0,
        0,
        0.0,
    )
}

fn decision(topology: MegakernelExecutionTopology) -> MegakernelTopologyDecision {
    MegakernelTopologyDecision {
        topology,
        memory_pressure_bps: 1_000,
        average_degree_bps: 20_000,
        launch_pressure_bps: 2_000,
    }
}

#[test]
fn cache_reuses_plan_for_same_graph_analysis_device_and_pressure_bucket() {
    let mut cache = CudaMegakernelPlanCache::new();
    let key = key(42, CudaMegakernelAnalysisKind::Ifds, 0.52, 2_400);
    let first = cache
        .get_or_insert_with(key, || decision(MegakernelExecutionTopology::FusedWave))
        .expect("Fix: CUDA megakernel plan-cache insert should fit telemetry counters.");
    let second = cache
        .get_or_insert_with(key, || {
            decision(MegakernelExecutionTopology::SparseFrontier)
        })
        .expect("Fix: CUDA megakernel plan-cache hit should fit telemetry counters.");

    assert_eq!(first, second);
    assert_eq!(second.topology, MegakernelExecutionTopology::FusedWave);
    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.entries, 1);
}

#[test]
fn device_key_is_derived_from_cuda_caps() {
    assert_eq!(
        CudaMegakernelDeviceKey::from(&synthetic_sm120_envelope_default()),
        device()
    );
}

#[test]
fn cache_separates_analysis_family_density_and_device_features() {
    let ifds = key(42, CudaMegakernelAnalysisKind::Ifds, 0.01, 1_000);
    let liveness = key(42, CudaMegakernelAnalysisKind::Liveness, 0.01, 1_000);
    let dense = key(42, CudaMegakernelAnalysisKind::Ifds, 0.95, 1_000);
    let mut other_device = device();
    other_device.sm_minor = 1;
    let device_changed = CudaMegakernelPlanCacheKey::new(
        42,
        CudaMegakernelAnalysisKind::Ifds,
        other_device,
        0.01,
        1_000,
        0,
        0,
        0.0,
    );

    assert_ne!(ifds, liveness);
    assert_ne!(ifds, dense);
    assert_ne!(ifds, device_changed);
}

#[test]
fn bounded_cache_evicts_lru_entry() {
    let mut cache = CudaMegakernelPlanCache::with_max_entries(2);
    let first = key(1, CudaMegakernelAnalysisKind::Dataflow, 0.1, 1_000);
    let second = key(2, CudaMegakernelAnalysisKind::Dataflow, 0.1, 1_000);
    let third = key(3, CudaMegakernelAnalysisKind::Dataflow, 0.1, 1_000);

    cache
        .get_or_insert_with(first, || {
            decision(MegakernelExecutionTopology::SparseFrontier)
        })
        .expect("Fix: CUDA megakernel plan-cache insert should fit telemetry counters.");
    cache
        .get_or_insert_with(second, || {
            decision(MegakernelExecutionTopology::HybridFrontier)
        })
        .expect("Fix: CUDA megakernel plan-cache insert should fit telemetry counters.");
    cache
        .get_or_insert_with(first, || {
            decision(MegakernelExecutionTopology::DenseFrontier)
        })
        .expect("Fix: CUDA megakernel plan-cache hit should fit telemetry counters.");
    cache
        .get_or_insert_with(third, || decision(MegakernelExecutionTopology::FusedWave))
        .expect("Fix: CUDA megakernel plan-cache eviction should fit telemetry counters.");

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 3);
    assert_eq!(stats.evictions, 1);
    assert_eq!(stats.entries, 2);
    let reloaded_second = cache
        .get_or_insert_with(second, || {
            decision(MegakernelExecutionTopology::DenseFrontier)
        })
        .expect("Fix: CUDA megakernel plan-cache reload should fit telemetry counters.");
    assert_eq!(
        reloaded_second.topology,
        MegakernelExecutionTopology::DenseFrontier
    );
}

#[test]
fn cache_selects_topology_and_reuses_pressure_bucket_plan() {
    let mut cache = CudaMegakernelPlanCache::new();
    let sample = crate::megakernel_scheduler::CudaMegakernelScheduleSample {
        dispatch_cost_ns: 1_000.0,
        frontier_density: 0.90,
        readback_bytes: 1 << 20,
    };
    let graph = vyre_driver::megakernel_execution::MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = vyre_driver::megakernel_execution::MegakernelMemoryBudget {
        required_bytes: 1_024,
        budget_bytes: 16_384,
    };
    let first = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            sample,
            graph,
            memory,
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology selection should fit telemetry counters.");
    let second = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            crate::megakernel_scheduler::CudaMegakernelScheduleSample {
                frontier_density: 0.91,
                ..sample
            },
            graph,
            vyre_driver::megakernel_execution::MegakernelMemoryBudget {
                required_bytes: 1_100,
                budget_bytes: 16_384,
            },
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology cache hit should fit telemetry counters.");

    assert_eq!(first, second);
    assert_eq!(first.topology, MegakernelExecutionTopology::FusedWave);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn cache_stabilizes_topology_across_adjacent_pressure_buckets() {
    let mut cache = CudaMegakernelPlanCache::new();
    let graph = vyre_driver::megakernel_execution::MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = vyre_driver::megakernel_execution::MegakernelMemoryBudget {
        required_bytes: 1_024,
        budget_bytes: 16_384,
    };
    let dense = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            crate::megakernel_scheduler::CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.70,
                readback_bytes: 512,
            },
            graph,
            memory,
            100.0,
            0.0,
        )
        .expect("Fix: CUDA megakernel topology selection should fit telemetry counters.");
    let near_dense = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            crate::megakernel_scheduler::CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.68,
                readback_bytes: 512,
            },
            graph,
            memory,
            100.0,
            0.0,
        )
        .expect("Fix: CUDA megakernel topology stabilization should fit telemetry counters.");

    assert_eq!(dense.topology, MegakernelExecutionTopology::DenseFrontier);
    assert_eq!(
        near_dense.topology,
        MegakernelExecutionTopology::DenseFrontier
    );
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn cache_reselects_when_memory_pressure_bucket_changes() {
    let mut cache = CudaMegakernelPlanCache::new();
    let sample = crate::megakernel_scheduler::CudaMegakernelScheduleSample {
        dispatch_cost_ns: 1_000.0,
        frontier_density: 0.90,
        readback_bytes: 1 << 20,
    };
    let graph = vyre_driver::megakernel_execution::MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let low_pressure = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            sample,
            graph,
            vyre_driver::megakernel_execution::MegakernelMemoryBudget {
                required_bytes: 1_024,
                budget_bytes: 16_384,
            },
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology selection should fit telemetry counters.");
    let red_zone = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            sample,
            graph,
            vyre_driver::megakernel_execution::MegakernelMemoryBudget {
                required_bytes: 15_500,
                budget_bytes: 16_384,
            },
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology reselection should fit telemetry counters.");

    assert_eq!(
        low_pressure.topology,
        MegakernelExecutionTopology::FusedWave
    );
    assert_eq!(
        red_zone.topology,
        MegakernelExecutionTopology::SparseFrontier
    );
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn cache_pressure_bucket_uses_exact_u128_math() {
    let low = CudaMegakernelPlanCacheKey::new(
        1,
        CudaMegakernelAnalysisKind::Dataflow,
        device(),
        0.5,
        super::megakernel_plan_cache_records::pressure_bps(1_u64 << 62, 1_u64 << 63),
        0,
        0,
        0.0,
    );
    let high = CudaMegakernelPlanCacheKey::new(
        1,
        CudaMegakernelAnalysisKind::Dataflow,
        device(),
        0.5,
        super::megakernel_plan_cache_records::pressure_bps(1_u64 << 63, 1_u64 << 63),
        0,
        0,
        0.0,
    );

    assert_eq!(low.memory_pressure_bucket, 5);
    assert_eq!(high.memory_pressure_bucket, 10);
}

#[test]
fn cache_reselects_when_readback_launch_or_fusion_pressure_changes() {
    let mut cache = CudaMegakernelPlanCache::new();
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let memory = vyre_driver::megakernel_execution::MegakernelMemoryBudget {
        required_bytes: 1_024,
        budget_bytes: 16_384,
    };
    let low_pressure = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 0,
            },
            graph,
            memory,
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology selection should fit telemetry counters.");
    let high_pressure = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 1 << 20,
            },
            graph,
            memory,
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology pressure split should fit telemetry counters.");

    assert_ne!(
        low_pressure.topology,
        MegakernelExecutionTopology::FusedWave
    );
    assert_eq!(
        high_pressure.topology,
        MegakernelExecutionTopology::FusedWave
    );
    assert_eq!(cache.stats().hits, 0);
    assert_eq!(cache.stats().misses, 2);
}

#[test]
fn cache_never_selects_fused_wave_without_grid_sync_support() {
    let mut cache = CudaMegakernelPlanCache::new();
    let mut no_grid_sync = device();
    no_grid_sync.supports_grid_sync = false;

    let plan = cache
        .get_or_select_topology(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            no_grid_sync,
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 1 << 20,
            },
            MegakernelGraphShape {
                node_count: 1_000,
                edge_count: 4_000,
            },
            vyre_driver::megakernel_execution::MegakernelMemoryBudget {
                required_bytes: 1_024,
                budget_bytes: 16_384,
            },
            250.0,
            0.95,
        )
        .expect("Fix: CUDA megakernel topology selection should fit telemetry counters.");

    assert_ne!(
        plan.topology,
        MegakernelExecutionTopology::FusedWave,
        "Fix: CUDA megakernel planner must not select cooperative fused-wave topology when the device key says grid sync is unavailable."
    );
}

#[test]
fn cached_execution_plan_reuses_topology_bucket_and_validates_memory() {
    let mut cache = CudaMegakernelPlanCache::new();
    let sample = CudaMegakernelScheduleSample {
        dispatch_cost_ns: 1_000.0,
        frontier_density: 0.90,
        readback_bytes: 1 << 20,
    };
    let graph = MegakernelGraphShape {
        node_count: 1_000,
        edge_count: 4_000,
    };
    let first = cache
        .get_or_plan_execution(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            sample,
            graph,
            byte_layout(2_048, 128 * 1024),
            250.0,
            0.95,
        )
        .expect("Fix: cache-backed fused CUDA execution plan should fit the explicit budget.");
    let second = cache
        .get_or_plan_execution(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            CudaMegakernelScheduleSample {
                frontier_density: 0.91,
                ..sample
            },
            graph,
            byte_layout(2_048, 128 * 1024),
            250.0,
            0.95,
        )
        .expect("Fix: equivalent CUDA execution pressure bucket should reuse the cached topology and still validate memory.");

    assert_eq!(first.topology, MegakernelExecutionTopology::FusedWave);
    assert_eq!(second.topology, MegakernelExecutionTopology::FusedWave);
    assert_eq!(second.memory.scratch_bytes, 8_192);
    assert!(!second.downgraded_to_sparse);
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn cached_execution_plan_downgrades_non_sparse_topology_when_exact_budget_fails() {
    let mut cache = CudaMegakernelPlanCache::new();
    let plan = cache
        .get_or_plan_execution(
            99,
            CudaMegakernelAnalysisKind::Dataflow,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density: 0.50,
                readback_bytes: 1 << 20,
            },
            MegakernelGraphShape {
                node_count: 1_000,
                edge_count: 4_000,
            },
            byte_layout(10_000, 80_000),
            250.0,
            0.90,
        )
        .expect(
            "Fix: sparse CUDA downgrade must fit after cached fused topology exceeds exact budget.",
        );

    assert_eq!(plan.topology, MegakernelExecutionTopology::SparseFrontier);
    assert!(plan.downgraded_to_sparse);
    assert_eq!(plan.memory.scratch_bytes, 10_000);
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().entries, 1);
}

#[test]
fn cache_rebases_lru_serial_instead_of_failing_dispatch() {
    let mut cache = CudaMegakernelPlanCache::with_max_entries(2);
    let first = key(1, CudaMegakernelAnalysisKind::Ifds, 0.10, 1_000);
    let second = key(2, CudaMegakernelAnalysisKind::Ifds, 0.20, 1_000);
    cache
        .get_or_insert_with(first, || {
            decision(MegakernelExecutionTopology::SparseFrontier)
        })
        .expect("Fix: first plan insert should fit");
    cache
        .get_or_insert_with(second, || {
            decision(MegakernelExecutionTopology::DenseFrontier)
        })
        .expect("Fix: second plan insert should fit");
    cache.serial = u64::MAX;

    cache
        .get_or_insert_with(first, || decision(MegakernelExecutionTopology::FusedWave))
        .expect("Fix: LRU serial exhaustion must rebase instead of failing the CUDA dispatch path");

    let first_seen = cache
        .entries
        .get(&first)
        .expect("Fix: first entry must remain")
        .last_seen;
    let second_seen = cache
        .entries
        .get(&second)
        .expect("Fix: second entry must remain")
        .last_seen;
    assert!(first_seen > second_seen);
    assert_eq!(cache.stats().hits, 1);
}

#[test]
fn cache_counters_pin_instead_of_failing_dispatch() {
    let mut cache = CudaMegakernelPlanCache::new();
    let key = key(3, CudaMegakernelAnalysisKind::Ifds, 0.10, 1_000);
    cache
        .get_or_insert_with(key, || {
            decision(MegakernelExecutionTopology::SparseFrontier)
        })
        .expect("Fix: plan insert should fit");
    cache.hits = u64::MAX;

    cache
        .get_or_insert_with(key, || decision(MegakernelExecutionTopology::DenseFrontier))
        .expect("Fix: counter exhaustion must not fail the CUDA dispatch path");

    assert_eq!(cache.stats().hits, u64::MAX);
}
