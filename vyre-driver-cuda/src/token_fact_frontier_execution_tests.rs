use super::*;
use crate::frontier_typed_ir_adapter::adapt_frontier_typed_ir_to_cuda;
use crate::token_fact_frontier_execution::estimated_queue_expansion_items;
use vyre_driver::device_work_queue::DeviceWorkQueueError;
use vyre_driver::ResidentGraphReuseTelemetry;
use vyre_libs::device::device_resident_token_fact_graph::{
    plan_device_resident_token_fact_graph, plan_device_resident_token_fact_graph_layout,
    DeviceResidentTokenFactGraphLayout, TokenFactEdge, TokenFactEdgeKind, TokenFactNode,
    TokenFactNodeKind, TOKEN_FACT_DEGREE_PROFILE_BUCKETS,
};
use vyre_libs::scheduling::frontier_typed_ir::{FrontierDomain, FrontierTypedPlan, FrontierWave};

#[test]
fn planner_combines_token_fact_residency_with_frontier_barriers() {
    let graph = plan_device_resident_token_fact_graph(
        &[
            TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 16),
            TokenFactNode::new(2, TokenFactNodeKind::Semantic, 16, 16),
            TokenFactNode::new(3, TokenFactNodeKind::Fact, 32, 16),
        ],
        &[
            TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact),
            TokenFactEdge::new(2, 3, TokenFactEdgeKind::FactDependency),
        ],
        48,
    )
    .expect("Fix: token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: token/fact graph should adapt");
    let frontier_plan = FrontierTypedPlan {
        waves: vec![
            FrontierWave {
                index: 0,
                domains: vec![FrontierDomain::Parser],
                node_ids: vec![10],
                active_items: 4,
            },
            FrontierWave {
                index: 1,
                domains: vec![FrontierDomain::Semantic],
                node_ids: vec![20],
                active_items: 4,
            },
            FrontierWave {
                index: 2,
                domains: vec![FrontierDomain::Dataflow],
                node_ids: vec![30],
                active_items: 4,
            },
        ],
    };
    let frontier_input = adapt_frontier_typed_ir_to_cuda(&frontier_plan, 8, 16, 8)
        .expect("Fix: frontier plan should adapt");
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xfeed,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 100_000.0,
            frontier_density: 0.10,
            readback_bytes: 24,
        },
        graph_layout,
        &frontier_input,
        8_192,
        1_000.0,
        0.0,
    )
    .expect("Fix: token/fact frontier execution should plan");

    assert_eq!(plan.frontier.barriers.global_barriers, 2);
    assert_eq!(plan.work_queue.queue_bytes, 14 * 4);
    assert_eq!(plan.resident_work_queue_bytes, 72);
    assert_eq!(plan.resident_payload_bytes, 48);
    assert!(plan.total_required_bytes >= plan.frontier.execution.memory.required_bytes);
}

#[test]
fn planner_sizes_resident_work_queue_for_edge_expansion_headroom() {
    let nodes = (0_u32..5)
        .map(|index| {
            TokenFactNode::new(
                index + 1,
                TokenFactNodeKind::Fact,
                u64::from(index) * 16,
                16,
            )
        })
        .collect::<Vec<_>>();
    let edges = complete_directed_edges(5, TokenFactEdgeKind::FactDependency);
    let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 80)
        .expect("Fix: fanout-heavy token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: fanout-heavy token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 16,
            scratch_bytes: 16,
            output_bytes: 16,
        }],
        active_items: vec![4],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xbeef,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10_000.0,
            frontier_density: 0.10,
            readback_bytes: 64,
        },
        graph_layout,
        &frontier_input,
        4_096,
        250.0,
        0.0,
    )
    .expect("Fix: CUDA token/fact planner should reserve edge-derived queue headroom");

    assert_eq!(plan.work_queue.queue_bytes, (4 + 16) * 4);
    assert_eq!(plan.resident_work_queue_bytes, (4 + 16) * 4 + 16);
    assert!(
        plan.frontier.execution.memory.required_bytes
            <= plan.frontier.execution.memory.budget_bytes
    );
}

#[test]
fn planner_avoids_total_edge_queue_reservation_for_sparse_dense_graph_frontiers() {
    let nodes = (0_u32..100)
        .map(|index| {
            TokenFactNode::new(
                index + 1,
                TokenFactNodeKind::Fact,
                u64::from(index) * 16,
                16,
            )
        })
        .collect::<Vec<_>>();
    let edges = complete_directed_edges(100, TokenFactEdgeKind::FactDependency);
    let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 1_600)
        .expect("Fix: dense token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: dense token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 16,
            scratch_bytes: 16,
            output_bytes: 16,
        }],
        active_items: vec![1],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xdad0,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10_000.0,
            frontier_density: 0.01,
            readback_bytes: 64,
        },
        graph_layout,
        &frontier_input,
        220_000,
        250.0,
        0.0,
    )
    .expect("Fix: sparse frontier over dense graph should not reserve every graph edge");

    assert_eq!(
        plan.work_queue.queue_bytes,
        100 * 4,
        "Fix: sparse active frontier should reserve initial item plus one average out-degree, not total graph edges"
    );
    assert_eq!(plan.resident_work_queue_bytes, 416);
}

#[test]
fn planner_reserves_hub_degree_headroom_for_sparse_star_frontier() {
    let nodes = (0_u32..100)
        .map(|index| {
            TokenFactNode::new(
                index + 1,
                TokenFactNodeKind::Fact,
                u64::from(index) * 16,
                16,
            )
        })
        .collect::<Vec<_>>();
    let edges = (2_u32..=100)
        .map(|to| TokenFactEdge::new(1, to, TokenFactEdgeKind::FactDependency))
        .collect::<Vec<_>>();
    let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 1_600)
        .expect("Fix: hub-heavy token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: hub-heavy token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 16,
            scratch_bytes: 16,
            output_bytes: 16,
        }],
        active_items: vec![1],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xdad1,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10_000.0,
            frontier_density: 0.01,
            readback_bytes: 64,
        },
        graph_layout,
        &frontier_input,
        30_000,
        250.0,
        0.0,
    )
    .expect("Fix: sparse hub frontier should reserve max-degree expansion headroom");

    assert_eq!(graph_layout.max_out_degree, 99);
    assert_eq!(plan.work_queue.queue_bytes, 100 * 4);
    assert_eq!(plan.resident_work_queue_bytes, 416);
}

#[test]
fn planner_uses_top_degree_profile_for_power_law_frontier_headroom() {
    let nodes = (0_u32..128)
        .map(|index| {
            TokenFactNode::new(
                index + 1,
                TokenFactNodeKind::Fact,
                u64::from(index) * 16,
                16,
            )
        })
        .collect::<Vec<_>>();
    let mut edges = (2_u32..=65)
        .map(|to| TokenFactEdge::new(1, to, TokenFactEdgeKind::FactDependency))
        .collect::<Vec<_>>();
    for from in 2_u32..=128 {
        for step in 1_u32..=4 {
            let to = ((from - 1 + step) % 128) + 1;
            edges.push(TokenFactEdge::new(
                from,
                to,
                TokenFactEdgeKind::FactDependency,
            ));
        }
    }
    let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 2_048)
        .expect("Fix: power-law token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: power-law token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 16,
            scratch_bytes: 16,
            output_bytes: 16,
        }],
        active_items: vec![4],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xdad2,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10_000.0,
            frontier_density: 0.03125,
            readback_bytes: 64,
        },
        graph_layout,
        &frontier_input,
        100_000,
        250.0,
        0.0,
    )
    .expect("Fix: power-law frontier should reserve top-degree rather than max-degree*N");

    assert_eq!(graph_layout.max_out_degree, 64);
    assert_eq!(graph_layout.top_out_degree_prefix_sums[2], 76);
    assert_eq!(plan.work_queue.queue_bytes, (4 + 76) * 4);
    assert_eq!(plan.resident_work_queue_bytes, (4 + 76) * 4 + 16);
}

#[test]
fn queue_expansion_estimate_caps_dense_frontier_at_total_edges() {
    assert_eq!(
        estimated_queue_expansion_items(
            200,
            vyre_driver::megakernel_execution::MegakernelGraphShape {
                node_count: 100,
                edge_count: 9_900,
            },
            99,
            [9_900; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
        )
        .expect("Fix: dense frontier queue expansion estimate should fit"),
        9_900
    );
}

#[test]
fn queue_expansion_estimate_prefers_top_degree_profile_when_rank_is_available() {
    let mut profile = [512_u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS];
    profile[0] = 64;
    profile[1] = 68;
    profile[2] = 76;

    assert_eq!(
        estimated_queue_expansion_items(
            4,
            vyre_driver::megakernel_execution::MegakernelGraphShape {
                node_count: 128,
                edge_count: 572,
            },
            64,
            profile,
        )
        .expect("Fix: profiled power-law expansion should fit"),
        76
    );
}

#[test]
fn queue_expansion_estimate_uses_total_edges_when_node_count_is_missing() {
    assert_eq!(
        estimated_queue_expansion_items(
            1,
            vyre_driver::megakernel_execution::MegakernelGraphShape {
                node_count: 0,
                edge_count: 128,
            },
            0,
            [0; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
        )
        .expect("Fix: malformed zero-node graph should fall back to total-edge headroom"),
        128
    );
}

#[test]
fn queue_expansion_estimate_rejects_active_average_degree_overflow() {
    assert_eq!(
        estimated_queue_expansion_items(
            2,
            vyre_driver::megakernel_execution::MegakernelGraphShape {
                node_count: 1,
                edge_count: u64::MAX,
            },
            u64::MAX,
            [0; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
        )
        .expect_err("overflowed active frontier expansion should fail before queue planning"),
        CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "active frontier edge expansion",
        }
    );
}

#[test]
fn planner_clamps_queue_expansion_after_graph_and_frontier_reserve() {
    let nodes = (0_u32..10)
        .map(|index| {
            TokenFactNode::new(
                index + 1,
                TokenFactNodeKind::Fact,
                u64::from(index) * 16,
                16,
            )
        })
        .collect::<Vec<_>>();
    let edges = complete_directed_edges(10, TokenFactEdgeKind::FactDependency);
    let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 160)
        .expect("Fix: dense token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: dense token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 16,
            scratch_bytes: 16,
            output_bytes: 16,
        }],
        active_items: vec![4],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xcafe,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10_000.0,
            frontier_density: 0.10,
            readback_bytes: 64,
        },
        graph_layout,
        &frontier_input,
        2_100,
        250.0,
        0.0,
    )
    .expect("Fix: queue expansion should clamp to the resident budget left after graph bytes");

    assert_eq!(plan.work_queue.queue_bytes, 17 * 4);
    assert_eq!(plan.resident_work_queue_bytes, 84);
    assert_eq!(plan.frontier.execution.memory.required_bytes, 1_808);
    assert_eq!(plan.frontier.execution.memory.budget_bytes, 1_856);
}

#[test]
fn planner_rejects_overflowed_edge_expansion_queue_capacity() {
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 8,
            scratch_bytes: 8,
            output_bytes: 8,
        }],
        active_items: vec![1],
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    assert_eq!(
        plan_cuda_token_fact_frontier_execution(
            &mut cache,
            0xfeed,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1.0,
                frontier_density: 0.0,
                readback_bytes: 0,
            },
            DeviceResidentTokenFactGraphLayout {
                node_count: 1,
                edge_count: u64::MAX,
                node_record_bytes: 32,
                edge_record_bytes: 16,
                max_out_degree: u64::MAX,
                top_out_degree_prefix_sums: [0; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
                node_bytes: 32,
                edge_bytes: 0,
                payload_bytes: 0,
                resident_bytes: 32,
            },
            &frontier_input,
            u64::MAX,
            0.0,
            0.0,
        )
        .expect_err("overflowed edge expansion capacity should fail before CUDA planning"),
        CudaTokenFactFrontierExecutionError::WorkQueue(DeviceWorkQueueError::ByteCountOverflow {
            field: "queue expansion capacity",
        })
    );
}

#[test]
fn planner_rejects_payload_that_exceeds_budget_before_frontier_planning() {
    let graph = plan_device_resident_token_fact_graph(
        &[TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 64)],
        &[],
        64,
    )
    .expect("Fix: token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: Vec::new(),
        active_items: Vec::new(),
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    assert_eq!(
        plan_cuda_token_fact_frontier_execution(
            &mut cache,
            0xfeed,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1.0,
                frontier_density: 0.0,
                readback_bytes: 0,
            },
            graph_layout,
            &frontier_input,
            63,
            0.0,
            0.0,
        )
        .expect_err("payload over budget should fail before cache planning"),
        CudaTokenFactFrontierExecutionError::PayloadExceedsBudget {
            payload_bytes: 64,
            budget_bytes: 63,
        }
    );
}

#[test]
fn planner_rejects_invalid_public_token_fact_layout_envelope() {
    let frontier_input = CudaFrontierTypedIrInput {
        waves: Vec::new(),
        active_items: Vec::new(),
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();
    let sample = CudaMegakernelScheduleSample {
        dispatch_cost_ns: 1.0,
        frontier_density: 0.0,
        readback_bytes: 0,
    };

    assert_eq!(
        plan_cuda_token_fact_frontier_execution(
            &mut cache,
            0xfeed,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device(),
            sample,
            DeviceResidentTokenFactGraphLayout {
                node_count: 0,
                edge_count: 0,
                node_record_bytes: 32,
                edge_record_bytes: 16,
                max_out_degree: 0,
                top_out_degree_prefix_sums: [0; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
                node_bytes: 0,
                edge_bytes: 0,
                payload_bytes: 0,
                resident_bytes: 0,
            },
            &frontier_input,
            8_192,
            0.0,
            0.0,
        )
        .expect_err("empty resident topology should fail before CUDA planning"),
        CudaTokenFactFrontierExecutionError::ZeroResidentGraphBytes
    );

    assert_eq!(
        plan_cuda_token_fact_frontier_execution(
            &mut cache,
            0xfeed,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device(),
            sample,
            DeviceResidentTokenFactGraphLayout {
                node_count: 1,
                edge_count: 1,
                node_record_bytes: 32,
                edge_record_bytes: 16,
                max_out_degree: 1,
                top_out_degree_prefix_sums: [1; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
                node_bytes: 32,
                edge_bytes: 16,
                payload_bytes: 8,
                resident_bytes: 55,
            },
            &frontier_input,
            8_192,
            0.0,
            0.0,
        )
        .expect_err("mismatched resident byte envelope should fail before CUDA planning"),
        CudaTokenFactFrontierExecutionError::ResidentGraphByteEnvelopeMismatch {
            expected_bytes: 56,
            actual_bytes: 55,
        }
    );
}

#[test]
fn planner_accounts_warm_resident_graph_without_upload_pressure() {
    let graph = plan_device_resident_token_fact_graph(
        &[TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 16)],
        &[],
        16,
    )
    .expect("Fix: token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: Vec::new(),
        active_items: Vec::new(),
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let cold = plan_cuda_token_fact_frontier_execution_envelope(
        &mut cache,
        0xfeed,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 1.0,
            frontier_density: 0.0,
            readback_bytes: 0,
        },
        graph_layout,
        CudaTokenFactGraphResidency::ColdUpload,
        &frontier_input,
        8_192,
        0.0,
        0.0,
    )
    .expect("Fix: cold token/fact graph should plan");
    let warm = plan_cuda_token_fact_frontier_execution_envelope(
        &mut cache,
        0xfeed,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 1.0,
            frontier_density: 0.0,
            readback_bytes: 0,
        },
        graph_layout,
        CudaTokenFactGraphResidency::WarmResident,
        &frontier_input,
        8_192,
        0.0,
        0.0,
    )
    .expect("Fix: warm token/fact graph should plan");

    assert_eq!(cold.resident_graph_bytes, 32);
    assert_eq!(cold.graph_upload_bytes, 32);
    assert_eq!(cold.avoided_graph_upload_bytes, 0);
    assert_eq!(
        cold.graph_reuse,
        ResidentGraphReuseTelemetry::cold_upload(32)
    );
    assert_eq!(warm.resident_graph_bytes, 32);
    assert_eq!(warm.graph_upload_bytes, 0);
    assert_eq!(warm.avoided_graph_upload_bytes, 32);
    assert_eq!(
        warm.graph_reuse,
        ResidentGraphReuseTelemetry::warm_reuse(32)
    );
    assert_eq!(warm.total_resident_bytes, cold.total_resident_bytes);
}

#[test]
fn planner_rejects_frontier_waves_without_matching_active_item_counts() {
    let graph = plan_device_resident_token_fact_graph(
        &[TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 16)],
        &[],
        16,
    )
    .expect("Fix: token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: vec![vyre_driver::megakernel_frontier::MegakernelFrontierWave {
            frontier_bytes: 8,
            scratch_bytes: 8,
            output_bytes: 8,
        }],
        active_items: Vec::new(),
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    assert_eq!(
        plan_cuda_token_fact_frontier_execution(
            &mut cache,
            0xfeed,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device(),
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1.0,
                frontier_density: 0.0,
                readback_bytes: 0,
            },
            graph_layout,
            &frontier_input,
            8_192,
            0.0,
            0.0,
        )
        .expect_err("mismatched active-item counts should fail before queue planning"),
        CudaTokenFactFrontierExecutionError::ActiveItemWaveCountMismatch {
            waves: 1,
            active_items: 0,
        }
    );
}

#[test]
fn planner_does_not_allocate_resident_work_queue_for_empty_frontier() {
    let graph = plan_device_resident_token_fact_graph(
        &[TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 16)],
        &[],
        16,
    )
    .expect("Fix: token/fact graph should pack");
    let graph_layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
        .expect("Fix: token/fact graph should adapt");
    let frontier_input = CudaFrontierTypedIrInput {
        waves: Vec::new(),
        active_items: Vec::new(),
        dependencies: Vec::new(),
    };
    let mut cache = CudaMegakernelPlanCache::new();

    let plan = plan_cuda_token_fact_frontier_execution(
        &mut cache,
        0xfeed,
        CudaMegakernelAnalysisKind::ParserFrontend,
        device(),
        CudaMegakernelScheduleSample {
            dispatch_cost_ns: 1.0,
            frontier_density: 0.0,
            readback_bytes: 0,
        },
        graph_layout,
        &frontier_input,
        8_192,
        0.0,
        0.0,
    )
    .expect("Fix: empty frontier should not need a resident work queue");

    assert_eq!(plan.work_queue.queue_bytes, 0);
    assert_eq!(plan.work_queue.control_bytes, 0);
    assert_eq!(plan.resident_work_queue_bytes, 0);
    assert!(plan.work_queue.final_only_host_sync);
}

fn complete_directed_edges(node_count: u32, kind: TokenFactEdgeKind) -> Vec<TokenFactEdge> {
    let mut edges = Vec::new();
    for from in 1..=node_count {
        for to in 1..=node_count {
            if from != to {
                edges.push(TokenFactEdge::new(from, to, kind));
            }
        }
    }
    edges
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
