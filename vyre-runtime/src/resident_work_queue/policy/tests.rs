use super::cache::{LaunchRecommendationCache, LaunchRecommendationCacheKey};
use super::launch::best_cost_index;
use super::*;

mod cache_contracts {
    use super::*;

    #[test]
    fn launch_cache_update_does_not_duplicate_entries() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest::direct(128, 64, 256);
        let key = LaunchRecommendationCacheKey { policy, request };
        let rec = policy
            .recommend(request)
            .expect("Fix: policy should accept non-zero adapter limits");
        let mut cache = LaunchRecommendationCache::default();

        cache.insert(key, rec);
        cache.insert(key, rec);

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn launch_cache_get_promotes_hot_key_before_eviction() {
        let policy = ResidentLaunchPolicy::standard();
        let hot_request = ResidentLaunchRequest::direct(1, 64, 256);
        let hot_key = LaunchRecommendationCacheKey {
            policy,
            request: hot_request,
        };
        let hot_rec = policy
            .recommend(hot_request)
            .expect("Fix: policy should accept non-zero adapter limits");
        let mut cache = LaunchRecommendationCache::default();

        cache.insert(hot_key, hot_rec);
        for queue_len in 2..=128 {
            let request = ResidentLaunchRequest::direct(queue_len, 64, 256);
            let rec = policy
                .recommend(request)
                .expect("Fix: policy should accept non-zero adapter limits");
            cache.insert(LaunchRecommendationCacheKey { policy, request }, rec);
        }
        assert!(cache.get(&hot_key).is_some());
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 0);

        let cold_request = ResidentLaunchRequest::direct(129, 64, 256);
        let cold_rec = policy
            .recommend(cold_request)
            .expect("Fix: policy should accept non-zero adapter limits");
        cache.insert(
            LaunchRecommendationCacheKey {
                policy,
                request: cold_request,
            },
            cold_rec,
        );

        assert!(cache.get(&hot_key).is_some());
        assert_eq!(cache.hits, 2);
        assert_eq!(cache.len(), 128);
    }

    #[test]
    fn launch_cache_records_misses_without_mutating_capacity() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest::direct(128, 64, 256);
        let missing = LaunchRecommendationCacheKey { policy, request };
        let mut cache = LaunchRecommendationCache::default();

        assert!(cache.get(&missing).is_none());

        assert_eq!(cache.hits, 0);
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn launch_policy_exposes_thread_local_cache_stats() {
        ResidentLaunchPolicy::reset_launch_cache_for_thread();
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest::direct(512, 64, 256);

        let initial = ResidentLaunchPolicy::launch_cache_stats();
        assert_eq!(initial.entries, 0);
        assert_eq!(initial.hits, 0);
        assert_eq!(initial.misses, 0);

        let first = policy
            .recommend(request)
            .expect("Fix: valid policy request must recommend");
        let after_miss = ResidentLaunchPolicy::launch_cache_stats();
        assert_eq!(after_miss.entries, 1);
        assert_eq!(after_miss.hits, 0);
        assert_eq!(after_miss.misses, 1);

        let second = policy
            .recommend(request)
            .expect("Fix: cached policy request must recommend");
        let after_hit = ResidentLaunchPolicy::launch_cache_stats();
        assert_eq!(first, second);
        assert_eq!(after_hit.entries, 1);
        assert_eq!(after_hit.hits, 1);
        assert_eq!(after_hit.misses, 1);

        ResidentLaunchPolicy::reset_launch_cache_for_thread();
    }
}

mod hysteresis_contracts {
    use super::*;

    #[test]
    fn stable_recommendation_holds_sparse_topology_inside_frontier_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest {
            queue_len: 8_192,
            requested_worker_groups: 128,
            max_workgroup_size_x: 256,
            graph_node_count: 100_000,
            graph_edge_count: 250_000,
            frontier_density_bps: policy.sparse_frontier_threshold_bps + 125,
            ..ResidentLaunchRequest::direct(8_192, 128, 256)
        };
        let stateless = policy
            .recommend(request)
            .expect("Fix: stateless launch recommendation should accept valid adapter limits");
        let stable = policy
            .recommend_with_previous_topology(request, ResidentQueueTopology::SparseFrontier)
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(stateless.topology, ResidentQueueTopology::HybridFrontier);
        assert_eq!(stable.topology, ResidentQueueTopology::SparseFrontier);
    }

    #[test]
    fn stable_recommendation_releases_sparse_topology_outside_frontier_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend_with_previous_topology(
                ResidentLaunchRequest {
                    queue_len: 8_192,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 100_000,
                    graph_edge_count: 250_000,
                    frontier_density_bps: policy.sparse_frontier_threshold_bps + 300,
                    ..ResidentLaunchRequest::direct(8_192, 128, 256)
                },
                ResidentQueueTopology::SparseFrontier,
            )
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
    }

    #[test]
    fn stable_recommendation_holds_hybrid_topology_inside_sparse_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend_with_previous_topology(
                ResidentLaunchRequest {
                    queue_len: 8_192,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 100_000,
                    graph_edge_count: 250_000,
                    frontier_density_bps: policy.sparse_frontier_threshold_bps - 125,
                    ..ResidentLaunchRequest::direct(8_192, 128, 256)
                },
                ResidentQueueTopology::HybridFrontier,
            )
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
    }

    #[test]
    fn stable_recommendation_holds_hybrid_topology_inside_dense_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend_with_previous_topology(
                ResidentLaunchRequest {
                    queue_len: 16_384,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 16_384,
                    graph_edge_count: 250_000,
                    frontier_density_bps: policy.dense_frontier_threshold_bps + 125,
                    ..ResidentLaunchRequest::direct(16_384, 128, 256)
                },
                ResidentQueueTopology::HybridFrontier,
            )
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
    }

    #[test]
    fn stable_recommendation_holds_dense_topology_inside_frontier_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest {
            queue_len: 16_384,
            requested_worker_groups: 128,
            max_workgroup_size_x: 256,
            graph_node_count: 16_384,
            graph_edge_count: 250_000,
            frontier_density_bps: policy.dense_frontier_threshold_bps - 125,
            ..ResidentLaunchRequest::direct(16_384, 128, 256)
        };
        let stateless = policy
            .recommend(request)
            .expect("Fix: stateless launch recommendation should accept valid adapter limits");
        let stable = policy
            .recommend_with_previous_topology(request, ResidentQueueTopology::DenseFrontier)
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(stateless.topology, ResidentQueueTopology::HybridFrontier);
        assert_eq!(stable.topology, ResidentQueueTopology::DenseFrontier);
    }

    #[test]
    fn stable_recommendation_preserves_fused_dense_when_hot_graph_stays_near_dense() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend_with_previous_topology(
                ResidentLaunchRequest {
                    queue_len: 131_072,
                    requested_worker_groups: 256,
                    max_workgroup_size_x: 256,
                    graph_node_count: 32_768,
                    graph_edge_count: 500_000,
                    frontier_density_bps: policy.dense_frontier_threshold_bps - 125,
                    hot_window_count: policy.hot_window_threshold,
                    ..ResidentLaunchRequest::direct(131_072, 256, 256)
                },
                ResidentQueueTopology::FusedDense,
            )
            .expect("Fix: stable fused dense recommendation should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
        assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
    }

    #[test]
    fn stable_recommendation_holds_memory_constrained_topology_inside_pressure_hysteresis() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest {
            queue_len: 16_384,
            requested_worker_groups: 128,
            max_workgroup_size_x: 256,
            graph_node_count: 16_384,
            graph_edge_count: 250_000,
            frontier_density_bps: 9_000,
            memory_pressure_bps: policy.memory_pressure_threshold_bps - 125,
            ..ResidentLaunchRequest::direct(16_384, 128, 256)
        };
        let stateless = policy
            .recommend(request)
            .expect("Fix: stateless launch recommendation should accept valid adapter limits");
        let stable = policy
            .recommend_with_previous_topology(request, ResidentQueueTopology::MemoryConstrained)
            .expect("Fix: stable launch recommendation should accept valid adapter limits");

        assert_eq!(stateless.topology, ResidentQueueTopology::DenseFrontier);
        assert_eq!(stable.topology, ResidentQueueTopology::MemoryConstrained);
        assert!(
            stable.worker_groups < stateless.worker_groups,
            "stable memory-constrained topology must preserve worker shedding near pressure threshold"
        );
    }
}

mod priority_diffusion_contracts {
    use super::*;

    #[test]
    fn priority_accounting_reports_structured_drain_before_overflow() {
        let accounting = PriorityRequeueAccounting {
            requeue_count: u64::MAX - 8,
            aged_promotions: 3,
            max_priority_age: 64,
        };
        let recommendation = accounting.drain_recommendation();

        assert!(recommendation.should_drain);
        assert_eq!(
            recommendation.reason,
            PriorityDrainReason::RequeueCounterNearLimit
        );
        assert_eq!(recommendation.requeue_count, u64::MAX - 8);
        assert_eq!(recommendation.aged_promotions, 3);
        assert_eq!(recommendation.max_priority_age, 64);
        assert_eq!(recommendation.requeue_counter_headroom, 8);
        assert_eq!(recommendation.aged_promotion_counter_headroom, u64::MAX - 3);
        assert_eq!(recommendation.fix, PRIORITY_COUNTER_DRAIN_FIX);
    }

    #[test]
    fn priority_accounting_reports_no_drain_for_empty_counters() {
        let recommendation = PriorityRequeueAccounting::default().drain_recommendation();

        assert!(!recommendation.should_drain);
        assert_eq!(recommendation.reason, PriorityDrainReason::None);
        assert_eq!(recommendation.requeue_count, 0);
        assert_eq!(recommendation.aged_promotions, 0);
        assert_eq!(recommendation.max_priority_age, 0);
        assert_eq!(recommendation.requeue_counter_headroom, u64::MAX);
        assert_eq!(recommendation.aged_promotion_counter_headroom, u64::MAX);
        assert_eq!(recommendation.fix, PRIORITY_COUNTER_DRAIN_FIX);
    }

    #[test]
    fn diffuse_priority_mismatched_restrictions_preserve_input_shape() {
        let input = [3.0, 1.0, 2.0];
        let restrictions = [1.0, 0.5];
        let mut out = Vec::with_capacity(input.len());
        let mut scratch = Vec::with_capacity(input.len());

        try_diffuse_priority_across_siblings_into(
            &input,
            &restrictions,
            0.5,
            4,
            &mut out,
            &mut scratch,
        )
        .expect("Fix: diffusion staging must succeed for three siblings");

        assert_eq!(out, input);
        assert!(scratch.is_empty());
        assert_eq!(out.capacity(), input.len());
    }

    #[test]
    fn diffuse_priority_reuses_exact_scratch_capacity() {
        let input = [4.0, 2.0, 1.0];
        let restrictions = [1.0, 1.0, 1.0];
        let mut out = Vec::with_capacity(input.len());
        let mut scratch = Vec::with_capacity(input.len());
        let out_ptr = out.as_ptr();
        let scratch_ptr = scratch.as_ptr();

        try_diffuse_priority_across_siblings_into(
            &input,
            &restrictions,
            0.25,
            2,
            &mut out,
            &mut scratch,
        )
        .expect("Fix: diffusion staging must succeed for three siblings");

        assert_eq!(out.len(), input.len());
        assert_eq!(scratch.len(), input.len());
        assert_eq!(out.capacity(), input.len());
        assert_eq!(scratch.capacity(), input.len());
        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(scratch.as_ptr(), scratch_ptr);
    }
}

mod recommendation_contracts {
    use super::*;

    #[test]
    fn policy_recommends_padded_geometry_and_hit_capacity() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 300,
                requested_worker_groups: 64,
                max_workgroup_size_x: 256,
                requested_hit_capacity: 0,
                expected_hits_per_item: 3,
                ..ResidentLaunchRequest::direct(300, 64, 256)
            })
            .expect("Fix: policy should accept non-zero adapter limits");
        assert_eq!(rec.geometry.workgroup_size_x, 64);
        assert_eq!(rec.geometry.slot_count, 320);
        assert_eq!(rec.geometry.dispatch_grid, [5, 1, 1]);
        assert_eq!(rec.hit_capacity, 1800);
        assert_eq!(rec.estimated_peak_device_bytes, 28_800);
        assert_eq!(rec.device_memory_budget_bytes, 0);
        assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
    }

    #[test]
    fn telemetry_pressure_selects_jit_and_priority_aging() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 8192,
                requested_worker_groups: 64,
                max_workgroup_size_x: 256,
                hot_opcode_count: 8,
                requeue_count: 1,
                max_priority_age: 64,
                ..ResidentLaunchRequest::direct(8192, 64, 256)
            })
            .expect("Fix: policy should accept non-zero adapter limits");
        assert_eq!(rec.pressure, ResidentQueuePressure::Saturated);
        assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
        assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
        assert!(rec.promote_hot_opcodes);
        assert!(rec.age_priority_work);
    }

    #[test]
    fn dense_large_hot_graph_selects_fused_dense_topology() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 131_072,
                requested_worker_groups: 256,
                max_workgroup_size_x: 256,
                graph_node_count: 32_768,
                graph_edge_count: 500_000,
                frontier_density_bps: 7_500,
                hot_window_count: policy.hot_window_threshold,
                ..ResidentLaunchRequest::direct(131_072, 256, 256)
            })
            .expect("Fix: fused dense topology should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
        assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
    }

    #[test]
    fn topology_evidence_reports_graphblas_switch_inputs_and_parity_contract() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest {
            queue_len: 131_072,
            requested_worker_groups: 256,
            max_workgroup_size_x: 256,
            graph_node_count: 32_768,
            graph_edge_count: 500_000,
            frontier_density_bps: 7_500,
            hot_window_count: policy.hot_window_threshold,
            resident_device_bytes: 64 * 1024 * 1024,
            ..ResidentLaunchRequest::direct(131_072, 256, 256)
        };
        let (rec, evidence) = policy
            .recommend_with_topology_evidence(request)
            .expect("Fix: topology evidence should be emitted for valid launch telemetry");

        assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
        assert_eq!(evidence.schema_version, TOPOLOGY_EVIDENCE_SCHEMA_VERSION);
        assert_eq!(evidence.selected_topology, rec.topology);
        assert_eq!(evidence.queue_pressure, rec.pressure);
        assert_eq!(evidence.frontier_density_bps, 7_500);
        assert_eq!(evidence.semiring_frontier_density_bps, 7_500);
        assert_eq!(
            evidence.graphblas_switch_class,
            ResidentGraphBlasSwitchClass::Dense
        );
        assert_eq!(evidence.resident_device_bytes, 64 * 1024 * 1024);
        assert_eq!(
            evidence.estimated_peak_device_bytes,
            rec.estimated_peak_device_bytes
        );
        assert!(evidence.output_parity_required);
        assert!(evidence.is_complete());
    }

    #[test]
    fn promotion_evidence_reports_fused_window_lowerer_contract() {
        let policy = ResidentLaunchPolicy::standard();
        let request = ResidentLaunchRequest {
            queue_len: 1024,
            requested_worker_groups: 64,
            max_workgroup_size_x: 256,
            hot_window_count: policy.hot_window_threshold,
            ..ResidentLaunchRequest::direct(1024, 64, 256)
        };
        let (rec, evidence) = policy
            .recommend_with_promotion_evidence(request)
            .expect("Fix: promotion evidence should be emitted for valid hot-window telemetry");

        assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
        assert!(rec.promote_hot_windows);
        assert_eq!(
            evidence.schema_version,
            HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION
        );
        assert_eq!(evidence.queue_len, 1024);
        assert_eq!(evidence.hot_window_count, policy.hot_window_threshold);
        assert_eq!(evidence.hot_window_threshold, policy.hot_window_threshold);
        assert_eq!(evidence.hot_opcode_count, 0);
        assert_eq!(evidence.hot_opcode_threshold, policy.hot_opcode_threshold);
        assert_eq!(evidence.execution_mode, ResidentExecutionMode::Jit);
        assert_eq!(evidence.promotion_route, ResidentPromotionRoute::WindowJit);
        assert!(evidence.promote_hot_windows);
        assert!(!evidence.promote_hot_opcodes);
        assert!(evidence.fused_descriptor_window_required);
        assert!(evidence.output_parity_required);
        assert!(evidence.is_complete());
    }

    #[test]
    fn high_memory_pressure_overrides_dense_frontier() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 16_384,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 16_384,
                graph_edge_count: 250_000,
                frontier_density_bps: 9_000,
                memory_pressure_bps: policy.memory_pressure_threshold_bps,
                ..ResidentLaunchRequest::direct(16_384, 128, 256)
            })
            .expect("Fix: memory-constrained topology should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
        assert!(
            rec.worker_groups < 128,
            "memory-constrained topology must lower worker-group pressure, got {}",
            rec.worker_groups
        );
        assert_eq!(
            rec.hit_capacity, 16_384,
            "memory-constrained topology must avoid the normal sparse-hit over-allocation multiplier"
        );
    }

    #[test]
    fn explicit_hit_capacity_survives_memory_constrained_worker_shedding() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 16_384,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                requested_hit_capacity: 65_536,
                memory_pressure_bps: 10_000,
                ..ResidentLaunchRequest::direct(16_384, 128, 256)
            })
            .expect(
                "Fix: memory-constrained explicit-capacity launch should accept valid adapter limits",
            );

        assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
        assert_eq!(rec.hit_capacity, 65_536);
        assert_eq!(rec.worker_groups, 64);
    }

    #[test]
    fn device_memory_budget_rejects_oversized_hit_plan_before_allocation() {
        let policy = ResidentLaunchPolicy::standard();
        let err = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 1024,
                requested_worker_groups: 64,
                max_workgroup_size_x: 256,
                expected_hits_per_item: 4,
                resident_device_bytes: 1024,
                device_memory_budget_bytes: 64 * 1024,
                ..ResidentLaunchRequest::direct(1024, 64, 256)
            })
            .expect_err("Fix: launch policy must reject plans that exceed explicit device budget");

        match err {
            vyre_driver::BackendError::DeviceOutOfMemory {
                requested,
                available,
            } => {
                assert_eq!(requested, 132_096);
                assert_eq!(available, 64 * 1024);
            }
            other => panic!("expected DeviceOutOfMemory for budget overflow, got {other:?}"),
        }
    }

    #[test]
    fn device_memory_budget_infers_pressure_without_manual_bps() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 1024,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                resident_device_bytes: 900_000,
                device_memory_budget_bytes: 1_000_000,
                ..ResidentLaunchRequest::direct(1024, 128, 256)
            })
            .expect("Fix: budget-aware policy should accept launches under the byte budget");

        assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
        assert!(
            rec.worker_groups < 128,
            "inferred memory pressure must shed worker groups before launch"
        );
        assert_eq!(rec.estimated_peak_device_bytes, 916_384);
        assert_eq!(rec.device_memory_budget_bytes, 1_000_000);
    }

    #[test]
    fn dense_frontier_without_hot_fusion_stays_dense() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 16_384,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 16_384,
                graph_edge_count: 250_000,
                frontier_density_bps: policy.dense_frontier_threshold_bps,
                ..ResidentLaunchRequest::direct(16_384, 128, 256)
            })
            .expect("Fix: dense topology should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::DenseFrontier);
    }

    #[test]
    fn mid_density_frontier_selects_hybrid_topology() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 8192,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 8192,
                graph_edge_count: 32_768,
                frontier_density_bps: policy.sparse_frontier_threshold_bps + 1,
                ..ResidentLaunchRequest::direct(8192, 128, 256)
            })
            .expect("Fix: hybrid topology should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
    }

    #[test]
    fn missing_frontier_telemetry_infers_density_from_queue_and_graph_scale() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 90_000,
                requested_worker_groups: 256,
                max_workgroup_size_x: 256,
                graph_node_count: 100_000,
                graph_edge_count: 750_000,
                hot_opcode_count: policy.hot_opcode_threshold,
                frontier_density_bps: 0,
                ..ResidentLaunchRequest::direct(90_000, 256, 256)
            })
            .expect("Fix: inferred-density topology should accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
        assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
    }

    #[test]
    fn sparse_frontier_density_sheds_worker_pressure_without_losing_warp_floor() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 100_000,
                requested_worker_groups: 256,
                max_workgroup_size_x: 256,
                graph_node_count: 1_000_000,
                graph_edge_count: 4_000_000,
                frontier_density_bps: 100,
                ..ResidentLaunchRequest::direct(100_000, 256, 256)
            })
            .expect("Fix: sparse density worker shedding must accept valid adapter limits");

        assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
        assert_eq!(rec.worker_groups, 51);
        assert_eq!(rec.geometry.workgroup_size_x, 51);
        assert_eq!(rec.geometry.dispatch_grid, [51, 1, 1]);
    }

    #[test]
    fn sparse_frontier_worker_shedding_preserves_warp_floor_for_tiny_density() {
        let policy = ResidentLaunchPolicy::standard();
        let rec = policy
            .recommend(ResidentLaunchRequest {
                queue_len: 1_000,
                requested_worker_groups: 256,
                max_workgroup_size_x: 256,
                graph_node_count: 1_000_000,
                graph_edge_count: 4_000_000,
                frontier_density_bps: 1,
                ..ResidentLaunchRequest::direct(1_000, 256, 256)
            })
            .expect("Fix: sparse density worker shedding must retain a useful GPU width");

        assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
        assert_eq!(rec.worker_groups, 32);
        assert_eq!(rec.geometry.workgroup_size_x, 32);
    }
}

// Inline: `best_cost_index` is crate-private and the public knobs refuse an
// empty candidate set before reaching it, so no integration test can hand it
// the empty slice its own contract has to answer for.
mod autotune_selection_contracts {
    use super::*;

    /// No measured cost selects nothing.
    ///
    /// The empty case used to be a `debug_assert` in front of `costs[0]`,
    /// which is absent from a release build, so the shipped binary indexed
    /// an empty slice.
    #[test]
    fn no_measured_cost_selects_nothing() {
        assert_eq!(best_cost_index(&[]), None);
    }

    /// The lowest cost wins, and the first of a tie keeps the selection stable.
    #[test]
    fn the_lowest_cost_wins_and_a_tie_keeps_the_earlier_candidate() {
        assert_eq!(best_cost_index(&[3.0]), Some(0));
        assert_eq!(best_cost_index(&[3.0, 1.0, 2.0]), Some(1));
        assert_eq!(best_cost_index(&[1.0, 5.0, 1.0]), Some(0));
        assert_eq!(best_cost_index(&[5.0, 4.0, 3.0, 2.0]), Some(3));
    }

    /// Every position is reachable, so the scan reports the index it scanned.
    ///
    /// The scan skips the first cost and counts from the rest, so an index
    /// that is off by one selects the neighbour of the cheapest candidate at
    /// every position except the first, which is exactly the case a single
    /// example misses.
    #[test]
    fn the_reported_index_is_the_position_of_the_lowest_cost() {
        let width = 6;
        for cheapest in 0..width {
            let costs: Vec<f64> = (0..width)
                .map(|index| if index == cheapest { 1.0 } else { 9.0 })
                .collect();
            assert_eq!(
                best_cost_index(&costs),
                Some(cheapest),
                "Fix: the lowest cost at {cheapest} of {width} selected the wrong candidate"
            );
        }
    }

    /// A cost that is not a number never beats a measured one.
    #[test]
    fn an_unmeasurable_cost_never_wins() {
        assert_eq!(best_cost_index(&[f64::NAN, 2.0]), Some(1));
        assert_eq!(best_cost_index(&[2.0, f64::NAN]), Some(0));
        assert_eq!(best_cost_index(&[f64::NAN, f64::NAN]), Some(0));
    }

    /// The public knobs return the candidate that the cheapest cost sits against.
    #[test]
    fn the_autotune_knobs_return_the_candidate_paired_with_the_lowest_cost() {
        let policy = ResidentLaunchPolicy::standard();
        assert_eq!(
            policy.autotune_workgroup_size(&[64, 128, 256], &[3.0, 1.0, 2.0], 32),
            128
        );
        assert_eq!(
            policy.autotune_hit_capacity_multiplier(&[2, 4, 8], &[5.0, 4.0, 1.0]),
            8
        );
    }

    /// A knob with nothing measured keeps the value it was given.
    #[test]
    fn the_autotune_knobs_keep_the_current_value_when_nothing_was_measured() {
        let policy = ResidentLaunchPolicy::standard();
        assert_eq!(policy.autotune_workgroup_size(&[64, 128], &[], 32), 32);
        assert_eq!(policy.autotune_workgroup_size(&[], &[1.0], 32), 32);
        assert_eq!(
            policy.autotune_hit_capacity_multiplier(&[2, 4], &[]),
            policy.hit_capacity_multiplier
        );
        assert_eq!(
            policy.autotune_hit_capacity_multiplier(&[], &[1.0]),
            policy.hit_capacity_multiplier
        );
    }

    /// More candidates than costs selects only among the costs that exist.
    #[test]
    fn a_candidate_without_a_cost_is_not_selected() {
        let policy = ResidentLaunchPolicy::standard();
        assert_eq!(
            policy.autotune_workgroup_size(&[64, 128, 256], &[2.0, 1.0], 32),
            128
        );
    }
}
