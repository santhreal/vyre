use super::*;

fn query(
    query: u32,
    graph_layout_hash: u64,
    traversal_key: u64,
    graph_upload_bytes: u64,
    frontier_bytes: u64,
    scratch_bytes: u64,
    output_bytes: u64,
) -> MultiQuery {
    MultiQuery {
        query,
        graph_layout_hash,
        traversal_key,
        graph_upload_bytes,
        frontier_bytes,
        scratch_bytes,
        output_bytes,
    }
}

#[test]
fn multi_query_batches_compatible_queries_over_one_resident_graph() {
    let plan = plan_multi_query_execution(
        &[
            query(3, 0xabc, 0x10, 4_096, 64, 128, 32),
            query(1, 0xabc, 0x10, 4_096, 32, 64, 16),
            query(2, 0xabc, 0x10, 4_096, 48, 96, 24),
        ],
        8_192,
    )
    .expect("Fix: compatible queries should batch");

    assert_eq!(plan.launch_count, 1);
    assert_eq!(plan.avoided_launches, 2);
    assert_eq!(plan.avoided_host_fences, 2);
    assert_eq!(plan.avoided_graph_upload_bytes, 8_192);
    assert_eq!(
        plan.graph_reuse,
        ResidentGraphReuseTelemetry::from_counters(1, 2, 4_096, 8_192)
    );
    assert_eq!(plan.groups[0].queries, vec![1, 2, 3]);
    assert_eq!(
        plan.groups[0].graph_reuse,
        ResidentGraphReuseTelemetry::from_counters(1, 2, 4_096, 8_192)
    );
    assert_eq!(plan.groups[0].frontier_bytes, 144);
    assert_eq!(plan.groups[0].peak_scratch_bytes, 128);
    assert_eq!(plan.groups[0].output_bytes, 72);
    assert!(plan.final_only_host_fence_per_group);
}

#[test]
fn multi_query_splits_compatible_group_to_fit_resident_budget_without_reuploading_graph() {
    let plan = plan_multi_query_execution(
        &[
            query(1, 0xabc, 0x10, 100, 100, 10, 10),
            query(2, 0xabc, 0x10, 100, 100, 10, 10),
            query(3, 0xabc, 0x10, 100, 100, 10, 10),
        ],
        350,
    )
    .expect("Fix: compatible multi-query queries should split into budget-fit resident chunks");

    assert_eq!(plan.launch_count, 2);
    assert_eq!(plan.avoided_launches, 1);
    assert_eq!(plan.avoided_host_fences, 1);
    assert_eq!(plan.avoided_graph_upload_bytes, 200);
    assert_eq!(
        plan.graph_reuse,
        ResidentGraphReuseTelemetry::from_counters(1, 2, 100, 200)
    );
    assert_eq!(plan.peak_resident_bytes, 330);
    assert_eq!(plan.groups[0].queries, vec![1, 2]);
    assert_eq!(plan.groups[0].graph_upload_bytes, 100);
    assert_eq!(plan.groups[0].resident_bytes, 330);
    assert_eq!(plan.groups[1].queries, vec![3]);
    assert_eq!(plan.groups[1].graph_upload_bytes, 0);
    assert_eq!(plan.groups[1].resident_bytes, 220);
    assert!(plan.final_only_host_fence_per_group);
}

#[test]
fn multi_query_later_chunks_still_count_resident_graph_memory() {
    assert_eq!(
        plan_multi_query_execution(
            &[
                query(1, 0xabc, 0x10, 100, 100, 10, 10),
                query(2, 0xabc, 0x10, 100, 100, 10, 10),
            ],
            150,
        )
        .expect_err("later resident chunk still needs graph memory and should exceed budget"),
        MultiQueryExecutionError::OverBudget {
            graph_layout_hash: 0xabc,
            traversal_key: 0x10,
            required_bytes: 220,
            budget_bytes: 150,
        }
    );
}

#[test]
fn multi_query_splits_incompatible_graph_or_traversal_keys() {
    let plan = plan_multi_query_execution(
        &[
            query(1, 0xdef, 0x10, 1_024, 32, 64, 16),
            query(2, 0xabc, 0x20, 1_024, 32, 64, 16),
            query(3, 0xabc, 0x10, 1_024, 32, 64, 16),
        ],
        4_096,
    )
    .expect("Fix: incompatible queries should become separate groups");

    assert_eq!(plan.launch_count, 3);
    assert_eq!(plan.avoided_launches, 0);
    assert_eq!(plan.avoided_graph_upload_bytes, 1_024);
    assert_eq!(
        plan.graph_reuse,
        ResidentGraphReuseTelemetry::from_counters(2, 1, 2_048, 1_024)
    );
    assert_eq!(plan.groups[0].graph_upload_bytes, 1_024);
    assert_eq!(plan.groups[1].graph_upload_bytes, 0);
    assert_eq!(plan.groups[2].graph_upload_bytes, 1_024);
    assert_eq!(
        plan.groups
            .iter()
            .map(|group| (group.graph_layout_hash, group.traversal_key))
            .collect::<Vec<_>>(),
        vec![(0xabc, 0x10), (0xabc, 0x20), (0xdef, 0x10)]
    );
}

#[test]
fn reused_query_bucket_returns_to_pool_when_reservation_fails() {
    let retained = vec![query(42, 0xabc, 0x10, 4_096, 8, 16, 4)];
    let mut free_query_buckets = vec![retained.clone()];

    let err = take_reserved_query_bucket(&mut free_query_buckets, usize::MAX)
        .expect_err("impossible query bucket reservation must fail");

    assert!(
        matches!(
            err,
            MultiQueryExecutionError::StorageReserveFailed {
                field: "multi-query grouped query bucket",
                ..
            }
        ),
        "Fix: query bucket reservation failure must surface the grouped-bucket field"
    );
    assert_eq!(
        free_query_buckets,
        vec![retained],
        "failed reservation must return the reusable multi-query query bucket to scratch"
    );
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// WHY: generated plans close grouping, identity, budget, ordering, and
/// aggregate-accounting variants without depending on implementation shape.
#[test]
fn generated_multi_query_plans_preserve_grouping_budget_and_identity_contracts() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for case_index in 0..768usize {
        let query_count = 1 + (next_u64(&mut state) as usize % 64);
        let mut graph_bytes_by_hash = [0_u64; 8];
        let mut queries = Vec::new();
        for index in 0..query_count {
            let graph_slot = (next_u64(&mut state) as usize % graph_bytes_by_hash.len()) + 1;
            let graph_upload_bytes = if graph_bytes_by_hash[graph_slot - 1] == 0 {
                128 + next_u64(&mut state) % 16_384
            } else {
                graph_bytes_by_hash[graph_slot - 1]
            };
            graph_bytes_by_hash[graph_slot - 1] = graph_upload_bytes;
            queries.push(query(
                index as u32,
                graph_slot as u64,
                1 + next_u64(&mut state) % 5,
                graph_upload_bytes,
                next_u64(&mut state) % 512,
                next_u64(&mut state) % 1_024,
                next_u64(&mut state) % 256,
            ));
        }

        let budget = graph_bytes_by_hash.iter().copied().sum::<u64>()
            + (query_count as u64 * 2_048)
            + 16_384;
        let plan = plan_multi_query_execution(&queries, budget)
            .expect("Fix: generated multi-query plan should fit generous budget");
        assert_eq!(
            plan.launch_count as usize,
            plan.groups.len(),
            "case {case_index}"
        );
        assert!(plan.final_only_host_fence_per_group, "case {case_index}");
        assert!(
            plan.groups
                .windows(2)
                .all(|pair| (pair[0].graph_layout_hash, pair[0].traversal_key)
                    <= (pair[1].graph_layout_hash, pair[1].traversal_key)),
            "case {case_index}"
        );
        let mut seen = vec![false; query_count];
        let mut avoided_launches = 0_u32;
        let mut avoided_host_fences = 0_u32;
        let mut peak_resident_bytes = 0_u64;
        for group in &plan.groups {
            assert!(group.resident_bytes <= budget, "case {case_index}");
            assert!(
                group.queries.windows(2).all(|pair| pair[0] <= pair[1]),
                "case {case_index}"
            );
            avoided_launches = avoided_launches
                .checked_add(group.avoided_launches)
                .expect("Fix: generated avoided launch sum should fit u32");
            avoided_host_fences = avoided_host_fences
                .checked_add(group.avoided_host_fences)
                .expect("Fix: generated avoided fence sum should fit u32");
            peak_resident_bytes = peak_resident_bytes.max(group.resident_bytes);
            for query in &group.queries {
                let slot = *query as usize;
                assert!(slot < query_count, "case {case_index}");
                assert!(!seen[slot], "case {case_index}");
                seen[slot] = true;
            }
        }
        assert!(seen.into_iter().all(|value| value), "case {case_index}");
        assert_eq!(plan.avoided_launches, avoided_launches, "case {case_index}");
        assert_eq!(
            plan.avoided_host_fences, avoided_host_fences,
            "case {case_index}"
        );
        assert_eq!(
            plan.peak_resident_bytes, peak_resident_bytes,
            "case {case_index}"
        );
    }
}

/// WHY: every rejected input class must fail at the planner boundary with
/// the exact typed error instead of producing a partial execution plan.
#[test]
fn multi_query_rejects_invalid_inputs_and_budget_overflow() {
    assert_eq!(
        plan_multi_query_execution(&[query(1, 0, 1, 8, 1, 1, 1)], 128)
            .expect_err("missing graph hash should fail"),
        MultiQueryExecutionError::ZeroGraphHash { query: 1 }
    );
    assert_eq!(
        plan_multi_query_execution(&[query(1, 1, 1, 0, 1, 1, 1)], 128)
            .expect_err("zero graph bytes should fail"),
        MultiQueryExecutionError::ZeroGraphUploadBytes { query: 1 }
    );
    assert_eq!(
        plan_multi_query_execution(
            &[query(1, 1, 1, 8, 1, 1, 1), query(2, 1, 2, 16, 1, 1, 1)],
            128,
        )
        .expect_err("same graph hash with conflicting bytes should fail"),
        MultiQueryExecutionError::GraphUploadBytesMismatch {
            graph_layout_hash: 1,
            expected_bytes: 8,
            actual_bytes: 16,
            query: 2,
        }
    );
    assert_eq!(
        plan_multi_query_execution(
            &[query(1, 1, 1, 8, 1, 1, 1), query(1, 1, 1, 8, 1, 1, 1)],
            128,
        )
        .expect_err("duplicate query should fail"),
        MultiQueryExecutionError::DuplicateQuery { query: 1 }
    );
    assert_eq!(
        plan_multi_query_execution(&[query(2, 1, 1, 128, 16, 16, 16)], 127)
            .expect_err("over-budget group should fail"),
        MultiQueryExecutionError::OverBudget {
            graph_layout_hash: 1,
            traversal_key: 1,
            required_bytes: 176,
            budget_bytes: 127,
        }
    );
}
