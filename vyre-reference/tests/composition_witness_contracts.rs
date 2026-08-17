//! Independent known-answer tests for composition witnesses.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::composition_witness::{
    adler32_chunk_witness, adler32_combine_chunks_witness, adler32_finalize_witness,
    adler32_witness, amg_solve_to_tolerance_witness_into,
    amg_solve_to_tolerance_witness_with_scratch_into, amg_v_cycle_witness,
    argmax_of_marginals_witness, argmin_cost_witness, backdoor_descendants_check_witness,
    bellman_shortest_path_witness, betti_persistence_witness, bhattacharyya_coefficient_witness,
    bigint_add_carry_witness, bigint_add_carry_witness_into, bitset_saturation_ratio_witness,
    canonicalize_union_find_witness, chebyshev_filter_witness, chebyshev_filter_witness_into,
    cluster_projection_matrix_witness_into, compose_ir_arrows_witness, compose_passes_witness,
    compose_passes_witness_into, composition_associates_witness, conformal_threshold_witness,
    conv1d_witness, conv1d_witness_into, count_sketch_query_witness, count_sketch_update_witness,
    crc32_witness, csr_backward_closure_witness, csr_backward_step_with_change_witness,
    csr_backward_traverse_witness, csr_backward_traverse_witness_into, csr_bfs_witness,
    csr_bidirectional_closure_witness, csr_bidirectional_closure_witness_into,
    csr_bidirectional_step_witness, csr_bidirectional_step_witness_into,
    csr_closure_with_step_hook_witness, csr_forward_or_changed_closure_with_step_hook_witness,
    csr_forward_or_changed_closure_witness, csr_forward_or_changed_witness,
    csr_forward_or_changed_witness_into, csr_forward_step_with_change_witness,
    csr_forward_traverse_witness, csr_forward_traverse_witness_into,
    csr_frontier_degree_sum_witness, csr_persistent_closure_detailed_witness,
    csr_persistent_closure_witness, csr_persistent_closure_witness_with_scratch_into,
    csr_queue_split_low_forward_witness, csr_queue_strided_forward_witness, ddnnf_evaluate_witness,
    dedup_regions_witness, dense_bitmatrix_step_witness, dense_bitmatrix_step_witness_into,
    dense_boolean_matvec_witness, dense_reachability_bitsets_witness, dense_scc_components_witness,
    differentiable_argmax_witness, differentiable_argmax_witness_into,
    differentiable_autotune_gradient_witness_into,
    differentiable_autotune_pick_config_witness_into, do_intervention_delete_incoming_witness_into,
    dominator_frontier_witness, dominator_idoms_witness, dominator_sets_idoms_witness,
    dominator_tree_witness, dp_clip_per_sample_witness, dp_clip_per_sample_witness_into,
    evaluate_condition_witness, evaluate_formula_witness, exploded_ifds_csr_witness,
    fisher_rao_distance_witness, fnv1a32_witness, fnv1a64_witness, fractional_derivative_witness,
    frontier_absorb_witness, frontier_domain_popcount_witness, frontier_popcount_witness,
    frontier_step_sharded_witness, frontier_to_queue_witness, frontier_to_queue_witness_into,
    functor_apply_witness, functor_apply_witness_into, fusion_affinity_witness,
    fusion_affinity_witness_into, gaussian_rdp_step_witness, grunwald_letnikov_kernel_witness,
    homotopy_euler_predictor_witness, hypervector_majority_bundle_witness,
    hypervector_xor_bind_witness, i4x8_batched_matmul_f32_scaled_witness,
    i4x8_batched_matmul_top1_f32_scaled_witness, i4x8_batched_matvec_f32_scaled_witness,
    i4x8_dot_f32_scaled_witness, i4x8_dot_i32_witness, i4x8_matvec_f32_scaled_witness,
    identity_arrow_witness, identity_functor_witness, identity_functor_witness_into,
    identity_matrix_witness_into, idoms_to_dominator_sets_witness, iht_top_k_witness,
    interval_merge_witness, jacobi_solve_to_tolerance_witness_into, kernel_to_fixed_16_16_witness,
    kernel_to_fixed_16_16_witness_into, kfac_block_inverse_witness, knn_csr_witness,
    l2p_zeroth_all_witness, launch_dominance_witness, linear_homotopy_witness,
    m2l_zeroth_all_witness, matmul_u32_witness, matroid_exchange_bfs_step_witness,
    matroid_intersection_augmentation_witness, matroid_select_optimal_subset_witness,
    matroid_select_optimal_subset_witness_into, merge_frontier_out_witness_into,
    modified_gram_schmidt_witness, mori_zwanzig_coarsen_via_clustering_witness_into,
    mori_zwanzig_project_witness, motif_witness, mp_edge_clip_witness, mp_edge_clip_witness_into,
    natural_gradient_autotune_step_witness_into, natural_gradient_block_apply_witness,
    natural_gradient_block_apply_witness_into, negative_truncator_coeffs_witness,
    negative_truncator_coeffs_witness_into, newton_schulz_inverse_sqrt_witness,
    newton_schulz_inverse_sqrt_witness_into, ntt_forward_witness, ntt_inverse_witness,
    pack_i4x8_witness, partition_frontier_by_vertex_witness_into, passes_commute_on_witness,
    path_reconstruct_witness, persistent_fixpoint_witness, prefix_scan_witness,
    prefix_scan_witness_into, privacy_epsilon_from_rdp_witness, qsvt_apply_witness,
    qsvt_apply_witness_into, qsvt_block_encode_witness, qsvt_block_encode_witness_into,
    rdp_to_dp_witness, reachable_witness, reduce_max_f32_witness, reduce_sum_f32_witness,
    region_of_witness, resolve_bigint_carry_chain_witness, resolve_bigint_carry_chain_witness_into,
    resolve_family_witness, scale_aware_pressure_witness, scallop_join_fixpoint_witness,
    scc_decompose_witness, schedule_via_homotopy_witness, schedule_via_scale_aware_samples_witness,
    select_retention_set_witness, select_retention_set_witness_into, semiring_gemm_witness,
    sheaf_diffusion_equilibrium_witness_into, sheaf_diffusion_step_witness,
    sheaf_diffusion_step_witness_into, sheaf_dominant_spectrum_witness_into,
    sheaf_fusion_incompatible_witness, sheaf_fusion_incompatible_witness_into,
    sheaf_spectral_gap_witness_into, sheaf_suggested_cluster_count_witness,
    should_fuse_chain_witness, simplicial_triangle_message_witness, sinkhorn_clustering_witness,
    sinkhorn_col_residual_witness, sinkhorn_iterate_f64_witness, sinkhorn_iterate_witness,
    sinkhorn_row_residual_witness, softmax_witness, softmax_witness_into,
    sos_gram_construct_witness, sos_gram_construct_witness_into, stochastic_decode_witness,
    stochastic_encode_witness, stream_compact_witness, stream_compact_witness_into,
    sum_product_evaluate_witness, tensor_scc_witness, tensor_train_contract_step_witness,
    tensor_train_contract_step_witness_into, tensor_train_fusion_pressure_witness,
    tensor_train_fusion_pressure_witness_with_scratch, toposort_csr_into_witness,
    toposort_csr_with_scratch_into_witness, toposort_csr_witness, toposort_witness,
    try_amg_solve_to_tolerance_witness_into, try_amg_solve_to_tolerance_witness_with_scratch_into,
    try_amg_v_cycle_witness_with_scratch_into, try_argmin_cost_witness,
    try_cluster_projection_matrix_witness_into, try_count_sketch_query_into_witness,
    try_ddnnf_evaluate_witness, try_differentiable_autotune_gradient_witness_into,
    try_exploded_ifds_csr_witness_into, try_fractional_derivative_witness_into,
    try_frontier_absorb_witness_into, try_gaussian_rdp_step_witness_into,
    try_grunwald_letnikov_kernel_witness_into, try_identity_matrix_witness_into,
    try_jacobi_solve_to_tolerance_witness_into, try_kernel_to_fixed_16_16_witness_into,
    try_l2p_zeroth_all_witness_into, try_m2l_zeroth_all_witness_into,
    try_match_post_process_witness, try_p2m_zeroth_moment_witness_into,
    try_sinkhorn_iterate_f64_witness_into, try_sinkhorn_iterate_witness,
    try_stochastic_encode_witness_into, try_tensor_flow_forward_witness_into,
    try_tensor_train_contract_step_witness, try_tensor_train_full_chain_witness_into,
    union_find_alias_witness, unpack_i4x8_witness, vector_graph_traverse_from_seed_witness,
    vector_top_k_witness, vietoris_rips_edge_filter_witness, vietoris_rips_edges_witness,
    vsa_fingerprint_witness, AmgSolveScratchWitness, ExplodedIfdsScratchWitness,
    MegakernelScaleSampleWitness, NewtonSchulzScratchWitness, RuleConditionWitness,
    RuleEvaluationContextWitness, RuleFormulaWitness,
};
use vyre_reference::{reference_eval, value::Value};
use vyre_spec::Semiring;

#[test]
fn prefix_scan_witness_known_answers() {
    let input = vec![1, 2, 3, 4, 5];

    // Inclusive sum: [1, 3, 6, 10, 15]
    let inc = prefix_scan_witness(&input, true, |a, b| a + b, 0);
    assert_eq!(inc, vec![1, 3, 6, 10, 15]);

    // Exclusive sum: [0, 1, 3, 6, 10]
    let exc = prefix_scan_witness(&input, false, |a, b| a + b, 0);
    assert_eq!(exc, vec![0, 1, 3, 6, 10]);

    // Prefix max: [3, 3, 5, 5, 8]
    let input2 = vec![3, 1, 5, 2, 8];
    let pmax = prefix_scan_witness(&input2, true, u32::max, 0);
    assert_eq!(pmax, vec![3, 3, 5, 5, 8]);
}

#[test]
fn semiring_gemm_witness_known_answers() {
    // 2x2 identity matrix times 2x2 matrix under Real
    let eye = vec![1, 0, 0, 1];
    let m = vec![3, 7, 2, 5];
    let prod = semiring_gemm_witness(&eye, &m, 2, 2, 2, Semiring::Real);
    assert_eq!(prod, vec![3, 7, 2, 5]);

    // Boolean Or-And matrix multiplication (transitive reachability step)
    let a = vec![1, 1, 0, 1];
    let b = vec![0, 1, 1, 0];
    let bool_prod = semiring_gemm_witness(&a, &b, 2, 2, 2, Semiring::BoolOr);
    // [ (1&0)|(1&1)=1, (1&1)|(1&0)=1 ]
    // [ (0&0)|(1&1)=1, (0&1)|(1&0)=0 ]
    assert_eq!(bool_prod, vec![1, 1, 1, 0]);

    // MinPlus (Tropical / Shortest path step)
    let dist_a = vec![0, 3, 7, 0];
    let dist_b = vec![0, 5, 2, 0];
    let min_plus = semiring_gemm_witness(&dist_a, &dist_b, 2, 2, 2, Semiring::MinPlus);
    // c[0][0] = min(0+0, 3+2) = 0
    // c[0][1] = min(0+5, 3+0) = 3
    // c[1][0] = min(7+0, 0+2) = 2
    // c[1][1] = min(7+5, 0+0) = 0
    assert_eq!(min_plus, vec![0, 3, 2, 0]);

    // Max-plus accumulates with max; max-times shares max accumulation but multiplies terms.
    let max_plus = semiring_gemm_witness(&[2, 5], &[3, 4], 1, 1, 2, Semiring::MaxPlus);
    let max_times = semiring_gemm_witness(&[2, 5], &[3, 4], 1, 1, 2, Semiring::MaxTimes);
    assert_eq!(max_plus, vec![9]);
    assert_eq!(max_times, vec![20]);
}

#[test]
fn csr_bfs_witness_known_graph_topologies() {
    // Triangle graph: 0 -> 1 -> 2 -> 0
    let row_offsets = vec![0, 1, 2, 3];
    let col_indices = vec![1, 2, 0];

    let dists = csr_bfs_witness(3, &row_offsets, &col_indices, 0);
    assert_eq!(dists, vec![0, 1, 2]);

    // Disconnected graph: node 0 -> 1, node 2 isolated
    let row_offsets_disc = vec![0, 1, 1, 1];
    let col_indices_disc = vec![1];
    let dists_disc = csr_bfs_witness(3, &row_offsets_disc, &col_indices_disc, 0);
    assert_eq!(dists_disc, vec![0, 1, u32::MAX]);
}

#[test]
fn interpreter_matches_independent_witness_on_matrix_vector() {
    // Program computing a 2x2 matrix-vector product
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("mat", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("vec", 1, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::output("out", 2, DataType::U32).with_count(2),
        ],
        [1, 1, 1],
        vec![
            // out[0] = mat[0]*vec[0] + mat[1]*vec[1]
            Node::store(
                "out",
                Expr::u32(0),
                Expr::add(
                    Expr::mul(
                        Expr::load("mat", Expr::u32(0)),
                        Expr::load("vec", Expr::u32(0)),
                    ),
                    Expr::mul(
                        Expr::load("mat", Expr::u32(1)),
                        Expr::load("vec", Expr::u32(1)),
                    ),
                ),
            ),
            // out[1] = mat[2]*vec[0] + mat[3]*vec[1]
            Node::store(
                "out",
                Expr::u32(1),
                Expr::add(
                    Expr::mul(
                        Expr::load("mat", Expr::u32(2)),
                        Expr::load("vec", Expr::u32(0)),
                    ),
                    Expr::mul(
                        Expr::load("mat", Expr::u32(3)),
                        Expr::load("vec", Expr::u32(1)),
                    ),
                ),
            ),
        ],
    );

    let mat = vec![2u32, 3, 4, 5];
    let v = vec![10u32, 20];

    let outputs = reference_eval(
        &program,
        &[
            Value::Bytes(bytemuck_slice(&mat)),
            Value::Bytes(bytemuck_slice(&v)),
        ],
    )
    .expect("reference evaluation must succeed");

    // Independent witness calculation:
    let witness = semiring_gemm_witness(&mat, &v, 2, 1, 2, Semiring::Real);
    // [ 2*10 + 3*20 = 80, 4*10 + 5*20 = 140 ]
    assert_eq!(witness, vec![80, 140]);

    let out_bytes = outputs[0].to_bytes();
    let mut out_u32s = vec![0u32; 2];
    out_u32s[0] = u32::from_le_bytes(out_bytes[0..4].try_into().unwrap());
    out_u32s[1] = u32::from_le_bytes(out_bytes[4..8].try_into().unwrap());

    assert_eq!(out_u32s, witness);
}

fn bytemuck_slice(u32s: &[u32]) -> std::sync::Arc<[u8]> {
    let mut bytes = Vec::with_capacity(u32s.len() * 4);
    for &x in u32s {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.into()
}
#[test]
fn csr_traversal_forward_backward_bidirectional_contracts() {
    // Graph: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let row_offsets = [0, 2, 3, 4, 4];
    let col_indices = [1, 2, 3, 3];
    let edge_kind_mask = [1, 2, 1, 2];

    // Forward from node 0 with allow_mask = 1 (only 0->1 followed)
    let fwd =
        csr_forward_traverse_witness(4, &row_offsets, &col_indices, &edge_kind_mask, &[0b0001], 1);
    assert_eq!(fwd, vec![0b0010]);

    // Backward from node 3 with allow_mask = 3 (reaches predecessors 1 and 2)
    let bwd =
        csr_backward_traverse_witness(4, &row_offsets, &col_indices, &edge_kind_mask, &[0b1000], 3);
    assert_eq!(bwd, vec![0b0110]);

    // Bidirectional step from node 1 with allow_mask = 3
    // Forward from 1 reaches 3; Backward from 1 reaches 0
    let bidi = csr_bidirectional_step_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0010],
        3,
    );
    assert_eq!(bidi, vec![0b1001]);

    // Forward step with change
    let (fwd_ch, ch) = csr_forward_step_with_change_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        3,
    );
    assert_eq!(fwd_ch, vec![0b1111]);
    assert_eq!(ch, 1);
}

#[test]
fn csr_closure_and_fixpoint_contracts() {
    // Linear chain: 0 -> 1 -> 2 -> 3
    let row_offsets = [0, 1, 2, 3, 3];
    let col_indices = [1, 2, 3];
    let edge_kind_mask = [1, 1, 1];

    let mut hook_calls = 0;
    let closed = csr_closure_with_step_hook_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        1,
        10,
        &[0b0001],
        |_| hook_calls += 1,
    );
    assert_eq!(closed, vec![0b1111]);
    assert_eq!(hook_calls, 3);

    // Persistent closure detailed
    let detailed = csr_persistent_closure_detailed_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
    );
    assert_eq!(detailed.frontier, vec![0b1111]);
    assert_eq!(detailed.changed, 1);
    assert!(detailed.converged);
    assert_eq!(detailed.stop_iteration, 4);
    assert_eq!(detailed.active_per_iteration, vec![2, 3, 4, 4]);
    assert_eq!(detailed.active_density, vec![2, 3, 4, 4, 4, 4, 4, 4, 4, 4]);

    // Persistent closure with scratch into
    let mut frontier_scratch = Vec::new();
    let mut step_scratch = Vec::new();
    let changed_scratch = csr_persistent_closure_witness_with_scratch_into(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
        &mut frontier_scratch,
        &mut step_scratch,
    );
    assert_eq!(frontier_scratch, vec![0b1111]);
    assert_eq!(changed_scratch, 1);
    assert_eq!(step_scratch, vec![0]);

    // Persistent closure compact
    let (frontier, changed) = csr_persistent_closure_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
    );
    assert_eq!(frontier, vec![0b1111]);
    assert_eq!(changed, 1);
    // Single in-place forward step with changed reporting
    let mut fwd_ch_into = Vec::new();
    let ch_into_flag = csr_forward_or_changed_witness_into(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        &mut fwd_ch_into,
    );
    assert_eq!(fwd_ch_into, vec![0b1111]);
    assert_eq!(ch_into_flag, 1);
    // Single in-place forward step with changed reporting
    let (fwd_ch_out, ch_flag) = csr_forward_or_changed_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
    );
    assert_eq!(fwd_ch_out, vec![0b1111]);
    assert_eq!(ch_flag, 1);

    // Persistent fixpoint helper
    let (fp, iters) = persistent_fixpoint_witness(&[1, 2, 3], 5, |curr| {
        curr.iter().map(|&x| x.saturating_add(1).min(3)).collect()
    });
    assert_eq!(fp, vec![3, 3, 3]);
    assert_eq!(iters, 3);

    // Forward or changed closure
    let mut step_count = 0;
    let res = csr_forward_or_changed_closure_with_step_hook_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
        |_| step_count += 1,
    );
    assert_eq!(res, vec![0b1111]);
    assert_eq!(step_count, 2);

    let res_no_hook = csr_forward_or_changed_closure_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
    );
    assert_eq!(res_no_hook, vec![0b1111]);
}

#[test]
fn csr_bidirectional_step_and_closure_scratch_reuse_and_validation_contracts() {
    // Chain: 0 -> 1 -> 2 -> 3
    let row_offsets = [0, 1, 2, 3, 3];
    let col_indices = [1, 2, 3];
    let edge_kind_mask = [1, 1, 1];

    // Pre-allocated step buffer
    let mut step_out = Vec::with_capacity(16);
    let step_ptr = step_out.as_ptr();
    csr_bidirectional_step_witness_into(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0010], // node 1
        1,
        &mut step_out,
    );
    // Forward from 1 -> 2; Backward from 1 -> 0
    assert_eq!(step_out, vec![0b0101]);
    assert_eq!(
        step_out.as_ptr(),
        step_ptr,
        "step_out pointer preserved (zero reallocations)"
    );
    assert!(step_out.capacity() >= 16, "capacity preserved");

    // Multi-iteration bidirectional closure with caller scratch reuse
    let mut current = Vec::with_capacity(32);
    let mut next = Vec::with_capacity(32);
    let current_ptr = current.as_ptr();
    let next_ptr = next.as_ptr();

    csr_bidirectional_closure_witness_into(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001], // seed at node 0
        1,
        10,
        &mut current,
        &mut next,
    );
    assert_eq!(current, vec![0b1111]); // Full transitive closure
    assert_eq!(
        current.as_ptr(),
        current_ptr,
        "current buffer capacity reused across iterations"
    );
    assert_eq!(
        next.as_ptr(),
        next_ptr,
        "next scratch buffer capacity reused across iterations"
    );

    // Owned variant matches into variant
    let owned = csr_bidirectional_closure_witness(
        4,
        &row_offsets,
        &col_indices,
        &edge_kind_mask,
        &[0b0001],
        1,
        10,
    );
    assert_eq!(owned, current);

    // Malformed CSR metadata is rejected before either caller-owned buffer changes.
    let mut guarded_current = vec![0xAAAA_AAAA];
    let mut guarded_next = vec![0x5555_5555];
    let current_before = guarded_current.clone();
    let next_before = guarded_next.clone();
    let malformed_offsets = [0, 2, 1, 3, 3];
    let malformed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        csr_bidirectional_closure_witness_into(
            4,
            &malformed_offsets,
            &col_indices,
            &edge_kind_mask,
            &[0b0001],
            1,
            10,
            &mut guarded_current,
            &mut guarded_next,
        );
    }));
    assert!(
        malformed.is_err(),
        "non-monotonic CSR offsets must be rejected"
    );
    assert_eq!(guarded_current, current_before);
    assert_eq!(guarded_next, next_before);

    let oversized_seed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        csr_bidirectional_closure_witness_into(
            4,
            &row_offsets,
            &col_indices,
            &edge_kind_mask,
            &[0b0001, 0],
            1,
            10,
            &mut guarded_current,
            &mut guarded_next,
        );
    }));
    assert!(
        oversized_seed.is_err(),
        "frontier words outside the node domain must be rejected"
    );
    assert_eq!(guarded_current, current_before);
    assert_eq!(guarded_next, next_before);
}

#[test]
fn frontier_queue_and_degree_sum_contracts() {
    let frontier = [0b1011_0101]; // nodes 0, 2, 4, 5, 7
    let (queue, active) = frontier_to_queue_witness(&frontier, 8, 3);
    assert_eq!(queue, vec![0, 2, 4]);
    assert_eq!(active, 5);

    // Degree sum over frontier
    let row_offsets = [0, 2, 3, 5, 6, 8, 9, 10, 12];
    let sum = csr_frontier_degree_sum_witness(&frontier, &row_offsets, 8);
    // node 0: 2, node 2: 2, node 4: 2, node 5: 1, node 7: 2 => 9
    assert_eq!(sum, 9);

    // Resolve family
    let tags = [0x01, 0x02, 0x01, 0x04];
    let fam = resolve_family_witness(&tags, 0x01);
    assert_eq!(fam, vec![0b0101]);
}

#[test]
fn csr_frontier_shard_contracts() {
    let frontier = [0b1011_0101u32]; // nodes 0, 2, 4, 5, 7 in an 8-node graph
    let mut guarded_shards = vec![vec![0xDEAD_BEEFu32; 4]; 2];
    let guarded_shards_before = guarded_shards.clone();

    // Zero shard count must fail closed with validate-before-mutation preserved
    let zero_shards =
        partition_frontier_by_vertex_witness_into(&frontier, 8, 0, &mut guarded_shards);
    assert!(zero_shards.is_err(), "zero shard count must fail closed");
    assert_eq!(
        guarded_shards, guarded_shards_before,
        "output must be untouched on error"
    );

    // Mis-sized frontier must fail closed
    let bad_size = partition_frontier_by_vertex_witness_into(&[0u32; 5], 8, 2, &mut guarded_shards);
    assert!(bad_size.is_err(), "mis-sized frontier must fail closed");
    assert_eq!(
        guarded_shards, guarded_shards_before,
        "output must be untouched on error"
    );

    // Valid 2-shard partition of 8 nodes:
    // shard 0 owns nodes 0..4 (indices 0, 1, 2, 3) -> nodes 0, 2 active -> 0b0101
    // shard 1 owns nodes 4..8 (indices 4, 5, 6, 7) -> nodes 4, 5, 7 active -> 0b1011_0000 = 0xB0
    partition_frontier_by_vertex_witness_into(&frontier, 8, 2, &mut guarded_shards)
        .expect("2-shard partition must succeed");
    assert_eq!(guarded_shards.len(), 2);
    assert_eq!(guarded_shards[0], vec![0b0000_0101u32]);
    assert_eq!(guarded_shards[1], vec![0b1011_0000u32]);

    // Merge roundtrip
    let mut merged = Vec::new();
    merge_frontier_out_witness_into(&guarded_shards, 1, &mut merged).expect("merge must succeed");
    assert_eq!(
        merged,
        vec![frontier[0]],
        "merge must exactly reconstruct the frontier"
    );

    // Merge order independence
    let reversed_shards = vec![guarded_shards[1].clone(), guarded_shards[0].clone()];
    let mut merged_reversed = Vec::new();
    merge_frontier_out_witness_into(&reversed_shards, 1, &mut merged_reversed)
        .expect("reversed merge");
    assert_eq!(
        merged, merged_reversed,
        "merge must be shard-order independent"
    );

    // Full sharded step expansion test against a single-device oracle
    let edge_offsets = [0, 2, 3, 5, 6, 8, 9, 10, 12];
    let edge_targets = [1, 2, 3, 0, 3, 0, 1, 2, 4, 5, 6, 7];
    let single_device_expand = |input: &[u32]| -> Vec<u32> {
        let mut out = vec![0u32; 1];
        for v in 0..8u32 {
            if input[(v >> 5) as usize] & (1 << (v & 31)) != 0 {
                let lo = edge_offsets[v as usize] as usize;
                let hi = edge_offsets[v as usize + 1] as usize;
                for &dst in &edge_targets[lo..hi] {
                    if dst < 8 {
                        out[(dst >> 5) as usize] |= 1 << (dst & 31);
                    }
                }
            }
        }
        out
    };

    let single_result = single_device_expand(&frontier);
    for shard_count in 1..=4usize {
        let sharded_result =
            frontier_step_sharded_witness(&frontier, 8, shard_count, |_, masked| {
                Ok(single_device_expand(masked))
            })
            .expect("sharded expansion must succeed");
        assert_eq!(
            sharded_result, single_result,
            "sharded expansion must equal single-device expansion for shard_count={shard_count}"
        );
    }
}

#[test]
fn csr_queue_strided_and_split_contracts() {
    let edge_offsets = [0, 4, 5, 6, 6];
    let edge_targets = [1, 2, 3, 0, 3, 0];
    let edge_kinds = [1, 1, 1, 1, 1, 1];

    // Strided forward from queue [0, 1]
    let strided = csr_queue_strided_forward_witness(
        &[0, 1],
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kinds,
        4,
        1,
    );
    assert_eq!(strided, vec![0b1111]);

    // Split low forward with high threshold = 3
    // Node 0 has degree 4 (high), Node 1 has degree 1 (low)
    let (fout, high_q, high_count) = csr_queue_split_low_forward_witness(
        &[0, 1],
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kinds,
        &[0],
        4,
        1,
        3,
        1,
    );
    // Node 1 expands 1->3 into fout; Node 0 goes to high_q
    assert_eq!(fout, vec![0b1000]);
    assert_eq!(high_q, vec![0]);
    assert_eq!(high_count, 1);
}

#[test]
fn dense_boolean_matvec_contract() {
    let frontier = [0x01]; // byte 0 = 1
    let mut lut = vec![0u32; 256 * 2];
    lut[1 * 2] = 0xAA;
    lut[1 * 2 + 1] = 0x55;

    let out = dense_boolean_matvec_witness(&frontier, &lut, 1, 2);
    assert_eq!(out, vec![0xAA, 0x55]);
}

#[test]
fn dominator_witnesses_contracts() {
    // Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let edges = [(0, 1), (0, 2), (1, 3), (2, 3)];

    let idoms_u32 = dominator_tree_witness(4, 0, &edges);
    assert_eq!(idoms_u32, vec![0, 0, 0, 0]);

    let idoms_opt = dominator_idoms_witness(4, 0, &edges);
    assert_eq!(idoms_opt, vec![Some(0), Some(0), Some(0), Some(0)]);

    let idoms_sets = dominator_sets_idoms_witness(4, 0, &edges);
    assert_eq!(idoms_sets, vec![Some(0), Some(0), Some(0), Some(0)]);

    let sets = idoms_to_dominator_sets_witness(&idoms_opt, 4);
    assert_eq!(sets[3], vec![0, 3]);

    // Unreachable root case
    let unreach_idoms = dominator_idoms_witness(4, 10, &edges);
    assert_eq!(unreach_idoms, vec![None; 4]);
    let unreach_sets = dominator_sets_idoms_witness(4, 10, &edges);
    assert_eq!(unreach_sets, vec![None; 4]);

    // Dominator frontier:
    // Dom closure: 0 dominates {0,1,2,3}, 1 dominates {1}, 2 dominates {2}, 3 dominates {3}
    let dom_offsets = [0, 4, 5, 6, 7];
    let dom_targets = [0, 1, 2, 3, 1, 2, 3];
    let pred_offsets = [0, 0, 1, 2, 4];
    let pred_targets = [0, 0, 1, 2];
    let df_1 = dominator_frontier_witness(
        4,
        &dom_offsets,
        &dom_targets,
        &pred_offsets,
        &pred_targets,
        &[0b0010],
    );
    // DF(1) = {3}
    assert_eq!(df_1, vec![0b1000]);
}

#[test]
fn betti_persistence_contract() {
    // Triangle graph: 3 vertices, 3 edges => b0 = 1, b1 = 1, edges = 3
    let mask = [0, 1, 1, 1, 0, 1, 1, 1, 0];
    let (b0, b1, edges) = betti_persistence_witness(&mask, 3);
    assert_eq!((b0, b1, edges), (1, 1, 3));

    // Empty graph
    assert_eq!(betti_persistence_witness(&[], 0), (0, 0, 0));
}

#[test]
fn matroid_witnesses_contracts() {
    let exchange_adj = [0, 1, 0, 0];
    let sources = [1, 0];
    let sinks = [0, 1];
    let set_x = [0, 0];

    let aug = matroid_intersection_augmentation_witness(&exchange_adj, &sources, &sinks, &set_x, 2);
    assert_eq!(aug, vec![1, 1]);

    let opt = matroid_select_optimal_subset_witness(&exchange_adj, &sources, &sinks, &set_x, 2, 5)
        .unwrap();
    assert_eq!(opt, vec![1, 1]);

    let (bfs_step, changed) = matroid_exchange_bfs_step_witness(&[1, 0], &exchange_adj, &[0, 0], 2);
    assert_eq!(bfs_step, vec![0, 1]);
    assert!(changed);
}

#[test]
fn motif_and_path_reconstruct_contracts() {
    let edge_offsets = [0, 2, 3, 3];
    let edge_targets = [1, 2, 2];
    let edge_kinds = [1, 2, 1];

    // Motif: 0 -(1)-> 1 and 1 -(1)-> 2
    let motif = [(0, 1, 1), (1, 1, 2)];
    let res = motif_witness(3, &edge_offsets, &edge_targets, &edge_kinds, &motif);
    assert_eq!(res, vec![1, 1, 1]);

    // Path reconstruct: 3 -> 2 -> 1 -> 0 -> 0 (root)
    let parents = [0, 0, 1, 2];
    let (path, len) = path_reconstruct_witness(&parents, 3, 6);
    assert_eq!(len, 4);
    assert_eq!(path, vec![3, 2, 1, 0, 0, 0]);
}

#[test]
fn exploded_ifds_csr_contract() {
    let (offsets, targets) =
        exploded_ifds_csr_witness(1, 2, 2, &[(0, 0, 1)], &[], &[(0, 0, 1)], &[(0, 0, 1)]);
    // 1 proc, 2 blocks, 2 facts => 4 nodes: (0,0,0)=0, (0,0,1)=1, (0,1,0)=2, (0,1,1)=3
    // Fact 0 survives (not killed): 0 -> 2
    // Fact 1 is killed: no 1 -> 3
    // GEN fact 1 at block 0: 0 -> 3
    assert_eq!(offsets.len(), 5);
    assert_eq!(offsets[0], 0);
    assert_eq!(targets, vec![2, 3]);

    let mut row_scratch = Vec::new();
    let mut col_scratch = Vec::new();
    let mut ifds_scratch = ExplodedIfdsScratchWitness::new();
    try_exploded_ifds_csr_witness_into(
        1,
        2,
        2,
        &[(0, 0, 1)],
        &[],
        &[(0, 0, 1)],
        &[(0, 0, 1)],
        &mut row_scratch,
        &mut col_scratch,
        &mut ifds_scratch,
    )
    .unwrap();
    assert_eq!(row_scratch, offsets);
    assert_eq!(col_scratch, targets);
}

#[test]
fn scc_and_reachability_contracts() {
    // 2-cycle between 0 and 1, isolated 2
    let adj = [0, 1, 0, 1, 0, 0, 0, 0, 0];
    let sccs = dense_scc_components_witness(&adj, 3);
    assert_eq!(sccs[0], sccs[1]);
    assert_eq!(sccs[2], 2);

    let (fwd, bwd) = dense_reachability_bitsets_witness(&adj, 0, 3);
    assert_eq!(fwd, vec![0b0011]);
    assert_eq!(bwd, vec![0b0011]);

    let stamped = scc_decompose_witness(3, &fwd, &bwd, &[u32::MAX; 3], 0);
    assert_eq!(stamped, vec![0, 0, u32::MAX]);
}

#[test]
fn union_find_and_functor_and_sheaf_contracts() {
    let parent_init = [0, 1, 2, 3];
    let roots = union_find_alias_witness(&parent_init, &[0, 2], &[1, 3]);
    assert_eq!(roots, vec![0, 0, 2, 2]);
    assert_eq!(
        canonicalize_union_find_witness(&[0, 0, 2, 2]),
        vec![0, 0, 2, 2]
    );

    // Functor apply
    let mapped = functor_apply_witness(&[10, 20, 30], &[2, 0, 1], 4);
    assert_eq!(mapped, vec![20, 30, 10, 0]);
    let mut mapped_out = Vec::with_capacity(4);
    let mapped_ptr = mapped_out.as_ptr();
    functor_apply_witness_into(&[10, 20, 30], &[2, 0, 1], 4, &mut mapped_out);
    assert_eq!(mapped_out, mapped);
    assert_eq!(mapped_out.as_ptr(), mapped_ptr);

    // Pass composition & identity
    let id = identity_functor_witness(3);
    assert_eq!(id, vec![0, 1, 2]);
    let mut id_out = Vec::new();
    identity_functor_witness_into(3, &mut id_out);
    assert_eq!(id_out, id);

    let view = [10, 20, 30];
    let g = [2, 0, 1];
    let f = [1, 2, 0];
    let composed = compose_passes_witness(&view, &g, 3, &f, 3);
    assert_eq!(composed, vec![10, 20, 30]);
    let mut comb_buf = Vec::new();
    let mut comp_out = Vec::new();
    compose_passes_witness_into(&view, &g, 3, &f, 3, &mut comb_buf, &mut comp_out);
    assert_eq!(comp_out, composed);

    assert!(passes_commute_on_witness(
        &view, &id, 3, &id, &id, 3, &id, 3
    ));
    // Sheaf diffusion
    let stalks = [1.0, 2.0];
    let diag = [0.5, 0.25];
    let diffused = sheaf_diffusion_step_witness(&stalks, &diag, 0.1);
    assert!((diffused[0] - (1.0 - 0.1 * 0.5 * 1.0)).abs() < 1e-6);
    assert!((diffused[1] - (2.0 - 0.1 * 0.25 * 2.0)).abs() < 1e-6);
    let mut step_out = Vec::with_capacity(2);
    let step_ptr = step_out.as_ptr();
    sheaf_diffusion_step_witness_into(&stalks, &diag, 0.1, &mut step_out);
    assert_eq!(step_out, diffused);
    assert_eq!(step_out.as_ptr(), step_ptr);

    let mut equilibrium = Vec::with_capacity(2);
    let mut equilibrium_scratch = Vec::with_capacity(2);
    let iterations = sheaf_diffusion_equilibrium_witness_into(
        &stalks,
        &diag,
        0.1,
        f64::INFINITY,
        4,
        &mut equilibrium,
        &mut equilibrium_scratch,
    );
    assert_eq!(iterations, 1);
    assert_eq!(equilibrium, diffused);

    assert_eq!(
        sheaf_fusion_incompatible_witness(&stalks, &diffused, 0.02),
        vec![1, 1]
    );
    let mut flags = vec![9, 9, 9];
    sheaf_fusion_incompatible_witness_into(&stalks, &diffused, 1.0, &mut flags);
    assert_eq!(flags, vec![0, 0]);

    // Backdoor check
    assert!(backdoor_descendants_check_witness(&[1, 0], &[1, 0]));
    assert!(!backdoor_descendants_check_witness(&[1, 0], &[0, 1]));
}

struct TestRuleCtx;
impl RuleEvaluationContextWitness for TestRuleCtx {
    fn pattern_count(&self, pattern_id: u32) -> u32 {
        if pattern_id == 7 {
            5
        } else {
            0
        }
    }
    fn file_size(&self) -> u64 {
        1024
    }
    fn field_value(&self, name: &str) -> Option<&str> {
        if name == "path" {
            Some("src/lib.rs")
        } else {
            None
        }
    }
}

#[test]
fn rule_formula_and_condition_witness_contracts() {
    let ctx = TestRuleCtx;
    assert!(evaluate_condition_witness(
        &RuleConditionWitness::LiteralTrue,
        &ctx
    ));
    assert!(!evaluate_condition_witness(
        &RuleConditionWitness::LiteralFalse,
        &ctx
    ));
    assert!(evaluate_condition_witness(
        &RuleConditionWitness::PatternExists { pattern_id: 7 },
        &ctx
    ));
    assert!(!evaluate_condition_witness(
        &RuleConditionWitness::PatternExists { pattern_id: 8 },
        &ctx
    ));
    assert!(evaluate_condition_witness(
        &RuleConditionWitness::PatternCountGt {
            pattern_id: 7,
            threshold: 4
        },
        &ctx
    ));
    assert!(!evaluate_condition_witness(
        &RuleConditionWitness::PatternCountGt {
            pattern_id: 7,
            threshold: 5
        },
        &ctx
    ));
    assert!(evaluate_condition_witness(
        &RuleConditionWitness::FileSizeLt(2048),
        &ctx
    ));
    assert!(!evaluate_condition_witness(
        &RuleConditionWitness::FileSizeLt(1024),
        &ctx
    ));
    assert!(evaluate_condition_witness(
        &RuleConditionWitness::SubstringMatch {
            haystack: "path".into(),
            needle: "lib".into()
        },
        &ctx
    ));

    let f = RuleFormulaWitness::And(
        Box::new(RuleFormulaWitness::Condition(
            RuleConditionWitness::PatternCountGte {
                pattern_id: 7,
                threshold: 3,
            },
        )),
        Box::new(RuleFormulaWitness::Condition(
            RuleConditionWitness::FileSizeLt(2048),
        )),
    );
    assert!(evaluate_formula_witness(&f, &ctx));

    let f_false = RuleFormulaWitness::Not(Box::new(f));
    assert!(!evaluate_formula_witness(&f_false, &ctx));
}

#[test]
fn ddnnf_evaluate_contract() {
    // Simple circuit: (X0 AND (NOT X1))
    // Nodes:
    // 0: Literal true X0 (kind=1, var=0)
    // 1: Literal false X1 (kind=2, var=1)
    // 2: AND(0, 1) (kind=3, child_offset=0, child_count=2)
    let nodes = [(1, 0, 0), (2, 0, 0), (3, 0, 2)];
    let node_vars = [0, 1, 0];
    let children = [0, 1];
    let toposort = [0, 1, 2];

    // Case 1: X0=1, X1=0 => (1 AND 1) = 1
    let eval1 = ddnnf_evaluate_witness(&nodes, &node_vars, &children, &[1, 0], &toposort);
    assert_eq!(eval1[2], 1);

    // Case 2: X0=1, X1=1 => (1 AND 0) = 0
    let eval2 = ddnnf_evaluate_witness(&nodes, &node_vars, &children, &[1, 1], &toposort);
    assert_eq!(eval2[2], 0);

    // Error on out of bounds variable
    assert!(try_ddnnf_evaluate_witness(&nodes, &node_vars, &children, &[1], &toposort).is_err());
}

#[test]
fn bigint_add_carry_and_resolve_contracts() {
    let a = [0xFFFF_FFFF, 0xFFFF_FFFF, 0];
    let b = [1, 0, 0];
    let (sums, carries) = bigint_add_carry_witness(&a, &b).unwrap();
    assert_eq!(sums, vec![0, 0xFFFF_FFFF, 0]);
    assert_eq!(carries, vec![1, 0, 0]);

    let (resolved, carry_out) = resolve_bigint_carry_chain_witness(&sums, &carries).unwrap();
    assert_eq!(resolved, vec![0, 0, 1]);
    assert_eq!(carry_out, 0);

    // Overflowing high carry
    let max_a = [0xFFFF_FFFF, 0xFFFF_FFFF];
    let max_b = [0xFFFF_FFFF, 0xFFFF_FFFF];
    let (max_sums, max_carries) = bigint_add_carry_witness(&max_a, &max_b).unwrap();
    let (_resolved_max, max_carry_out) =
        resolve_bigint_carry_chain_witness(&max_sums, &max_carries).unwrap();
    assert_eq!(max_carry_out, 1);

    // Length mismatch error
    assert!(bigint_add_carry_witness(&[1], &[1, 2]).is_err());
    assert!(resolve_bigint_carry_chain_witness(&[1], &[1, 2]).is_err());
}

#[test]
fn hypervector_contracts() {
    let a = [0xAAAA_5555, 0x1234_5678];
    let b = [0xFFFF_0000, 0x0000_FFFF];
    let bound = hypervector_xor_bind_witness(&a, &b);
    assert_eq!(
        bound,
        vec![0xAAAA_5555 ^ 0xFFFF_0000, 0x1234_5678 ^ 0x0000_FFFF]
    );

    let v1 = vec![0b1100_u32];
    let v2 = vec![0b1010_u32];
    let v3 = vec![0b0110_u32];
    let bundled = hypervector_majority_bundle_witness(&[v1, v2, v3]);
    assert_eq!(bundled, vec![0b1110]);
}
#[test]
fn count_sketch_and_ntt_contracts() {
    let mut table = vec![0_u32; 6];
    count_sketch_update_witness(&mut table, &[0, 1], &[1, -1], 2, 3);
    assert_eq!(
        count_sketch_query_witness(&table, &[0, 1], &[1, -1], 2, 3),
        1
    );

    let mut scratch = vec![91, 92];
    assert!(try_count_sketch_query_into_witness(&[], &[], &[], 0, 3, &mut scratch).is_err());
    assert_eq!(
        scratch,
        vec![91, 92],
        "invalid dimensions must not mutate caller scratch"
    );

    let mut wrapping_table = vec![i32::MAX as u32];
    count_sketch_update_witness(&mut wrapping_table, &[0], &[1], 1, 1);
    assert_eq!(wrapping_table, vec![i32::MIN as u32]);
    assert_eq!(
        count_sketch_query_witness(&[i32::MIN as u32], &[0], &[-1], 1, 1),
        i32::MIN,
        "count-sketch arithmetic follows the GPU's two's-complement wrapping"
    );

    let mut values = [1, 2, 3, 4];
    ntt_forward_witness(&mut values);
    assert_eq!(values, [10, 173_167_434, 998_244_351, 825_076_915]);
    ntt_inverse_witness(&mut values);
    assert_eq!(values, [1, 2, 3, 4]);
}

#[test]
fn chebyshev_filter_and_kfac_contracts() {
    // Chebyshev on 2x2 Laplacian
    let laplacian = [1.0_f32, 0.0, 0.0, 1.0];
    let signal = [2.0_f32, 3.0];
    let coeffs = [0.5_f32, 1.0, 0.25];
    let filtered = chebyshev_filter_witness(&laplacian, &signal, &coeffs, 2, 2);
    assert_eq!(filtered.len(), 2);
    // T0 = [2, 3], T1 = L*x = [2, 3], T2 = 2*L*T1 - T0 = [2, 3]
    // filtered = 0.5*[2,3] + 1.0*[2,3] + 0.25*[2,3] = 1.75*[2,3] = [3.5, 5.25]
    assert!((filtered[0] - 3.5).abs() < 1e-6);
    assert!((filtered[1] - 5.25).abs() < 1e-6);

    // Chebyshev into scratch with pointer-capacity preservation
    let mut out = Vec::with_capacity(4);
    let mut t_prev = Vec::with_capacity(4);
    let mut t_curr = Vec::with_capacity(4);
    let mut t_next = Vec::with_capacity(4);
    let pointers = [
        out.as_ptr(),
        t_prev.as_ptr(),
        t_curr.as_ptr(),
        t_next.as_ptr(),
    ];
    chebyshev_filter_witness_into(
        &laplacian,
        &signal,
        &coeffs,
        2,
        2,
        &mut out,
        &mut t_prev,
        &mut t_curr,
        &mut t_next,
    );
    assert_eq!(out, vec![3.5, 5.25]);
    assert_eq!(t_prev, vec![2.0, 3.0]); // T1
    assert_eq!(t_curr, vec![2.0, 3.0]); // T2
    assert_eq!(t_next, vec![2.0, 3.0]); // T2
    assert_eq!(out.as_ptr(), pointers[0]);
    assert_eq!(t_prev.as_ptr(), pointers[1]);
    assert_eq!(t_curr.as_ptr(), pointers[2]);
    assert_eq!(t_next.as_ptr(), pointers[3]);

    // K-FAC block inverse
    let blocks = [2.0_f32, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 10.0];
    let inv = kfac_block_inverse_witness(&blocks, 2, 2);
    assert_eq!(inv, vec![0.5, 0.0, 0.0, 0.25, 0.2, 0.0, 0.0, 0.1]);

    // MP edge clip
    let values = [1.0_f64, 2.5, 5.0];
    let clipped = mp_edge_clip_witness(&values, 3.0);
    assert_eq!(clipped, vec![1.0, 2.5, 3.0]);
}

#[test]
fn fractional_kernel_and_fixed_point_and_privacy_witness_contracts() {
    let kernel = grunwald_letnikov_kernel_witness(1.0, 3);
    assert_eq!(kernel.len(), 3);
    assert!((kernel[0] - 1.0).abs() < 1e-6);
    assert!((kernel[1] - (-1.0)).abs() < 1e-6);
    assert!(kernel[2].abs() < 1e-6);

    let mut kernel_into = Vec::with_capacity(3);
    let ptr = kernel_into.as_ptr();
    try_grunwald_letnikov_kernel_witness_into(1.0, 3, &mut kernel_into).unwrap();
    assert_eq!(kernel_into, kernel);
    assert_eq!(kernel_into.as_ptr(), ptr);

    let fixed = kernel_to_fixed_16_16_witness(&[1.0, -0.5], 1.0, 1.0);
    assert_eq!(fixed, vec![65536, -32768i32 as u32]);

    let mut fixed_into = Vec::with_capacity(2);
    let ptr_fixed = fixed_into.as_ptr();
    kernel_to_fixed_16_16_witness_into(&[1.0, -0.5], 1.0, 1.0, &mut fixed_into);
    assert_eq!(fixed_into, fixed);
    assert_eq!(fixed_into.as_ptr(), ptr_fixed);

    // Non-positive step returns Ok(()) with empty output and no mutation
    let mut empty_into = Vec::with_capacity(4);
    empty_into.push(999);
    try_kernel_to_fixed_16_16_witness_into(&[1.0, 2.0], 0.0, 1.0, &mut empty_into).unwrap();
    assert!(empty_into.is_empty());

    // Fractional derivative known contract
    let derivative = fractional_derivative_witness(&[0.0, 1.0, 2.0], 1.0, 1.0);
    assert_eq!(derivative, vec![0.0, 1.0, 1.0]);

    // Privacy RDP to DP
    let eps = rdp_to_dp_witness(0.0, 2.0, std::f64::consts::E.recip());
    assert!((eps - 1.0).abs() < 1e-6);
    assert_eq!(
        privacy_epsilon_from_rdp_witness(0.0, 2.0, std::f64::consts::E.recip()),
        eps
    );
    assert!(privacy_epsilon_from_rdp_witness(0.0, 0.5, 0.1).is_infinite());
}

#[test]
fn qsvt_truncator_and_fusion_affinity_witness_contracts() {
    let coeffs1 = negative_truncator_coeffs_witness(1);
    assert_eq!(coeffs1.len(), 1);
    assert!((coeffs1[0] - (-1.0 / std::f64::consts::PI)).abs() < 1e-6);

    let coeffs8 = negative_truncator_coeffs_witness(8);
    assert_eq!(coeffs8.len(), 8);
    assert_eq!(coeffs8[3], 0.0);
    assert_eq!(coeffs8[5], 0.0);
    assert_eq!(coeffs8[7], 0.0);

    let mut coeffs_into = Vec::with_capacity(8);
    let ptr = coeffs_into.as_ptr();
    negative_truncator_coeffs_witness_into(8, &mut coeffs_into);
    assert_eq!(coeffs_into, coeffs8);
    assert_eq!(coeffs_into.as_ptr(), ptr);

    let residual = [1.0_f64, -2.5, 0.0, 0.5];
    let affinity = fusion_affinity_witness(&residual);
    assert_eq!(affinity, vec![-1.0, -2.5, 0.0, -0.5]);

    let mut affinity_into = Vec::with_capacity(4);
    let ptr_aff = affinity_into.as_ptr();
    fusion_affinity_witness_into(&residual, &mut affinity_into);
    assert_eq!(affinity_into, affinity);
    assert_eq!(affinity_into.as_ptr(), ptr_aff);
}
#[test]
fn conformal_and_argmax_and_dp_clip_contracts() {
    let scores = [10_u32, 20, 30, 40, 50];
    let threshold = conformal_threshold_witness(&scores, 0.2);
    assert_eq!(threshold, 50);

    let gains = [10_u32, 50, 30];
    let picked = [0_u32, 1, 0];
    let (winner, gain) = argmax_of_marginals_witness(&gains, &picked);
    assert_eq!(winner, 2);
    assert_eq!(gain, 30);

    let grads = [3.0_f64, 4.0, 1.0, 1.0];
    let norms = [5.0_f64, 1.414];
    let clipped = dp_clip_per_sample_witness(&grads, &norms, 2.5, 2, 2);
    assert!((clipped[0] - 1.5).abs() < 1e-6);
    assert!((clipped[1] - 2.0).abs() < 1e-6);
    assert!((clipped[2] - 1.0).abs() < 1e-6);
    assert!((clipped[3] - 1.0).abs() < 1e-6);
}

#[test]
fn natural_gradient_and_tensor_train_and_scc_contracts() {
    let matrix = [2.0_f64, 0.0, 0.0, 3.0];
    let grad = [4.0_f64, 5.0];
    let ng = natural_gradient_block_apply_witness(&matrix, &grad, 2);
    assert_eq!(ng, vec![8.0, 15.0]);

    let mut ng_out = Vec::new();
    natural_gradient_block_apply_witness_into(&matrix, &grad, 2, &mut ng_out);
    assert_eq!(ng_out, vec![8.0, 15.0]);

    let acc = [1.0_f64, 2.0];
    let core = [1.0_f64, 0.0, 0.0, 1.0];
    let contracted = tensor_train_contract_step_witness(&acc, &core, 2, 2);
    assert_eq!(contracted, vec![1.0, 2.0]);
    assert!(try_tensor_train_contract_step_witness(&acc, &core, 0, 2).is_err());

    // Tensor SCC: node 0 -> 1, node 1 -> 2, seed = node 0, group_mask = 0b011
    let matrix_rows = [0b0010_u32, 0b0100, 0b0000];
    let reached = tensor_scc_witness(&matrix_rows, 0b0001, 0b0011, 10);
    assert_eq!(reached, 0b0011);
}

#[test]
fn stream_compact_and_interval_and_iht_contracts() {
    let payloads = [10_u32, 20, 30, 40];
    let flags = [1_u32, 0, 1, 0];
    let (compacted, count) = stream_compact_witness(&payloads, &flags);
    assert_eq!(compacted, vec![10, 30]);
    assert_eq!(count, 2);

    let mins_a = [5_u32, 10];
    let maxs_a = [15_u32, 20];
    let mins_b = [2_u32, 12];
    let maxs_b = [18_u32, 25];
    let (mins, maxs) = interval_merge_witness(&mins_a, &maxs_a, &mins_b, &maxs_b);
    assert_eq!(mins, vec![2, 10]);
    assert_eq!(maxs, vec![18, 25]);

    let values = [1.0_f64, -5.0, 3.0, -2.0];
    let (kept, thresh) = iht_top_k_witness(&values, 2);
    assert_eq!(kept, vec![0.0, -5.0, 3.0, 0.0]);
    assert_eq!(thresh, 3.0);
}

#[test]
fn scallop_and_bellman_contracts() {
    // Scallop 2x2 fixpoint
    let state = [1_u32, 0, 0, 1];
    let rules = [1_u32, 1, 0, 1];
    let (closure, iters) = scallop_join_fixpoint_witness(&state, &rules, 2, 1, 10);
    assert_eq!(closure, vec![1, 1, 0, 1]);
    assert!(iters <= 10);

    // Bellman shortest path
    let sources = [0_u32, 1];
    let destinations = [1_u32, 2];
    let weights = [5_u32, 3];
    let initial = [0_u32, u32::MAX, u32::MAX];
    let (dist, _) =
        bellman_shortest_path_witness(&sources, &destinations, &weights, &initial, 3, 5);
    assert_eq!(dist, vec![0, 5, 8]);
}

#[test]
fn amg_and_sinkhorn_contracts() {
    // 2-level AMG V-cycle
    let fine_matrix = [2.0_f64, -1.0, -1.0, 2.0];
    let fine_rhs = [1.0_f64, 1.0];
    let initial = [0.0_f64, 0.0];
    let restriction = [0.5_f64, 0.5];
    let prolongation = [1.0_f64, 1.0];
    let coarse_matrix = [1.0_f64];
    let solved = amg_v_cycle_witness(
        &fine_matrix,
        &fine_rhs,
        &initial,
        &restriction,
        &prolongation,
        &coarse_matrix,
        0.67,
        2,
        1,
    );
    assert_eq!(solved.len(), 2);
    assert!(solved.iter().all(|&v| v.is_finite()));

    // Sinkhorn iteration
    let k = [1_u32, 2, 3, 4];
    let k_t = [1_u32, 3, 2, 4];
    let a = [10_u32, 20];
    let b = [10_u32, 20];
    let u_init = [1_u32, 1];
    let v_init = [1_u32, 1];
    let (u, v, _) = sinkhorn_iterate_witness(&k, &k_t, &a, &b, &u_init, &v_init, 2, 2, 5);
    assert_eq!(u.len(), 2);
    assert_eq!(v.len(), 2);
    assert!(try_sinkhorn_iterate_witness(&k, &k_t, &a, &b, &u_init, &v_init, 0, 2, 5).is_err());

    // Sinkhorn clustering
    let region_features = [1.0_f32, 0.0, 0.0, 1.0];
    let centroids = [1.0_f32, 0.0, 0.0, 1.0];
    let weights = [1.0_f32, 1.0];
    let capacities = [1.0_f32, 1.0];
    let clusters = sinkhorn_clustering_witness(
        &region_features,
        &centroids,
        &weights,
        &capacities,
        2,
        2,
        2,
        3,
        1.0,
    );
    assert_eq!(clusters, vec![0, 1]);
}

#[test]
fn int4_quantized_contracts() {
    let lanes = [1_i32, -2, 3, -4, 5, -6, 7, -8];
    let packed = pack_i4x8_witness(&lanes);
    assert_eq!(packed.len(), 1);
    let unpacked = unpack_i4x8_witness(&packed, 8);
    assert_eq!(unpacked, lanes);

    let dot_i32 = i4x8_dot_i32_witness(&packed, &packed, 8);
    assert_eq!(dot_i32, 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64);

    let dot_f32 = i4x8_dot_f32_scaled_witness(&packed, &packed, 0.5, 0.25, 8);
    assert!((dot_f32 - (dot_i32 as f32 * 0.125)).abs() < 1e-6);

    let matvec = i4x8_matvec_f32_scaled_witness(&packed, &[1.0; 8], &[0.5], 1, 8);
    let expected_sum = lanes.iter().sum::<i32>() as f32 * 0.5;
    assert!((matvec[0] - expected_sum).abs() < 1e-6);

    let batched_matvec =
        i4x8_batched_matvec_f32_scaled_witness(&packed, &[1.0; 8], &[0.5], 1, 1, 8);
    assert_eq!(batched_matvec, matvec);

    let (scores, indices) =
        i4x8_batched_matmul_top1_f32_scaled_witness(&packed, &packed, &[1.0], &[1.0], 1, 1, 8);
    assert_eq!(scores, vec![dot_i32 as f32]);
    assert_eq!(indices, vec![0]);
    let batched_matmul =
        i4x8_batched_matmul_f32_scaled_witness(&packed, &packed, &[1.0], &[1.0], 1, 1, 8);
    assert_eq!(batched_matmul, vec![dot_i32 as f32]);
}

#[test]
fn simplicial_and_vietoris_and_sos_contracts() {
    // Simplicial triangle message
    let edge_features = [1.0_f64, 2.0, 3.0];
    let triangle_edges = [0_u32, 1, 2];
    let msg = simplicial_triangle_message_witness(&edge_features, &triangle_edges, 3, 1, 1);
    // jk(0) - ik(1) + ij(2) = 1.0 - 2.0 + 3.0 = 2.0
    assert_eq!(msg, vec![2.0]);

    // Vietoris-Rips
    let dists = [0.0_f64, 1.0, 1.0, 0.0];
    let mask = vietoris_rips_edge_filter_witness(&dists, 1.5, 2);
    assert_eq!(mask, vec![0, 1, 0, 0]);
    let edges = vietoris_rips_edges_witness(&mask, 2);
    assert_eq!(edges, vec![(0, 1)]);

    // SOS Gram
    let monomial_pairs = [0_u32, 1, 1, 2];
    let poly_coeffs = [10_u32, 20, 30];
    let gram = sos_gram_construct_witness(&monomial_pairs, &poly_coeffs, 2);
    assert_eq!(gram, vec![10, 20, 20, 30]);
}

#[test]
fn qsvt_and_mori_zwanzig_and_homotopy_contracts() {
    let matrix = [3.0_f64, 0.0, 0.0, 4.0];
    let (encoded, norm) = qsvt_block_encode_witness(&matrix, 2);
    assert!((norm - 5.0).abs() < 1e-6);
    assert!((encoded[0] - 0.6).abs() < 1e-6);
    assert!((encoded[3] - 0.8).abs() < 1e-6);

    let coeffs = [0.5_f64, 1.0];
    let vector = [1.0_f64, 2.0];
    let qsvt_out = qsvt_apply_witness(&encoded, &vector, &coeffs, 2).unwrap();
    // c0*v + c1*A*v = 0.5*[1,2] + 1.0*[0.6, 1.6] = [1.1, 2.6]
    assert!((qsvt_out[0] - 1.1).abs() < 1e-6);
    assert!((qsvt_out[1] - 2.6).abs() < 1e-6);

    let proj = mori_zwanzig_project_witness(&[1.0, 0.0, 0.0, 0.0], &[3.0, 4.0], 2);
    assert_eq!(proj, vec![3.0, 0.0]);

    let state = [1.0_f64, 2.0];
    let vel = [0.5_f64, -1.0];
    let predicted = homotopy_euler_predictor_witness(&state, &vel, 2.0);
    assert_eq!(predicted, vec![2.0, 0.0]);

    let homo = linear_homotopy_witness(&[0.0_f64, 10.0], &[10.0, 20.0], 0.3);
    assert_eq!(homo, vec![3.0, 13.0]);
}

#[test]
fn sum_product_and_softmax_and_conv1d_contracts() {
    // Sum-product circuit:
    // 0: leaf (val=2.0), 1: leaf (val=3.0)
    // 2: sum(0*0.5 + 1*1.0) = 1.0 + 3.0 = 4.0
    // 3: product(0, 1) = 2.0 * 3.0 = 6.0
    let kinds = [0_u32, 0, 1, 2];
    let child_offsets = [0_u32, 0, 0, 2];
    let child_counts = [0_u32, 0, 2, 2];
    let children = [0_u32, 1, 0, 1];
    let weights = [0.5_f64, 1.0, 1.0, 1.0];
    let leaf_values = [2.0_f64, 3.0, 0.0, 0.0];
    let topological_order = [0_u32, 1, 2, 3];
    let spc_eval = sum_product_evaluate_witness(
        &kinds,
        &child_offsets,
        &child_counts,
        &children,
        &weights,
        &leaf_values,
        &topological_order,
    );
    assert_eq!(spc_eval, vec![2.0, 3.0, 4.0, 6.0]);

    // Softmax & diff argmax
    let sm = softmax_witness(&[0.0_f64, 0.0]);
    assert_eq!(sm, vec![0.5, 0.5]);

    let diff_argmax = differentiable_argmax_witness(&[1.0_f64, 2.0], 1.0);
    assert!(diff_argmax[1] > diff_argmax[0]);
    assert!((diff_argmax.iter().sum::<f64>() - 1.0).abs() < 1e-6);

    // 1D Conv
    let input = [10_u32, 20, 30, 40];
    let conv_weights = [1_u32, 2, 1]; // radius = 1
    let convolved = conv1d_witness(&input, &conv_weights, 1);
    // index 0: clamped src [-1, 0, 1] -> [0, 0, 1] -> input[0]*1 + input[0]*2 + input[1]*1 = 10*1 + 10*2 + 20*1 = 50
    // index 1: src [0, 1, 2] -> input[0]*1 + input[1]*2 + input[2]*1 = 10 + 40 + 30 = 80
    // index 2: src [1, 2, 3] -> input[1]*1 + input[2]*2 + input[3]*1 = 20 + 60 + 40 = 120
    // index 3: src [2, 3, 4] -> clamped [2, 3, 3] -> input[2]*1 + input[3]*2 + input[3]*1 = 30 + 80 + 40 = 150
    assert_eq!(convolved, vec![50, 80, 120, 150]);

    // Arrow composition
    let eye = identity_arrow_witness(2);
    assert_eq!(eye, vec![1.0, 0.0, 0.0, 1.0]);
    let a = [1.0_f64, 2.0, 3.0, 4.0];
    let composed = compose_ir_arrows_witness(&a, &eye, 2, 2, 2);
    assert_eq!(composed, a.to_vec());
    assert!(composition_associates_witness(&a, &a, &a, 2, 2, 2, 2));
}

#[test]
fn hash_witnesses_standard_known_vectors() {
    assert_eq!(crc32_witness(b"123456789"), 0xCBF43926);
    assert_eq!(fnv1a32_witness(b"abc"), 0x1A47E90B);
    assert_eq!(fnv1a64_witness(b"abc"), 0xE71FA2190541574B);
    assert_eq!(adler32_witness(b"Wikipedia"), 0x11E60398);

    let chunk = adler32_chunk_witness(b"Wikipedia");
    assert_eq!(adler32_finalize_witness(chunk.a, chunk.b), 0x11E60398);

    let left = adler32_chunk_witness(b"Wiki");
    let right = adler32_chunk_witness(b"pedia");
    let combined = adler32_combine_chunks_witness(left, right);
    assert_eq!(adler32_finalize_witness(combined.a, combined.b), 0x11E60398);

    let a = [1, 2, 3, 4];
    let b = [5, 6, 7, 8];
    let c = matmul_u32_witness(&a, &b, None, 2, 2, 2);
    assert_eq!(c, vec![19, 22, 43, 50]);

    let c_bias = matmul_u32_witness(&a, &b, Some(&[10, 20]), 2, 2, 2);
    assert_eq!(c_bias, vec![29, 42, 53, 70]);

    let mut rdp = Vec::new();
    try_gaussian_rdp_step_witness_into(&[2.0], &[1.0], &mut rdp).unwrap();
    assert_eq!(rdp, vec![1.0]);
}

#[test]
fn stochastic_bitstream_contracts() {
    // Deterministic 0.25 roundtrip tolerance
    let bs = stochastic_encode_witness(0.25, 1024, 42);
    let p = stochastic_decode_witness(&bs, 1024);
    assert!((p - 0.25).abs() < 0.05);

    // Zero probability yields all-zero bitstream
    let bs_zero = stochastic_encode_witness(0.0, 256, 1);
    assert!(bs_zero.iter().all(|&w| w == 0));

    // 65-bit output truncates stale tail to 3 words while preserving preallocated pointer
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&[u32::MAX; 16]);
    let ptr = out.as_ptr();
    try_stochastic_encode_witness_into(0.0, 65, 42, &mut out).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out.as_ptr(), ptr);
    assert!(out.iter().all(|&word| word == 0));

    // usize::MAX returns overflow error while preserving caller output unchanged
    let mut preserve = vec![10, 20, 30];
    let err = try_stochastic_encode_witness_into(0.5, usize::MAX, 1, &mut preserve);
    assert!(err.is_err());
    assert_eq!(preserve, vec![10, 20, 30]);
}

#[test]
fn region_ownership_handles_exact_and_insertion_points() {
    let starts = [0, 5, 11];
    assert_eq!(region_of_witness(0, &starts), 0);
    assert_eq!(region_of_witness(4, &starts), 0);
    assert_eq!(region_of_witness(5, &starts), 1);
    assert_eq!(region_of_witness(10, &starts), 1);
    assert_eq!(region_of_witness(11, &starts), 2);
    assert_eq!(region_of_witness(u32::MAX, &starts), 2);
}

#[test]
fn vsa_fingerprint_witness_known_answers() {
    let kind = [0xDEAD_BEEFu32, 0x1111_1111];
    let sig = [0x1234_5678u32, 0x2222_2222];
    let region = [0x9ABC_DEF0u32, 0x4444_4444];
    let fp = vsa_fingerprint_witness(&kind, &sig, &region);
    assert_eq!(
        fp,
        vec![
            0xDEAD_BEEF ^ 0x1234_5678 ^ 0x9ABC_DEF0,
            0x1111_1111 ^ 0x2222_2222 ^ 0x4444_4444,
        ]
    );
}

#[test]
fn matroid_select_optimal_subset_witness_into_reuses_storage() {
    let n = 3;
    let mut adj = vec![0u32; 9];
    adj[0 * 3 + 1] = 1;
    adj[1 * 3 + 2] = 1;
    let sources = vec![1, 0, 0];
    let sinks = vec![0, 0, 1];
    let seed = vec![0u32; 3];
    let mut current = Vec::new();
    let mut next = Vec::new();

    matroid_select_optimal_subset_witness_into(
        &adj,
        &sources,
        &sinks,
        &seed,
        n,
        8,
        &mut current,
        &mut next,
    )
    .expect("matroid selection succeeds");

    assert_eq!(current.len(), 3);
    assert!(current[0] != 0 || current.iter().sum::<u32>() >= 1);
}

#[test]
fn reduce_f32_witness_known_answers() {
    assert_eq!(reduce_sum_f32_witness(&[1.25, -2.0, 5.5]), 4.75);
    assert_eq!(reduce_max_f32_witness(&[-7.0, 3.5, 2.0]), 3.5);
}

#[test]
fn csr_backward_step_and_closure_known_answers() {
    let node_count = 3;
    let offsets = [0, 1, 2, 2];
    let targets = [1, 2];
    let masks = [1, 1];
    let seed = [0b100]; // node 2 active
    let (step_out, changed) =
        csr_backward_step_with_change_witness(node_count, &offsets, &targets, &masks, &seed, 1);
    assert_eq!(changed, 1);
    assert_eq!(step_out, vec![0b110]); // nodes 1, 2 active

    let closure_out =
        csr_backward_closure_witness(node_count, &offsets, &targets, &masks, &seed, 1);
    assert_eq!(closure_out, vec![0b111]); // nodes 0, 1, 2 all reach 2
}

#[test]
fn sinkhorn_f64_witness_known_answers_and_storage_reuse() {
    let k = vec![1.0, 2.0, 3.0, 4.0];
    let a = vec![1.0, 1.0];
    let b = vec![1.0, 1.0];
    let (u, v, iters) = sinkhorn_iterate_f64_witness(&k, &a, &b, 1e-6, 100);
    assert!(iters > 0);
    assert_eq!(u.len(), 2);
    assert_eq!(v.len(), 2);
    let r_row = sinkhorn_row_residual_witness(&k, &u, &v, &a);
    let r_col = sinkhorn_col_residual_witness(&k, &u, &v, &b);
    assert!(r_row < 1e-4);
    assert!(r_col < 1e-4);

    let mut u_out = Vec::with_capacity(2);
    let mut v_out = Vec::with_capacity(2);
    let mut u_old = Vec::with_capacity(2);
    let iters_into = try_sinkhorn_iterate_f64_witness_into(
        &k, &a, &b, 1e-6, 100, &mut u_out, &mut v_out, &mut u_old,
    )
    .expect("sinkhorn into succeeds");
    assert_eq!(iters_into, iters);
    assert_eq!(u_out, u);
    assert_eq!(v_out, v);
}

#[test]
fn fractional_derivative_witness_known_answers() {
    let f = vec![1.0, 2.0, 3.0, 4.0];
    let res = fractional_derivative_witness(&f, 1.0, 1.0);
    assert_eq!(res.len(), 4);
    assert!((res[0] - 1.0).abs() < 1e-6);
    assert!((res[1] - 1.0).abs() < 1e-6);
    assert!((res[2] - 1.0).abs() < 1e-6);
    assert!((res[3] - 1.0).abs() < 1e-6);

    let mut kernel = Vec::new();
    let mut out = Vec::new();
    try_fractional_derivative_witness_into(&f, 1.0, 1.0, &mut kernel, &mut out)
        .expect("fractional derivative into succeeds");
    assert_eq!(out, res);
}

#[test]
fn newton_schulz_witness_known_answers_and_storage_reuse() {
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    let inv_sqrt = newton_schulz_inverse_sqrt_witness(&eye, 2, 5);
    assert_eq!(inv_sqrt.len(), 4);
    assert!((inv_sqrt[0] - 1.0).abs() < 1e-5);
    assert!(inv_sqrt[1].abs() < 1e-5);
    assert!(inv_sqrt[2].abs() < 1e-5);
    assert!((inv_sqrt[3] - 1.0).abs() < 1e-5);

    let mut out = Vec::with_capacity(4);
    let mut scratch = NewtonSchulzScratchWitness::new();
    newton_schulz_inverse_sqrt_witness_into(&eye, 2, 5, &mut out, &mut scratch);
    assert_eq!(out, inv_sqrt);
}

#[test]
fn info_geometry_and_projection_witnesses_known_answers() {
    let p = vec![0.25, 0.25, 0.25, 0.25];
    let q = vec![0.25, 0.25, 0.25, 0.25];
    let bc = bhattacharyya_coefficient_witness(&p, &q);
    assert!((bc - 1.0).abs() < 1e-6);
    let dist = fisher_rao_distance_witness(&p, &q);
    assert!(dist.abs() < 1e-6);

    let y = vec![1.0, 0.0, 0.0, 1.0];
    let q_orth = modified_gram_schmidt_witness(&y, 2, 2);
    assert_eq!(q_orth.len(), 4);
    assert!((q_orth[0] - 1.0).abs() < 1e-6);
    assert!((q_orth[3] - 1.0).abs() < 1e-6);

    let rdp = gaussian_rdp_step_witness(&[2.0], &[1.0]);
    assert_eq!(rdp, vec![1.0]);
}

#[test]
fn stream_compact_witness_known_answers_and_storage_reuse() {
    let payloads = [10, 20, 30, 40, 50];
    let flags = [0, 1, 1, 0, 1];
    let (compacted, count) = stream_compact_witness(&payloads, &flags);
    assert_eq!(compacted, vec![20, 30, 50]);
    assert_eq!(count, 3);

    let mut out = Vec::with_capacity(8);
    let ptr = out.as_ptr();
    let live = stream_compact_witness_into(&payloads, &flags, &mut out);
    assert_eq!(out, vec![20, 30, 50]);
    assert_eq!(live, 3);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn pattern_region_and_post_process_neutral_contracts() {
    let triples = vec![(0, 5, 10), (0, 7, 12), (1, 2, 4)];
    let deduped = dedup_regions_witness(triples);
    assert_eq!(deduped, vec![(0, 5, 12), (1, 2, 4)]);

    let ranges = [vyre_foundation::match_result::ByteRange::new(1, 0, 3)];
    let haystack = b"abcdef";
    let processed = try_match_post_process_witness(&ranges, haystack).unwrap();
    assert_eq!(processed.len(), 1);
    assert_eq!(processed[0].pattern_id, 1);
    assert_eq!(processed[0].start, 0);
    assert_eq!(processed[0].end, 3);
}

#[test]
fn canonical_into_witnesses_storage_reuse_and_truncation() {
    // Prefix scan into
    let mut scan_out = Vec::with_capacity(16);
    scan_out.extend([999_u32; 16]);
    let scan_ptr = scan_out.as_ptr();
    prefix_scan_witness_into(&[1, 2, 3, 4], true, |a, b| a + b, 0, &mut scan_out);
    assert_eq!(scan_out, vec![1, 3, 6, 10]);
    assert_eq!(scan_out.as_ptr(), scan_ptr);

    // Conv1D into
    let mut conv_out = Vec::with_capacity(16);
    conv_out.extend([999_u32; 16]);
    let conv_ptr = conv_out.as_ptr();
    conv1d_witness_into(&[10, 20, 30, 40], &[1, 2, 1], 1, &mut conv_out);
    assert_eq!(conv_out, vec![50, 80, 120, 150]);
    assert_eq!(conv_out.as_ptr(), conv_ptr);

    // Tensor-train step and chain into
    let mut tt_out = Vec::with_capacity(8);
    tt_out.extend([99.0; 8]);
    let tt_ptr = tt_out.as_ptr();
    tensor_train_contract_step_witness_into(
        &[1.0, 2.0],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2,
        3,
        &mut tt_out,
    );
    assert_eq!(tt_out, vec![9.0, 12.0, 15.0]);
    assert_eq!(tt_out.as_ptr(), tt_ptr);

    let cores = vec![vec![3.0, 5.0], vec![7.0, 11.0]];
    let ranks = [1, 1, 1];
    let dims = [2, 2];
    let mut acc = Vec::with_capacity(4);
    let mut next = Vec::with_capacity(4);
    let val = try_tensor_train_full_chain_witness_into(
        &cores,
        &ranks,
        &dims,
        &[1, 1],
        &mut acc,
        &mut next,
    )
    .unwrap();
    assert!((val - 55.0).abs() < 1e-10);

    // QSVT block encode into
    let mut qsvt_out = Vec::with_capacity(8);
    qsvt_out.extend([99.0; 8]);
    let qsvt_ptr = qsvt_out.as_ptr();
    let norm = qsvt_block_encode_witness_into(&[3.0, 0.0, 0.0, 4.0], 2, &mut qsvt_out);
    assert!((norm - 5.0).abs() < 1e-6);
    assert_eq!(qsvt_out.len(), 4);
    assert_eq!(qsvt_out.as_ptr(), qsvt_ptr);

    // Sheaf spectral clustering witnesses into
    let mut sheaf_v = Vec::with_capacity(8);
    sheaf_v.extend([99.0; 8]);
    let sheaf_ptr = sheaf_v.as_ptr();
    let lambda = sheaf_dominant_spectrum_witness_into(&[0.1, 0.5, 0.9, 0.3], 32, &mut sheaf_v);
    assert!((lambda - 0.9).abs() < 1e-10);
    assert_eq!(sheaf_v, vec![0.0, 0.0, 1.0, 0.0]);
    assert_eq!(sheaf_v.as_ptr(), sheaf_ptr);

    let gap = sheaf_spectral_gap_witness_into(&[0.1, 0.5, 0.9, 0.3], 32, &mut sheaf_v);
    assert!((gap - 1.0).abs() < 1e-10);

    let cluster_count = sheaf_suggested_cluster_count_witness(&sheaf_v);
    assert_eq!(cluster_count, 1);

    // CSR forward and backward into
    let mut csr_out = Vec::with_capacity(4);
    csr_out.extend([u32::MAX; 4]);
    let csr_ptr = csr_out.as_ptr();
    csr_forward_traverse_witness_into(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[1, 1, 1, 1],
        &[0b0001],
        0xFFFF_FFFF,
        &mut csr_out,
    );
    assert_eq!(csr_out, vec![0b0110]);
    assert_eq!(csr_out.as_ptr(), csr_ptr);

    csr_backward_traverse_witness_into(
        4,
        &[0, 2, 3, 4, 4],
        &[1, 2, 3, 3],
        &[1, 1, 1, 1],
        &[0b1000],
        0xFFFF_FFFF,
        &mut csr_out,
    );
    assert_eq!(csr_out, vec![0b0110]);
    assert_eq!(csr_out.as_ptr(), csr_ptr);

    // Causal into
    let adj = [0, 1, 0, 0];
    let mut causal_out = Vec::with_capacity(8);
    causal_out.extend([99; 8]);
    let causal_ptr = causal_out.as_ptr();
    do_intervention_delete_incoming_witness_into(&adj, &[0, 1], 2, &mut causal_out);
    assert_eq!(causal_out, vec![0, 0, 0, 0]);
    assert_eq!(causal_out.as_ptr(), causal_ptr);

    // Math into functions: dp_clip, mp_edge_clip, sos_gram, softmax, qsvt_apply, diff_argmax, bigint
    let mut dp_out = Vec::with_capacity(8);
    dp_out.extend([99.0; 8]);
    let dp_ptr = dp_out.as_ptr();
    dp_clip_per_sample_witness_into(&[3.0, 4.0], &[5.0], 1.0, 1, 2, &mut dp_out);
    assert_eq!(dp_out.len(), 2);
    assert!((dp_out[0] - 0.6).abs() < 1e-10);
    assert!((dp_out[1] - 0.8).abs() < 1e-10);
    assert_eq!(dp_out.as_ptr(), dp_ptr);

    let mut mp_out = Vec::with_capacity(8);
    mp_out.extend([99.0; 8]);
    let mp_ptr = mp_out.as_ptr();
    mp_edge_clip_witness_into(&[1.0, 5.0, 2.0], 3.0, &mut mp_out);
    assert_eq!(mp_out, vec![1.0, 3.0, 2.0]);
    assert_eq!(mp_out.as_ptr(), mp_ptr);

    let mut sos_out = Vec::with_capacity(8);
    sos_out.extend([99; 8]);
    let sos_ptr = sos_out.as_ptr();
    sos_gram_construct_witness_into(&[0, 1, 1, 0], &[10, 20], 2, &mut sos_out);
    assert_eq!(sos_out, vec![10, 20, 20, 10]);
    assert_eq!(sos_out.as_ptr(), sos_ptr);

    let mut sm_out = Vec::with_capacity(8);
    sm_out.extend([99.0; 8]);
    let sm_ptr = sm_out.as_ptr();
    softmax_witness_into(&[0.0, 0.0], &mut sm_out);
    assert_eq!(sm_out.len(), 2);
    assert!((sm_out[0] - 0.5).abs() < 1e-10);
    assert_eq!(sm_out.as_ptr(), sm_ptr);

    let mut qsvt_out = Vec::with_capacity(8);
    qsvt_out.extend([99.0; 8]);
    let qsvt_ptr = qsvt_out.as_ptr();
    qsvt_apply_witness_into(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0], &[0.5], 2, &mut qsvt_out).unwrap();
    assert_eq!(qsvt_out, vec![0.5, 1.0]);
    assert_eq!(qsvt_out.as_ptr(), qsvt_ptr);

    let mut scaled = Vec::with_capacity(8);
    let mut argmax_out = Vec::with_capacity(8);
    scaled.extend([99.0; 8]);
    argmax_out.extend([99.0; 8]);
    let argmax_ptr = argmax_out.as_ptr();
    differentiable_argmax_witness_into(&[1.0, 1.0], 1.0, &mut scaled, &mut argmax_out);
    assert_eq!(argmax_out.len(), 2);
    assert!((argmax_out[0] - 0.5).abs() < 1e-10);
    assert_eq!(argmax_out.as_ptr(), argmax_ptr);

    let mut sums = Vec::with_capacity(8);
    let mut carries = Vec::with_capacity(8);
    sums.extend([99; 8]);
    carries.extend([99; 8]);
    let sums_ptr = sums.as_ptr();
    bigint_add_carry_witness_into(&[u32::MAX], &[1], &mut sums, &mut carries).unwrap();
    assert_eq!(sums, vec![0]);
    assert_eq!(carries, vec![1]);
    assert_eq!(sums.as_ptr(), sums_ptr);

    let mut resolved = Vec::with_capacity(8);
    resolved.extend([99; 8]);
    let resolved_ptr = resolved.as_ptr();
    let carry_out =
        resolve_bigint_carry_chain_witness_into(&sums, &carries, &mut resolved).unwrap();
    assert_eq!(carry_out, 1);
    assert_eq!(resolved, vec![0]);
    assert_eq!(resolved.as_ptr(), resolved_ptr);

    let mut queue = Vec::with_capacity(8);
    queue.extend([99; 8]);
    let queue_ptr = queue.as_ptr();
    let active = frontier_to_queue_witness_into(&[0b1011_0101], 8, 3, &mut queue);
    assert_eq!(active, 5);
    assert_eq!(queue, vec![0, 2, 4]);
    assert_eq!(queue.as_ptr(), queue_ptr);
}

#[test]
fn toposort_and_reachable_witness_contracts() {
    // 0 depends on 1, 1 depends on 2 -> 2, 1, 0
    let edges = [(0, 1), (1, 2)];
    let order = toposort_witness(3, &edges).expect("acyclic");
    assert_eq!(order, vec![2, 1, 0]);

    // Cycle detection
    let cycle_edges = [(0, 1), (1, 0)];
    assert!(toposort_witness(2, &cycle_edges).is_err());

    // CSR topological sort
    let offsets = [0, 2, 3, 3];
    let targets = [1, 2, 2];
    let csr_order = toposort_csr_witness(3, &offsets, &targets).expect("valid CSR DAG");
    assert_eq!(csr_order.len(), 3);

    let mut reused_order = Vec::with_capacity(8);
    reused_order.extend([99; 8]);
    let ptr = reused_order.as_ptr();
    toposort_csr_into_witness(3, &offsets, &targets, &mut reused_order)
        .expect("valid CSR DAG into");
    assert_eq!(reused_order.len(), 3);
    assert_eq!(reused_order.as_ptr(), ptr);

    // Reachable witness
    let reach_edges = [(0, 1), (1, 2), (3, 4)];
    let reached = reachable_witness(5, &reach_edges, &[0]).expect("valid reachable");
    assert_eq!(reached, [0, 1, 2].into_iter().collect());

    // Out of range sources reported as reachable from themselves
    let out_sources = reachable_witness(0, &[], &[5, 10]).expect("out of range sources");
    assert_eq!(out_sources, [5, 10].into_iter().collect());
}

#[test]
fn toposort_csr_with_scratch_into_witness_contracts() {
    let offsets = [0, 2, 3, 4, 4];
    let targets = [1, 2, 3, 3];

    let mut order = Vec::with_capacity(16);
    order.extend_from_slice(&[99, 98, 97, 96, 95]);
    let mut indegree = Vec::with_capacity(16);
    indegree.extend_from_slice(&[88, 87, 86, 85, 84, 83]);
    let mut queue = Vec::with_capacity(16);
    queue.extend_from_slice(&[77, 76, 75, 74]);

    let order_ptr = order.as_ptr();
    let indegree_ptr = indegree.as_ptr();
    let queue_ptr = queue.as_ptr();

    toposort_csr_with_scratch_into_witness(
        4,
        &offsets,
        &targets,
        &mut order,
        &mut indegree,
        &mut queue,
    )
    .expect("toposort_csr_with_scratch_into_witness should succeed on valid DAG");

    assert_eq!(order.as_ptr(), order_ptr);
    assert_eq!(indegree.as_ptr(), indegree_ptr);
    assert_eq!(queue.as_ptr(), queue_ptr);

    assert_eq!(order.len(), 4);
    assert_eq!(order[0], 0);
    assert_eq!(order[3], 3);
    assert!((order[1] == 1 && order[2] == 2) || (order[1] == 2 && order[2] == 1));

    assert!(
        queue.is_empty(),
        "Kahn worklist queue must be completely drained on success"
    );

    let small_offsets = [0, 1, 1];
    let small_targets = [1];
    toposort_csr_with_scratch_into_witness(
        2,
        &small_offsets,
        &small_targets,
        &mut order,
        &mut indegree,
        &mut queue,
    )
    .expect("smaller repeated call should succeed");

    assert_eq!(order.as_ptr(), order_ptr);
    assert_eq!(indegree.as_ptr(), indegree_ptr);
    assert_eq!(queue.as_ptr(), queue_ptr);
    assert_eq!(order, vec![0, 1]);
    assert_eq!(indegree, vec![0, 0]);
    assert!(queue.is_empty());

    let prev_order = order.clone();
    let prev_indegree = indegree.clone();
    queue.extend_from_slice(&[123, 456]);
    let prev_queue = queue.clone();

    let bad_offsets = [0, 2, 1];
    let err = toposort_csr_with_scratch_into_witness(
        2,
        &bad_offsets,
        &small_targets,
        &mut order,
        &mut indegree,
        &mut queue,
    );
    assert!(err.is_err());
    assert_eq!(
        order, prev_order,
        "malformed CSR must not mutate order output"
    );
    assert_eq!(
        indegree, prev_indegree,
        "malformed CSR must not mutate indegree scratch"
    );
    assert_eq!(
        queue, prev_queue,
        "malformed CSR must not mutate queue scratch"
    );

    let out_of_range_targets = [5];
    let err2 = toposort_csr_with_scratch_into_witness(
        2,
        &small_offsets,
        &out_of_range_targets,
        &mut order,
        &mut indegree,
        &mut queue,
    );
    assert!(err2.is_err());
    assert_eq!(order, prev_order);
    assert_eq!(indegree, prev_indegree);
    assert_eq!(queue, prev_queue);
}
#[test]
fn amg_and_jacobi_solve_to_tolerance_contracts() {
    // Identity matrix 4x4
    let mut a = vec![0.0_f64; 16];
    for i in 0..4 {
        a[i * 4 + i] = 1.0;
    }
    let b = [1.0_f64, 2.0, 3.0, 4.0];
    let x0 = [0.0_f64; 4];
    let r_mat = [0.5_f64, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
    let p_mat = [1.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let a_c = [1.0_f64, 0.0, 0.0, 1.0];

    let mut out = Vec::with_capacity(8);
    out.extend([99.0; 8]);
    let mut scratch = Vec::with_capacity(8);
    scratch.extend([88.0; 8]);
    let out_ptr = out.as_ptr();
    let scratch_ptr = scratch.as_ptr();

    let cycles = amg_solve_to_tolerance_witness_into(
        &a,
        &b,
        &x0,
        &r_mat,
        &p_mat,
        &a_c,
        0.67,
        4,
        2,
        1e-4,
        10,
        &mut out,
        &mut scratch,
    );
    assert!((1..=10).contains(&cycles));
    assert_eq!(out.len(), 4);
    for i in 0..4 {
        assert!((out[i] - b[i]).abs() < 1e-4);
    }
    assert_eq!(out.as_ptr(), out_ptr);
    assert_eq!(scratch.as_ptr(), scratch_ptr);

    // Malformed input no-mutation contract
    let prev_out = out.clone();
    let prev_scratch = scratch.clone();
    let bad_res = try_amg_solve_to_tolerance_witness_into(
        &a[..10],
        &b,
        &x0,
        &r_mat,
        &p_mat,
        &a_c,
        0.67,
        4,
        2,
        1e-4,
        10,
        &mut out,
        &mut scratch,
    );
    assert!(bad_res.is_err());
    assert_eq!(out, prev_out);
    assert_eq!(scratch, prev_scratch);

    // Jacobi solve to tolerance
    let a_jacobi = [2.0_f64, 0.0, 0.0, 2.0];
    let b_jacobi = [6.0_f64, 8.0];
    let x0_jacobi = [0.0_f64, 0.0];
    let iters = jacobi_solve_to_tolerance_witness_into(
        &a_jacobi,
        &b_jacobi,
        &x0_jacobi,
        1.0,
        2,
        1e-6,
        10,
        &mut out,
        &mut scratch,
    );
    assert_eq!(iters, 1);
    assert_eq!(out, vec![3.0, 4.0]);
    assert_eq!(out.as_ptr(), out_ptr);

    // Jacobi malformed no-mutation
    let prev_out2 = out.clone();
    let bad_jacobi = try_jacobi_solve_to_tolerance_witness_into(
        &a_jacobi[..2],
        &b_jacobi,
        &x0_jacobi,
        1.0,
        2,
        1e-6,
        10,
        &mut out,
        &mut scratch,
    );
    assert!(bad_jacobi.is_err());
    assert_eq!(out, prev_out2);

    // AmgSolveScratchWitness and AmgVcycleScratchWitness reuse and validation contracts
    let mut solve_scratch = AmgSolveScratchWitness::new();
    solve_scratch.reserve(4, 2);
    solve_scratch.v_cycle.fine.extend([11.0; 4]);
    solve_scratch.v_cycle.residual.extend([22.0; 4]);
    solve_scratch.v_cycle.coarse_rhs.extend([33.0; 2]);
    solve_scratch.v_cycle.coarse.extend([44.0; 2]);
    solve_scratch.v_cycle.coarse_next.extend([55.0; 2]);
    solve_scratch.next_iterate.extend([66.0; 4]);

    let fine_ptr = solve_scratch.v_cycle.fine.as_ptr();
    let res_ptr = solve_scratch.v_cycle.residual.as_ptr();
    let crhs_ptr = solve_scratch.v_cycle.coarse_rhs.as_ptr();
    let coarse_ptr = solve_scratch.v_cycle.coarse.as_ptr();
    let cnext_ptr = solve_scratch.v_cycle.coarse_next.as_ptr();
    let next_iter_ptr = solve_scratch.next_iterate.as_ptr();

    let cycles2 = amg_solve_to_tolerance_witness_with_scratch_into(
        &a,
        &b,
        &x0,
        &r_mat,
        &p_mat,
        &a_c,
        0.67,
        4,
        2,
        1e-4,
        10,
        &mut solve_scratch,
        &mut out,
    );
    assert!((1..=10).contains(&cycles2));
    assert_eq!(out.len(), 4);
    assert_eq!(out.as_ptr(), out_ptr);
    assert_eq!(solve_scratch.v_cycle.fine.as_ptr(), fine_ptr);
    assert_eq!(solve_scratch.v_cycle.residual.as_ptr(), res_ptr);
    assert_eq!(solve_scratch.v_cycle.coarse_rhs.as_ptr(), crhs_ptr);
    assert_eq!(solve_scratch.v_cycle.coarse.as_ptr(), coarse_ptr);
    assert_eq!(solve_scratch.v_cycle.coarse_next.as_ptr(), cnext_ptr);
    assert_eq!(solve_scratch.next_iterate.as_ptr(), next_iter_ptr);

    // try_amg_solve_to_tolerance_witness_with_scratch_into malformed input: every scratch Vec and out must remain unchanged
    let prev_out3 = out.clone();
    let prev_scratch2 = solve_scratch.clone();
    let bad_solve_scratch = try_amg_solve_to_tolerance_witness_with_scratch_into(
        &a[..10],
        &b,
        &x0,
        &r_mat,
        &p_mat,
        &a_c,
        0.67,
        4,
        2,
        1e-4,
        10,
        &mut solve_scratch,
        &mut out,
    );
    assert!(bad_solve_scratch.is_err());
    assert_eq!(out, prev_out3);
    assert_eq!(solve_scratch.v_cycle.fine, prev_scratch2.v_cycle.fine);
    assert_eq!(
        solve_scratch.v_cycle.residual,
        prev_scratch2.v_cycle.residual
    );
    assert_eq!(
        solve_scratch.v_cycle.coarse_rhs,
        prev_scratch2.v_cycle.coarse_rhs
    );
    assert_eq!(solve_scratch.v_cycle.coarse, prev_scratch2.v_cycle.coarse);
    assert_eq!(
        solve_scratch.v_cycle.coarse_next,
        prev_scratch2.v_cycle.coarse_next
    );
    assert_eq!(solve_scratch.next_iterate, prev_scratch2.next_iterate);
    assert_eq!(solve_scratch, prev_scratch2);

    // try_amg_v_cycle_witness_with_scratch_into malformed input: every scratch Vec and out must remain unchanged
    let mut v_cycle_scratch = prev_scratch2.v_cycle.clone();
    let prev_v_cycle_scratch = v_cycle_scratch.clone();
    let bad_v_cycle = try_amg_v_cycle_witness_with_scratch_into(
        &a[..10],
        &b,
        &x0,
        &r_mat,
        &p_mat,
        &a_c,
        0.67,
        4,
        2,
        &mut v_cycle_scratch,
        &mut out,
    );
    assert!(bad_v_cycle.is_err());
    assert_eq!(out, prev_out3);
    assert_eq!(v_cycle_scratch.fine, prev_v_cycle_scratch.fine);
    assert_eq!(v_cycle_scratch.residual, prev_v_cycle_scratch.residual);
    assert_eq!(v_cycle_scratch.coarse_rhs, prev_v_cycle_scratch.coarse_rhs);
    assert_eq!(v_cycle_scratch.coarse, prev_v_cycle_scratch.coarse);
    assert_eq!(
        v_cycle_scratch.coarse_next,
        prev_v_cycle_scratch.coarse_next
    );
    assert_eq!(v_cycle_scratch, prev_v_cycle_scratch);
}

#[test]
fn differentiable_autotune_witness_contracts() {
    // argmin_cost_witness
    assert_eq!(argmin_cost_witness(&[10.0, 5.0, 20.0]), 1);
    // total_cmp tie breaking: first occurrence of minimum wins
    assert_eq!(argmin_cost_witness(&[5.0, 10.0, 5.0]), 0);
    assert!(try_argmin_cost_witness(&[]).is_err());

    // Gradient witness into
    let costs = [1.0_f64, 2.0, 3.0];
    let mut neg_costs = Vec::with_capacity(8);
    let mut out = Vec::with_capacity(8);
    neg_costs.extend([99.0; 8]);
    out.extend([99.0; 8]);
    let neg_ptr = neg_costs.as_ptr();
    let out_ptr = out.as_ptr();

    differentiable_autotune_gradient_witness_into(&costs, 1.0, &mut neg_costs, &mut out);
    assert_eq!(out.len(), 3);
    // Gradient w.r.t. cost is negative softmax probability
    assert!(out.iter().all(|&v| v < 0.0));
    assert!((out.iter().sum::<f64>() + 1.0).abs() < 1e-6);
    assert_eq!(neg_costs.as_ptr(), neg_ptr);
    assert_eq!(out.as_ptr(), out_ptr);

    // Invalid temperature returns error and does not mutate
    let prev_out = out.clone();
    assert!(try_differentiable_autotune_gradient_witness_into(
        &costs,
        0.0,
        &mut neg_costs,
        &mut out
    )
    .is_err());
    assert!(try_differentiable_autotune_gradient_witness_into(
        &costs,
        -1.0,
        &mut neg_costs,
        &mut out
    )
    .is_err());
    assert_eq!(out, prev_out);

    // Pick config witness into
    let mut scaled = Vec::with_capacity(8);
    scaled.extend([99.0; 8]);
    let scaled_ptr = scaled.as_ptr();
    differentiable_autotune_pick_config_witness_into(
        &costs,
        1.0,
        &mut neg_costs,
        &mut scaled,
        &mut out,
    );
    assert_eq!(out.len(), 3);
    assert!(out[0] > out[1] && out[1] > out[2]); // lower cost = higher prob
    assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    assert_eq!(scaled.as_ptr(), scaled_ptr);
    assert_eq!(out.as_ptr(), out_ptr);
}

#[test]
fn natural_gradient_and_identity_matrix_contracts() {
    let m_inv_sqrt = [1.0_f64, 0.0, 0.0, 1.0];
    let grad = [2.0_f64, 3.0];
    let mut out = Vec::with_capacity(16);
    out.extend([99.0; 8]);
    let out_ptr = out.as_ptr();

    natural_gradient_autotune_step_witness_into(&m_inv_sqrt, &grad, 2, 0.1, &mut out);
    assert!((out[0] + 0.2).abs() < 1e-12);
    assert!((out[1] + 0.3).abs() < 1e-12);
    assert_eq!(out.as_ptr(), out_ptr);

    // Identity matrix
    identity_matrix_witness_into(3, &mut out);
    assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(out.as_ptr(), out_ptr);

    assert!(try_identity_matrix_witness_into(0, &mut out).is_ok());
    assert_eq!(out, Vec::<f64>::new());
}

#[test]
fn tensor_train_fusion_pressure_contracts() {
    assert_eq!(tensor_train_fusion_pressure_witness(&[]), 0.0);

    let mut acc = Vec::with_capacity(8);
    let mut next = Vec::with_capacity(8);
    acc.extend([99.0; 8]);
    next.extend([99.0; 8]);
    let acc_ptr = acc.as_ptr();
    let next_ptr = next.as_ptr();

    let pressure = tensor_train_fusion_pressure_witness_with_scratch(&[2, 3], &mut acc, &mut next);
    assert!((pressure - 6.0).abs() < 1e-10);

    // Zero-rank buffer is skipped
    let pressure_zero =
        tensor_train_fusion_pressure_witness_with_scratch(&[2, 0, 3], &mut acc, &mut next);
    assert!((pressure_zero - 6.0).abs() < 1e-10);
    // Buffers were reused
    assert_eq!(acc.as_ptr(), acc_ptr);
    assert_eq!(next.as_ptr(), next_ptr);

    // Threshold semantics for should_fuse_chain_witness
    assert!(!should_fuse_chain_witness(&[], 5.0));
    assert!(should_fuse_chain_witness(&[4, 4], 5.0));
    assert!(!should_fuse_chain_witness(&[4, 4], 3.0));
    assert!(should_fuse_chain_witness(&[8, 8, 8], 16.0));
    assert!(!should_fuse_chain_witness(&[8, 8, 8], 4.0));
}

#[test]
fn mori_zwanzig_clustering_witness_contracts() {
    let assignments = [0_u32, 0, 1, 1];
    let mut cluster_sizes = Vec::with_capacity(8);
    let mut projection = Vec::with_capacity(16);
    let mut out = Vec::with_capacity(8);
    cluster_sizes.extend([99; 8]);
    projection.extend([99.0; 16]);
    out.extend([99.0; 8]);

    let p_ptr = projection.as_ptr();
    let out_ptr = out.as_ptr();

    cluster_projection_matrix_witness_into(&assignments, 4, 2, &mut cluster_sizes, &mut projection);
    assert_eq!(projection.len(), 16);
    assert!((projection[0] - 0.5).abs() < 1e-10);
    assert!((projection[1] - 0.5).abs() < 1e-10);
    assert_eq!(projection[2], 0.0);
    assert_eq!(projection[3], 0.0);
    assert_eq!(projection.as_ptr(), p_ptr);

    // Out of bounds cluster assignment error
    let bad_assignments = [0_u32, 2, 0, 1];
    assert!(try_cluster_projection_matrix_witness_into(
        &bad_assignments,
        4,
        2,
        &mut cluster_sizes,
        &mut projection
    )
    .is_err());

    // Coarsen via clustering
    let state = [10.0_f64, 20.0, 100.0, 200.0];
    mori_zwanzig_coarsen_via_clustering_witness_into(
        &state,
        &assignments,
        4,
        2,
        &mut cluster_sizes,
        &mut projection,
        &mut out,
    );
    assert_eq!(out.len(), 4);
    assert!((out[0] - 15.0).abs() < 1e-10);
    assert!((out[1] - 15.0).abs() < 1e-10);
    assert!((out[2] - 150.0).abs() < 1e-10);
    assert!((out[3] - 150.0).abs() < 1e-10);
    assert_eq!(out.as_ptr(), out_ptr);
}

#[test]
fn frontier_bitset_witness_contracts() {
    // Popcount
    let frontier = [0b1010_u32, 0xFFFF_0000, 0];
    assert_eq!(frontier_popcount_witness(&frontier).unwrap(), 2 + 16);

    // Domain popcount with tail masking
    // 35 nodes => 2 words, tail mask is 0b0111 (3 bits)
    let domain_frontier = [0xFFFF_FFFF_u32, 0b1111_1111];
    let count = frontier_domain_popcount_witness(&domain_frontier, 35).unwrap();
    // Word 0: 32 bits, Word 1: 0b1111_1111 & 0b0111 = 0b0111 (3 bits) => 35 bits total
    assert_eq!(count, 35);

    // Domain popcount rejects wrong shape
    assert!(frontier_domain_popcount_witness(&domain_frontier, 70).is_err());

    // Absorb contract: validates before mutating visited, reserves next_wave
    let mut visited = vec![0b0001_u32, 0b0001];
    let neighbors = vec![0b0111_u32, 0b1000_0111];
    let mut next_wave = Vec::with_capacity(4);
    next_wave.extend([99_u32; 4]);
    let next_wave_ptr = next_wave.as_ptr();

    let (added_any, added_popcount) =
        try_frontier_absorb_witness_into(&mut visited, &neighbors, 35, &mut next_wave).unwrap();
    assert!(added_any);
    assert_eq!(added_popcount, 4);
    assert_eq!(next_wave, vec![0b0110, 0b0110]);
    assert_eq!(visited, vec![0b0111, 0b0111]);
    assert_eq!(next_wave.as_ptr(), next_wave_ptr);

    // Validation-before-mutation: bad neighbor length leaves all caller storage untouched.
    let bad_neighbors = vec![0b0111_u32];
    let before_visited = visited.clone();
    let before_next_wave = next_wave.clone();
    assert!(
        try_frontier_absorb_witness_into(&mut visited, &bad_neighbors, 35, &mut next_wave).is_err()
    );
    assert_eq!(visited, before_visited);
    assert_eq!(next_wave, before_next_wave);

    // A smaller repeated call truncates stale output without replacing its allocation.
    let mut small_visited = [0_u32];
    let (small_added, small_count) =
        frontier_absorb_witness(&mut small_visited, &[1], 1, &mut next_wave);
    assert!(small_added);
    assert_eq!(small_count, 1);
    assert_eq!(small_visited, [1]);
    assert_eq!(next_wave, [1]);
    assert_eq!(next_wave.as_ptr(), next_wave_ptr);

    // Saturation ratio contract
    let sat_words = vec![0xAAAA_AAAA_u32; 64];
    assert!((bitset_saturation_ratio_witness(&sat_words) - 0.5).abs() < 1e-12);
    assert_eq!(bitset_saturation_ratio_witness(&[]), 0.0);
}

#[test]
fn tensor_flow_forward_and_vector_graph_witness_contracts() {
    // Tensor flow forward contract
    let edge_offsets = [0, 1, 2];
    let edge_targets = [1, 0];
    let edge_kind_mask = [1, 1];
    let tensor_in = [0b01]; // node 0, ctx 0, fld 0 active
    let mut out = Vec::new();
    try_tensor_flow_forward_witness_into(
        2,
        &edge_offsets,
        &edge_targets,
        &edge_kind_mask,
        &tensor_in,
        1,
        1,
        1,
        &mut out,
    )
    .unwrap();
    // Flow forwards from node 0 to node 1: node 1 active => bit 1 set => 0b10
    assert_eq!(out, vec![0b10]);

    // Vector KNN and traversal contracts
    let vectors = [0.0, 1.0, 2.0, 3.0, 4.0];
    let (csr_offsets, csr_targets) = knn_csr_witness(&vectors, 1, 2);
    assert_eq!(csr_offsets, vec![0, 2, 4, 6, 8, 10]);
    assert_eq!(csr_targets.len(), 10);

    let query = [0.1];
    let top_k = vector_top_k_witness(&vectors, 1, &query, 0..5, 2);
    assert_eq!(top_k.len(), 2);
    assert_eq!(top_k[0].0, 0); // node 0 is closest to 0.1
    assert_eq!(top_k[1].0, 1); // node 1 is second closest

    let reached = vector_graph_traverse_from_seed_witness(0, 5, &csr_offsets, &csr_targets);
    assert_eq!(reached, vec![true, true, true, true, true]);
}
#[test]
fn megakernel_schedule_witness_contracts() {
    // Homotopy continuation contract
    let costs = vec![1.0, 2.0, 3.0];
    let schedule = schedule_via_homotopy_witness(&costs, 100, 0.2);
    assert_eq!(schedule.len(), 3);
    for &v in &schedule {
        assert!((0.0..=1.0).contains(&v));
        assert!(v > 0.3);
    }
    assert!(schedule[2] > schedule[1]);
    assert!(schedule[1] > schedule[0]);

    // Zero steps returns zeros
    let zero_steps = schedule_via_homotopy_witness(&costs, 0, 0.1);
    assert_eq!(zero_steps, vec![0.0, 0.0, 0.0]);

    // Zero costs returns zeros
    let zero_costs = schedule_via_homotopy_witness(&[0.0, 0.0, 0.0], 100, 0.5);
    assert_eq!(zero_costs, vec![0.0, 0.0, 0.0]);

    // Scale-aware telemetry witness contract
    let sample_small = MegakernelScaleSampleWitness {
        dispatch_cost_ns: 10.0,
        frontier_density: 0.05,
        readback_bytes: 64,
    };
    let sample_large = MegakernelScaleSampleWitness {
        dispatch_cost_ns: 1000.0,
        frontier_density: 0.95,
        readback_bytes: 4096,
    };
    let scale_schedule =
        schedule_via_scale_aware_samples_witness(&[sample_small, sample_large], 25.0, 64, 0.25);
    assert_eq!(scale_schedule.len(), 2);
    assert!(
        scale_schedule[1] > scale_schedule[0],
        "dense, readback-heavy candidate must receive stronger fusion pressure"
    );

    // Launch dominance and scale aware pressure mathematical properties
    assert_eq!(launch_dominance_witness(0.0, 10.0), 0.0);
    assert_eq!(launch_dominance_witness(10.0, 0.0), 1.0);
    assert!((launch_dominance_witness(10.0, 10.0) - 0.5).abs() < 1e-12);

    let pressure_low = scale_aware_pressure_witness(0.1, 0.1, 0.1, 0.1);
    let pressure_high = scale_aware_pressure_witness(0.9, 0.9, 0.9, 0.9);
    assert!(pressure_high > pressure_low);
}

#[test]
fn dense_bitmatrix_step_and_select_retention_set_contracts() {
    // Dense bitmatrix step on 4 nodes: 0->1, 1->2, 2->3
    // Reverse adjacency rows:
    // node 0: no incoming (0b0000)
    // node 1: from 0 (0b0001)
    // node 2: from 1 (0b0010)
    // node 3: from 2 (0b0100)
    let adj = vec![0b0000, 0b0001, 0b0010, 0b0100];
    let frontier = vec![0b0001]; // node 0 active
    let step = dense_bitmatrix_step_witness(&frontier, &adj, 4);
    assert_eq!(step, vec![0b0010]); // node 1 active

    let mut step_into = Vec::new();
    dense_bitmatrix_step_witness_into(&frontier, &adj, 4, &mut step_into);
    assert_eq!(step_into, vec![0b0010]);

    // Select retention set greedy argmax
    let mut gains = vec![5_u32, 9, 1, 7];
    let picked = select_retention_set_witness(&mut gains, 4, 3);
    assert_eq!(picked, vec![1, 1, 0, 1]);
    assert_eq!(gains, vec![0, 0, 1, 0]);

    let mut gains2 = vec![0_u32, 3, 0];
    let mut picked2 = Vec::new();
    select_retention_set_witness_into(&mut gains2, 3, 1, &mut picked2);
    assert_eq!(picked2, vec![0, 1, 0]);
    assert_eq!(gains2, vec![0, 0, 0]);
}

#[test]
fn fmm_zeroth_witnesses_known_answers_reuse_and_no_mutation() {
    let charges = [1.0, 2.0, 3.0];
    let assignments = [0, 1, 0];
    let mut moments = Vec::with_capacity(8);
    moments.push(99.0);
    let moments_ptr = moments.as_ptr();
    try_p2m_zeroth_moment_witness_into(&charges, &assignments, &mut moments)
        .expect("valid P2M inputs");
    assert_eq!(moments, [4.0, 2.0]);
    assert_eq!(moments.as_ptr(), moments_ptr);

    let distances = [0.0, 2.0, 4.0, 0.0];
    let local = m2l_zeroth_all_witness(&moments, &distances);
    assert_eq!(local, [1.0, 1.0]);
    assert_eq!(
        l2p_zeroth_all_witness(&local, &assignments, assignments.len() as u32),
        [1.0, 1.0, 1.0]
    );

    let mut preserved = vec![7.0, 8.0];
    assert!(try_p2m_zeroth_moment_witness_into(&charges, &[0, 1], &mut preserved).is_err());
    assert_eq!(preserved, [7.0, 8.0]);
    assert!(try_m2l_zeroth_all_witness_into(&moments, &[0.0], &mut preserved).is_err());
    assert_eq!(preserved, [7.0, 8.0]);
    assert!(try_l2p_zeroth_all_witness_into(&local, &[0, 2], 2, &mut preserved).is_err());
    assert_eq!(preserved, [7.0, 8.0]);
}
