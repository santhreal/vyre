//! Independent, obviously correct sequential mathematical witnesses for composite operations.
//!
//! Per Section 183.3, reference witnesses use simple sequential mathematical algorithms
//! without Blelloch scheduling, workgroup decomposition, frontier queues, or other GPU optimizations.
//! Composed Programs continue to run through the generic reference interpreter, with independent
//! known-answer cases where interpreter parity alone would compare an implementation with itself.

mod bitset;
mod causal;
mod csr;
mod csr_step;
mod encoding;
mod geometry;
mod graph;
mod graph_dataflow;
mod graph_dominator;
mod graph_matroid;
mod graph_sheaf;
mod graph_vector;
mod hash;
mod hash_transform;
mod math;
mod math_amg;
mod math_analysis;
mod math_physics;
mod math_quant;
mod math_sinkhorn;
mod math_tensor;
mod parsing;
mod pattern;
mod reasoning;
mod reduction;
mod scheduling;
mod text;

// Re-export bitset witnesses
pub use bitset::{
    bitset_and_inplace_witness, bitset_and_not_inplace_witness, bitset_and_not_witness,
    bitset_and_not_witness_into, bitset_and_witness, bitset_and_witness_into,
    bitset_clear_bit_inplace_witness, bitset_clear_bit_witness, bitset_contains_witness,
    bitset_copy_witness, bitset_difference_flag_witness, bitset_equal_witness, bitset_not_witness,
    bitset_not_witness_into, bitset_or_inplace_witness, bitset_or_witness, bitset_or_witness_into,
    bitset_popcount_witness, bitset_popcount_witness_into, bitset_saturation_ratio_witness,
    bitset_set_bit_inplace_witness, bitset_set_bit_witness, bitset_subset_of_witness,
    bitset_test_bit_witness, bitset_warm_start_witness, bitset_xor_inplace_witness,
    bitset_xor_witness, bitset_xor_witness_into, bitset_zero_inplace_witness, bitset_zero_witness,
    four_russians_binary_witness, four_russians_binary_witness_into,
    four_russians_dense_matvec_witness, four_russians_dense_matvec_witness_into,
    frontier_absorb_witness, frontier_domain_popcount_witness, frontier_popcount_witness,
    stochastic_decode_witness, stochastic_encode_witness, try_frontier_absorb_witness_into,
    try_stochastic_encode_witness_into,
};

// Re-export CSR graph and frontier witnesses
pub use csr::{
    csr_backward_closure_witness, csr_backward_step_with_change_witness,
    csr_backward_traverse_witness, csr_backward_traverse_witness_into, csr_bfs_witness,
    csr_bidirectional_closure_witness, csr_bidirectional_closure_witness_into,
    csr_bidirectional_step_witness, csr_bidirectional_step_witness_into,
    csr_closure_with_step_hook_witness, csr_forward_or_changed_closure_with_step_hook_witness,
    csr_forward_or_changed_closure_with_step_hook_witness_into,
    csr_forward_or_changed_closure_witness, csr_forward_or_changed_closure_witness_into,
    csr_forward_or_changed_witness, csr_forward_or_changed_witness_into,
    csr_forward_step_with_change_witness, csr_forward_traverse_witness,
    csr_forward_traverse_witness_into, csr_frontier_degree_sum_witness,
    csr_persistent_closure_detailed_witness, csr_persistent_closure_witness,
    csr_persistent_closure_witness_with_scratch_into, csr_queue_strided_forward_witness,
    csr_queue_strided_forward_witness_into, dense_bitmatrix_step_witness,
    dense_bitmatrix_step_witness_into, dense_boolean_matvec_witness, frontier_step_sharded_witness,
    frontier_to_queue_witness, frontier_to_queue_witness_into, merge_frontier_out_witness,
    merge_frontier_out_witness_into, node_kind_eq_witness, node_kind_eq_witness_into,
    partition_frontier_by_vertex_witness, partition_frontier_by_vertex_witness_into,
    persistent_fixpoint_into_witness, persistent_fixpoint_witness, resolve_family_witness,
    try_persistent_fixpoint_into_witness, CsrPersistentClosureWitness,
};

// Re-export graph analysis, dominator, homology, and matroid witnesses
pub use graph::{
    backdoor_descendants_check_witness, betti_persistence_witness, canonicalize_union_find_witness,
    csr_queue_split_low_forward_witness, ddnnf_evaluate_witness,
    dense_reachability_bitsets_witness, dense_scc_components_witness, dominator_frontier_witness,
    dominator_frontier_witness_into, dominator_idoms_witness, dominator_sets_idoms_witness,
    dominator_tree_witness, exploded_ifds_csr_witness, functor_apply_witness,
    functor_apply_witness_into, idoms_to_dominator_sets_witness, knn_csr_witness,
    level_wave_witness, matroid_exchange_bfs_step_witness, matroid_exchange_bfs_step_witness_into,
    matroid_intersection_augmentation_witness, matroid_intersection_augmentation_witness_into,
    matroid_select_optimal_subset_witness, matroid_select_optimal_subset_witness_into,
    motif_witness, motif_witness_into, path_reconstruct_witness, path_reconstruct_witness_into,
    reachable_witness, scc_decompose_witness, sheaf_diffusion_equilibrium_witness_into,
    sheaf_diffusion_step_witness, sheaf_diffusion_step_witness_into,
    sheaf_dominant_spectrum_witness, sheaf_dominant_spectrum_witness_into,
    sheaf_fusion_incompatible_witness, sheaf_fusion_incompatible_witness_into,
    sheaf_spectral_gap_witness, sheaf_spectral_gap_witness_into,
    sheaf_suggested_cluster_count_witness, tensor_bit_index_witness, tensor_flow_forward_witness,
    toposort_csr_into_witness, toposort_csr_with_scratch_into_witness, toposort_csr_witness,
    toposort_witness, try_ddnnf_evaluate_witness, try_ddnnf_evaluate_witness_into,
    try_exploded_ifds_csr_witness_into, try_path_reconstruct_batch_witness_into,
    try_tensor_flow_forward_witness, try_tensor_flow_forward_witness_into,
    union_find_alias_witness, vector_graph_traverse_from_seed_witness, vector_squared_l2_witness,
    vector_top_k_witness, ExplodedIfdsScratchWitness,
};

// Re-export causal graph, do-calculus, and impact prediction witnesses
pub use causal::{
    do_intervention_delete_incoming_witness, do_intervention_delete_incoming_witness_into,
    do_rule2_reverse_incoming_witness, do_rule2_reverse_incoming_witness_into,
    do_rule3_subgraph_witness, do_rule3_subgraph_witness_into, impact_from_surgery_witness_into,
    predict_impact_observation_form_witness, predict_impact_observation_form_witness_into,
    predict_impact_witness, predict_impact_witness_into, reachability_closure_witness_into,
};

// Re-export encoding, base64, RLE, and literal extraction witnesses
pub use encoding::{
    base64_decode_bytes_witness, base64_decode_packed_witness, base64_decode_packed_witness_into,
    hex_decode_packed_witness, inflate_stored_witness, rle_decode_witness, rle_decode_witness_into,
    rle_segment_lengths_witness, rle_segment_lengths_witness_into,
    rle_segment_start_offsets_witness, rle_segment_start_offsets_witness_into,
    try_base64_decode_packed_witness, try_base64_decode_packed_witness_into,
    try_rle_decode_witness_into, try_rle_segment_lengths_witness_into,
    try_rle_segment_start_offsets_witness_into, vsa_fingerprint_witness,
    ziftsieve_extract_literals_witness, Base64DecodeWitnessError, InflateStoredWitness,
    ZiftsieveLiteralWitness,
};

// Re-export text analysis and metrics witnesses
pub use text::{
    byte_histogram_witness, char_class_witness, encoding_classify_histogram_witness,
    line_index_witness, shannon_entropy_bits_per_byte_witness, utf8_histogram_shape_counts_witness,
    utf8_shape_counts_witness, utf8_validate_witness,
};

// Re-export pattern matching and bracket pairing witnesses
pub use pattern::{
    ascii_case_variants_witness, bracket_match_witness, bracket_match_witness_into,
    cap_regions_per_pattern_survivors_witness, classic_ac_bounded_ranges_scan_witness,
    classic_ac_candidate_end_byte_mask_words_witness,
    classic_ac_candidate_suffix2_mask_words_witness,
    classic_ac_candidate_suffix3_bloom_words_ci_witness,
    classic_ac_candidate_suffix3_bloom_words_witness, classic_ac_scan_counts_witness,
    classic_ac_scan_witness, classic_ac_suffix3_bloom_contains_witness,
    compact_first_per_region_pattern_survivors_witness, dedup_regions_survivor_flags_witness,
    dedup_regions_witness, dedup_regions_witness_in_place, dfa_scan_accept_witness,
    match_post_process_witness, planar_rewrite_schedule_witness, region_of_witness,
    sort_regions_witness, sort_regions_witness_in_place, subgroup_nfa_step_witness,
    subgroup_nfa_step_witness_into, try_match_post_process_records_into,
    try_match_post_process_witness, try_match_post_process_witness_into, WitnessPostProcessError,
    WitnessPostProcessedMatch,
};

// Re-export sequential parsing witnesses
pub use parsing::{
    is_structural_whitespace_witness, line_splice_classify_witness,
    line_splice_classify_witness_into, parse_lr_witness, whitespace_classify_word_witness,
    whitespace_classify_word_witness_into, LrAction, LrProduction, ParseLrWitnessError,
};

// Re-export hash compression witnesses
pub use hash::{
    adler32_chunk_witness, adler32_combine_chunks_witness, adler32_combine_state_witness,
    adler32_finalize_witness, adler32_initial_a_witness, adler32_initial_b_witness,
    adler32_update_byte_witness, adler32_witness, blake3_g_witness, blake3_round_witness,
    count_sketch_query_into_witness, count_sketch_query_witness, count_sketch_table_len,
    count_sketch_update_witness, crc32_combine_chunks_witness, crc32_combine_witness,
    crc32_finalize_witness, crc32_initial_state_witness, crc32_map_reduce_plan_witness,
    crc32_pack_chunks_witness, crc32_pair_reduce_chunk_words_witness,
    crc32_pair_reduce_chunks_witness, crc32_table_witness, crc32_unpack_chunks_witness,
    crc32_update_byte_witness, crc32_witness, fnv1a32_initial_state_witness,
    fnv1a32_mul_xor_word_witness, fnv1a32_update_byte_witness, fnv1a32_witness,
    fnv1a64_initial_state_witness, fnv1a64_update_byte_witness, fnv1a64_witness,
    hamming_similarity_witness, hypervector_majority_bundle_into_witness,
    hypervector_majority_bundle_witness, hypervector_xor_bind_into_witness,
    hypervector_xor_bind_witness, multi_hash_witness, ntt_bit_reverse_witness, ntt_forward_witness,
    ntt_inverse_witness, ntt_mod_add_witness, ntt_mod_mul_witness, ntt_mod_pow_witness,
    ntt_mod_sub_witness, sparse_fft_bin_hash_into_witness, sparse_fft_bin_hash_witness,
    sparse_fft_voting_recovery_into_witness, sparse_fft_voting_recovery_witness,
    try_count_sketch_query_into_witness, try_hypervector_majority_bundle_into_witness,
    try_hypervector_xor_bind_into_witness, try_sparse_fft_bin_hash_into_witness,
    try_sparse_fft_voting_recovery_into_witness, Adler32ChunkWitness, Crc32ChunkWitness,
    Crc32MapReducePlanWitness, Crc32MapReduceStepKindWitness, Crc32MapReduceStepWitness,
    ADLER32_MOD_WITNESS, CRC32_INIT_WITNESS, CRC32_POLY_WITNESS, FNV1A32_OFFSET_WITNESS,
    FNV1A32_PRIME_WITNESS, FNV1A64_OFFSET_WITNESS, FNV1A64_PRIME_WITNESS,
};
// Re-export sequential reasoning witnesses
pub use reasoning::{
    adjoint_pair_witness, adjustment_set_ordering_is_safe_witness,
    adjustment_set_pass_descendants_witness, compile_dnnf_witness, compose_passes_witness,
    compose_passes_witness_into, dnnf_is_satisfiable_witness, dnnf_is_tautology_witness,
    dnnf_model_count_witness, evaluate_condition_witness, evaluate_formula_witness,
    identity_functor_witness, identity_functor_witness_into, kan_extension_at_witness,
    kan_extension_table_witness, natural_transformation_count_witness, passes_commute_on_witness,
    yoneda_embedding_witness, zx_color_change_witness, zx_identity_removal_witness,
    zx_simplified_diagram_witness, zx_spider_fusion_witness, AdjointPair, DnnfDag, DnnfGate,
    FiniteCategory, FiniteFunctor, KanDirection, RuleConditionWitness,
    RuleEvaluationContextWitness, RuleFormulaWitness, ZxColor, ZxDiagram, ZxSpider,
};

// Re-export math, linear algebra, and filter witnesses
pub use math::{
    amari_alpha_step_witness, amari_alpha_step_witness_into, amg_residual_witness_into,
    amg_solve_to_tolerance_witness, amg_solve_to_tolerance_witness_into,
    amg_solve_to_tolerance_witness_with_scratch_into, amg_v_cycle_witness,
    amg_v_cycle_witness_into, amg_v_cycle_witness_with_scratch_into, argmax_of_marginals_witness,
    argmin_cost_witness, bellman_shortest_path_witness, bhattacharyya_coefficient_witness,
    bigint_add_carry_witness, bigint_add_carry_witness_into, chebyshev_filter_witness,
    chebyshev_filter_witness_into, cluster_projection_matrix_witness,
    cluster_projection_matrix_witness_into, compose_ir_arrows_witness,
    composition_associates_witness, conformal_rank_witness, conformal_threshold_witness,
    conv1d_witness, conv1d_witness_into, dense_matrix_multiply_witness,
    dense_matrix_multiply_witness_into, differentiable_argmax_witness,
    differentiable_argmax_witness_into, differentiable_autotune_gradient_witness,
    differentiable_autotune_gradient_witness_into, differentiable_autotune_pick_config_witness,
    differentiable_autotune_pick_config_witness_into, dp_clip_per_sample_witness,
    dp_clip_per_sample_witness_into, fisher_rao_distance_witness, fractional_derivative_witness,
    fusion_affinity_witness, fusion_affinity_witness_into, gaussian_rdp_step_witness,
    gaussian_rdp_step_witness_into, greedy_tensor_contract_order_witness,
    grunwald_letnikov_kernel_witness, hensel_lift_step_witness, homotopy_euler_predictor_witness,
    i4x8_batched_matmul_f32_scaled_witness, i4x8_batched_matmul_top1_f32_scaled_witness,
    i4x8_batched_matvec_f32_scaled_witness, i4x8_dot_f32_scaled_witness, i4x8_dot_i32_witness,
    i4x8_matvec_f32_scaled_witness, identity_arrow_witness, identity_matrix_witness,
    identity_matrix_witness_into, iht_top_k_witness, iht_top_k_witness_into, im2col_3x3_witness,
    im2col_3x3_witness_into, interval_merge_witness, is_psd_matrix_witness,
    jacobi_solve_to_tolerance_witness, jacobi_solve_to_tolerance_witness_into,
    kernel_to_fixed_16_16_witness, kernel_to_fixed_16_16_witness_into, kfac_block_inverse_witness,
    kfac_block_inverse_witness_into, l2p_zeroth_all_witness, l2p_zeroth_eval_witness,
    linear_homotopy_witness, m2l_zeroth_all_witness, m2l_zeroth_translate_witness,
    matmul_u32_witness, matmul_u32_witness_into, modified_gram_schmidt_witness,
    modified_gram_schmidt_witness_into, mori_zwanzig_coarsen_via_clustering_witness,
    mori_zwanzig_coarsen_via_clustering_witness_into, mori_zwanzig_project_witness,
    mori_zwanzig_project_witness_into, mp_edge_clip_witness, mp_edge_clip_witness_into,
    natural_gradient_autotune_step_witness, natural_gradient_autotune_step_witness_into,
    natural_gradient_block_apply_witness, natural_gradient_block_apply_witness_into,
    negative_truncator_coeffs_witness, negative_truncator_coeffs_witness_into,
    newton_schulz_inverse_sqrt_witness, newton_schulz_inverse_sqrt_witness_into,
    newton_schulz_y_step_witness, newton_schulz_y_step_witness_into,
    p2m_zeroth_moment_truncating_witness, p2m_zeroth_moment_truncating_witness_into,
    p2m_zeroth_moment_witness, pack_i4x8_witness, pack_i4x8_witness_into, predict_interval_witness,
    privacy_epsilon_from_rdp_witness, qsvt_apply_witness, qsvt_apply_witness_into,
    qsvt_apply_witness_with_scratch_into, qsvt_block_encode_witness,
    qsvt_block_encode_witness_into, rdp_to_dp_witness, resolve_bigint_carry_chain_witness,
    resolve_bigint_carry_chain_witness_into, rk4_step_witness, rk4_step_witness_into,
    rms_norm_linear_witness, scallop_join_fixpoint_witness, scallop_join_fixpoint_witness_into,
    score_denoise_step_witness, score_denoise_step_witness_into, select_retention_set_witness,
    select_retention_set_witness_into, semiring_gemm_witness, semiring_gemm_witness_into,
    should_fuse_chain_witness, simplicial_triangle_message_witness, sinkhorn_clustering_witness,
    sinkhorn_col_residual_witness, sinkhorn_iter_f64_in_place_witness_into,
    sinkhorn_iter_f64_step_witness, sinkhorn_iter_f64_step_witness_into,
    sinkhorn_iterate_f64_witness, sinkhorn_iterate_witness, sinkhorn_row_residual_witness,
    softmax_witness, softmax_witness_into, sos_gram_construct_witness,
    sos_gram_construct_witness_into, stream_compact_witness, stream_compact_witness_into,
    sum_product_evaluate_witness, sum_product_evaluate_witness_into, tensor_scc_witness,
    tensor_train_contract_step_witness, tensor_train_contract_step_witness_into,
    tensor_train_full_chain_witness, tensor_train_full_chain_witness_into,
    tensor_train_fusion_pressure_witness, tensor_train_fusion_pressure_witness_with_scratch,
    try_amg_solve_to_tolerance_witness_into, try_amg_solve_to_tolerance_witness_with_scratch_into,
    try_amg_v_cycle_witness_into, try_amg_v_cycle_witness_with_scratch_into,
    try_argmin_cost_witness, try_chebyshev_filter_witness, try_chebyshev_filter_witness_into,
    try_cluster_projection_matrix_witness_into, try_differentiable_autotune_gradient_witness_into,
    try_differentiable_autotune_pick_config_witness_into, try_fractional_derivative_witness,
    try_fractional_derivative_witness_into, try_gaussian_rdp_step_witness_into,
    try_grunwald_letnikov_kernel_witness, try_grunwald_letnikov_kernel_witness_into,
    try_identity_matrix_witness_into, try_jacobi_solve_to_tolerance_witness_into,
    try_kernel_to_fixed_16_16_witness_into, try_l2p_zeroth_all_witness_into,
    try_m2l_zeroth_all_witness_into, try_mori_zwanzig_coarsen_via_clustering_witness_into,
    try_natural_gradient_autotune_step_witness_into, try_natural_gradient_block_apply_witness_into,
    try_p2m_zeroth_moment_truncating_witness_into, try_p2m_zeroth_moment_witness_into,
    try_sinkhorn_iter_f64_in_place_witness_into, try_sinkhorn_iterate_f64_witness,
    try_sinkhorn_iterate_f64_witness_into, try_sinkhorn_iterate_witness,
    try_sinkhorn_iterate_witness_into, try_tensor_train_contract_step_witness,
    try_tensor_train_contract_step_witness_into, try_tensor_train_full_chain_witness,
    try_tensor_train_full_chain_witness_into,
    try_tensor_train_fusion_pressure_witness_with_scratch, unpack_i4x8_witness,
    unpack_i4x8_witness_into, vietoris_rips_edge_filter_witness, vietoris_rips_edges_witness,
    AmgSolveScratchWitness, AmgVcycleScratchWitness, NewtonSchulzScratchWitness,
};
// Re-export sequential frontier scheduling witnesses
pub use scheduling::{
    plan_frontier_typed_ir_witness, FrontierDependencyWitness, FrontierDomainWitness,
    FrontierNodeWitness, FrontierTypedPlanWitness, FrontierTypedPlanWitnessError,
    FrontierWaveWitness,
};

// Re-export scan, reduction, and array movement witnesses
pub use reduction::{
    exclusive_prefix_sum_witness, gather_witness, gather_witness_into, histogram_witness,
    histogram_witness_into, inclusive_prefix_sum_witness, inclusive_prefix_sum_witness_into,
    prefix_scan_witness, prefix_scan_witness_into, radix_sort_masked_witness, range_counts_witness,
    reduce_all_witness, reduce_any_witness, reduce_count_non_zero_witness, reduce_count_witness,
    reduce_max_f32_witness, reduce_max_witness, reduce_min_witness, reduce_sum_f32_witness,
    reduce_workgroup_any_witness, scatter_witness, scatter_witness_into,
    segment_reduce_sum_witness, segment_reduce_sum_witness_into,
    try_segment_reduce_sum_witness_into, wrapping_sum_witness,
};

// Re-export geometric and equivariant witnesses
pub use geometry::{clifford2_product_witness, tfn_scalar_mix_witness};
