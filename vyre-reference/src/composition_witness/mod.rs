//! Independent, obviously correct sequential mathematical witnesses for composite operations.
//!
//! Per Section 183.3, reference witnesses use simple sequential mathematical algorithms
//! without Blelloch scheduling, workgroup decomposition, frontier queues, or other GPU optimizations.
//! Composed Programs continue to run through the generic reference interpreter, with independent
//! known-answer cases where interpreter parity alone would compare an implementation with itself.

mod bitset;
mod causal;
mod csr;
mod encoding;
mod graph;
mod math;
mod pattern;
mod reduction;
mod text;

// Re-export bitset witnesses
pub use bitset::{
    bitset_and_not_witness, bitset_and_not_witness_into, bitset_and_witness,
    bitset_and_witness_into, bitset_clear_bit_witness, bitset_contains_witness,
    bitset_equal_witness, bitset_not_witness, bitset_not_witness_into, bitset_or_witness,
    bitset_or_witness_into, bitset_popcount_witness, bitset_popcount_witness_into,
    bitset_set_bit_witness, bitset_subset_of_witness, bitset_xor_witness, bitset_xor_witness_into,
    bitset_zero_witness,
};

// Re-export CSR graph and frontier witnesses
pub use csr::{
    csr_backward_traverse_witness, csr_bfs_witness, csr_closure_with_step_hook_witness,
    csr_forward_or_changed_witness, csr_forward_traverse_witness, csr_frontier_degree_sum_witness,
    csr_queue_strided_forward_witness, dense_boolean_matvec_witness, frontier_to_queue_witness,
    persistent_fixpoint_witness, resolve_family_witness,
};

// Re-export graph analysis, dominator, homology, and matroid witnesses
pub use graph::{
    betti_persistence_witness, dominator_idoms_witness, dominator_tree_witness,
    matroid_intersection_augmentation_witness,
};

// Re-export causal graph, do-calculus, and impact prediction witnesses
pub use causal::{
    do_intervention_delete_incoming_witness, do_rule2_reverse_incoming_witness,
    do_rule3_subgraph_witness, predict_impact_observation_form_witness, predict_impact_witness,
};

// Re-export encoding, base64, RLE, and literal extraction witnesses
pub use encoding::{
    base64_decode_bytes_witness, base64_decode_packed_witness, base64_decode_packed_witness_into,
    rle_decode_witness, rle_decode_witness_into, rle_segment_lengths_witness,
    rle_segment_start_offsets_witness, try_base64_decode_packed_witness,
    try_base64_decode_packed_witness_into, try_rle_decode_witness_into,
    try_rle_segment_lengths_witness_into, try_rle_segment_start_offsets_witness_into,
    ziftsieve_extract_literals_witness, Base64DecodeWitnessError, ZiftsieveLiteralWitness,
};

// Re-export text analysis and metrics witnesses
pub use text::{
    byte_histogram_witness, char_class_witness, line_index_witness,
    shannon_entropy_bits_per_byte_witness, utf8_shape_counts_witness, utf8_validate_witness,
};

// Re-export pattern matching and bracket pairing witnesses
pub use pattern::{
    bracket_match_witness, bracket_match_witness_into, match_post_process_witness,
    try_match_post_process_witness, try_match_post_process_witness_into,
};

// Re-export math, linear algebra, and filter witnesses
pub use math::{
    chebyshev_filter_witness, conformal_threshold_witness, hypervector_majority_bundle_witness,
    hypervector_xor_bind_witness, kfac_block_inverse_witness, semiring_gemm_witness,
};

// Re-export scan, reduction, and array movement witnesses
pub use reduction::{
    exclusive_prefix_sum_witness, gather_witness, gather_witness_into, histogram_witness,
    histogram_witness_into, inclusive_prefix_sum_witness, inclusive_prefix_sum_witness_into,
    prefix_scan_witness, range_counts_witness, reduce_all_witness, reduce_any_witness,
    reduce_count_non_zero_witness, reduce_count_witness, reduce_max_witness, reduce_min_witness,
    reduce_workgroup_any_witness, scatter_witness, scatter_witness_into, wrapping_sum_witness,
};
