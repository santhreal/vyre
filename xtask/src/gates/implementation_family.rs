//! Canonical implementation-family taxonomy for registered-op dedup audits.
//!
//! A family names the shared builder that emits an operation's IR shape. Two
//! operations in one family are one implementation reached under two names, so a
//! dedup audit that measures shape alone would report them as copies forever.
//!
//! The taxonomy is a table rather than a match so the audit can enumerate it and
//! prove every row still names a registered operation. A row whose operation was
//! renamed or deleted classifies nothing, and nothing says so: the audit keeps
//! passing and a reader takes the row as evidence the operation is covered.

/// One row per registered operation whose IR shape is emitted by a shared
/// builder, as `(operation id, family)`. The family is the path of the builder
/// that owns the emitted body.
pub const IMPLEMENTATION_FAMILY_ROWS: &[(&str, &str)] = &[
    (
        "vyre-primitives::bitset::and",
        "vyre-primitives::bitset::binary_word",
    ),
    (
        "vyre-primitives::bitset::and_not",
        "vyre-primitives::bitset::binary_word",
    ),
    (
        "vyre-primitives::bitset::or",
        "vyre-primitives::bitset::binary_word",
    ),
    (
        "vyre-primitives::bitset::stochastic_and_mul",
        "vyre-primitives::bitset::binary_word",
    ),
    (
        "vyre-primitives::bitset::xor",
        "vyre-primitives::bitset::binary_word",
    ),
    (
        "vyre-primitives::bitset::and_into",
        "vyre-primitives::bitset::target_operand_word",
    ),
    (
        "vyre-primitives::bitset::and_not_into",
        "vyre-primitives::bitset::target_operand_word",
    ),
    (
        "vyre-primitives::bitset::copy",
        "vyre-primitives::bitset::target_operand_word",
    ),
    (
        "vyre-primitives::bitset::or_into",
        "vyre-primitives::bitset::target_operand_word",
    ),
    (
        "vyre-primitives::bitset::xor_into",
        "vyre-primitives::bitset::target_operand_word",
    ),
    (
        "vyre-primitives::bitset::equal",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::bitset::subset_of",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::bitset::set_bit",
        "vyre-primitives::bitset::bit_update",
    ),
    (
        "vyre-primitives::bitset::clear_bit",
        "vyre-primitives::bitset::bit_update",
    ),
    (
        "vyre-primitives::predicate::literal_of",
        "vyre-primitives::nodeset_filter",
    ),
    (
        "vyre-primitives::predicate::node_kind_eq",
        "vyre-primitives::nodeset_filter",
    ),
    (
        "vyre-primitives::label::resolve_family",
        "vyre-primitives::nodeset_filter",
    ),
    (
        "vyre-primitives::graph::vast_walk_preorder",
        "vyre-primitives::graph::vast_tree_walk_order",
    ),
    (
        "vyre-primitives::graph::vast_walk_postorder",
        "vyre-primitives::graph::vast_tree_walk_order",
    ),
    // Every row below reaches `forward_body` or `backward_body` in
    // `csr_frontier_step`, including the excluding variant, whose only
    // difference is the excluded-source operand that body already takes.
    (
        "vyre-primitives::graph::csr_forward_traverse",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::graph::csr_forward_traverse_excluding",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::graph::csr_backward_traverse",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::graph::csr_frontier_degree_sum",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::graph::tensor_flow_forward",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::predicate::call_to",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::predicate::edge",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::predicate::return_value_of",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::predicate::arg_of",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::predicate::size_argument_of",
        "vyre-primitives::graph::csr_frontier_step",
    ),
    (
        "vyre-primitives::graph::csr_forward_or_changed",
        "vyre-primitives::graph::outgoing_frontier_or_changed",
    ),
    (
        "vyre-primitives::graph::csr_backward_or_changed",
        "vyre-primitives::graph::incoming_frontier_or_changed",
    ),
    (
        "vyre-primitives::graph::functor_apply",
        "vyre-primitives::graph::target_centric_functor_apply",
    ),
    (
        "vyre-primitives::hardware::workgroup_barrier",
        "vyre-primitives::hardware::barrier_identity_u32_program",
    ),
    (
        "vyre-primitives::hardware::storage_barrier",
        "vyre-primitives::hardware::barrier_identity_u32_program",
    ),
    (
        "vyre-primitives::hardware::bit_reverse_u32",
        "vyre-primitives::hardware::unary_u32_program",
    ),
    (
        "vyre-primitives::hardware::popcount_u32",
        "vyre-primitives::hardware::unary_u32_program",
    ),
    (
        "vyre-primitives::graph::monoidal_compose",
        "vyre-primitives::fixed_u32_matmul::u32_matmul_program",
    ),
    (
        "vyre-primitives::math::tensor_network_pair_contract",
        "vyre-primitives::fixed_u32_matmul::u32_matmul_program",
    ),
    (
        "vyre-primitives::math::semiring_gemm",
        "vyre-primitives::fixed_u32_matmul::u32_matmul_program",
    ),
    (
        "vyre-primitives::math::sinkhorn_scale",
        "vyre-primitives::math::u32_binary_map",
    ),
    (
        "vyre-primitives::math::gaussian_rdp_step",
        "vyre-primitives::math::u32_binary_map",
    ),
    (
        "vyre-primitives::math::iht_threshold",
        "vyre-primitives::math::u32_vector_scalar_map",
    ),
    (
        "vyre-primitives::math::mp_edge_clip",
        "vyre-primitives::math::u32_vector_scalar_map",
    ),
    (
        "vyre-primitives::reduce::sum",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::min",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::max",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::count",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::count_non_zero",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::any",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::all",
        "vyre-primitives::reduce::atomic_grid_stride_u32",
    ),
    (
        "vyre-primitives::reduce::gather",
        "vyre-primitives::reduce::indexed_move",
    ),
    (
        "vyre-primitives::reduce::scatter",
        "vyre-primitives::reduce::indexed_move",
    ),
    (
        "vyre-primitives::reduce::workgroup_sum_f32",
        "vyre-primitives::reduce::workgroup_tree",
    ),
    (
        "vyre-primitives::reduce::workgroup_sum_u32",
        "vyre-primitives::reduce::workgroup_tree",
    ),
    (
        "vyre-primitives::reduce::workgroup_max_f32",
        "vyre-primitives::reduce::workgroup_tree",
    ),
    (
        "vyre-libs::math::atomic::atomic_add_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_and_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_exchange_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_max_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_min_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_or_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::math::atomic::atomic_xor_u32",
        "vyre-libs::math::atomic::build_atomic_serial",
    ),
    (
        "vyre-libs::logical::nand",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::logical::nor",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::math::algebra::join",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::math::algebra::meet",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::math::algebra::minplus_mul",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::math::avg_floor",
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
    ),
    (
        "vyre-libs::math::lzcnt_u32",
        "vyre-libs::builder::elementwise::u32_elementwise_unary",
    ),
    (
        "vyre-libs::math::tzcnt_u32",
        "vyre-libs::builder::elementwise::u32_elementwise_unary",
    ),
    (
        "vyre-libs::math::wrapping_neg",
        "vyre-libs::builder::elementwise::u32_elementwise_unary",
    ),
    (
        "vyre-libs::quant::int8_pack",
        "vyre-libs::builder::elementwise::u32_elementwise_unary",
    ),
    (
        "vyre-libs::nn::gelu",
        "vyre-libs::nn::activation::f32_unary_activation_program",
    ),
    (
        "vyre-libs::nn::leaky_relu_sq",
        "vyre-libs::nn::activation::f32_unary_activation_program",
    ),
    (
        "vyre-libs::nn::residual_add",
        "vyre-libs::nn::activation::typed_binary_activation_program",
    ),
    (
        "vyre-libs::nn::sigmoid_gate",
        "vyre-libs::nn::activation::typed_sigmoid_gate_program",
    ),
    (
        "vyre-libs::nn::swiglu",
        "vyre-libs::nn::activation::typed_sigmoid_gate_program",
    ),
    (
        "vyre-libs::nn::skip_gate",
        "vyre-libs::builder::indexed_map",
    ),
    (
        "vyre-libs::optim::ema_apply",
        "vyre-libs::builder::indexed_map",
    ),
    (
        "vyre-libs::math::reduce_mean",
        "vyre-libs::builder::tiled_reduce",
    ),
    (
        "vyre-libs::nn::layer_norm",
        "vyre-libs::builder::tiled_reduce",
    ),
    (
        "vyre-libs::nn::rms_norm",
        "vyre-libs::builder::tiled_reduce",
    ),
    (
        "vyre-libs::nn::softmax",
        "vyre-libs::builder::tiled_reduce",
    ),
];

/// Family pairs that emit the same shape from deliberately separate builders.
///
/// Each row is one unordered pair, read in both directions. A pair belongs here
/// only when the two builders were compared and kept apart for a reason the
/// shape cannot express: a barrier that must not be reordered into a unary map,
/// a direction that must stay a direction, a gather that must not become a
/// scatter, an activation whose operand typing differs.
pub const DISTINCT_FAMILY_PAIRS: &[(&str, &str)] = &[
    (
        "vyre-primitives::hardware::barrier_identity_u32_program",
        "vyre-primitives::hardware::unary_u32_program",
    ),
    (
        "vyre-primitives::graph::outgoing_frontier_or_changed",
        "vyre-primitives::graph::incoming_frontier_or_changed",
    ),
    (
        "vyre-primitives::reduce::indexed_move",
        "vyre-primitives::graph::target_centric_functor_apply",
    ),
    (
        "vyre-libs::nn::activation::typed_binary_activation_program",
        "vyre-libs::nn::activation::typed_sigmoid_gate_program",
    ),
    (
        "vyre-libs::builder::elementwise::u32_elementwise_binary",
        "vyre-libs::builder::elementwise::u32_elementwise_unary",
    ),
];

/// Operation pairs whose IR shapes agree past the bucket key and whose
/// algorithms were read side by side and judged distinct, as
/// `(one, other, reason)`.
///
/// A shape verdict cannot tell a shared algorithm from a shared IR idiom: a
/// guarded lane index, a row-major loop nest, and straight-line unrolled
/// arithmetic have one fingerprint whatever work they do. So the shape verdict
/// is "unreviewed", not "duplicate", and the reviewer records the outcome. The
/// two outcomes are a shared builder, which is a family row above, and a
/// reviewed pair, which is a row here carrying the reason the shape cannot
/// express.
pub const REVIEWED_DISTINCT_OPERATIONS: &[(&str, &str, &str)] = &[
    (
        "vyre-primitives::graph::path_reconstruct",
        "vyre-primitives::text::encoding_classify",
        "a bounded serial loop that stores one element per step; one follows parent pointers to \
         materialize a path, the other reads 256 histogram bins to pick an encoding class",
    ),
    (
        "vyre-primitives::graph::functor_apply",
        "vyre-primitives::reduce::histogram",
        "one lane reading through an index table and storing one element; the functor carries a \
         schema column mapping and the histogram counts occurrences of the bin its lane owns",
    ),
    (
        "vyre-primitives::math::matrix_identity_fill",
        "vyre-primitives::parsing::planar_rewrite_schedule",
        "row-major two-dimensional index arithmetic under a per-cell predicate; the fill compares \
         row against column and stores a constant, the rewrite matches a k by k window against a \
         pattern and stores its replacement",
    ),
    (
        "vyre-primitives::math::tensor_train_decompose",
        "vyre-primitives::parsing::planar_rewrite_schedule",
        "the loop nest that walks one mode index range is all the two share; the decomposition \
         composes an eigensolve and partial dot products per mode, and the shape carries neither \
         the truncation nor the window match",
    ),
    (
        "vyre-primitives::decode::ziftsieve_literal_copy",
        "vyre-primitives::math::bigint_add_carry",
        "one lane per element over a contiguous range with a running offset; the copy moves bytes \
         to a prefix-summed destination and the addition propagates a carry between limbs, and a \
         carry chain cannot be a copy",
    ),
    (
        "vyre-libs::math::fft::fft4_complex",
        "vyre-primitives::hash::blake3_g",
        "straight-line unrolled arithmetic over a fixed small operand set has one fingerprint \
         whatever the arithmetic is; one is four complex butterflies over f32 twiddles, the other \
         is the BLAKE3 four-word mixing of add, xor and rotate over u32",
    ),
    (
        "vyre-primitives::graph::dominator_tree_intersect_step",
        "vyre-primitives::math::softmax_step",
        "a lane-zero guard around one serial pass that accumulates into a binding and a second \
         serial pass that stores per element; the softmax divides every element by the sum it \
         just totalled, and the relaxation sweep walks a predecessor CSR and intersects two idom \
         parents by climbing the deeper one, which no division expresses",
    ),
];

/// Reason two registered operations were reviewed and kept apart, read in both
/// directions.
#[must_use]
pub fn reviewed_distinct_operations(left_id: &str, right_id: &str) -> Option<&'static str> {
    REVIEWED_DISTINCT_OPERATIONS
        .iter()
        .find(|(one, other, _)| {
            (*one == left_id && *other == right_id) || (*one == right_id && *other == left_id)
        })
        .map(|(_, _, reason)| *reason)
}

/// Family id a source path belongs to, used to group similar implementations.
#[must_use]
pub fn implementation_family_id(op_id: &str) -> Option<&'static str> {
    IMPLEMENTATION_FAMILY_ROWS
        .iter()
        .find(|(id, _)| *id == op_id)
        .map(|(_, family)| *family)
}

/// Return whether two registered operations already use one shared implementation family.
#[must_use]
pub fn same_implementation_family(left_id: &str, right_id: &str) -> bool {
    let Some(left_family) = implementation_family_id(left_id) else {
        return false;
    };
    implementation_family_id(right_id) == Some(left_family)
}

/// Return whether similar scaffolding belongs to deliberately distinct shared families.
#[must_use]
pub fn known_distinct_implementation_families(left_id: &str, right_id: &str) -> bool {
    let Some(left_family) = implementation_family_id(left_id) else {
        return false;
    };
    let Some(right_family) = implementation_family_id(right_id) else {
        return false;
    };
    DISTINCT_FAMILY_PAIRS.iter().any(|(one, other)| {
        (*one == left_family && *other == right_family)
            || (*one == right_family && *other == left_family)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: a distinct-pair row names two families. If either family stops being
    /// claimed by any operation, the row keeps its length and stops separating
    /// anything, and the dedup audit silently loses the exemption it records.
    #[test]
    fn every_distinct_pair_names_a_family_some_operation_claims() {
        for (one, other) in DISTINCT_FAMILY_PAIRS {
            for family in [one, other] {
                assert!(
                    IMPLEMENTATION_FAMILY_ROWS
                        .iter()
                        .any(|(_, claimed)| claimed == family),
                    "Fix: no operation row claims the family `{family}`; delete the distinct-family pair that names it or restore the row that claimed it"
                );
            }
        }
    }

    /// WHY: one operation with two families would make its family answer depend
    /// on table order, so two operations could be in one family and not in it.
    #[test]
    fn no_operation_claims_two_families() {
        for (index, (id, family)) in IMPLEMENTATION_FAMILY_ROWS.iter().enumerate() {
            for (other_id, other_family) in IMPLEMENTATION_FAMILY_ROWS.iter().skip(index + 1) {
                assert!(
                    other_id != id || other_family == family,
                    "Fix: `{id}` claims both `{family}` and `{other_family}`; one operation has one shared builder"
                );
            }
        }
    }

    /// WHY: a family of one operation groups nothing, and reads as evidence that
    /// a shared builder is shared.
    #[test]
    fn a_family_that_groups_one_operation_is_paired_or_absent() {
        for (id, family) in IMPLEMENTATION_FAMILY_ROWS {
            let claimants = IMPLEMENTATION_FAMILY_ROWS
                .iter()
                .filter(|(_, claimed)| claimed == family)
                .count();
            let paired = DISTINCT_FAMILY_PAIRS
                .iter()
                .any(|(one, other)| one == family || other == family);
            assert!(
                claimants > 1 || paired,
                "Fix: `{family}` is claimed only by `{id}` and is in no distinct-family pair; either it groups nothing and the row goes, or the second claimant is missing"
            );
        }
    }

    #[test]
    fn a_shared_builder_groups_its_operations() {
        assert!(same_implementation_family(
            "vyre-primitives::predicate::edge",
            "vyre-primitives::graph::csr_forward_traverse_excluding"
        ));
        assert!(!same_implementation_family(
            "vyre-primitives::predicate::edge",
            "vyre-primitives::reduce::all"
        ));
    }

    #[test]
    fn a_distinct_pair_reads_in_both_directions() {
        assert!(known_distinct_implementation_families(
            "vyre-primitives::hardware::workgroup_barrier",
            "vyre-primitives::hardware::popcount_u32"
        ));
        assert!(known_distinct_implementation_families(
            "vyre-primitives::hardware::popcount_u32",
            "vyre-primitives::hardware::workgroup_barrier"
        ));
    }

    /// WHY: a reviewed pair whose sides are one operation would suppress that
    /// operation against itself, which the pair walk never asks about, so the
    /// row would read as a judgment nothing consults.
    #[test]
    fn a_reviewed_pair_names_two_operations() {
        for (one, other, _) in REVIEWED_DISTINCT_OPERATIONS {
            assert_ne!(
                one, other,
                "Fix: the reviewed-distinct row for `{one}` names one operation twice"
            );
        }
    }

    /// WHY: the lookup returns the first match, so a second row for one pair is
    /// a reason no reader of the audit will ever be shown.
    #[test]
    fn no_reviewed_pair_is_recorded_twice_in_either_order() {
        for (index, (one, other, _)) in REVIEWED_DISTINCT_OPERATIONS.iter().enumerate() {
            for (later_one, later_other, _) in REVIEWED_DISTINCT_OPERATIONS.iter().skip(index + 1) {
                let same = (one == later_one && other == later_other)
                    || (one == later_other && other == later_one);
                assert!(
                    !same,
                    "Fix: `{one}` and `{other}` are recorded twice; keep one row and merge the reasons"
                );
            }
        }
    }

    /// WHY: the reason is the whole content of the row. An empty one suppresses
    /// the finding and records nothing about why the shapes agree.
    #[test]
    fn every_reviewed_pair_carries_a_reason() {
        for (one, other, reason) in REVIEWED_DISTINCT_OPERATIONS {
            assert!(
                reason.len() > 40,
                "Fix: `{one}` and `{other}` carry the reason `{reason}`, which is too short to \
                 name what the shared shape cannot express"
            );
        }
    }

    #[test]
    fn a_reviewed_pair_reads_in_both_directions() {
        let forward = reviewed_distinct_operations(
            "vyre-libs::math::fft::fft4_complex",
            "vyre-primitives::hash::blake3_g",
        );
        let backward = reviewed_distinct_operations(
            "vyre-primitives::hash::blake3_g",
            "vyre-libs::math::fft::fft4_complex",
        );
        assert_eq!(forward, backward);
        assert!(forward.is_some());
        assert!(reviewed_distinct_operations(
            "vyre-primitives::hash::blake3_g",
            "vyre-primitives::reduce::all"
        )
        .is_none());
    }
}
