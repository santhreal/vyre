//! Canonical implementation-family taxonomy for registered-op dedup audits.

/// Return the shared builder family that owns an operation's emitted IR shape.
#[must_use]
pub(crate) fn implementation_family_id(op_id: &str) -> Option<&'static str> {
    match op_id {
        "vyre-primitives::bitset::and"
        | "vyre-primitives::bitset::and_not"
        | "vyre-primitives::bitset::or"
        | "vyre-primitives::bitset::stochastic_and_mul"
        | "vyre-primitives::bitset::xor" => Some("vyre-primitives::bitset::binary_word"),
        "vyre-primitives::bitset::and_into"
        | "vyre-primitives::bitset::and_not_into"
        | "vyre-primitives::bitset::copy"
        | "vyre-primitives::bitset::or_into"
        | "vyre-primitives::bitset::xor_into" => {
            Some("vyre-primitives::bitset::target_operand_word")
        }
        "vyre-primitives::bitset::equal" | "vyre-primitives::bitset::subset_of" => {
            Some("vyre-primitives::bitset::relation")
        }
        "vyre-primitives::bitset::set_bit" | "vyre-primitives::bitset::clear_bit" => {
            Some("vyre-primitives::bitset::bit_update")
        }
        "vyre-primitives::predicate::literal_of"
        | "vyre-primitives::predicate::node_kind_eq"
        | "vyre-primitives::label::resolve_family" => Some("vyre-primitives::nodeset_filter"),
        "vyre-primitives::graph::vast_walk_preorder"
        | "vyre-primitives::graph::vast_walk_postorder" => {
            Some("vyre-primitives::graph::vast_tree_walk_order")
        }
        "vyre-primitives::graph::csr_forward_traverse"
        | "vyre-primitives::graph::csr_backward_traverse"
        | "vyre-primitives::graph::csr_frontier_degree_sum"
        | "vyre-primitives::graph::tensor_flow_forward"
        | "vyre-primitives::predicate::call_to"
        | "vyre-primitives::predicate::edge"
        | "vyre-primitives::predicate::return_value_of"
        | "vyre-primitives::predicate::arg_of"
        | "vyre-primitives::predicate::size_argument_of" => {
            Some("vyre-primitives::graph::csr_frontier_step")
        }
        "vyre-primitives::graph::csr_forward_or_changed" => {
            Some("vyre-primitives::graph::outgoing_frontier_or_changed")
        }
        "vyre-primitives::graph::csr_backward_or_changed" => {
            Some("vyre-primitives::graph::incoming_frontier_or_changed")
        }
        "vyre-primitives::graph::functor_apply" => {
            Some("vyre-primitives::graph::target_centric_functor_apply")
        }
        "vyre-intrinsics::hardware::workgroup_barrier"
        | "vyre-intrinsics::hardware::storage_barrier" => {
            Some("vyre-intrinsics::hardware::barrier_identity_u32_program")
        }
        "vyre-intrinsics::hardware::bit_reverse_u32"
        | "vyre-intrinsics::hardware::popcount_u32" => {
            Some("vyre-intrinsics::hardware::unary_u32_program")
        }
        "vyre-primitives::graph::monoidal_compose"
        | "vyre-primitives::math::tensor_network_pair_contract"
        | "vyre-primitives::math::semiring_gemm" => {
            Some("vyre-primitives::fixed_u32_matmul::u32_matmul_program")
        }
        "vyre-primitives::math::sinkhorn_scale" | "vyre-primitives::math::gaussian_rdp_step" => {
            Some("vyre-primitives::math::u32_binary_map")
        }
        "vyre-primitives::math::iht_threshold" | "vyre-primitives::math::mp_edge_clip" => {
            Some("vyre-primitives::math::u32_vector_scalar_map")
        }
        "vyre-primitives::reduce::sum"
        | "vyre-primitives::reduce::min"
        | "vyre-primitives::reduce::max"
        | "vyre-primitives::reduce::count"
        | "vyre-primitives::reduce::count_non_zero"
        | "vyre-primitives::reduce::any"
        | "vyre-primitives::reduce::all" => Some("vyre-primitives::reduce::atomic_grid_stride_u32"),
        "vyre-primitives::reduce::gather" | "vyre-primitives::reduce::scatter" => {
            Some("vyre-primitives::reduce::indexed_move")
        }
        "vyre-primitives::reduce::workgroup_sum_f32"
        | "vyre-primitives::reduce::workgroup_sum_u32"
        | "vyre-primitives::reduce::workgroup_max_f32" => {
            Some("vyre-primitives::reduce::workgroup_tree")
        }
        "vyre-libs::math::atomic::atomic_add_u32"
        | "vyre-libs::math::atomic::atomic_and_u32"
        | "vyre-libs::math::atomic::atomic_exchange_u32"
        | "vyre-libs::math::atomic::atomic_max_u32"
        | "vyre-libs::math::atomic::atomic_min_u32"
        | "vyre-libs::math::atomic::atomic_or_u32"
        | "vyre-libs::math::atomic::atomic_xor_u32" => {
            Some("vyre-libs::math::atomic::build_atomic_serial")
        }
        "vyre-libs::logical::nand"
        | "vyre-libs::logical::nor"
        | "vyre-libs::math::algebra::join"
        | "vyre-libs::math::algebra::meet"
        | "vyre-libs::math::algebra::minplus_mul"
        | "vyre-libs::math::avg_floor" => {
            Some("vyre-libs::math::elementwise::u32_elementwise_binary")
        }
        "vyre-libs::math::lzcnt_u32"
        | "vyre-libs::math::tzcnt_u32"
        | "vyre-libs::math::wrapping_neg" => {
            Some("vyre-libs::math::elementwise::u32_elementwise_unary")
        }
        "vyre-libs::nn::gelu" | "vyre-libs::nn::leaky_relu_sq" => {
            Some("vyre-libs::nn::activation::f32_unary_activation_program")
        }
        "vyre-libs::nn::residual_add" => {
            Some("vyre-libs::nn::activation::typed_binary_activation_program")
        }
        "vyre-libs::nn::sigmoid_gate" | "vyre-libs::nn::swiglu" => {
            Some("vyre-libs::nn::activation::typed_sigmoid_gate_program")
        }
        "vyre-libs::nn::rms_norm" | "vyre-libs::nn::softmax" => {
            Some("vyre-libs::builder::strided_writeback_child")
        }
        "vyre-libs::parsing::c11_gnu_inline_asm_pass" => {
            Some("vyre-libs::parsing::c::atomic_collect_u32")
        }
        "vyre-libs::parsing::c_sema_scope.scope"
        | "vyre-libs::parsing::c_sema_scope.scope.brace"
        | "vyre-libs::parsing::c_sema_scope.scope.function_parameters"
        | "vyre-libs::parsing::c_sema_scope.decl"
        | "vyre-libs::parsing::c_sema_scope.identifier_intern" => {
            Some("vyre-libs::parsing::c_sema_scope_phase")
        }
        _ => None,
    }
}

/// Return whether two registered operations already use one shared implementation family.
#[must_use]
pub(crate) fn same_implementation_family(left_id: &str, right_id: &str) -> bool {
    let Some(left_family) = implementation_family_id(left_id) else {
        return false;
    };
    implementation_family_id(right_id) == Some(left_family)
}

/// Return whether similar scaffolding belongs to deliberately distinct shared families.
#[must_use]
pub(crate) fn known_distinct_implementation_families(left_id: &str, right_id: &str) -> bool {
    let Some(left_family) = implementation_family_id(left_id) else {
        return false;
    };
    let Some(right_family) = implementation_family_id(right_id) else {
        return false;
    };
    matches!(
        (left_family, right_family),
        (
            "vyre-intrinsics::hardware::barrier_identity_u32_program",
            "vyre-intrinsics::hardware::unary_u32_program"
        ) | (
            "vyre-intrinsics::hardware::unary_u32_program",
            "vyre-intrinsics::hardware::barrier_identity_u32_program"
        ) | (
            "vyre-primitives::graph::outgoing_frontier_or_changed",
            "vyre-primitives::graph::incoming_frontier_or_changed"
        ) | (
            "vyre-primitives::graph::incoming_frontier_or_changed",
            "vyre-primitives::graph::outgoing_frontier_or_changed"
        ) | (
            "vyre-primitives::reduce::indexed_move",
            "vyre-primitives::graph::target_centric_functor_apply"
        ) | (
            "vyre-primitives::graph::target_centric_functor_apply",
            "vyre-primitives::reduce::indexed_move"
        ) | (
            "vyre-libs::nn::activation::typed_binary_activation_program",
            "vyre-libs::nn::activation::typed_sigmoid_gate_program"
        ) | (
            "vyre-libs::nn::activation::typed_sigmoid_gate_program",
            "vyre-libs::nn::activation::typed_binary_activation_program"
        )
    )
}
