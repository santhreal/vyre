//! What one pair of registered operations shares: buffer contract,
//! implementation family, tier, and the verdict those facts produce.

use xtask::gates::implementation_family::{
    implementation_family_id, known_distinct_implementation_families, same_implementation_family,
};

use crate::gates::lego_audit::{OpInfo, Tier};

pub(super) fn same_buffer_contract(left: &OpInfo, right: &OpInfo) -> bool {
    left.buffer_signature == right.buffer_signature
}

pub(super) fn same_centralized_family(left: &OpInfo, right: &OpInfo) -> bool {
    same_implementation_family(&left.id, &right.id)
}

pub(super) fn known_distinct_implementation_family(left: &OpInfo, right: &OpInfo) -> bool {
    known_distinct_implementation_family_id(&left.id, &right.id)
}

fn known_distinct_implementation_family_id(left_id: &str, right_id: &str) -> bool {
    known_distinct_implementation_families(left_id, right_id)
}

pub(super) fn implementation_family(op: &OpInfo) -> Option<&'static str> {
    implementation_family_id(&op.id)
}

pub(super) fn pair_verdict(score: f64, same_contract: bool, same_family: bool) -> &'static str {
    if same_family {
        return match score {
            s if s >= 0.95 => {
                "CENTRALIZED FAMILY  -  same emitted kernel is already routed through a shared builder"
            }
            s if s >= 0.80 => {
                "CENTRALIZED FAMILY  -  similar emitted kernel already shares implementation plumbing"
            }
            _ => "loosely related centralized family",
        };
    }
    if !same_contract {
        return match score {
            s if s >= 0.95 => {
                "CONTRACT VARIANT  -  same body shape but different buffer contract; share helpers, do not merge ops"
            }
            s if s >= 0.80 => {
                "CONTRACT-SHAPE FAMILY  -  similar body under different buffer contract"
            }
            _ => "loosely related contract variant",
        };
    }
    match score {
        s if s >= 0.95 => "DUPLICATE  -  almost certainly the same shape; reuse instead",
        s if s >= 0.80 => "VERY SIMILAR  -  extract shared body to vyre-primitives or reuse",
        s if s >= 0.50 => "SIMILAR  -  same family; consider whether divergence is justified",
        _ => "loosely related",
    }
}

pub(super) fn tier_label(t: Tier) -> &'static str {
    match t {
        Tier::T2 => "T2",
        Tier::T2_5 => "T2.5",
        Tier::T3 => "T3",
        Tier::Other => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test keeps every operation routed through a shared builder in one audit taxonomy so emitted-shape similarity is not reported as reinvention.
    #[test]
    fn implementation_family_tracks_shared_builders() {
        assert_eq!(
            implementation_family_id("vyre-primitives::bitset::and"),
            implementation_family_id("vyre-primitives::bitset::stochastic_and_mul")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::predicate::size_argument_of"),
            implementation_family_id("vyre-primitives::graph::csr_backward_traverse")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::graph::csr_forward_traverse"),
            implementation_family_id("vyre-primitives::graph::csr_backward_traverse")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::graph::csr_forward_traverse"),
            implementation_family_id("vyre-primitives::graph::csr_frontier_degree_sum")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::graph::csr_forward_traverse"),
            implementation_family_id("vyre-primitives::graph::tensor_flow_forward")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::graph::vast_walk_preorder"),
            implementation_family_id("vyre-primitives::graph::vast_walk_postorder")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::math::semiring_gemm"),
            implementation_family_id("vyre-primitives::math::tensor_network_pair_contract")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::math::semiring_gemm"),
            implementation_family_id("vyre-primitives::graph::monoidal_compose")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::math::sinkhorn_scale"),
            implementation_family_id("vyre-primitives::math::gaussian_rdp_step")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::math::iht_threshold"),
            implementation_family_id("vyre-primitives::math::mp_edge_clip")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::predicate::node_kind_eq"),
            implementation_family_id("vyre-primitives::label::resolve_family")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::bitset::and_not"),
            implementation_family_id("vyre-primitives::bitset::or")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::bitset::and_into"),
            implementation_family_id("vyre-primitives::bitset::xor_into")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::bitset::copy"),
            implementation_family_id("vyre-primitives::bitset::and_into")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::bitset::set_bit"),
            implementation_family_id("vyre-primitives::bitset::clear_bit")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::reduce::workgroup_sum_f32"),
            implementation_family_id("vyre-primitives::reduce::workgroup_max_f32")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::reduce::sum"),
            implementation_family_id("vyre-primitives::reduce::any")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::reduce::sum"),
            implementation_family_id("vyre-primitives::reduce::count_non_zero")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::reduce::count"),
            implementation_family_id("vyre-primitives::reduce::all")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::reduce::gather"),
            implementation_family_id("vyre-primitives::reduce::scatter")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::math::atomic::atomic_or_u32"),
            implementation_family_id("vyre-libs::math::atomic::atomic_xor_u32")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::logical::nand"),
            implementation_family_id("vyre-libs::logical::nor")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::logical::nand"),
            implementation_family_id("vyre-libs::math::algebra::meet")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::math::algebra::join"),
            implementation_family_id("vyre-libs::math::avg_floor")
        );
        assert_eq!(
            implementation_family_id("vyre-primitives::hardware::bit_reverse_u32"),
            implementation_family_id("vyre-primitives::hardware::popcount_u32")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::math::lzcnt_u32"),
            implementation_family_id("vyre-libs::math::tzcnt_u32")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::math::wrapping_neg"),
            implementation_family_id("vyre-libs::math::lzcnt_u32")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::nn::gelu"),
            implementation_family_id("vyre-libs::nn::leaky_relu_sq")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::nn::rms_norm"),
            implementation_family_id("vyre-libs::nn::softmax")
        );
        assert_eq!(
            implementation_family_id("vyre-libs::parsing::c_sema_scope.scope"),
            implementation_family_id("vyre-libs::parsing::c_sema_scope.identifier_intern")
        );
        assert!(known_distinct_implementation_family_id(
            "vyre-primitives::hardware::workgroup_barrier",
            "vyre-primitives::hardware::bit_reverse_u32"
        ));
        assert!(known_distinct_implementation_family_id(
            "vyre-primitives::graph::csr_forward_or_changed",
            "vyre-primitives::graph::csr_backward_or_changed"
        ));
        assert!(known_distinct_implementation_family_id(
            "vyre-primitives::reduce::gather",
            "vyre-primitives::graph::functor_apply"
        ));
        assert!(!known_distinct_implementation_family_id(
            "vyre-primitives::hardware::workgroup_barrier",
            "vyre-primitives::hardware::storage_barrier"
        ));
    }

    #[test]
    fn unrelated_ops_do_not_gain_family_suppression() {
        assert_ne!(
            implementation_family_id("vyre-libs::math::atomic::atomic_or_u32"),
            implementation_family_id("vyre-primitives::bitset::and")
        );
        assert!(implementation_family_id("unknown::op").is_none());
    }
}
