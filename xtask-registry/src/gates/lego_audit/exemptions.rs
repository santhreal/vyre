//! Check 0: every exemption row answers to something the tree still holds.
//!
//! An exemption is a rule that a named subject is judged elsewhere. A row naming
//! something renamed or deleted stops exempting anything and nothing says so,
//! so the row reads as coverage of a subject nobody checks.

use super::*;

/// Id fragments that mark an op as one phase of a larger composition rather than
/// an op a caller reaches for.
pub(super) const PHASE_MARKERS: [&str; 6] = [
    "::hidden_projection",
    "::output_projection",
    "::softmax_stats",
    "::weight_write",
    "::v_cycle_phase",
    "::power_iteration_phase",
];

pub(super) fn is_internal_phase_op(id: &str) -> bool {
    PHASE_MARKERS.iter().any(|marker| id.contains(marker))
}

/// Explicit domain-owned Category-A leaves.
///
/// These operations emit pure, backend-neutral IR but have no lower registered
/// composition unit. Keeping this list explicit prevents an arbitrary flat
/// Tier-3 operation from bypassing the depth gate.
///
/// The three decode codecs are leaves because one module owns each codec and one
/// op id names it. Each was previously a `vyre-libs` builder wrapping a
/// registered `vyre-primitives` child, so the child Region was the second
/// module, not a lower composition unit; collapsing the pair left the emitting
/// body with nothing under it to name.
///
/// `nn::attention::online_softmax` is the `(m, l, o_acc)` recurrence itself.
/// The scalar and tiled attention entry points compose it and are no longer
/// leaves; splitting the recurrence further would name a fragment of one loop
/// nest that nothing else can call.
///
/// `llm::nucleus_select` is one serial draw in two passes over the same
/// candidate array. The first pass records the nucleus length and its mass in
/// registers and the second scales the uniform sample by that mass, which is
/// what renormalizes the nucleus without a third pass. A registered child is a
/// Region, a Region is a scope, so naming either pass would push those two
/// scalars through a scratch buffer to satisfy the shape of the rule and emit a
/// worse program than the one it judges.
pub(crate) const DECLARED_TIER3_LEAVES: [&str; 40] = [
    "vyre-libs::graph::ast_walk_postorder",
    "vyre-libs::graph::ast_walk_preorder",
    "vyre-libs::graph::csr_backward_or_changed",
    "vyre-libs::graph::csr_backward_traverse",
    "vyre-libs::graph::csr_forward_or_changed",
    "vyre-libs::graph::csr_forward_traverse",
    "vyre-libs::graph::csr_forward_traverse_excluding",
    "vyre-libs::graph::csr_frontier_degree_sum",
    "vyre-libs::graph::csr_queue_strided_forward_traverse",
    "vyre-libs::graph::dominator_frontier",
    "vyre-libs::graph::dominator_tree_depth",
    "vyre-libs::graph::dominator_tree_intersect_step",
    "vyre-libs::graph::motif",
    "vyre-libs::graph::path_reconstruct",
    "vyre-libs::graph::sum_product_evaluate",
    "vyre-libs::graph::sum_product_evaluate_leveled",
    "vyre-libs::graph::tensor_flow_forward",
    "vyre-libs::graph::union_find",
    "vyre-libs::graph::vast_walk_postorder",
    "vyre-libs::graph::vast_walk_preorder",
    "vyre-libs::llm::nucleus_select",
    "vyre-libs::math::bellman_shortest_path",
    "vyre-libs::math::fft::fft_radix2",
    "vyre-libs::math::fft::pointwise_complex_multiply_conjugate",
    "vyre-libs::math::fft::scale_conjugate_inverse",
    "vyre-libs::math::linalg::matmul_strassen_2x2",
    "vyre-libs::math::newton_schulz_poly5_f32",
    "vyre-libs::math::quantized::i4x8_batched_matmul_top1_f32_scaled",
    "vyre-libs::math::reduce_variance",
    "vyre-libs::math::scallop_join",
    "vyre-libs::math::sinkhorn_iterate",
    "vyre-libs::nn::attention::online_softmax",
    "vyre-libs::nn::linear_4bit_affine_grouped",
    "vyre-libs::nn::softmax_top_k",
    "vyre-libs::nn::top_k",
    // The bounded linear probe itself: one loop over a table of
    // (hash, earliest index) pairs, with no sub-body another op could name.
    // `ast_cse_structural_hash` is the wave that composes it and is a leaf for
    // the same reason one level up.
    "vyre-libs::parsing::ast_cse_hash_probe",
    "vyre-libs::parsing::ast_cse_structural_hash",
    "vyre-libs::parsing::core_delimiter_match",
    "vyre-libs::parsing::ssa_dominance_scan",
    "vyre-libs::pattern::bracket_match",
];

pub(crate) fn is_declared_tier3_leaf(id: &str) -> bool {
    id.starts_with("vyre-libs::bitset::")
        || id.starts_with("vyre-libs::reduce::")
        || id.starts_with("vyre-libs::hash::")
        || id.starts_with("vyre-libs::text::")
        || id.starts_with("vyre-libs::geom::")
        || id.starts_with("vyre-libs::decode::")
        || id.starts_with("vyre-libs::predicate::")
        || id.starts_with("vyre-libs::vfs::")
        || id.starts_with("vyre-libs::math::succinct::")
        || DECLARED_TIER3_LEAVES.contains(&id)
}

/// What a dead exemption row costs, and how to close it.
pub(super) const DEAD_EXEMPTION_FIX: &str =
    "delete the row: an exemption that matches no registered op exempts nothing, and it reads as coverage of an op that no longer exists";

/// Every exemption row must match something the tree still holds.
///
/// An exemption is a rule that a named subject is judged elsewhere: a phase of a
/// larger composition, a declared pure-IR leaf, an op whose shape comes from a
/// shared builder, a pair whose shapes were read side by side and judged
/// distinct, a directory that is plumbing rather than a dialect, or a leaf-stem
/// family that is acknowledged rather than renamed. A row naming something that
/// was renamed or deleted stops exempting anything, and nothing says so: the
/// list keeps its length, the audit keeps passing, and a reader takes the row as
/// evidence the subject is covered. Two rows were already in that state when the
/// check was written, and six more turned up when the plumbing directories and
/// the leaf-stem families came under the same rule.
///
/// A stem row is held to the condition it suppresses rather than to the mere
/// existence of the stem, because a family that shrank below the collision
/// threshold no longer needs acknowledging.
pub(super) fn check_0_every_exemption_is_live(report: &mut Report, ops: &[OpInfo]) {
    let libs_src = xtask::checkout::checkout_root()
        .join("vyre-libs")
        .join("src");
    for dir in dead_plumbing_rows(&libs_src) {
        report.find(Finding::new(
            format!("no directory `vyre-libs/src/{dir}` answers to the shared-plumbing row"),
            "delete the row: a plumbing row that matches no directory exempts nothing, and it reads as if a cross-dialect edge into it were already reviewed",
        ));
    }
    for dir in dead_substrate_rows(&libs_src) {
        report.find(Finding::new(
            format!("no directory `vyre-libs/src/{dir}` answers to the kernel-substrate row"),
            "delete the row: a substrate row that matches no directory exempts nothing, and it reads as if a cross-dialect edge into it were already reviewed",
        ));
    }
    for marker in PHASE_MARKERS {
        if !ops.iter().any(|op| op.id.contains(marker)) {
            report.find(Finding::new(
                format!("no registered op id contains the phase marker `{marker}`"),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
    for leaf in DECLARED_TIER3_LEAVES {
        if !ops.iter().any(|op| op.id == leaf) {
            report.find(Finding::new(
                format!("no registered op answers to the declared Tier-3 leaf `{leaf}`"),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
    let colliding = colliding_stems(ops);
    for stem in KNOWN_STEM_FAMILIES {
        if !colliding.contains_key(stem) {
            report.find(Finding::new(
                format!(
                    "the stem allowlist row `{stem}` suppresses nothing: fewer than \
                     {STEM_COLLISION_MIN} ops share it, or they already live under a `{stem}` \
                     namespace segment"
                ),
                DEAD_EXEMPTION_FIX,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This regression test keeps reviewed pure-IR leaves explicit instead of exempting every flat Tier-3 operation.
    #[test]
    fn declared_leaf_classification_is_exact() {
        assert!(is_declared_tier3_leaf("vyre-libs::nn::top_k"));
        assert!(!is_declared_tier3_leaf("vyre-libs::nn::unknown_flat_op"));
    }
}
