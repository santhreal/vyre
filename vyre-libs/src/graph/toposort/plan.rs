//! Dispatch plan and static-input identity over a validated CSR layout.

use vyre_foundation::ir::Program;

use crate::graph::u32_slice_fingerprint;

use super::csr::{validate_toposort_csr_inputs, ToposortCsrLayout};
use super::error::ToposortCsrError;
use super::program::toposort_program;
use super::{
    TOPOSORT_DISPATCH_GRID, TOPOSORT_INDEGREE_SCRATCH_BUFFER, TOPOSORT_OFFSETS_BUFFER,
    TOPOSORT_ORDER_OUT_BUFFER, TOPOSORT_QUEUE_SCRATCH_BUFFER, TOPOSORT_TARGETS_BUFFER,
};

/// Primitive-owned dispatch plan for CSR topological sort.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToposortCsrDispatchPlan {
    /// Validated CSR layout.
    pub layout: ToposortCsrLayout,
    /// Dispatch grid override.
    pub grid: [u32; 3],
    /// Words in the offsets input buffer.
    pub offset_words: usize,
    /// Words in the targets input buffer.
    pub target_words: usize,
    /// Words in each node-indexed scratch/output buffer.
    pub node_words: usize,
}

/// Primitive-owned identity for reusable topological-sort static inputs.
///
/// Dispatch wrappers use this key to decide whether the CSR graph inputs can
/// remain resident across calls. Keeping it in the primitive prevents each
/// wrapper from inventing a private fingerprint contract for the same graph
/// representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToposortCsrStaticInputKey {
    /// Number of nodes in the CSR graph.
    pub node_count: u32,
    /// Words in node-indexed scratch/output buffers.
    pub node_words: usize,
    /// Words in the CSR offsets buffer.
    pub offset_words: usize,
    /// Words in the CSR targets buffer.
    pub target_words: usize,
    /// Stable content fingerprint for CSR offsets.
    pub offsets_hash: u64,
    /// Stable content fingerprint for CSR targets.
    pub targets_hash: u64,
}

impl ToposortCsrDispatchPlan {
    /// Build the single-lane topological-sort program for this plan.
    #[must_use]
    pub fn program(&self) -> Program {
        toposort_program(
            self.layout.node_count,
            TOPOSORT_OFFSETS_BUFFER,
            TOPOSORT_TARGETS_BUFFER,
            TOPOSORT_INDEGREE_SCRATCH_BUFFER,
            TOPOSORT_QUEUE_SCRATCH_BUFFER,
            TOPOSORT_ORDER_OUT_BUFFER,
        )
    }

    /// Return the primitive-owned cache identity for this plan's static graph inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ToposortCsrError::BadCsr`] if the supplied CSR slices no
    /// longer match the validated dispatch plan shape.
    pub fn static_input_key(
        &self,
        offsets: &[u32],
        targets: &[u32],
    ) -> Result<ToposortCsrStaticInputKey, ToposortCsrError> {
        if offsets.len() != self.offset_words {
            return Err(ToposortCsrError::BadCsr {
                message: format!(
                    "Fix: toposort_csr static key expected {} offset words, got {}.",
                    self.offset_words,
                    offsets.len()
                ),
            });
        }
        if targets.len() != self.target_words {
            return Err(ToposortCsrError::BadCsr {
                message: format!(
                    "Fix: toposort_csr static key expected {} target words, got {}.",
                    self.target_words,
                    targets.len()
                ),
            });
        }
        Ok(ToposortCsrStaticInputKey {
            node_count: self.layout.node_count,
            node_words: self.node_words,
            offset_words: self.offset_words,
            target_words: self.target_words,
            offsets_hash: toposort_csr_slice_fingerprint(offsets),
            targets_hash: toposort_csr_slice_fingerprint(targets),
        })
    }
}

/// Stable primitive-owned fingerprint for CSR topological-sort u32 slices.
#[must_use]
pub fn toposort_csr_slice_fingerprint(values: &[u32]) -> u64 {
    u32_slice_fingerprint(values)
}

/// Validate primitive-native CSR inputs and return the full dispatch plan.
pub fn plan_toposort_csr_dispatch(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<ToposortCsrDispatchPlan, ToposortCsrError> {
    let layout = validate_toposort_csr_inputs(node_count, offsets, targets)?;
    Ok(ToposortCsrDispatchPlan {
        offset_words: layout.offset_words,
        target_words: layout.target_words,
        node_words: layout.node_words,
        layout,
        grid: TOPOSORT_DISPATCH_GRID,
    })
}

#[cfg(test)]
mod dispatch_plan_tests {
    use super::super::csr::{toposort_csr_into, validate_toposort_csr_order};
    use super::*;

    #[test]
    fn dispatch_plan_owns_scratch_sizes_and_grid() {
        let plan = plan_toposort_csr_dispatch(3, &[0, 2, 3, 3], &[1, 2, 2])
            .expect("Fix: valid DAG CSR should plan topological-sort dispatch");

        assert_eq!(plan.grid, TOPOSORT_DISPATCH_GRID);
        assert_eq!(plan.offset_words, 4);
        assert_eq!(plan.target_words, 3);
        assert_eq!(plan.node_words, 3);
        assert_eq!(plan.layout.node_count, 3);
    }

    #[test]
    fn empty_dispatch_plan_is_non_dispatchable_but_well_shaped() {
        let plan = plan_toposort_csr_dispatch(0, &[0], &[])
            .expect("Fix: canonical empty CSR should plan without dispatch");

        assert_eq!(plan.grid, TOPOSORT_DISPATCH_GRID);
        assert_eq!(plan.offset_words, 1);
        assert_eq!(plan.target_words, 0);
        assert_eq!(plan.node_words, 0);
        assert_eq!(plan.layout.node_count, 0);
    }

    #[test]
    fn csr_into_emits_order_accepted_by_public_validator() {
        let offsets = [0, 2, 3, 3];
        let targets = [1, 2, 2];
        let mut order = Vec::with_capacity(3);

        toposort_csr_into(3, &offsets, &targets, &mut order)
            .expect("Fix: valid DAG CSR should topologically sort.");

        validate_toposort_csr_order(3, &offsets, &targets, &order)
            .expect("Fix: toposort_csr_into output must satisfy the public order validator.");
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn csr_order_validator_rejects_dependency_inversion() {
        let err = validate_toposort_csr_order(3, &[0, 2, 3, 3], &[1, 2, 2], &[2, 1, 0])
            .expect_err("Fix: dependency-inverted CSR order must be rejected.");

        assert!(matches!(err, ToposortCsrError::BadOrder { .. }));
    }

    #[test]
    fn static_input_key_tracks_content_not_only_shape() {
        let plan = plan_toposort_csr_dispatch(4, &[0, 2, 3, 3, 3], &[1, 2, 3])
            .expect("Fix: valid CSR should plan topological-sort dispatch");
        let first = plan
            .static_input_key(&[0, 2, 3, 3, 3], &[1, 2, 3])
            .expect("Fix: static key should accept matching slices");
        let same = plan
            .static_input_key(&[0, 2, 3, 3, 3], &[1, 2, 3])
            .expect("Fix: identical CSR should produce identical key");
        let changed_targets = plan
            .static_input_key(&[0, 2, 3, 3, 3], &[2, 3, 3])
            .expect("Fix: same-shape CSR content change should still key");

        assert_eq!(first, same);
        assert_eq!(first.node_count, 4);
        assert_eq!(first.node_words, 4);
        assert_eq!(first.offset_words, 5);
        assert_eq!(first.target_words, 3);
        assert_ne!(first, changed_targets);
        assert_eq!(first.offsets_hash, changed_targets.offsets_hash);
        assert_ne!(first.targets_hash, changed_targets.targets_hash);
    }

    #[test]
    fn static_input_key_rejects_plan_slice_drift() {
        let plan = plan_toposort_csr_dispatch(3, &[0, 1, 2, 2], &[1, 2])
            .expect("Fix: valid CSR should plan topological-sort dispatch");

        let err = plan
            .static_input_key(&[0, 1, 2, 2], &[1])
            .expect_err("Fix: stale plan must not accept mismatched target slices");

        assert!(matches!(err, ToposortCsrError::BadCsr { .. }));
    }
}
