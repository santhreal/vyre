//! Cache identities and launch plans over a validated layout.

use vyre_foundation::ir::Program;

use crate::graph::padded_u32_slice_fingerprint as motif_padded_slice_fingerprint;
use crate::graph::program_graph::ProgramGraphShape;

use super::layout::{validate_motif_inputs, MotifLayout};
use super::pattern::MotifEdge;
use super::program::motif;
use super::MOTIF_DISPATCH_GRID;

/// Primitive-owned cache identity for motif Programs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MotifProgramCacheKey {
    /// Number of graph nodes baked into the generated Program.
    pub node_count: u32,
    /// Number of physical CSR edges baked into the generated Program shape.
    pub edge_count: u32,
    /// Motif edges lowered as Program constants.
    pub motif_edges: Vec<MotifEdge>,
    /// Witness output buffer name baked into the Program.
    pub witness_out: String,
}

/// Primitive-owned identity for reusable motif static graph inputs.
///
/// Motif edges are compiled into the generated Program and are therefore part
/// of [`MotifProgramCacheKey`]. This key only tracks staged CSR graph inputs so
/// dispatch wrappers can reuse static graph buffers across motif-program
/// changes without forking fingerprint rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MotifStaticInputKey {
    /// Number of graph nodes and witness words.
    pub node_count: u32,
    /// Number of u32 witness/scratch words staged by the wrapper.
    pub output_words: usize,
    /// Number of u32 words staged for edge targets and kind masks.
    pub edge_storage_words: usize,
    /// Stable fingerprint of the CSR offsets upload.
    pub edge_offsets_hash: u64,
    /// Stable fingerprint of the padded target upload.
    pub edge_targets_hash: u64,
    /// Stable fingerprint of the padded kind-mask upload.
    pub edge_kind_mask_hash: u64,
}

/// Validated motif launch plan without eager Program materialization.
pub struct MotifLaunchPlan {
    layout: MotifLayout,
    cache_key: MotifProgramCacheKey,
}

impl MotifLaunchPlan {
    /// Validated motif graph/pattern layout.
    #[must_use]
    pub const fn layout(&self) -> MotifLayout {
        self.layout
    }

    /// Stable cache identity for the generated Program.
    #[must_use]
    pub fn cache_key(&self) -> &MotifProgramCacheKey {
        &self.cache_key
    }

    /// Number of u32 words in motif scratch and witness outputs.
    #[must_use]
    pub const fn output_words(&self) -> usize {
        self.layout.output_words
    }

    /// Number of u32 words required by physical edge buffers after padding.
    #[must_use]
    pub const fn edge_storage_words(&self) -> usize {
        self.layout.edge_storage_words
    }

    /// Canonical one-workgroup dispatch grid.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        MOTIF_DISPATCH_GRID
    }

    /// Materialize the canonical primitive Program for this launch plan.
    #[must_use]
    pub fn program(&self) -> Program {
        motif(
            ProgramGraphShape::new(self.layout.node_count, self.layout.edge_count.max(1)),
            &self.cache_key.motif_edges,
            &self.cache_key.witness_out,
        )
    }

    /// Return the primitive-owned cache identity for static CSR graph inputs.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the supplied CSR slices no longer
    /// match the validated launch-plan shape.
    pub fn static_input_key(
        &self,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
    ) -> Result<MotifStaticInputKey, String> {
        if edge_offsets.len() != self.layout.node_count as usize + 1 {
            return Err(format!(
                "Fix: motif static key expected {} offset words, got {}.",
                self.layout.node_count as usize + 1,
                edge_offsets.len()
            ));
        }
        if edge_targets.len() != self.layout.edge_count as usize {
            return Err(format!(
                "Fix: motif static key expected {} target word(s), got {}.",
                self.layout.edge_count,
                edge_targets.len()
            ));
        }
        if edge_kind_mask.len() != self.layout.edge_count as usize {
            return Err(format!(
                "Fix: motif static key expected {} kind-mask word(s), got {}.",
                self.layout.edge_count,
                edge_kind_mask.len()
            ));
        }
        Ok(MotifStaticInputKey {
            node_count: self.layout.node_count,
            output_words: self.layout.output_words,
            edge_storage_words: self.layout.edge_storage_words,
            edge_offsets_hash: motif_padded_slice_fingerprint(edge_offsets, edge_offsets.len()),
            edge_targets_hash: motif_padded_slice_fingerprint(
                edge_targets,
                self.layout.edge_storage_words,
            ),
            edge_kind_mask_hash: motif_padded_slice_fingerprint(
                edge_kind_mask,
                self.layout.edge_storage_words,
            ),
        })
    }
}

/// Primitive-owned motif dispatch plan.
pub struct MotifDispatchPlan {
    layout: MotifLayout,
    program: Program,
}

impl MotifDispatchPlan {
    /// Validated motif graph/pattern layout.
    #[must_use]
    pub const fn layout(&self) -> MotifLayout {
        self.layout
    }

    /// Canonical primitive program for this motif.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Number of u32 words in motif scratch and witness outputs.
    #[must_use]
    pub const fn output_words(&self) -> usize {
        self.layout.output_words
    }

    /// Number of u32 words required by physical edge buffers after padding.
    #[must_use]
    pub const fn edge_storage_words(&self) -> usize {
        self.layout.edge_storage_words
    }

    /// Canonical one-workgroup dispatch grid.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        MOTIF_DISPATCH_GRID
    }
}

/// Validate motif inputs and build the canonical dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed CSR inputs, mismatched edge
/// masks, out-of-range destinations, or motif patterns too large for GPU
/// metadata.
pub fn plan_motif_launch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    witness_out: &str,
) -> Result<MotifLaunchPlan, String> {
    let layout = validate_motif_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;

    Ok(MotifLaunchPlan {
        layout,
        cache_key: MotifProgramCacheKey {
            node_count: layout.node_count,
            edge_count: layout.edge_count,
            motif_edges: motif_edges.to_vec(),
            witness_out: witness_out.to_string(),
        },
    })
}

/// Validate motif inputs and build the canonical dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic for malformed CSR inputs, mismatched edge
/// masks, out-of-range destinations, or motif patterns too large for GPU
/// metadata.
pub fn plan_motif_dispatch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    witness_out: &str,
) -> Result<MotifDispatchPlan, String> {
    let launch = plan_motif_launch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        witness_out,
    )?;
    let layout = launch.layout();
    let program = launch.program();
    Ok(MotifDispatchPlan { layout, program })
}

#[cfg(test)]
mod dispatch_contract_tests {
    use super::super::layout::validate_motif_witness;
    use super::*;

    #[test]
    fn static_input_key_tracks_graph_content_not_motif_program() {
        let first_motif = [MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        }];
        let second_motif = [MotifEdge {
            from: 1,
            kind_mask: 1,
            to: 2,
        }];
        let first = plan_motif_launch(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &first_motif, "w")
            .expect("Fix: first motif launch should plan");
        let second = plan_motif_launch(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &second_motif, "w")
            .expect("Fix: second motif launch should plan");

        assert_ne!(first.cache_key(), second.cache_key());
        assert_eq!(
            first
                .static_input_key(&[0, 1, 2, 2], &[1, 2], &[1, 1])
                .expect("Fix: first motif static key should build"),
            second
                .static_input_key(&[0, 1, 2, 2], &[1, 2], &[1, 1])
                .expect("Fix: second motif static key should build")
        );
    }

    #[test]
    fn static_input_key_refreshes_on_same_shape_graph_content_change() {
        let motif = [MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        }];
        let plan = plan_motif_launch(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &motif, "w")
            .expect("Fix: motif launch should plan");
        let first = plan
            .static_input_key(&[0, 1, 2, 2], &[1, 2], &[1, 1])
            .expect("Fix: first static key should build");
        let changed = plan
            .static_input_key(&[0, 1, 2, 2], &[2, 2], &[1, 1])
            .expect("Fix: same-shape changed graph should key");

        assert_eq!(first.edge_offsets_hash, changed.edge_offsets_hash);
        assert_eq!(first.edge_kind_mask_hash, changed.edge_kind_mask_hash);
        assert_ne!(first.edge_targets_hash, changed.edge_targets_hash);
        assert_ne!(first, changed);
    }

    #[test]
    fn static_input_key_rejects_shape_drift() {
        let motif = [MotifEdge {
            from: 0,
            kind_mask: 1,
            to: 1,
        }];
        let plan = plan_motif_launch(2, &[0, 1, 1], &[1], &[1], &motif, "w")
            .expect("Fix: motif launch should plan");
        let err = plan
            .static_input_key(&[0, 1, 1], &[], &[])
            .expect_err("Fix: stale motif plan must reject edge-array drift");

        assert!(err.contains("expected 1 target"));
    }

    #[test]
    fn witness_validation_rejects_non_boolean_backend_output() {
        let layout = validate_motif_inputs(3, &[0, 1, 2, 2], &[1, 2], &[1, 1], &[])
            .expect("Fix: valid graph should validate");

        validate_motif_witness(layout, &[0, 1, 0]).expect("Fix: boolean witness is valid");
        let err = validate_motif_witness(layout, &[0, 2, 0])
            .expect_err("Fix: non-boolean witness must be rejected");

        assert!(err.contains("witness[1]=2 is not boolean"));
    }
}
