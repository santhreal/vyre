//! Dominance-frontier dispatch layout planning, keying, and validation.

use vyre_foundation::ir::Program;

use super::{dominator_frontier_dispatch_grid, try_dominator_frontier};
use crate::bitset::bitset_words;

/// Validated dominance-frontier dispatch layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorFrontierLayout {
    /// Number of u32 words in the frontier/seed bitset.
    pub words: usize,
    /// Number of dominance-closure CSR edges.
    pub dom_edge_count: u32,
    /// Number of predecessor CSR edges.
    pub pred_edge_count: u32,
}

/// Program-shape key for dominance-frontier IR materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorFrontierProgramShape {
    /// Number of candidate nodes.
    pub node_count: u32,
    /// Number of dominance-closure CSR edges.
    pub dom_edge_count: u32,
    /// Number of predecessor CSR edges.
    pub pred_edge_count: u32,
}

/// Content fingerprint for one immutable dominance-frontier input slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorFrontierSliceFingerprint {
    len: usize,
    first: u32,
    last: u32,
    xor: u32,
    sum: u64,
}

/// Primitive-owned identity for immutable dominance-frontier dispatch inputs.
///
/// Dynamic seed/frontier buffers are intentionally excluded: wrappers refresh
/// those every dispatch. This key covers only graph shape and graph content
/// that determine whether static device inputs can be reused safely.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorFrontierStaticInputKey {
    shape: DominatorFrontierProgramShape,
    layout: DominatorFrontierLayout,
    dom_target_words: usize,
    pred_target_words: usize,
    frontier_words: usize,
    dom_offsets: DominatorFrontierSliceFingerprint,
    dom_targets: DominatorFrontierSliceFingerprint,
    pred_offsets: DominatorFrontierSliceFingerprint,
    pred_targets: DominatorFrontierSliceFingerprint,
}

/// Compute the primitive-owned fingerprint used for immutable dispatch inputs.
#[must_use]
pub fn dominator_frontier_slice_fingerprint(words: &[u32]) -> DominatorFrontierSliceFingerprint {
    let mut xor = 0u32;
    let mut sum = 0u64;
    for &word in words {
        xor ^= word;
        sum = sum.wrapping_add(u64::from(word));
    }
    DominatorFrontierSliceFingerprint {
        len: words.len(),
        first: words.first().copied().unwrap_or(0),
        last: words.last().copied().unwrap_or(0),
        xor,
        sum,
    }
}

/// Primitive-owned dominance-frontier launch plan without eager IR materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DominatorFrontierLaunchPlan {
    layout: DominatorFrontierLayout,
    shape: DominatorFrontierProgramShape,
    dispatch_grid: [u32; 3],
}

impl DominatorFrontierLaunchPlan {
    /// Validated CSR and bitset layout.
    #[must_use]
    pub const fn layout(&self) -> DominatorFrontierLayout {
        self.layout
    }

    /// Program-shape key for cache lookups.
    #[must_use]
    pub const fn shape(&self) -> DominatorFrontierProgramShape {
        self.shape
    }

    /// Exact GPU dispatch grid for this query.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        self.dispatch_grid
    }

    /// Number of u32 words in the seed/frontier bitsets.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.layout.words
    }

    /// Number of u32 target words required by the dominance-closure input.
    #[must_use]
    pub const fn dom_target_words(&self) -> usize {
        if self.layout.dom_edge_count == 0 {
            1
        } else {
            self.layout.dom_edge_count as usize
        }
    }

    /// Number of u32 target words required by the predecessor input.
    #[must_use]
    pub const fn pred_target_words(&self) -> usize {
        if self.layout.pred_edge_count == 0 {
            1
        } else {
            self.layout.pred_edge_count as usize
        }
    }

    /// Stable identity for immutable graph inputs associated with this plan.
    #[must_use]
    pub fn static_input_key(
        &self,
        dom_offsets: &[u32],
        dom_targets: &[u32],
        pred_offsets: &[u32],
        pred_targets: &[u32],
    ) -> DominatorFrontierStaticInputKey {
        DominatorFrontierStaticInputKey {
            shape: self.shape,
            layout: self.layout,
            dom_target_words: self.dom_target_words(),
            pred_target_words: self.pred_target_words(),
            frontier_words: self.frontier_words(),
            dom_offsets: dominator_frontier_slice_fingerprint(dom_offsets),
            dom_targets: dominator_frontier_slice_fingerprint(dom_targets),
            pred_offsets: dominator_frontier_slice_fingerprint(pred_offsets),
            pred_targets: dominator_frontier_slice_fingerprint(pred_targets),
        }
    }

    /// Build the dominance-frontier Program for this launch plan.
    pub fn program(&self, seed_buffer: &str, out_buffer: &str) -> Result<Program, String> {
        try_dominator_frontier(
            self.shape.node_count,
            self.shape.dom_edge_count,
            self.shape.pred_edge_count,
            seed_buffer,
            out_buffer,
        )
    }
}

/// Primitive-owned dominance-frontier dispatch plan with eager IR materialization.
pub struct DominatorFrontierDispatchPlan {
    launch: DominatorFrontierLaunchPlan,
    program: Program,
}

impl DominatorFrontierDispatchPlan {
    /// Validated CSR and bitset layout.
    #[must_use]
    pub const fn layout(&self) -> DominatorFrontierLayout {
        self.launch.layout()
    }

    /// Program-shape key for cache lookups.
    #[must_use]
    pub const fn shape(&self) -> DominatorFrontierProgramShape {
        self.launch.shape()
    }

    /// Program wired to the canonical primitive buffer layout.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Exact GPU dispatch grid for this query.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        self.launch.dispatch_grid()
    }

    /// Number of u32 words in the seed/frontier bitsets.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.launch.frontier_words()
    }

    /// Number of u32 target words required by the dominance-closure input.
    #[must_use]
    pub const fn dom_target_words(&self) -> usize {
        self.launch.dom_target_words()
    }

    /// Number of u32 target words required by the predecessor input.
    #[must_use]
    pub const fn pred_target_words(&self) -> usize {
        self.launch.pred_target_words()
    }
}

/// Validate inputs and build a dominance-frontier launch plan without
/// materializing IR.
///
/// # Errors
///
/// Returns an actionable diagnostic when either CSR is malformed, the seed
/// bitset is not exactly shaped for `node_count`, or the dispatch shape would
/// overflow.
pub fn plan_dominator_frontier_launch(
    node_count: u32,
    dom_offsets: &[u32],
    dom_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
    seed: &[u32],
) -> Result<DominatorFrontierLaunchPlan, String> {
    let layout = validate_dominator_frontier_inputs(
        node_count,
        dom_offsets,
        dom_targets,
        pred_offsets,
        pred_targets,
        seed,
    )?;
    let _offset_count = node_count.checked_add(1).ok_or_else(|| {
        format!(
            "dominator_frontier node_count={node_count} overflows CSR offset buffer count. Fix: shard the graph before GPU dispatch."
        )
    })?;

    Ok(DominatorFrontierLaunchPlan {
        layout,
        shape: DominatorFrontierProgramShape {
            node_count,
            dom_edge_count: layout.dom_edge_count,
            pred_edge_count: layout.pred_edge_count,
        },
        dispatch_grid: dominator_frontier_dispatch_grid(node_count),
    })
}

/// Validate inputs and build the canonical dominance-frontier dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when either CSR is malformed, the seed
/// bitset is not exactly shaped for `node_count`, or the generated dispatch
/// program would overflow its CSR launch shape.
pub fn plan_dominator_frontier_dispatch(
    node_count: u32,
    dom_offsets: &[u32],
    dom_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
    seed: &[u32],
    seed_buffer: &str,
    out_buffer: &str,
) -> Result<DominatorFrontierDispatchPlan, String> {
    let launch = plan_dominator_frontier_launch(
        node_count,
        dom_offsets,
        dom_targets,
        pred_offsets,
        pred_targets,
        seed,
    )?;
    let program = launch.program(seed_buffer, out_buffer)?;

    Ok(DominatorFrontierDispatchPlan { launch, program })
}

/// Validate a CSR buffer pair for `node_count` rows.
///
/// # Errors
///
/// Returns an actionable diagnostic when offsets are the wrong length,
/// non-monotonic, inconsistent with target count, or targets point outside
/// `0..node_count`.
pub fn validate_csr_shape(
    label: &str,
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<u32, String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!(
            "Fix: dominator_frontier {label} node_count + 1 overflows usize for node_count={node_count}."
        )
    })?;
    if offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: dominator_frontier {label} offsets length must be {expected_offsets}, got {}.",
            offsets.len()
        ));
    }
    let mut previous = 0u32;
    for (idx, &offset) in offsets.iter().enumerate() {
        if idx > 0 && offset < previous {
            return Err(format!(
                "Fix: dominator_frontier {label} offsets must be monotonic; offsets[{idx}]={offset} after {previous}."
            ));
        }
        previous = offset;
    }
    if offsets.last().copied().unwrap_or(0) as usize != targets.len() {
        return Err(format!(
            "Fix: dominator_frontier {label} final offset must equal target count {}, got {}.",
            targets.len(),
            offsets.last().copied().unwrap_or(0)
        ));
    }
    for (idx, &target) in targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: dominator_frontier {label} target[{idx}]={target} is outside node_count {node_count}."
            ));
        }
    }
    u32::try_from(targets.len()).map_err(|_| {
        format!(
            "Fix: dominator_frontier {label} target count {} exceeds u32 index space.",
            targets.len()
        )
    })
}

/// Validate the full dominance-frontier CPU/dispatch input contract.
///
/// # Errors
///
/// Returns an actionable diagnostic when either CSR is malformed or when the
/// seed bitset does not contain exactly the required number of words.
pub fn validate_dominator_frontier_inputs(
    node_count: u32,
    dom_offsets: &[u32],
    dom_targets: &[u32],
    pred_offsets: &[u32],
    pred_targets: &[u32],
    seed: &[u32],
) -> Result<DominatorFrontierLayout, String> {
    let words = bitset_words(node_count) as usize;
    if seed.len() != words {
        return Err(format!(
            "Fix: dominator_frontier expected seed length {words} words for {node_count} nodes, got {}.",
            seed.len()
        ));
    }
    let dom_edge_count = validate_csr_shape("dominance", node_count, dom_offsets, dom_targets)?;
    let pred_edge_count =
        validate_csr_shape("predecessor", node_count, pred_offsets, pred_targets)?;
    Ok(DominatorFrontierLayout {
        words,
        dom_edge_count,
        pred_edge_count,
    })
}

#[cfg(test)]
mod static_input_key_tests {
    use super::*;

    #[test]
    fn slice_fingerprint_tracks_interior_content_not_only_len_edges() {
        let baseline = dominator_frontier_slice_fingerprint(&[7, 11, 13, 17]);
        let changed = dominator_frontier_slice_fingerprint(&[7, 11, 19, 17]);

        assert_ne!(baseline, changed);
    }

    #[test]
    fn static_input_key_tracks_graph_content_but_not_dynamic_seed_bits() {
        let plan_a = plan_dominator_frontier_launch(
            4,
            &[0, 4, 5, 6, 7],
            &[0, 1, 2, 3, 1, 2, 3],
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
            &[0b0010],
        )
        .expect("Fix: valid dominator-frontier launch plan should build");
        let plan_b = plan_dominator_frontier_launch(
            4,
            &[0, 4, 5, 6, 7],
            &[0, 1, 2, 3, 1, 2, 3],
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
            &[0b0100],
        )
        .expect("Fix: seed-only changes should keep the same static launch shape");

        let baseline = plan_a.static_input_key(
            &[0, 4, 5, 6, 7],
            &[0, 1, 2, 3, 1, 2, 3],
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
        );
        let seed_only_change = plan_b.static_input_key(
            &[0, 4, 5, 6, 7],
            &[0, 1, 2, 3, 1, 2, 3],
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
        );
        let graph_content_change = plan_a.static_input_key(
            &[0, 4, 5, 6, 7],
            &[0, 1, 2, 2, 1, 2, 3],
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
        );

        assert_eq!(baseline, seed_only_change);
        assert_ne!(baseline, graph_content_change);
    }
}
