//! `csr_bidirectional`  -  one BFS step over BOTH forward + backward
//! edges of a ProgramGraph CSR. Used for undirected reachability
//! (e.g. component discovery, alias unification).

use super::padded_u32_slice_fingerprint as csr_bidirectional_padded_slice_fingerprint;
use vyre_foundation::composition::trap_program;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{DataType, Program};

use crate::bitset::bitset_words;
use crate::graph::csr_backward_traverse::csr_backward_traverse;
use crate::graph::csr_forward_traverse::csr_forward_traverse;
use crate::graph::csr_frontier_step::csr_frontier_step_dispatch_grid;
use crate::graph::program_graph::ProgramGraphShape;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_bidirectional";
/// Canonical dispatch input label for graph node scratch.
pub const CSR_BIDIRECTIONAL_NODES_BUFFER: &str = "csr_bidirectional nodes";
/// Canonical dispatch input label for CSR offsets.
pub const CSR_BIDIRECTIONAL_OFFSETS_BUFFER: &str = "csr_bidirectional edge_offsets";
/// Canonical dispatch input label for CSR targets.
pub const CSR_BIDIRECTIONAL_TARGETS_BUFFER: &str = "csr_bidirectional edge_targets";
/// Canonical dispatch input label for edge-kind masks.
pub const CSR_BIDIRECTIONAL_EDGE_KIND_BUFFER: &str = "csr_bidirectional edge_kind_mask";
/// Canonical dispatch input label for node tags.
pub const CSR_BIDIRECTIONAL_NODE_TAGS_BUFFER: &str = "csr_bidirectional node_tags";
/// Canonical dispatch input label for the incoming frontier.
pub const CSR_BIDIRECTIONAL_FRONTIER_IN_BUFFER: &str = "csr_bidirectional frontier_in";
/// Canonical dispatch output label for the outgoing frontier.
pub const CSR_BIDIRECTIONAL_FRONTIER_OUT_BUFFER: &str = "csr_bidirectional frontier_out";

/// Build a Program: emit one forward step + one backward step,
/// fused into one Region. Both writes target `frontier_out` so a
/// single dispatch covers both directions.
#[must_use]
pub fn csr_bidirectional(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
) -> Program {
    let fwd = csr_forward_traverse(shape, frontier_in, frontier_out, edge_kind_mask);
    let bwd = csr_backward_traverse(shape, frontier_in, frontier_out, edge_kind_mask);
    fuse_programs(&[fwd, bwd]).unwrap_or_else(|error| {
        trap_program(
            OP_ID,
            Some((frontier_out, DataType::U32)),
            format!("Fix: csr_bidirectional forward+backward fusion failed: {error}"),
        )
    })
}

/// Validated dispatch layout for bidirectional CSR traversal.
///
/// The primitive owns these derived values so dispatch wrappers do not fork
/// CSR/frontier layout rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrBidirectionalLayout {
    /// Number of nodes accepted by the primitive.
    pub node_count: u32,
    /// Number of `u32` frontier words required for `node_count`.
    pub words: usize,
    /// Number of node-index words required by graph-indexed scratch buffers.
    pub node_words: usize,
    /// Exact edge count declared by `edge_offsets[node_count]`.
    pub edge_count: u32,
    /// Number of u32 words required by physical edge buffers after padding.
    pub edge_storage_words: usize,
}

/// Primitive-owned dispatch plan for a bidirectional CSR step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrBidirectionalDispatchPlan {
    /// Validated CSR/frontier layout.
    pub layout: CsrBidirectionalLayout,
    /// Edge-kind mask accepted by this step.
    pub allow_mask: u32,
    /// Dispatch grid override.
    pub grid: [u32; 3],
    /// Words required by graph-node scratch buffers.
    pub node_words: usize,
    /// Words required by padded edge buffers.
    pub edge_storage_words: usize,
    /// Words required by input/output frontiers.
    pub frontier_words: usize,
}

/// Primitive-owned program identity for bidirectional CSR dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrBidirectionalProgramKey {
    /// Validated CSR/frontier layout represented by this program.
    pub layout: CsrBidirectionalLayout,
    /// Edge-kind mask accepted by this step.
    pub allow_mask: u32,
}

/// Primitive-owned identity for reusable bidirectional CSR static inputs.
///
/// Dispatch wrappers stage node scratch and frontier buffers dynamically, but
/// CSR offsets, targets, and edge-kind masks are static graph inputs. This key
/// keeps content identity next to the primitive-owned layout and padded edge
/// storage contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrBidirectionalStaticInputKey {
    /// Program identity selected by the primitive dispatch planner.
    pub program_key: CsrBidirectionalProgramKey,
    /// Words in the CSR offsets buffer.
    pub edge_offset_words: usize,
    /// Words in each padded edge-indexed input.
    pub edge_storage_words: usize,
    /// Stable fingerprint of the edge-offset upload.
    pub edge_offsets_hash: u64,
    /// Stable fingerprint of the padded edge-target upload.
    pub edge_targets_hash: u64,
    /// Stable fingerprint of the padded edge-kind upload.
    pub edge_kind_mask_hash: u64,
}

impl CsrBidirectionalDispatchPlan {
    /// Stable key for caching the generated primitive program.
    #[must_use]
    pub const fn program_key(&self) -> CsrBidirectionalProgramKey {
        CsrBidirectionalProgramKey {
            layout: self.layout,
            allow_mask: self.allow_mask,
        }
    }

    /// Build the fused forward/backward traversal program for this plan.
    #[must_use]
    pub fn program(&self) -> Program {
        csr_bidirectional(
            ProgramGraphShape::new(self.layout.node_count, self.layout.edge_count.max(1)),
            CSR_BIDIRECTIONAL_FRONTIER_IN_BUFFER,
            CSR_BIDIRECTIONAL_FRONTIER_OUT_BUFFER,
            self.allow_mask,
        )
    }

    /// Return true when both logical edge arrays already match the physical
    /// edge-buffer storage required by this plan and can be dispatched without
    /// staging padded scratch.
    #[cfg(test)]
    #[must_use]
    pub const fn edge_buffers_can_dispatch_unpadded(
        &self,
        edge_targets_len: usize,
        edge_kind_mask_len: usize,
    ) -> bool {
        can_dispatch_edge_buffers_without_padding(
            edge_targets_len,
            edge_kind_mask_len,
            self.edge_storage_words,
        )
    }

    /// Return the primitive-owned cache identity for this plan's static CSR graph inputs.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the supplied CSR slices no longer
    /// match the validated dispatch plan shape.
    pub fn static_input_key(
        &self,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
    ) -> Result<CsrBidirectionalStaticInputKey, String> {
        let expected_offsets = self.layout.node_words.checked_add(1).ok_or_else(|| {
            format!(
                "Fix: csr_bidirectional static key node_words + 1 overflows usize for node_words={}.",
                self.layout.node_words
            )
        })?;
        if edge_offsets.len() != expected_offsets {
            return Err(format!(
                "Fix: csr_bidirectional static key expected {expected_offsets} offset word(s), got {}.",
                edge_offsets.len()
            ));
        }
        let expected_edges = self.layout.edge_count as usize;
        if edge_targets.len() != expected_edges {
            return Err(format!(
                "Fix: csr_bidirectional static key expected {expected_edges} edge target word(s), got {}.",
                edge_targets.len()
            ));
        }
        if edge_kind_mask.len() != expected_edges {
            return Err(format!(
                "Fix: csr_bidirectional static key expected {expected_edges} edge kind word(s), got {}.",
                edge_kind_mask.len()
            ));
        }
        Ok(CsrBidirectionalStaticInputKey {
            program_key: self.program_key(),
            edge_offset_words: expected_offsets,
            edge_storage_words: self.edge_storage_words,
            edge_offsets_hash: csr_bidirectional_padded_slice_fingerprint(
                edge_offsets,
                expected_offsets,
            ),
            edge_targets_hash: csr_bidirectional_padded_slice_fingerprint(
                edge_targets,
                self.edge_storage_words,
            ),
            edge_kind_mask_hash: csr_bidirectional_padded_slice_fingerprint(
                edge_kind_mask,
                self.edge_storage_words,
            ),
        })
    }
}

/// Return true when both edge arrays have the exact required physical edge
/// storage width and can be borrowed directly by dispatch wrappers.
///
/// Empty logical edge arrays intentionally return false for the canonical
/// one-word padded storage case, keeping that padding contract owned by the
/// primitive instead of each dispatch consumer.
#[cfg(test)]
#[must_use]
pub const fn can_dispatch_edge_buffers_without_padding(
    edge_targets_len: usize,
    edge_kind_mask_len: usize,
    edge_storage_words: usize,
) -> bool {
    edge_targets_len == edge_storage_words && edge_kind_mask_len == edge_storage_words
}

/// Validate the public CSR/frontier inputs consumed by the bidirectional
/// traversal primitive.
///
/// Returns the full dispatch layout so wrappers can build padded device buffers
/// without re-parsing the CSR contract locally.
///
/// # Errors
///
/// Returns an actionable diagnostic when offsets, edge arrays, frontier width,
/// or destinations violate the primitive's contract.
pub fn validate_csr_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
) -> Result<CsrBidirectionalLayout, String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!(
            "Fix: csr_bidirectional node_count + 1 overflows usize for node_count={node_count}."
        )
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: csr_bidirectional requires edge_offsets.len() == node_count + 1, got len={}, node_count={node_count}.",
            edge_offsets.len()
        ));
    }

    let expected_frontier_words = bitset_words(node_count) as usize;
    if frontier_in.len() != expected_frontier_words {
        return Err(format!(
            "Fix: csr_bidirectional expected frontier length {expected_frontier_words} words for {node_count} nodes, got {}.",
            frontier_in.len()
        ));
    }

    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: csr_bidirectional requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }

    if let Some(&first) = edge_offsets.first() {
        if first != 0 {
            return Err(format!(
                "Fix: csr_bidirectional requires edge_offsets[0] == 0, got {first}."
            ));
        }
    }
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: csr_bidirectional offsets must be monotonic; offsets[{index}]={} > offsets[{}]={}.",
                pair[0],
                index + 1,
                pair[1]
            ));
        }
    }

    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    if edge_targets.len() != edge_count {
        return Err(format!(
            "Fix: csr_bidirectional final offset declares edge_count={edge_count}, but targets_len={} and kind_mask_len={}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    for (index, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: csr_bidirectional edge_targets[{index}]={target} is outside node_count {node_count}."
            ));
        }
    }
    let edge_count = u32::try_from(edge_count).map_err(|_| {
        format!("Fix: csr_bidirectional edge count {edge_count} exceeds u32 index space.")
    })?;
    Ok(CsrBidirectionalLayout {
        node_count,
        words: expected_frontier_words,
        node_words: node_count as usize,
        edge_count,
        edge_storage_words: edge_targets.len().max(1),
    })
}

/// Validate inputs and return the complete dispatch plan for one bidirectional step.
pub fn plan_csr_bidirectional_step(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
) -> Result<CsrBidirectionalDispatchPlan, String> {
    let layout = validate_csr_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
    )?;
    Ok(CsrBidirectionalDispatchPlan {
        node_words: layout.node_words,
        edge_storage_words: layout.edge_storage_words,
        frontier_words: layout.words,
        grid: csr_frontier_step_dispatch_grid(layout.node_count),
        allow_mask,
        layout,
    })
}

/// Run a bidirectional CSR closure loop from a primitive-owned dispatch plan.
///
/// The caller supplies one step executor: CPU references can execute the
/// validated primitive oracle, while GPU wrappers can dispatch a prepared
/// program. Initialization, max-iteration handling, frontier merge semantics,
/// and reusable-buffer reservation stay single-sourced here.
///
/// # Errors
///
/// Returns caller-mapped errors for malformed seed width, reservation failure,
/// step execution failure, or frontier shape drift.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn run_csr_bidirectional_closure_plan_with_step<E, MapError, Step>(
    plan: &CsrBidirectionalDispatchPlan,
    seed: &[u32],
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
    mut map_error: MapError,
    mut step: Step,
) -> Result<(), E>
where
    MapError: FnMut(String) -> E,
    Step: FnMut(&[u32], &mut Vec<u32>) -> Result<(), E>,
{
    if seed.len() != plan.frontier_words {
        return Err(map_error(format!(
            "Fix: csr_bidirectional closure expected seed length {} words for {} nodes, got {}.",
            plan.frontier_words,
            plan.layout.node_count,
            seed.len()
        )));
    }
    crate::plumbing::host::scratch::reserve_items_with(
        current,
        plan.frontier_words,
        "csr_bidirectional closure runner",
        "current frontier",
        |message| map_error(message),
    )?;
    crate::plumbing::host::scratch::reserve_items_with(
        next,
        plan.frontier_words,
        "csr_bidirectional closure runner",
        "next frontier",
        |message| map_error(message),
    )?;

    current.clear();
    current.extend_from_slice(seed);
    next.clear();
    if plan.layout.node_count == 0 || max_iters == 0 {
        return Ok(());
    }

    for _ in 0..max_iters {
        next.clear();
        step(current, next)?;
        if !try_merge_frontier_or_changed(current, next).map_err(&mut map_error)? {
            return Ok(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod dispatch_plan_tests {
    use super::*;

    #[test]
    fn dispatch_plan_owns_buffer_sizes_grid_and_mask() {
        let plan = plan_csr_bidirectional_step(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0010],
            0x55AA_00FF,
        )
        .expect("Fix: valid bidirectional CSR step should produce dispatch plan");

        // 4 nodes / 256 threads per workgroup = ceil(4/256) = 1 block, not 4.
        assert_eq!(plan.grid, [1, 1, 1]);
        assert_eq!(plan.node_words, 4);
        assert_eq!(plan.edge_storage_words, 3);
        assert_eq!(plan.frontier_words, 1);
        assert_eq!(plan.allow_mask, 0x55AA_00FF);
        assert_eq!(plan.layout.edge_count, 3);
    }

    /// Regression: grid X was set to `node_count` instead of
    /// `ceil(node_count / CSR_FRONTIER_STEP_WORKGROUP_SIZE[0])`.
    /// For node_count=257 the old code emitted [257,1,1] (257 blocks × 256
    /// threads = 65,792 invocations) instead of [2,1,1] (2 blocks × 256
    /// threads = 512 invocations), a 256x over-dispatch.
    #[test]
    fn grid_x_is_ceil_node_count_div_workgroup_size_not_node_count() {
        // 4 nodes: ceil(4/256) == 1
        let plan_small = plan_csr_bidirectional_step(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0010],
            u32::MAX,
        )
        .expect("Fix: valid 4-node bidirectional CSR step should produce dispatch plan");
        assert_eq!(
            plan_small.grid,
            [1, 1, 1],
            "4 nodes: expected ceil(4/256)=1 block, got {:?}",
            plan_small.grid
        );

        // 257 nodes: ceil(257/256) == 2; old buggy code emits [257,1,1].
        // Build a chain: 0→1→2→…→256 (257 nodes, 256 edges).
        let mut offsets = Vec::with_capacity(258);
        offsets.push(0u32);
        for i in 0..256u32 {
            offsets.push(i + 1);
        }
        offsets.push(256u32); // node 256 has no outgoing edge
        let targets: Vec<u32> = (1u32..=256).collect();
        let kinds: Vec<u32> = vec![1u32; 256];
        let frontier = vec![0u32; crate::bitset::bitset_words(257) as usize];

        let plan_large =
            plan_csr_bidirectional_step(257, &offsets, &targets, &kinds, &frontier, u32::MAX)
                .expect("Fix: valid 257-node bidirectional CSR step should produce dispatch plan");
        assert_eq!(
            plan_large.grid,
            [2, 1, 1],
            "257 nodes: expected ceil(257/256)=2 blocks, got {:?}",
            plan_large.grid
        );
    }

    #[test]
    fn dispatch_plan_pads_empty_edges_without_zero_sized_buffers() {
        let plan = plan_csr_bidirectional_step(1, &[0, 0], &[], &[], &[0], u32::MAX)
            .expect("Fix: edgeless one-node graph should still have dispatch buffers");

        assert_eq!(plan.grid, [1, 1, 1]);
        assert_eq!(plan.edge_storage_words, 1);
        assert_eq!(plan.frontier_words, 1);
        assert_eq!(plan.layout.edge_count, 0);
        assert!(!plan.edge_buffers_can_dispatch_unpadded(0, 0));
    }

    #[test]
    fn edge_buffer_unpadded_policy_is_primitive_owned() {
        assert!(can_dispatch_edge_buffers_without_padding(3, 3, 3));
        assert!(!can_dispatch_edge_buffers_without_padding(0, 0, 1));
        assert!(!can_dispatch_edge_buffers_without_padding(3, 2, 3));
        assert!(!can_dispatch_edge_buffers_without_padding(2, 3, 3));
    }

    #[test]
    fn static_input_key_tracks_graph_content_and_padded_edge_storage() {
        let plan = plan_csr_bidirectional_step(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0010],
            0x55AA_00FF,
        )
        .expect("Fix: valid bidirectional CSR step should produce dispatch plan");

        let first = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1, 1])
            .expect("Fix: matching static CSR slices should key");
        let same = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1, 1])
            .expect("Fix: matching static CSR slices should key");
        let changed_targets = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[2, 3, 0], &[1, 1, 1])
            .expect("Fix: same-shape target content should key");
        let changed_kind = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 2, 1])
            .expect("Fix: same-shape kind content should key");

        assert_eq!(first, same);
        assert_ne!(first, changed_targets);
        assert_ne!(first, changed_kind);
        assert_eq!(first.program_key, plan.program_key());
        assert_eq!(first.edge_offset_words, 5);
        assert_eq!(first.edge_storage_words, 3);
    }

    #[test]
    fn static_input_key_normalizes_empty_edges_to_padded_upload() {
        let plan = plan_csr_bidirectional_step(1, &[0, 0], &[], &[], &[0], u32::MAX)
            .expect("Fix: edgeless one-node graph should still have dispatch buffers");
        let key = plan
            .static_input_key(&[0, 0], &[], &[])
            .expect("Fix: empty edge buffers should key through padded primitive storage");

        assert_eq!(key.edge_offset_words, 2);
        assert_eq!(key.edge_storage_words, 1);
    }

    #[test]
    fn static_input_key_rejects_shape_drift() {
        let plan = plan_csr_bidirectional_step(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0010],
            u32::MAX,
        )
        .expect("Fix: valid bidirectional CSR step should produce dispatch plan");

        let err = plan
            .static_input_key(&[0, 1, 2, 3], &[1, 2, 3], &[1, 1, 1])
            .unwrap_err();
        assert!(err.contains("expected 5 offset word"));

        let err = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2], &[1, 1, 1])
            .unwrap_err();
        assert!(err.contains("expected 3 edge target"));

        let err = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1])
            .unwrap_err();
        assert!(err.contains("expected 3 edge kind"));
    }

    #[test]
    fn closure_runner_stops_after_fixpoint_and_reuses_buffers() {
        let plan = plan_csr_bidirectional_step(4, &[0, 0, 0, 0, 0], &[], &[], &[0b0001], u32::MAX)
            .expect("Fix: valid empty-edge CSR plan should build");
        let mut current = Vec::with_capacity(4);
        let mut next = Vec::with_capacity(4);
        let mut calls = 0usize;

        run_csr_bidirectional_closure_plan_with_step(
            &plan,
            &[0b0001],
            9,
            &mut current,
            &mut next,
            |message| message,
            |_frontier, out| {
                calls += 1;
                out.extend_from_slice(&[0]);
                Ok(())
            },
        )
        .expect("Fix: closure runner should accept matching frontier shapes");

        assert_eq!(calls, 1);
        assert_eq!(current, vec![0b0001]);
        assert!(current.capacity() >= 4);
        assert!(next.capacity() >= 4);
    }

    #[test]
    fn closure_runner_rejects_seed_width_drift_without_clobbering_buffers() {
        let plan = plan_csr_bidirectional_step(4, &[0, 0, 0, 0, 0], &[], &[], &[0], u32::MAX)
            .expect("Fix: valid empty-edge CSR plan should build");
        let mut current = vec![0xAA55_AA55];
        let mut next = vec![0x55AA_55AA];

        let err = run_csr_bidirectional_closure_plan_with_step(
            &plan,
            &[0, 1],
            1,
            &mut current,
            &mut next,
            |message| message,
            |_frontier, _out| Ok(()),
        )
        .expect_err("seed width drift must be rejected before mutation");

        assert!(err.contains("expected seed length"));
        assert_eq!(current, vec![0xAA55_AA55]);
        assert_eq!(next, vec![0x55AA_55AA]);
    }
}

/// Merge a bidirectional step frontier into the accumulated closure.
///
/// Returns `true` when at least one bit was newly set. This helper owns the
/// fixpoint-merge semantics so dispatch consumers do not fork closure logic.
///
/// # Panics
///
/// Panics when the two frontier slices differ in length. That is a caller
/// contract violation: both slices must be bitsets for the same `node_count`.
#[cfg(test)]
#[must_use]
pub fn merge_frontier_or_changed(current: &mut [u32], next: &[u32]) -> bool {
    // Fail fast on a caller contract violation (mismatched bitset lengths).
    // `unwrap_or(false)` would silently report "no change" for an unmergeable
    // pair, hiding a fixpoint bug. Use `try_merge_frontier_or_changed` to handle
    // it structurally.
    try_merge_frontier_or_changed(current, next).unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible variant of [`merge_frontier_or_changed`].
#[cfg(test)]
pub fn try_merge_frontier_or_changed(current: &mut [u32], next: &[u32]) -> Result<bool, String> {
    if current.len() != next.len() {
        return Err(format!(
            "Fix: bidirectional frontier merge requires equal bitset word counts, got current={} next={}.",
            current.len(),
            next.len()
        ));
    }
    let mut changed = false;
    for (dst, src) in current.iter_mut().zip(next.iter()) {
        let merged = *dst | *src;
        changed |= merged != *dst;
        *dst = merged;
    }
    Ok(changed)
}

fn csr_bidir_u32_to_usize(value: u32, label: &'static str) -> Result<usize, String> {
    usize::try_from(value).map_err(|source| {
        format!("Fix: csr_bidirectional {label} value {value} cannot fit host usize: {source}.")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr_closure_inputs::{graphs, CsrClosureInputs};
    use vyre_reference::composition_witness::{
        csr_bidirectional_closure_witness_into, csr_bidirectional_step_witness_into,
    };

    fn try_cpu_ref_into(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        let layout = validate_csr_inputs(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
        )?;
        csr_bidirectional_step_witness_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            out,
        );
        if out.len() < layout.words as usize {
            out.resize(layout.words as usize, 0);
        }
        Ok(())
    }

    fn try_cpu_ref(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        try_cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            &mut out,
        )?;
        Ok(out)
    }

    fn cpu_ref_into(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
        out: &mut Vec<u32>,
    ) {
        try_cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
            out,
        )
        .expect("cpu_ref_into failed");
    }

    fn cpu_ref(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Vec<u32> {
        try_cpu_ref(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_in,
            allow_mask,
        )
        .expect("cpu_ref failed")
    }

    fn try_cpu_ref_closure_into(
        inputs: CsrClosureInputs<'_>,
        seed: &[u32],
        current: &mut Vec<u32>,
        next: &mut Vec<u32>,
    ) -> Result<(), String> {
        let _layout = validate_csr_inputs(
            inputs.graph.node_count,
            inputs.graph.edge_offsets,
            inputs.graph.edge_targets,
            inputs.graph.edge_kind_mask,
            seed,
        )?;
        csr_bidirectional_closure_witness_into(
            inputs.graph.node_count,
            inputs.graph.edge_offsets,
            inputs.graph.edge_targets,
            inputs.graph.edge_kind_mask,
            seed,
            inputs.allow_mask,
            inputs.max_iters,
            current,
            next,
        );
        Ok(())
    }

    fn try_cpu_ref_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Result<Vec<u32>, String> {
        let mut current = Vec::new();
        let mut next = Vec::new();
        try_cpu_ref_closure_into(inputs, seed, &mut current, &mut next)?;
        Ok(current)
    }

    fn cpu_ref_closure_into(
        inputs: CsrClosureInputs<'_>,
        seed: &[u32],
        current: &mut Vec<u32>,
        next: &mut Vec<u32>,
    ) {
        try_cpu_ref_closure_into(inputs, seed, current, next).expect("cpu_ref_closure_into failed");
    }

    fn cpu_ref_closure(inputs: CsrClosureInputs<'_>, seed: &[u32]) -> Vec<u32> {
        try_cpu_ref_closure(inputs, seed).expect("cpu_ref_closure failed")
    }

    #[test]
    fn forward_step_propagates() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0001],
            u32::MAX,
        );
        // 0's forward neighbor = 1 → bit 1 set.
        assert!(out[0] & 0b0010 != 0);
    }

    #[test]
    fn empty_seed_yields_empty_step() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0],
            u32::MAX,
        );
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn allow_mask_zero_blocks_all() {
        let g = graphs::CHAIN_4;
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0001],
            0,
        );
        assert_eq!(out, vec![0]);
    }

    #[test]
    fn bidirectional_includes_both_directions() {
        let g = graphs::CHAIN_4;
        // From {1}, forward reaches {2}; backward reaches {0}.
        let out = cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0010],
            u32::MAX,
        );
        assert!(out[0] & 0b0001 != 0, "bwd should reach node 0");
        assert!(out[0] & 0b0100 != 0, "fwd should reach node 2");
    }

    #[test]
    fn closure_reaches_full_linear_component() {
        let out = cpu_ref_closure(CsrClosureInputs::allow_all(graphs::CHAIN_4, 5), &[0b0001]);
        assert_eq!(out, vec![0b1111]);
    }

    #[test]
    fn closure_into_reuses_caller_buffers() {
        let mut current = Vec::with_capacity(8);
        let mut next = Vec::with_capacity(8);
        cpu_ref_closure_into(
            CsrClosureInputs::allow_all(graphs::CHAIN_4, 5),
            &[0b0001],
            &mut current,
            &mut next,
        );
        assert_eq!(current, vec![0b1111]);
        assert_eq!(current.capacity(), 8);
        assert_eq!(next.capacity(), 8);
    }

    #[test]
    fn merge_frontier_reports_change_and_or_merges_words() {
        let mut current = [0b0001u32, 0b1000];
        let next = [0b0110u32, 0b1000];
        assert!(merge_frontier_or_changed(&mut current, &next));
        assert_eq!(current, [0b0111, 0b1000]);
        assert!(!merge_frontier_or_changed(&mut current, &next));
    }

    #[test]
    fn try_merge_frontier_rejects_mismatched_word_counts_without_panic() {
        let mut current = [0u32];
        let next = [1u32, 2];
        let err = try_merge_frontier_or_changed(&mut current, &next)
            .expect_err("mismatched frontier word counts must be a typed error");
        assert!(err.contains("equal bitset word counts"));
        assert_eq!(current, [0u32]);
    }

    #[test]
    #[should_panic(
        expected = "Fix: bidirectional frontier merge requires equal bitset word counts"
    )]
    fn merge_frontier_rejects_mismatched_word_counts() {
        let mut current = [0u32];
        let next = [1u32, 2];
        let _ = merge_frontier_or_changed(&mut current, &next);
    }

    #[test]
    fn validate_csr_inputs_accepts_empty_and_canonical_graphs() {
        assert_eq!(
            validate_csr_inputs(0, &[0], &[], &[], &[]).unwrap(),
            CsrBidirectionalLayout {
                node_count: 0,
                words: 0,
                node_words: 0,
                edge_count: 0,
                edge_storage_words: 1,
            }
        );

        let g = graphs::CHAIN_4;
        assert_eq!(
            validate_csr_inputs(
                g.node_count,
                g.edge_offsets,
                g.edge_targets,
                g.edge_kind_mask,
                &[0]
            )
            .unwrap(),
            CsrBidirectionalLayout {
                node_count: 4,
                words: 1,
                node_words: 4,
                edge_count: 3,
                edge_storage_words: 3,
            }
        );
    }

    #[test]
    fn validate_csr_inputs_rejects_frontier_and_csr_contract_violations() {
        let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[1], &[]).unwrap_err();
        assert!(err.contains("expected frontier length"));

        let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[], &[0]).unwrap_err();
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));

        let err = validate_csr_inputs(2, &[0, 2, 1], &[1], &[1], &[0]).unwrap_err();
        assert!(err.contains("offsets must be monotonic"));

        let err = validate_csr_inputs(2, &[0, 1, 1], &[5], &[1], &[0]).unwrap_err();
        assert!(err.contains("outside node_count"));
    }

    #[test]
    fn try_cpu_ref_into_rejects_bad_csr_without_clobbering_output() {
        let mut out = vec![0xCAFE_BABEu32];
        let capacity = out.capacity();
        let err = try_cpu_ref_into(2, &[0, 1, 1], &[1], &[], &[0], u32::MAX, &mut out)
            .expect_err("mismatched edge arrays must return an error");
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));
        assert_eq!(out, vec![0xCAFE_BABEu32]);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn try_cpu_ref_closure_rejects_bad_seed_without_clobbering_buffers() {
        let mut current = vec![0xCAFE_BABEu32];
        let mut next = vec![0xDEAD_BEEFu32];
        let current_capacity = current.capacity();
        let next_capacity = next.capacity();
        let err = try_cpu_ref_closure_into(
            CsrClosureInputs::allow_all(graphs::CHAIN_4, 4),
            &[],
            &mut current,
            &mut next,
        )
        .expect_err("bad seed width must be rejected");
        assert!(err.contains("expected frontier length"));
        assert_eq!(current, vec![0xCAFE_BABEu32]);
        assert_eq!(next, vec![0xDEAD_BEEFu32]);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);
    }

    #[test]
    fn fallible_cpu_reference_matches_compatibility_wrappers() {
        let g = graphs::CHAIN_4;
        let step = try_cpu_ref(
            g.node_count,
            g.edge_offsets,
            g.edge_targets,
            g.edge_kind_mask,
            &[0b0010],
            u32::MAX,
        )
        .expect("Fix: operation must return Err on failure; tests may use expect only with Fix: recovery text - valid step should succeed");
        assert_eq!(
            step,
            cpu_ref(
                g.node_count,
                g.edge_offsets,
                g.edge_targets,
                g.edge_kind_mask,
                &[0b0010],
                u32::MAX
            )
        );

        let inputs = CsrClosureInputs::allow_all(graphs::CHAIN_4, 5);
        let closure = try_cpu_ref_closure(inputs, &[0b0001])
            .expect("Fix: operation must return Err on failure; tests may use expect only with Fix: recovery text - valid closure should succeed");
        assert_eq!(closure, cpu_ref_closure(inputs, &[0b0001]));
    }

    #[test]
    fn cpu_ref_into_validates_before_resizing_output() {
        let mut out = vec![0xCAFE_BABEu32];
        let original_capacity = out.capacity();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cpu_ref_into(u32::MAX, &[0], &[], &[], &[], u32::MAX, &mut out);
        }));

        assert!(result.is_err(), "malformed CSR must still be rejected");
        assert_eq!(
            out,
            vec![0xCAFE_BABEu32],
            "invalid input must not clear or resize caller output before validation"
        );
        assert_eq!(
            out.capacity(),
            original_capacity,
            "invalid input must not allocate based on hostile node_count"
        );
    }
}
