//! `persistent_bfs`  -  on-device multi-step BFS frontier expansion.
//!
//! The kernel copies `frontier_in` into `frontier_out`, then performs up to
//! `max_iters` forward traversal steps, accumulating reachable nodes into
//! `frontier_out` via atomic OR.  The first `min(max_iters, 4)` iterations
//! are unrolled and use a workgroup-local `wg_scratch` buffer to coalesce
//! per-workgroup change detection between steps.
//!
use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::graph::persistent_bfs_step::persistent_bfs_step_child_prefixed_with_active;
use crate::graph::program_graph::{ProgramGraphShape, BINDING_PRIMITIVE_START};
use crate::hash::fnv1a::{fnv1a64_initial_state, fnv1a64_update_byte};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::persistent_bfs";
/// Canonical op id for batched persistent BFS over many seed frontiers.
pub const BATCH_OP_ID: &str = "vyre-primitives::graph::persistent_bfs_batch";

/// Canonical binding index for the input frontier bitset.
pub const BINDING_FRONTIER_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output frontier bitset.
pub const BINDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 1;
/// Canonical binding index for the global changed flag.
pub const BINDING_CHANGED: u32 = BINDING_PRIMITIVE_START + 2;
/// Canonical workgroup size for persistent BFS programs.
pub const PERSISTENT_BFS_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
/// Canonical single-query dispatch grid.
pub const PERSISTENT_BFS_SINGLE_DISPATCH_GRID: [u32; 3] = [1, 1, 1];

/// Validated persistent-BFS graph layout metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentBfsLayout {
    /// Number of graph nodes accepted by the primitive.
    pub node_count: u32,
    /// Number of logical CSR edges.
    pub edge_count: u32,
    /// Number of u32 words in one frontier bitset.
    pub words: usize,
    /// Number of u32 words in one frontier bitset, narrowed for cache keys.
    pub words_u32: u32,
    /// Number of u32 words required by node-indexed scratch buffers.
    pub node_words: usize,
    /// Number of u32 words required by physical edge buffers after padding.
    pub edge_storage_words: usize,
}

/// Validated flat-frontier batch metadata for persistent BFS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentBfsBatchLayout {
    /// Number of queries in the batch, narrowed for GPU grid dimensions.
    pub query_count: u32,
    /// Total number of u32 words in the flat `[query][word]` frontier array.
    pub total_words: usize,
}

/// Validated single-frontier metadata for resident persistent BFS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentBfsFrontierLayout {
    /// Number of u32 words in the frontier bitset.
    pub words: usize,
    /// Number of u32 words in the frontier bitset, narrowed for primitive metadata.
    pub words_u32: u32,
}

/// Primitive program-cache class for persistent-BFS dispatch plans.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PersistentBfsPlanCacheKind {
    /// One seed frontier for one graph.
    Single,
    /// Many seed frontiers batched over one graph.
    Batch,
}

/// Primitive-owned persistent-BFS program cache key.
///
/// Dispatch wrappers add only backend feature bits; graph identity, frontier
/// width, query count, masks, iteration budget, and plan class are owned here
/// so every backend caches the same primitive program shapes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PersistentBfsPlanCacheKey {
    /// Stable discriminator for the cached program layout.
    ///
    /// Content-addressed graph staging should use [`persistent_bfs_layout_hash`].
    /// Program caches should prefer [`persistent_bfs_program_layout_hash`] so
    /// same-shape CSR contents reuse the same compiled persistent-BFS program.
    pub layout_hash: u64,
    /// Number of graph nodes in the primitive program shape.
    pub node_count: u32,
    /// Number of logical graph edges in the primitive program shape.
    pub edge_count: u32,
    /// Number of frontier words per query.
    pub words_per_query: u32,
    /// Number of queries represented by the program.
    pub query_count: u32,
    /// Edge-kind allow mask compiled into the primitive program.
    pub allow_mask: u32,
    /// Iteration budget compiled into the primitive program.
    pub max_iters: u32,
    /// Backend/device feature key supplied by the dispatch wrapper.
    pub device_features: u64,
    /// Single-query or batched-query plan kind.
    pub kind: PersistentBfsPlanCacheKind,
}

/// Primitive-owned identity for immutable non-resident persistent-BFS inputs.
///
/// Dynamic frontier input/output and changed buffers are intentionally omitted:
/// dispatch wrappers refresh those every call. This key covers graph contents
/// and shape that decide when static CSR/device inputs must be refreshed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PersistentBfsStaticInputKey {
    /// Stable graph-content hash from [`persistent_bfs_layout_hash`].
    pub layout_hash: u64,
    /// Number of graph nodes.
    pub node_count: u32,
    /// Number of logical CSR edges.
    pub edge_count: u32,
    /// Number of frontier words.
    pub words: u32,
}

/// Primitive-owned non-resident persistent-BFS dispatch plan.
pub struct PersistentBfsDispatchPlan {
    layout: PersistentBfsLayout,
    layout_hash: u64,
    allow_mask: u32,
    max_iters: u32,
}

impl PersistentBfsDispatchPlan {
    /// Validated graph/frontier layout.
    #[must_use]
    pub const fn layout(&self) -> PersistentBfsLayout {
        self.layout
    }

    /// Stable graph-layout hash for plan caches.
    #[must_use]
    pub const fn layout_hash(&self) -> u64 {
        self.layout_hash
    }

    /// Number of words in each frontier bitset.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.layout.words
    }

    /// Number of u32 words required by node-indexed scratch buffers.
    #[must_use]
    pub const fn node_words(&self) -> usize {
        self.layout.node_words
    }

    /// Number of u32 words required by edge-indexed buffers after zero padding.
    #[must_use]
    pub const fn edge_storage_words(&self) -> usize {
        self.layout.edge_storage_words
    }

    /// Single-query dispatch grid.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        PERSISTENT_BFS_SINGLE_DISPATCH_GRID
    }

    /// Program graph shape with primitive-owned empty-edge padding.
    #[must_use]
    pub fn program_shape(&self) -> ProgramGraphShape {
        ProgramGraphShape::new(self.layout.node_count, self.layout.edge_count.max(1))
    }

    /// Build the canonical primitive program for this plan.
    #[must_use]
    pub fn program(&self, frontier_in: &str, frontier_out: &str) -> Program {
        persistent_bfs(
            self.program_shape(),
            frontier_in,
            frontier_out,
            self.allow_mask,
            self.max_iters,
        )
    }

    /// Build the primitive-owned program-cache key for this dispatch plan.
    #[must_use]
    pub const fn cache_key(&self, device_features: u64) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash: self.layout_hash,
            node_count: self.layout.node_count,
            edge_count: self.layout.edge_count,
            words_per_query: self.layout.words_u32,
            query_count: 1,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Single,
        }
    }

    /// Build a shape-only program cache key for this non-resident dispatch plan.
    ///
    /// The persistent-BFS program depends on graph dimensions, frontier width,
    /// traversal options, dispatch kind, and device features. CSR edge contents
    /// are dispatch inputs, not shader source, so they must not fragment the
    /// compiled-program cache.
    #[must_use]
    pub fn program_cache_key(&self, device_features: u64) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash: persistent_bfs_program_layout_hash(
                self.layout.node_count,
                self.layout.edge_count,
                self.layout.words_u32,
                1,
                PersistentBfsPlanCacheKind::Single,
            ),
            node_count: self.layout.node_count,
            edge_count: self.layout.edge_count,
            words_per_query: self.layout.words_u32,
            query_count: 1,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Single,
        }
    }

    /// Stable identity for immutable graph inputs associated with this plan.
    #[must_use]
    pub const fn static_input_key(&self) -> PersistentBfsStaticInputKey {
        PersistentBfsStaticInputKey {
            layout_hash: self.layout_hash,
            node_count: self.layout.node_count,
            edge_count: self.layout.edge_count,
            words: self.layout.words_u32,
        }
    }
}

/// Primitive-owned resident single-frontier persistent-BFS dispatch plan.
pub struct PersistentBfsResidentDispatchPlan {
    frontier_layout: PersistentBfsFrontierLayout,
    node_count: u32,
    edge_count: u32,
    allow_mask: u32,
    max_iters: u32,
}

impl PersistentBfsResidentDispatchPlan {
    /// Validated resident frontier layout.
    #[must_use]
    pub const fn frontier_layout(&self) -> PersistentBfsFrontierLayout {
        self.frontier_layout
    }

    /// Number of words in the frontier bitset.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.frontier_layout.words
    }

    /// Number of frontier words narrowed for cache keys.
    #[must_use]
    pub const fn words_u32(&self) -> u32 {
        self.frontier_layout.words_u32
    }

    /// Single-query dispatch grid.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        PERSISTENT_BFS_SINGLE_DISPATCH_GRID
    }

    /// Program graph shape with primitive-owned empty-edge padding.
    #[must_use]
    pub fn program_shape(&self) -> ProgramGraphShape {
        ProgramGraphShape::new(self.node_count, self.edge_count.max(1))
    }

    /// Build the canonical primitive program for this resident plan.
    #[must_use]
    pub fn program(&self, frontier_in: &str, frontier_out: &str) -> Program {
        persistent_bfs(
            self.program_shape(),
            frontier_in,
            frontier_out,
            self.allow_mask,
            self.max_iters,
        )
    }

    /// Build the primitive-owned program-cache key for this resident dispatch plan.
    #[must_use]
    pub const fn cache_key(
        &self,
        layout_hash: u64,
        device_features: u64,
    ) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash,
            node_count: self.node_count,
            edge_count: self.edge_count,
            words_per_query: self.frontier_layout.words_u32,
            query_count: 1,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Single,
        }
    }

    /// Build a shape-only program cache key for this resident dispatch plan.
    #[must_use]
    pub fn program_cache_key(&self, device_features: u64) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash: persistent_bfs_program_layout_hash(
                self.node_count,
                self.edge_count,
                self.frontier_layout.words_u32,
                1,
                PersistentBfsPlanCacheKind::Single,
            ),
            node_count: self.node_count,
            edge_count: self.edge_count,
            words_per_query: self.frontier_layout.words_u32,
            query_count: 1,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Single,
        }
    }
}

/// Primitive-owned resident batched persistent-BFS dispatch plan.
pub struct PersistentBfsResidentBatchDispatchPlan {
    batch_layout: PersistentBfsBatchLayout,
    node_count: u32,
    edge_count: u32,
    words_per_query: u32,
    allow_mask: u32,
    max_iters: u32,
}

impl PersistentBfsResidentBatchDispatchPlan {
    /// Validated flat-frontier batch layout.
    #[must_use]
    pub const fn batch_layout(&self) -> PersistentBfsBatchLayout {
        self.batch_layout
    }

    /// Query count as `usize` for host buffers.
    #[must_use]
    pub const fn query_count(&self) -> usize {
        self.batch_layout.query_count as usize
    }

    /// Query count narrowed for GPU grid dimensions and cache keys.
    #[must_use]
    pub const fn query_count_u32(&self) -> u32 {
        self.batch_layout.query_count
    }

    /// Total flat frontier words across every query.
    #[must_use]
    pub const fn total_words(&self) -> usize {
        self.batch_layout.total_words
    }

    /// Batch dispatch grid.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        if self.batch_layout.query_count == 0 {
            [1, 1, 1]
        } else {
            [1, self.batch_layout.query_count, 1]
        }
    }

    /// Program graph shape with primitive-owned empty-edge padding.
    #[must_use]
    pub fn program_shape(&self) -> ProgramGraphShape {
        ProgramGraphShape::new(self.node_count, self.edge_count.max(1))
    }

    /// Build the canonical primitive batch program for this resident plan.
    #[must_use]
    pub fn program(&self, frontier_in: &str, frontier_out: &str, changed: &str) -> Program {
        persistent_bfs_batch(
            self.program_shape(),
            frontier_in,
            frontier_out,
            changed,
            self.batch_layout.query_count,
            self.allow_mask,
            self.max_iters,
        )
    }

    /// Number of words per query narrowed for cache keys.
    #[must_use]
    pub const fn words_per_query(&self) -> u32 {
        self.words_per_query
    }

    /// Build the primitive-owned program-cache key for this resident batch plan.
    #[must_use]
    pub const fn cache_key(
        &self,
        layout_hash: u64,
        device_features: u64,
    ) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash,
            node_count: self.node_count,
            edge_count: self.edge_count,
            words_per_query: self.words_per_query,
            query_count: self.batch_layout.query_count,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Batch,
        }
    }

    /// Build a shape-only program cache key for this resident batch plan.
    #[must_use]
    pub fn program_cache_key(&self, device_features: u64) -> PersistentBfsPlanCacheKey {
        PersistentBfsPlanCacheKey {
            layout_hash: persistent_bfs_program_layout_hash(
                self.node_count,
                self.edge_count,
                self.words_per_query,
                self.batch_layout.query_count,
                PersistentBfsPlanCacheKind::Batch,
            ),
            node_count: self.node_count,
            edge_count: self.edge_count,
            words_per_query: self.words_per_query,
            query_count: self.batch_layout.query_count,
            allow_mask: self.allow_mask,
            max_iters: self.max_iters,
            device_features,
            kind: PersistentBfsPlanCacheKind::Batch,
        }
    }
}

/// Validate full non-resident persistent-BFS inputs and derive the dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when the graph CSR, edge masks, or seed
/// frontier do not match the primitive contract.
pub fn plan_persistent_bfs_dispatch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<PersistentBfsDispatchPlan, String> {
    let layout = validate_persistent_bfs_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
    )?;
    Ok(PersistentBfsDispatchPlan {
        layout,
        layout_hash: persistent_bfs_layout_hash(
            layout.node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
        ),
        allow_mask,
        max_iters,
    })
}

/// Validate a resident graph frontier and derive the single-query dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when `frontier_in` does not match the
/// resident graph's frontier width.
pub fn plan_persistent_bfs_resident_dispatch(
    node_count: u32,
    edge_count: u32,
    words_per_query: usize,
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<PersistentBfsResidentDispatchPlan, String> {
    Ok(PersistentBfsResidentDispatchPlan {
        frontier_layout: validate_persistent_bfs_frontier(words_per_query, frontier_in)?,
        node_count,
        edge_count,
        allow_mask,
        max_iters,
    })
}

/// Validate resident batched frontiers and derive the batch dispatch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when the flat frontier buffer does not match
/// `query_count * words_per_query` or when the batch cannot fit GPU grid
/// dimensions.
pub fn plan_persistent_bfs_resident_batch_dispatch(
    node_count: u32,
    edge_count: u32,
    words_per_query: usize,
    frontier_inputs: &[u32],
    query_count: usize,
    allow_mask: u32,
    max_iters: u32,
) -> Result<PersistentBfsResidentBatchDispatchPlan, String> {
    let batch_layout =
        validate_persistent_bfs_batch_frontiers(words_per_query, frontier_inputs, query_count)?;
    let words_per_query = u32::try_from(words_per_query).map_err(|_| {
        format!(
            "Fix: persistent_bfs_batch words_per_query {words_per_query} exceeds u32::MAX; shard the graph before GPU dispatch."
        )
    })?;
    Ok(PersistentBfsResidentBatchDispatchPlan {
        batch_layout,
        node_count,
        edge_count,
        words_per_query,
        allow_mask,
        max_iters,
    })
}

/// Copy a seed frontier into caller-owned output storage.
///
/// Reservation happens before mutation, so a failed allocation does not clobber
/// the previous frontier. Dispatch wrappers use this for zero-iteration and
/// validation-only fast paths without forking seed-copy semantics.
///
/// # Errors
///
/// Returns the caller-mapped reservation error.
pub fn copy_persistent_bfs_seed_frontier_into<E, MapError>(
    frontier_out: &mut Vec<u32>,
    frontier_in: &[u32],
    context: &'static str,
    mut map_error: MapError,
) -> Result<(), E>
where
    MapError: FnMut(String) -> E,
{
    crate::graph::scratch::reserve_graph_items_with(
        frontier_out,
        frontier_in.len(),
        context,
        "persistent BFS seed frontier",
        |message| map_error(message),
    )?;
    frontier_out.clear();
    frontier_out.extend_from_slice(frontier_in);
    Ok(())
}

/// Copy flat batched seed frontiers and clear per-query changed flags.
///
/// Both output buffers reserve before mutation, preventing allocation failures
/// from destroying reusable frontier or changed-flag storage.
///
/// # Errors
///
/// Returns the caller-mapped reservation error.
pub fn copy_persistent_bfs_batch_seed_and_clear_changed_into<E, MapError>(
    frontier_outputs: &mut Vec<u32>,
    frontier_inputs: &[u32],
    changed_outputs: &mut Vec<u32>,
    query_count: usize,
    context: &'static str,
    mut map_error: MapError,
) -> Result<(), E>
where
    MapError: FnMut(String) -> E,
{
    crate::graph::scratch::reserve_graph_items_with(
        frontier_outputs,
        frontier_inputs.len(),
        context,
        "persistent BFS batch frontier",
        |message| map_error(message),
    )?;
    crate::graph::scratch::reserve_graph_items_with(
        changed_outputs,
        query_count,
        context,
        "persistent BFS batch changed flags",
        |message| map_error(message),
    )?;
    frontier_outputs.clear();
    frontier_outputs.extend_from_slice(frontier_inputs);
    changed_outputs.clear();
    changed_outputs.resize(query_count, 0);
    Ok(())
}

/// Validate a persistent-BFS changed flag read back from a backend.
///
/// # Errors
///
/// Returns an actionable diagnostic when the scalar flag is not boolean.
pub fn validate_persistent_bfs_changed_flag(changed: u32) -> Result<(), String> {
    if changed > 1 {
        return Err(format!(
            "Fix: persistent BFS changed flag readback must be 0 or 1, got {changed}. Treat this as malformed GPU readback or a backend bug."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod dispatch_plan_contract_tests {
    use super::*;

    #[test]
    fn static_input_key_tracks_graph_content_not_frontier_seed() {
        let offsets = [0, 1, 1];
        let targets = [1];
        let masks = [7];
        let seed_a = [1];
        let seed_b = [2];
        let plan_a =
            plan_persistent_bfs_dispatch(2, &offsets, &targets, &masks, &seed_a, u32::MAX, 4)
                .expect("Fix: valid persistent BFS plan should build");
        let plan_b =
            plan_persistent_bfs_dispatch(2, &offsets, &targets, &masks, &seed_b, u32::MAX, 4)
                .expect("Fix: seed-only changes should keep static input identity");
        let changed_graph =
            plan_persistent_bfs_dispatch(2, &offsets, &[0], &masks, &seed_a, u32::MAX, 4)
                .expect("Fix: same-shape changed graph should still validate");

        assert_eq!(plan_a.static_input_key(), plan_b.static_input_key());
        assert_ne!(plan_a.static_input_key(), changed_graph.static_input_key());
    }

    #[test]
    fn seed_copy_reserves_before_mutating_output() {
        let mut out = vec![9, 9, 9];
        copy_persistent_bfs_seed_frontier_into(&mut out, &[1, 2], "test seed copy", |message| {
            message
        })
        .expect("Fix: small seed copy should succeed");

        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn batch_seed_copy_clears_changed_flags() {
        let mut frontier = vec![9, 9, 9];
        let mut changed = vec![1, 1, 1];
        copy_persistent_bfs_batch_seed_and_clear_changed_into(
            &mut frontier,
            &[1, 2, 3, 4],
            &mut changed,
            2,
            "test batch seed copy",
            |message| message,
        )
        .expect("Fix: small batch seed copy should succeed");

        assert_eq!(frontier, vec![1, 2, 3, 4]);
        assert_eq!(changed, vec![0, 0]);
    }

    #[test]
    fn changed_flag_validation_rejects_non_boolean_values() {
        validate_persistent_bfs_changed_flag(0).expect("zero is a valid changed flag");
        validate_persistent_bfs_changed_flag(1).expect("one is a valid changed flag");
        let err = validate_persistent_bfs_changed_flag(2)
            .expect_err("non-boolean changed flags must be rejected");
        assert!(err.contains("0 or 1"));
    }
}

/// Words needed to hold a bitset over `node_count` nodes.
#[must_use]
pub const fn bitset_words(node_count: u32) -> u32 {
    crate::bitset::bitset_words(node_count)
}

/// Build the IR `Program` for persistent BFS.
///
/// The kernel copies `frontier_in` into `frontier_out`, then performs up
/// to `max_iters` forward traversal steps.  The first four iterations are
/// unrolled with inter-step workgroup barriers and a shared `wg_scratch`
/// array; any additional iterations run in a plain bounded loop.
///
/// `changed` is a single u32 word that is set to `1` if *any* step produced
/// a new reachable node.
#[must_use]
pub fn persistent_bfs(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    let words = bitset_words(shape.node_count);
    let t = Expr::gid_x();

    let unrolled_iter = |iter: u32| -> Node {
        persistent_bfs_step_child_prefixed_with_active(
            OP_ID,
            shape,
            frontier_out,
            "changed",
            "wg_scratch",
            "wg_active",
            edge_kind_mask,
            &format!("unroll_{iter}"),
        )
    };

    let mut entry: Vec<Node> = vec![
        // Seed frontier_out from frontier_in.
        Node::if_then(
            Expr::eq(Expr::local_x(), Expr::u32(0)),
            vec![Node::loop_for(
                "seed_word_idx",
                Expr::u32(0),
                Expr::u32(words),
                vec![Node::store(
                    frontier_out,
                    Expr::var("seed_word_idx"),
                    Expr::load(frontier_in, Expr::var("seed_word_idx")),
                )],
            )],
        ),
        // Zero the global changed flag.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(0)),
            vec![
                Node::store("changed", Expr::u32(0), Expr::u32(0)),
                Node::store("wg_active", Expr::u32(0), Expr::u32(1)),
            ],
        ),
        // Barrier clears fusion hazards from the plain store above before the
        // first atomic access inside the unrolled steps.
        Node::barrier(),
    ];

    let unroll_count = max_iters.min(4);
    for iter in 0..unroll_count {
        entry.push(unrolled_iter(iter));
    }

    let remaining = max_iters.saturating_sub(unroll_count);
    if remaining > 0 {
        entry.push(Node::loop_for(
            "iter",
            Expr::u32(0),
            Expr::u32(remaining),
            vec![Node::if_then(
                Expr::ne(
                    Expr::load("wg_active", Expr::u32(0)),
                    Expr::u32(0),
                ),
                vec![
                    Node::let_bind("local_changed", Expr::u32(0)),
                    Node::if_then(
                        Expr::lt(t.clone(), Expr::u32(shape.node_count)),
                        vec![
                            crate::graph::csr_forward_or_changed::csr_forward_or_changed_child_prefixed(
                                OP_ID,
                                shape,
                                frontier_out,
                                "local_changed",
                                edge_kind_mask,
                                "remaining_csr",
                            ),
                        ],
                    ),
                    Node::if_then(
                        Expr::eq(t.clone(), Expr::u32(0)),
                        vec![Node::store(
                            "wg_active",
                            Expr::u32(0),
                            Expr::var("local_changed"),
                        )],
                    ),
                    Node::if_then(
                        Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
                        vec![Node::let_bind(
                            "_",
                            Expr::atomic_or("changed", Expr::u32(0), Expr::u32(1)),
                        )],
                    ),
                ],
            )],
        ));
    }

    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_in,
            BINDING_FRONTIER_IN,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_FRONTIER_OUT,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            "changed",
            BINDING_CHANGED,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );
    buffers.push(BufferDecl::workgroup("wg_scratch", 256, DataType::U32));
    buffers.push(BufferDecl::workgroup("wg_active", 1, DataType::U32));

    Program::wrapped(
        buffers,
        PERSISTENT_BFS_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(entry),
        }],
    )
}

/// Build a batched persistent-BFS Program.
///
/// Frontier buffers are flat `[query][word]` arrays. The launch topology is
/// one workgroup per query on `grid.y`; inside each query the same persistent
/// CSR expansion contract as [`persistent_bfs`] is applied to that query's
/// frontier slice.
#[must_use]
pub fn persistent_bfs_batch(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Program {
    match try_persistent_bfs_batch(
        shape,
        frontier_in,
        frontier_out,
        changed,
        query_count,
        edge_kind_mask,
        max_iters,
    ) {
        Ok(program) => program,
        Err(_) => inert_persistent_bfs_batch_program(shape, frontier_in, frontier_out, changed),
    }
}

/// Build a batched persistent-BFS Program with checked flat-frontier sizing.
pub fn try_persistent_bfs_batch(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
    query_count: u32,
    edge_kind_mask: u32,
    max_iters: u32,
) -> Result<Program, String> {
    let words = bitset_words(shape.node_count).max(1);
    let q = Expr::gid_y();
    let base = Expr::mul(q.clone(), Expr::u32(words));

    let src = "batch_src";
    let word_idx = "batch_word_idx";
    let bit_mask = "batch_bit_mask";
    let src_word = "batch_src_word";
    let edge_start = "batch_edge_start";
    let edge_end = "batch_edge_end";
    let edge_iter = "batch_edge";
    let kind_mask = "batch_kind_mask";
    let dst = "batch_dst";
    let dst_word_idx = "batch_dst_word_idx";
    let dst_bit = "batch_dst_bit";
    let old = "batch_old";
    let local_changed = "batch_local_changed";
    let active = "batch_active";

    let per_source = vec![
        Node::let_bind(word_idx, Expr::shr(Expr::var(src), Expr::u32(5))),
        Node::let_bind(
            bit_mask,
            Expr::shl(Expr::u32(1), Expr::bitand(Expr::var(src), Expr::u32(31))),
        ),
        Node::let_bind(
            src_word,
            Expr::load(frontier_out, Expr::add(base.clone(), Expr::var(word_idx))),
        ),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var(src_word), Expr::var(bit_mask)),
                Expr::u32(0),
            ),
            vec![
                Node::let_bind(edge_start, Expr::load("pg_edge_offsets", Expr::var(src))),
                Node::let_bind(
                    edge_end,
                    Expr::load("pg_edge_offsets", Expr::add(Expr::var(src), Expr::u32(1))),
                ),
                Node::loop_for(
                    edge_iter,
                    Expr::var(edge_start),
                    Expr::var(edge_end),
                    vec![
                        Node::let_bind(
                            kind_mask,
                            Expr::load("pg_edge_kind_mask", Expr::var(edge_iter)),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var(kind_mask), Expr::u32(edge_kind_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    dst,
                                    Expr::load("pg_edge_targets", Expr::var(edge_iter)),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var(dst), Expr::u32(shape.node_count)),
                                    vec![
                                        Node::let_bind(
                                            dst_word_idx,
                                            Expr::shr(Expr::var(dst), Expr::u32(5)),
                                        ),
                                        Node::let_bind(
                                            dst_bit,
                                            Expr::shl(
                                                Expr::u32(1),
                                                Expr::bitand(Expr::var(dst), Expr::u32(31)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            old,
                                            Expr::atomic_or(
                                                frontier_out,
                                                Expr::add(base.clone(), Expr::var(dst_word_idx)),
                                                Expr::var(dst_bit),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::eq(
                                                Expr::bitand(Expr::var(old), Expr::var(dst_bit)),
                                                Expr::u32(0),
                                            ),
                                            vec![Node::assign(local_changed, Expr::u32(1))],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ];

    let iter_body = vec![
        Node::let_bind(local_changed, Expr::u32(0)),
        Node::if_then(
            Expr::ne(Expr::var(active), Expr::u32(0)),
            vec![Node::if_then(
                Expr::eq(Expr::local_x(), Expr::u32(0)),
                vec![Node::loop_for(
                    src,
                    Expr::u32(0),
                    Expr::u32(shape.node_count),
                    per_source,
                )],
            )],
        ),
        Node::assign(active, Expr::var(local_changed)),
        Node::if_then(
            Expr::eq(Expr::var(local_changed), Expr::u32(1)),
            vec![Node::let_bind(
                "batch_changed_old",
                Expr::atomic_or(changed, q.clone(), Expr::u32(1)),
            )],
        ),
        Node::barrier(),
    ];

    let entry: Vec<Node> = vec![
        Node::if_then(
            Expr::eq(Expr::local_x(), Expr::u32(0)),
            vec![Node::loop_for(
                "batch_copy_word",
                Expr::u32(0),
                Expr::u32(words),
                vec![Node::store(
                    frontier_out,
                    Expr::add(base.clone(), Expr::var("batch_copy_word")),
                    Expr::load(
                        frontier_in,
                        Expr::add(base.clone(), Expr::var("batch_copy_word")),
                    ),
                )],
            )],
        ),
        Node::if_then(
            Expr::eq(Expr::local_x(), Expr::u32(0)),
            vec![Node::store(changed, q.clone(), Expr::u32(0))],
        ),
        Node::barrier(),
        Node::let_bind(active, Expr::u32(1)),
        Node::loop_for("batch_iter", Expr::u32(0), Expr::u32(max_iters), iter_body),
    ];

    let total_words = checked_batch_frontier_words(words, query_count, BATCH_OP_ID)?;
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_in,
            BINDING_FRONTIER_IN,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(total_words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_FRONTIER_OUT,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(total_words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            BINDING_CHANGED,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(query_count.max(1)),
    );

    Ok(Program::wrapped(
        buffers,
        PERSISTENT_BFS_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(BATCH_OP_ID),
            source_region: None,
            body: Arc::new(entry),
        }],
    ))
}

fn checked_batch_frontier_words(
    words_per_query: u32,
    query_count: u32,
    op_id: &'static str,
) -> Result<u32, String> {
    words_per_query.checked_mul(query_count.max(1)).ok_or_else(|| {
        format!(
            "{op_id} frontier words overflow u32: words_per_query={words_per_query}, query_count={query_count}. Fix: shard the BFS query batch before GPU dispatch."
        )
    })
}

fn inert_persistent_bfs_batch_program(
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    changed: &str,
) -> Program {
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_in,
            BINDING_FRONTIER_IN,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(1),
    );
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_FRONTIER_OUT,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            BINDING_CHANGED,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );

    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(BATCH_OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::return_()]),
        }],
    )
}

/// CPU reference: run BFS up to `max_iters` steps, accumulating into a
/// running bitset.  Returns the final frontier and a sticky `changed`
/// flag (`1` if any step added new nodes, else `0`).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32) {
    try_cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    )
    .expect("persistent_bfs CPU oracle received malformed input or could not reserve output")
}

/// Fallible CPU reference for persistent BFS.
///
/// This is the primitive-owned entry point for parity wrappers that must reject
/// hostile CSR/frontier inputs without panicking.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<(Vec<u32>, u32), String> {
    let mut out = Vec::new();
    let changed = try_cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
        &mut out,
    )?;
    Ok((out, changed))
}

/// Caller-owned workspace for repeated persistent-BFS CPU oracle runs.
///
/// Conformance and CUDA parity sweeps call this oracle across large generated
/// graph corpora. Reusing the per-iteration frontier scratch avoids a heap
/// allocation per proof case while preserving the allocating compatibility API.
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Debug, Default, Clone)]
pub struct PersistentBfsCpuScratch {
    /// Temporary frontier produced by one CSR expansion step.
    pub step: Vec<u32>,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl PersistentBfsCpuScratch {
    /// Create an empty reusable persistent-BFS workspace.
    pub fn new() -> Self {
        Self::default()
    }
}

/// CPU reference into caller-owned output storage.
///
/// Runs BFS up to `max_iters` steps, accumulating into `frontier_out`. Returns
/// a sticky changed flag (`1` if any step added new nodes, else `0`).
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    frontier_out: &mut Vec<u32>,
) -> u32 {
    let mut scratch = PersistentBfsCpuScratch::default();
    try_cpu_ref_into_with_scratch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
        frontier_out,
        &mut scratch,
    )
    .expect("persistent_bfs CPU oracle received malformed input or could not reserve output")
}

/// Fallible CPU reference into caller-owned output storage.
///
/// On error, `frontier_out` is left unchanged. This lets integration tests and
/// dispatch wrappers treat malformed graph/frontier data as a typed finding
/// instead of a panic or partially clobbered oracle output.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    frontier_out: &mut Vec<u32>,
) -> Result<u32, String> {
    let mut scratch = PersistentBfsCpuScratch::default();
    try_cpu_ref_into_with_scratch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
        frontier_out,
        &mut scratch,
    )
}

/// Fallible CPU reference into caller-owned output and scratch storage.
///
/// On validation error, `frontier_out` and `scratch` are left unchanged. This
/// lets integration tests and dispatch wrappers treat malformed graph/frontier
/// data as a typed finding instead of a panic or partially clobbered oracle
/// state.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_cpu_ref_into_with_scratch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    frontier_out: &mut Vec<u32>,
    scratch: &mut PersistentBfsCpuScratch,
) -> Result<u32, String> {
    let layout = validate_persistent_bfs_inputs(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
    )?;
    let words = layout.words;
    crate::graph::scratch::reserve_graph_items(
        frontier_out,
        words,
        "persistent BFS CPU oracle",
        "frontier output",
    )?;
    crate::graph::scratch::reserve_graph_items(
        &mut scratch.step,
        words,
        "persistent BFS CPU oracle",
        "per-iteration frontier scratch",
    )?;
    frontier_out.clear();
    frontier_out.extend_from_slice(frontier_in);
    frontier_out.resize(words, 0);
    scratch.step.clear();
    scratch.step.resize(words, 0);
    let mut changed = 0u32;

    for _ in 0..max_iters {
        crate::graph::csr_forward_traverse::cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            frontier_out,
            allow_mask,
            &mut scratch.step,
        );
        let mut step_changed = false;
        for w in 0..words {
            let old = frontier_out[w];
            frontier_out[w] |= scratch.step[w];
            if frontier_out[w] != old {
                step_changed = true;
            }
        }
        if step_changed {
            changed = 1;
        } else {
            break;
        }
    }
    Ok(changed)
}

/// Validate a persistent-BFS CSR graph layout.
///
/// # Errors
///
/// Returns an actionable diagnostic when offsets are malformed, masks and
/// targets diverge, the edge count exceeds u32 indexing, or an edge target is
/// outside `0..node_count`.
pub fn validate_persistent_bfs_graph_layout(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<PersistentBfsLayout, String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!("Fix: persistent_bfs node_count + 1 overflows usize for node_count={node_count}.")
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: persistent_bfs expected {expected_offsets} CSR offsets for {node_count} nodes, got {}.",
            edge_offsets.len()
        ));
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: persistent_bfs requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    let edge_count = u32::try_from(edge_targets.len()).map_err(|_| {
        format!(
            "Fix: persistent_bfs edge count {} exceeds u32 index space.",
            edge_targets.len()
        )
    })?;
    let final_offset = edge_offsets[expected_offsets - 1] as usize;
    if final_offset != edge_targets.len() {
        return Err(format!(
            "Fix: persistent_bfs final CSR offset {final_offset} must equal edge_count {}.",
            edge_targets.len()
        ));
    }
    for (row, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: persistent_bfs CSR offsets are non-monotonic at row {row}: {} > {}.",
                pair[0], pair[1]
            ));
        }
    }
    for (idx, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: persistent_bfs CSR target[{idx}]={target} is outside node_count {node_count}."
            ));
        }
    }
    let words_u32 = bitset_words(node_count);
    Ok(PersistentBfsLayout {
        node_count,
        edge_count,
        words: words_u32 as usize,
        words_u32,
        node_words: node_count as usize,
        edge_storage_words: edge_targets.len().max(1),
    })
}

/// Validate the full non-resident persistent-BFS dispatch/input contract.
///
/// # Errors
///
/// Returns an actionable diagnostic when the graph layout is malformed or the
/// seed frontier length does not match the graph.
pub fn validate_persistent_bfs_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
) -> Result<PersistentBfsLayout, String> {
    let layout = validate_persistent_bfs_graph_layout(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    )?;
    if frontier_in.len() != layout.words {
        return Err(format!(
            "Fix: persistent_bfs expected frontier length {} words for {node_count} nodes, got {}.",
            layout.words,
            frontier_in.len()
        ));
    }
    Ok(layout)
}

/// Validate flat-frontier batch shape for persistent BFS.
///
/// The frontier buffer is laid out as `[query][word]`, where
/// `words_per_query` is derived from the already-validated graph layout.
///
/// # Errors
///
/// Returns an actionable diagnostic when query count cannot be represented by
/// GPU grid dimensions, the flat word count overflows, or the supplied
/// frontier buffer length does not match `words_per_query * query_count`.
pub fn validate_persistent_bfs_batch_frontiers(
    words_per_query: usize,
    frontier_inputs: &[u32],
    query_count: usize,
) -> Result<PersistentBfsBatchLayout, String> {
    let query_count_u32 = u32::try_from(query_count).map_err(|_| {
        format!(
            "Fix: persistent_bfs_batch query_count {query_count} exceeds u32::MAX; shard the BFS query batch before GPU dispatch."
        )
    })?;
    let total_words = words_per_query.checked_mul(query_count).ok_or_else(|| {
        format!(
            "Fix: persistent_bfs_batch word count overflows usize for {words_per_query} words/query and {query_count} queries; shard the BFS query batch before GPU dispatch."
        )
    })?;
    if frontier_inputs.len() != total_words {
        return Err(format!(
            "Fix: persistent_bfs_batch expected {total_words} frontier word(s), got {}.",
            frontier_inputs.len()
        ));
    }
    Ok(PersistentBfsBatchLayout {
        query_count: query_count_u32,
        total_words,
    })
}

/// Validate a single persistent-BFS frontier against an already-validated graph layout.
///
/// # Errors
///
/// Returns an actionable diagnostic when the graph frontier width cannot be
/// represented by primitive metadata, or when the supplied frontier length
/// does not match the graph frontier width.
pub fn validate_persistent_bfs_frontier(
    words_per_query: usize,
    frontier_in: &[u32],
) -> Result<PersistentBfsFrontierLayout, String> {
    let words_u32 = u32::try_from(words_per_query).map_err(|_| {
        format!(
            "Fix: persistent_bfs frontier word count {words_per_query} exceeds u32::MAX; shard the graph before GPU dispatch."
        )
    })?;
    if frontier_in.len() != words_per_query {
        return Err(format!(
            "Fix: persistent_bfs expected frontier length {words_per_query} word(s), got {}.",
            frontier_in.len()
        ));
    }
    Ok(PersistentBfsFrontierLayout {
        words: words_per_query,
        words_u32,
    })
}

fn fnv1a64_mix_u32(hash: &mut u64, value: u32) {
    for byte in value.to_le_bytes() {
        *hash = fnv1a64_update_byte(*hash, byte);
    }
}

/// Stable FNV-1a hash of a persistent-BFS graph layout.
#[must_use]
pub fn persistent_bfs_layout_hash(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> u64 {
    let mut hash = fnv1a64_initial_state();
    fnv1a64_mix_u32(&mut hash, node_count);
    fnv1a64_mix_u32(&mut hash, edge_offsets.len() as u32);
    for &value in edge_offsets {
        fnv1a64_mix_u32(&mut hash, value);
    }
    fnv1a64_mix_u32(&mut hash, edge_targets.len() as u32);
    for &value in edge_targets {
        fnv1a64_mix_u32(&mut hash, value);
    }
    fnv1a64_mix_u32(&mut hash, edge_kind_mask.len() as u32);
    for &value in edge_kind_mask {
        fnv1a64_mix_u32(&mut hash, value);
    }
    hash
}

/// Stable FNV-1a hash of the persistent-BFS program shape.
///
/// This intentionally excludes CSR contents. The generated program is the same
/// for any graph with the same dimensions, frontier width, query count, and
/// dispatch kind; edge data is carried in buffers.
#[must_use]
pub fn persistent_bfs_program_layout_hash(
    node_count: u32,
    edge_count: u32,
    words_per_query: u32,
    query_count: u32,
    kind: PersistentBfsPlanCacheKind,
) -> u64 {
    let mut hash = fnv1a64_initial_state();
    fnv1a64_mix_u32(&mut hash, 0x5042_4653);
    fnv1a64_mix_u32(&mut hash, node_count);
    fnv1a64_mix_u32(&mut hash, edge_count);
    fnv1a64_mix_u32(&mut hash, words_per_query);
    fnv1a64_mix_u32(&mut hash, query_count);
    fnv1a64_mix_u32(
        &mut hash,
        match kind {
            PersistentBfsPlanCacheKind::Single => 0,
            PersistentBfsPlanCacheKind::Batch => 1,
        },
    );
    hash
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || persistent_bfs(ProgramGraphShape::new(4, 4), "fin", "fout", 0xFFFF_FFFF, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),          // pg_nodes
                to_bytes(&[0, 2, 3, 4, 4]),       // pg_edge_offsets
                to_bytes(&[1, 2, 3, 3]),          // pg_edge_targets
                to_bytes(&[1, 1, 1, 1]),          // pg_edge_kind_mask
                to_bytes(&[0, 0, 0, 0]),          // pg_node_tags
                to_bytes(&[0b0001]),              // frontier_in = {0}
                to_bytes(&[0]),                   // frontier_out
                to_bytes(&[0]),                   // changed
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            // After 4 iterations the graph 0→1,0→2,1→3,2→3 is fully closed.
            vec![vec![
                to_bytes(&[0b1111]),              // frontier_out = {0,1,2,3}
                to_bytes(&[1]),                   // changed
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_bfs_reaches_closure() {
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            4,
        );
        assert_eq!(frontier, vec![0b1111]);
        assert_eq!(changed, 1);
    }

    #[test]
    fn cpu_ref_into_reuses_frontier_storage() {
        let mut frontier = Vec::with_capacity(8);
        let changed = cpu_ref_into(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            8,
            &mut frontier,
        );
        let capacity = frontier.capacity();
        assert_eq!(frontier, vec![0b1111]);
        assert_eq!(changed, 1);

        let changed = cpu_ref_into(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0],
            0xFFFF_FFFF,
            8,
            &mut frontier,
        );
        assert_eq!(frontier.capacity(), capacity);
        assert_eq!(frontier, vec![0]);
        assert_eq!(changed, 0);
    }

    #[test]
    fn try_cpu_ref_into_with_scratch_reuses_step_storage_and_clears_stale_state() {
        let mut frontier = Vec::with_capacity(8);
        let mut step = Vec::with_capacity(8);
        step.extend_from_slice(&[0xDEAD_BEEF, 0xCAFE_BABE, 0xBADC_0FFE]);
        let mut scratch = PersistentBfsCpuScratch { step };
        let frontier_capacity = frontier.capacity();
        let step_capacity = scratch.step.capacity();

        let changed = try_cpu_ref_into_with_scratch(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            8,
            &mut frontier,
            &mut scratch,
        )
        .expect("Fix: valid persistent BFS chain must run with reusable scratch.");
        assert_eq!(frontier, vec![0b1111]);
        assert_eq!(changed, 1);
        assert_eq!(frontier.capacity(), frontier_capacity);
        assert_eq!(scratch.step.capacity(), step_capacity);
        assert_eq!(scratch.step.len(), 1);

        let changed = try_cpu_ref_into_with_scratch(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0],
            0xFFFF_FFFF,
            8,
            &mut frontier,
            &mut scratch,
        )
        .expect("Fix: second persistent BFS run must clear stale step bits.");
        assert_eq!(frontier, vec![0]);
        assert_eq!(changed, 0);
        assert_eq!(frontier.capacity(), frontier_capacity);
        assert_eq!(scratch.step.capacity(), step_capacity);
        assert_eq!(
            scratch.step,
            vec![0],
            "Fix: reusable step scratch must be resized to live words and cleared by traversal."
        );
    }

    #[test]
    fn try_cpu_ref_into_rejects_bad_input_without_clobbering_frontier() {
        let mut frontier = vec![0xDEAD_BEEF];
        let capacity = frontier.capacity();

        let err = try_cpu_ref_into(
            4,
            &[0, 1, 2],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            8,
            &mut frontier,
        )
        .expect_err("Fix: fallible persistent BFS oracle must reject malformed CSR inputs");

        assert!(err.contains("CSR offsets"));
        assert_eq!(frontier, vec![0xDEAD_BEEF]);
        assert_eq!(frontier.capacity(), capacity);
    }

    #[test]
    fn try_cpu_ref_into_with_scratch_rejects_bad_input_without_clobbering_storage() {
        let mut frontier = vec![0xDEAD_BEEF];
        let mut scratch = PersistentBfsCpuScratch {
            step: vec![0xCAFE_BABE, 0xBADC_0FFE],
        };

        let err = try_cpu_ref_into_with_scratch(
            4,
            &[0, 1, 2],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            8,
            &mut frontier,
            &mut scratch,
        )
        .expect_err("Fix: fallible persistent BFS oracle must reject malformed CSR inputs.");

        assert!(err.contains("CSR offsets"));
        assert_eq!(
            frontier,
            vec![0xDEAD_BEEF],
            "Fix: validation failures must not clobber reusable frontier output."
        );
        assert_eq!(
            scratch.step,
            vec![0xCAFE_BABE, 0xBADC_0FFE],
            "Fix: validation failures must not clear reusable step scratch."
        );
    }

    #[test]
    fn fallible_cpu_ref_matches_compatibility_oracle_on_generated_chains() {
        for node_count in [0_u32, 1, 2, 3, 31, 32, 33, 64, 65, 257] {
            let mut offsets = Vec::with_capacity(node_count as usize + 1);
            let mut targets = Vec::new();
            let mut masks = Vec::new();
            offsets.push(0);
            for node in 0..node_count {
                if node + 1 < node_count {
                    targets.push(node + 1);
                    masks.push(1);
                }
                offsets.push(targets.len() as u32);
            }
            let words = bitset_words(node_count) as usize;
            let mut seed = vec![0; words];
            if node_count != 0 {
                seed[0] = 1;
            }

            let expected = cpu_ref(
                node_count,
                &offsets,
                &targets,
                &masks,
                &seed,
                0xFFFF_FFFF,
                node_count.saturating_add(1),
            );
            let actual = try_cpu_ref(
                node_count,
                &offsets,
                &targets,
                &masks,
                &seed,
                0xFFFF_FFFF,
                node_count.saturating_add(1),
            )
            .expect("Fix: generated valid persistent BFS chain should run fallibly");
            assert_eq!(actual, expected, "node_count={node_count}");
        }
    }

    #[test]
    fn generated_try_cpu_ref_into_with_scratch_matches_allocating_reference() {
        let mut frontier = Vec::new();
        let mut scratch = PersistentBfsCpuScratch::new();

        for case in 0..1024usize {
            let node_count = (case % 67) as u32;
            let mut offsets = Vec::with_capacity(node_count as usize + 1);
            let mut targets = Vec::new();
            let mut masks = Vec::new();
            offsets.push(0);
            for src in 0..node_count {
                for dst in 0..node_count {
                    let mixed = case
                        .wrapping_mul(43)
                        .wrapping_add((src as usize).wrapping_mul(17))
                        .wrapping_add((dst as usize).wrapping_mul(29));
                    if src != dst && (mixed % 23 == 0 || (case % 19 == 0 && dst == src + 1)) {
                        targets.push(dst);
                        masks.push(if mixed % 2 == 0 { 1 } else { 2 });
                    }
                }
                offsets.push(targets.len() as u32);
            }

            let words = bitset_words(node_count) as usize;
            let mut seed = vec![0; words];
            for node in 0..node_count {
                let mixed = case
                    .wrapping_mul(11)
                    .wrapping_add((node as usize).wrapping_mul(7));
                if mixed % 13 == 0 || (node == 0 && node_count != 0) {
                    seed[(node / 32) as usize] |= 1u32 << (node % 32);
                }
            }
            let allow_mask = if case % 3 == 0 { 1 } else { 0xFFFF_FFFF };
            let max_iters = (case % 11) as u32;
            let expected = try_cpu_ref(
                node_count, &offsets, &targets, &masks, &seed, allow_mask, max_iters,
            )
            .expect("Fix: generated persistent BFS graph must be valid for allocating oracle.");
            let changed = try_cpu_ref_into_with_scratch(
                node_count,
                &offsets,
                &targets,
                &masks,
                &seed,
                allow_mask,
                max_iters,
                &mut frontier,
                &mut scratch,
            )
            .expect("Fix: generated persistent BFS graph must run with reusable scratch.");
            assert_eq!(
                (frontier.clone(), changed),
                expected,
                "Fix: scratch-backed persistent BFS diverged from allocating oracle at case {case}."
            );
        }
    }

    #[test]
    fn reusable_layout_validation_rejects_bad_csr_and_frontier() {
        let err = validate_persistent_bfs_graph_layout(2, &[0, 2, 1], &[1], &[1]).unwrap_err();
        assert!(err.contains("final CSR offset") || err.contains("non-monotonic"));

        let err = validate_persistent_bfs_graph_layout(2, &[0, 1, 1], &[2], &[1]).unwrap_err();
        assert!(err.contains("outside node_count"));

        let err = validate_persistent_bfs_inputs(33, &[0; 34], &[], &[], &[0]).unwrap_err();
        assert!(err.contains("frontier length 2 words"));
    }

    #[test]
    fn reusable_graph_layout_returns_dispatch_shape() {
        assert_eq!(
            validate_persistent_bfs_graph_layout(33, &[0; 34], &[], &[]).unwrap(),
            PersistentBfsLayout {
                node_count: 33,
                edge_count: 0,
                words: 2,
                words_u32: 2,
                node_words: 33,
                edge_storage_words: 1,
            }
        );
        assert_eq!(
            validate_persistent_bfs_inputs(4, &[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1, 1], &[0])
                .unwrap(),
            PersistentBfsLayout {
                node_count: 4,
                edge_count: 3,
                words: 1,
                words_u32: 1,
                node_words: 4,
                edge_storage_words: 3,
            }
        );
    }

    #[test]
    fn dispatch_plans_pin_grid_cache_shape_and_program_builders() {
        let edge_offsets = [0, 1, 2, 3, 3];
        let edge_targets = [1, 2, 3];
        let edge_kind_mask = [1, 1, 1];
        let plan = plan_persistent_bfs_dispatch(
            4,
            &edge_offsets,
            &edge_targets,
            &edge_kind_mask,
            &[0b0001],
            0xFFFF_FFFF,
            8,
        )
        .expect("Fix: canonical persistent-BFS dispatch plan should validate");

        assert_eq!(plan.layout().node_count, 4);
        assert_eq!(plan.layout().edge_count, 3);
        assert_eq!(plan.frontier_words(), 1);
        assert_eq!(plan.node_words(), 4);
        assert_eq!(plan.edge_storage_words(), 3);
        assert_eq!(plan.dispatch_grid(), PERSISTENT_BFS_SINGLE_DISPATCH_GRID);
        assert_eq!(
            plan.layout_hash(),
            persistent_bfs_layout_hash(4, &edge_offsets, &edge_targets, &edge_kind_mask)
        );
        assert_eq!(
            plan.cache_key(0xCAFE),
            PersistentBfsPlanCacheKey {
                layout_hash: plan.layout_hash(),
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 1,
                allow_mask: 0xFFFF_FFFF,
                max_iters: 8,
                device_features: 0xCAFE,
                kind: PersistentBfsPlanCacheKind::Single,
            }
        );
        assert_eq!(
            plan.program_cache_key(0xCAFE),
            PersistentBfsPlanCacheKey {
                layout_hash: persistent_bfs_program_layout_hash(
                    4,
                    3,
                    1,
                    1,
                    PersistentBfsPlanCacheKind::Single,
                ),
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 1,
                allow_mask: 0xFFFF_FFFF,
                max_iters: 8,
                device_features: 0xCAFE,
                kind: PersistentBfsPlanCacheKind::Single,
            }
        );
        assert_eq!(
            plan.program("frontier_in", "frontier_out").workgroup_size,
            PERSISTENT_BFS_WORKGROUP_SIZE
        );

        let empty_edge_plan =
            plan_persistent_bfs_dispatch(2, &[0, 0, 0], &[], &[], &[0], 0xFFFF_FFFF, 1)
                .expect("Fix: zero-edge persistent-BFS graph is a valid dispatch shape");
        assert_eq!(empty_edge_plan.layout().edge_count, 0);
        assert_eq!(empty_edge_plan.edge_storage_words(), 1);
        assert_eq!(
            empty_edge_plan
                .program("frontier_in", "frontier_out")
                .workgroup_size,
            PERSISTENT_BFS_WORKGROUP_SIZE
        );

        let resident = plan_persistent_bfs_resident_dispatch(4, 3, 1, &[0b0001], 0xFF, 4)
            .expect("Fix: resident single-frontier plan should validate");
        assert_eq!(resident.frontier_words(), 1);
        assert_eq!(resident.words_u32(), 1);
        assert_eq!(
            resident.dispatch_grid(),
            PERSISTENT_BFS_SINGLE_DISPATCH_GRID
        );
        assert_eq!(
            resident.cache_key(0xABCD, 0x10),
            PersistentBfsPlanCacheKey {
                layout_hash: 0xABCD,
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 1,
                allow_mask: 0xFF,
                max_iters: 4,
                device_features: 0x10,
                kind: PersistentBfsPlanCacheKind::Single,
            }
        );
        assert_eq!(
            resident.program_cache_key(0x10),
            PersistentBfsPlanCacheKey {
                layout_hash: persistent_bfs_program_layout_hash(
                    4,
                    3,
                    1,
                    1,
                    PersistentBfsPlanCacheKind::Single,
                ),
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 1,
                allow_mask: 0xFF,
                max_iters: 4,
                device_features: 0x10,
                kind: PersistentBfsPlanCacheKind::Single,
            }
        );

        let batch = plan_persistent_bfs_resident_batch_dispatch(4, 3, 1, &[1, 2], 2, 0xFF, 4)
            .expect("Fix: resident batch plan should validate");
        assert_eq!(batch.query_count(), 2);
        assert_eq!(batch.query_count_u32(), 2);
        assert_eq!(batch.total_words(), 2);
        assert_eq!(batch.words_per_query(), 1);
        assert_eq!(batch.dispatch_grid(), [1, 2, 1]);
        assert_eq!(
            batch
                .program("frontier_in", "frontier_out", "changed")
                .workgroup_size,
            PERSISTENT_BFS_WORKGROUP_SIZE
        );
        assert_eq!(
            batch.cache_key(0xABCD, 0x20),
            PersistentBfsPlanCacheKey {
                layout_hash: 0xABCD,
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 2,
                allow_mask: 0xFF,
                max_iters: 4,
                device_features: 0x20,
                kind: PersistentBfsPlanCacheKind::Batch,
            }
        );
        assert_eq!(
            batch.program_cache_key(0x20),
            PersistentBfsPlanCacheKey {
                layout_hash: persistent_bfs_program_layout_hash(
                    4,
                    3,
                    1,
                    2,
                    PersistentBfsPlanCacheKind::Batch,
                ),
                node_count: 4,
                edge_count: 3,
                words_per_query: 1,
                query_count: 2,
                allow_mask: 0xFF,
                max_iters: 4,
                device_features: 0x20,
                kind: PersistentBfsPlanCacheKind::Batch,
            }
        );
    }

    #[test]
    fn reusable_batch_frontier_validation_accepts_empty_and_canonical_batches() {
        assert_eq!(
            validate_persistent_bfs_batch_frontiers(2, &[], 0).unwrap(),
            PersistentBfsBatchLayout {
                query_count: 0,
                total_words: 0,
            }
        );

        assert_eq!(
            validate_persistent_bfs_batch_frontiers(2, &[1, 0, 2, 0, 4, 0], 3).unwrap(),
            PersistentBfsBatchLayout {
                query_count: 3,
                total_words: 6,
            }
        );
    }

    #[test]
    fn reusable_batch_frontier_validation_rejects_bad_shape_and_overflow() {
        let err = validate_persistent_bfs_batch_frontiers(2, &[1, 0, 2], 2).unwrap_err();
        assert!(err.contains("expected 4 frontier word"));

        let err = validate_persistent_bfs_batch_frontiers(usize::MAX, &[], 2).unwrap_err();
        assert!(err.contains("word count overflows usize"));

        let err =
            validate_persistent_bfs_batch_frontiers(1, &[], u32::MAX as usize + 1).unwrap_err();
        assert!(err.contains("query_count"));
    }

    #[test]
    fn reusable_single_frontier_validation_accepts_canonical_frontier() {
        assert_eq!(
            validate_persistent_bfs_frontier(2, &[1, 0]).unwrap(),
            PersistentBfsFrontierLayout {
                words: 2,
                words_u32: 2,
            }
        );
    }

    #[test]
    fn reusable_single_frontier_validation_rejects_bad_shape_and_overflow() {
        let err = validate_persistent_bfs_frontier(2, &[1]).unwrap_err();
        assert!(err.contains("expected frontier length 2 word"));

        let err = validate_persistent_bfs_frontier(u32::MAX as usize + 1, &[]).unwrap_err();
        assert!(err.contains("frontier word count"));
    }

    #[test]
    fn layout_hash_distinguishes_edges_and_masks() {
        let a = persistent_bfs_layout_hash(2, &[0, 1, 1], &[1], &[1]);
        let b = persistent_bfs_layout_hash(2, &[0, 1, 1], &[1], &[2]);
        let c = persistent_bfs_layout_hash(2, &[0, 1, 1], &[0], &[1]);
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn program_cache_key_reuses_same_shape_graph_variants() {
        let a = plan_persistent_bfs_dispatch(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[1],
            0xFFFF_FFFF,
            8,
        )
        .unwrap();
        let b = plan_persistent_bfs_dispatch(
            4,
            &[0, 1, 2, 3, 3],
            &[2, 3, 0],
            &[1, 1, 1],
            &[1],
            0xFFFF_FFFF,
            8,
        )
        .unwrap();

        assert_ne!(a.layout_hash(), b.layout_hash());
        assert_ne!(a.cache_key(0xCAFE), b.cache_key(0xCAFE));
        assert_eq!(a.program_cache_key(0xCAFE), b.program_cache_key(0xCAFE));
    }

    #[test]
    fn empty_frontier_stays_empty() {
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0],
            0xFFFF_FFFF,
            4,
        );
        assert_eq!(frontier, vec![0]);
        assert_eq!(changed, 0);
    }

    #[test]
    fn edge_mask_limits_reachability() {
        // 0→1 (mask 0b10), 0→2 (mask 0b01), 1→3 (mask 0b01), 2→3 (mask 0b01)
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[0b10, 0b01, 0b01, 0b01],
            &[0b0001],
            0b01,
            4,
        );
        // From 0, only 0→2 is allowed. Then 2→3 is allowed.
        assert_eq!(frontier, vec![0b1101]);
        assert_eq!(changed, 1);
    }

    #[test]
    fn max_iters_caps_expansion() {
        // Chain: 0→1, 1→2, 2→3. Frontier = {0}.
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            2,
        );
        // After 2 steps: {0,1,2}
        assert_eq!(frontier, vec![0b0111]);
        assert_eq!(changed, 1);
    }

    #[test]
    fn zero_max_iters_is_noop() {
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            0,
        );
        assert_eq!(frontier, vec![0b0001]);
        assert_eq!(changed, 0);
    }

    #[test]
    fn program_builds_and_validates() {
        let program = persistent_bfs(ProgramGraphShape::new(8, 8), "fin", "fout", 0xFF, 4);
        assert_eq!(program.workgroup_size, [256, 1, 1]);
        // 5 canonical PG buffers + frontier_in + frontier_out + changed + wg_scratch + wg_active
        assert_eq!(program.buffers().len(), 10);
    }

    #[test]
    fn program_carries_device_side_convergence_flag() {
        let program = persistent_bfs(ProgramGraphShape::new(8, 8), "fin", "fout", 0xFF, 8);
        let debug = format!("{:?}", program.entry);
        assert!(
            debug.contains("wg_active"),
            "persistent_bfs must gate later device work through a workgroup-resident active flag"
        );
    }

    #[test]
    fn persistent_bfs_seed_copy_covers_frontiers_larger_than_one_workgroup() {
        let source = include_str!("persistent_bfs.rs");
        let single_source = source
            .split("pub fn persistent_bfs(")
            .nth(1)
            .expect("Fix: persistent_bfs builder source must be present")
            .split("/// Build a batched persistent-BFS Program.")
            .next()
            .expect("Fix: persistent_bfs builder source must precede batch builder");

        assert!(
            single_source.contains("Node::loop_for(\n                \"seed_word_idx\""),
            "Fix: persistent_bfs must copy every frontier word, not only the first workgroup lane range."
        );
        assert!(
            !single_source.contains("Node::let_bind(\"seed_word_idx\", t.clone())"),
            "Fix: persistent_bfs seed copy must not be capped by gid_x."
        );
    }

    #[test]
    fn batch_program_carries_per_query_convergence_flag() {
        let program = persistent_bfs_batch(
            ProgramGraphShape::new(8, 8),
            "fin",
            "fout",
            "changed",
            4,
            0xFF,
            8,
        );
        let debug = format!("{:?}", program.entry);
        assert!(
            debug.contains("batch_active"),
            "persistent_bfs_batch must gate later per-query device work through an active flag"
        );
    }

    #[test]
    fn persistent_bfs_batch_seed_copy_covers_frontiers_larger_than_one_workgroup() {
        let source = include_str!("persistent_bfs.rs");
        let batch_source = source
            .split("pub fn try_persistent_bfs_batch(")
            .nth(1)
            .expect("Fix: checked batch builder source must be present")
            .split("fn checked_batch_frontier_words(")
            .next()
            .expect("Fix: checked batch builder source must precede sizing helper");

        assert!(
            batch_source.contains("Node::loop_for(\n                \"batch_copy_word\""),
            "Fix: persistent_bfs_batch must copy every frontier word for each query."
        );
        assert!(
            !batch_source.contains("Expr::lt(t.clone(), Expr::u32(words))"),
            "Fix: persistent_bfs_batch seed copy must not be capped by the first workgroup lane range."
        );
    }

    #[test]
    fn checked_batch_builder_rejects_flat_frontier_overflow() {
        let error = try_persistent_bfs_batch(
            ProgramGraphShape::new(u32::MAX, 0),
            "fin",
            "fout",
            "changed",
            33,
            0xFF,
            1,
        )
        .expect_err("checked batched persistent BFS builder must reject flat frontier overflow");

        assert!(
            error.contains("frontier words overflow u32"),
            "error should describe the flat frontier overflow: {error}"
        );
    }

    #[test]
    fn legacy_batch_builder_does_not_panic_on_flat_frontier_overflow() {
        let program = persistent_bfs_batch(
            ProgramGraphShape::new(u32::MAX, 0),
            "fin",
            "fout",
            "changed",
            33,
            0xFF,
            1,
        );

        assert_eq!(program.workgroup_size, [256, 1, 1]);
    }

    #[test]
    fn persistent_bfs_batch_release_source_has_checked_builder_without_panics() {
        let source = include_str!("persistent_bfs.rs");
        let batch_source = source
            .split("/// Build a batched persistent-BFS Program.")
            .nth(1)
            .expect("Fix: persistent BFS batch builder source must be present")
            .split("/// CPU reference:")
            .next()
            .expect("Fix: persistent BFS batch builder source must precede CPU oracle");

        assert!(
            batch_source.contains("pub fn try_persistent_bfs_batch(")
                && !batch_source.contains(concat!("panic", "!("))
                && !batch_source.contains(".unwrap_or_else("),
            "Fix: persistent_bfs_batch must expose checked release API and avoid production panics."
        );
    }
}
