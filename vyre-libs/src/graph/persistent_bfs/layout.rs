use vyre_foundation::ir::{BufferAccess, BufferDecl};

use super::hash::persistent_bfs_program_layout_hash;
use crate::graph::program_graph::{word_buffer, ProgramGraphShape, BINDING_PRIMITIVE_START};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::persistent_bfs";
/// Canonical op id for batched persistent BFS over many seed frontiers.
pub const BATCH_OP_ID: &str = "vyre-libs::graph::persistent_bfs_batch";

/// Canonical binding index for the input frontier bitset.
pub const BINDING_FRONTIER_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output frontier bitset.
pub const BINDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 1;
/// Canonical binding index for the global changed flag.
pub const BINDING_CHANGED: u32 = BINDING_PRIMITIVE_START + 2;
/// Canonical binding index for the converged flag.
///
/// `1` if the frontier reached a fixpoint (a step added nothing) before the
/// `max_iters` budget was exhausted, `0` if the loop ran all `max_iters` steps
/// while still growing (an under-approximated closure) or `max_iters == 0`.
/// This is the device readback that lets a host caller reject a partial closure
/// loudly instead of silently trusting a frontier the kernel never drove to a
/// fixpoint. Mirrors the `vyre-reference` persistent BFS witness convergence contract.
pub const BINDING_CONVERGED: u32 = BINDING_PRIMITIVE_START + 3;
/// Canonical binding index for the optional per-iteration frontier-density array.
///
/// Present only in the density-instrumented program variants
/// ([`super::program::persistent_bfs_with_density`] and
/// [`super::program::try_persistent_bfs_batch_with_density`]). It is a
/// `max_iters`-length (single) or `query_count * max_iters` (batch) u32 array
/// where entry `i` holds the popcount of the frontier after traversal step `i`
/// (per query for the batch variant). Because reachability growth is monotone, a
/// host caller reconstructs every `FrontierDensityTelemetry` aggregate (active
/// total, per-step delta, peak, last) from this array plus the seed popcount,
/// with no per-iteration device readback loop. The base
/// [`super::program::persistent_bfs`] programs omit this buffer entirely, so
/// their ABI is unchanged.
pub const BINDING_DENSITY_ACTIVE: u32 = BINDING_PRIMITIVE_START + 4;
/// Canonical name for the per-iteration frontier-density array output buffer.
pub const DENSITY_ACTIVE_BUFFER: &str = "density_active";
/// Canonical workgroup size for persistent BFS programs.
pub const PERSISTENT_BFS_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];
/// One-block dispatch grid used by the compact single-workgroup BFS path.
pub(crate) const PERSISTENT_BFS_SINGLE_DISPATCH_GRID: [u32; 3] = [1, 1, 1];

/// The persistent-BFS buffer bundle: two frontier bitsets, a changed array, a
/// converged array, and the optional per-iteration density array.
///
/// The three program variants differ only in these word counts, never in the
/// binding order or the access, so each one names its counts here instead of
/// restating the bundle and risking a disagreement a backend would see only as a
/// mis-sized allocation.
pub(crate) struct PersistentBfsBuffers<'a> {
    /// Read-only input frontier.
    pub frontier_in: &'a str,
    /// Read-write output frontier.
    pub frontier_out: &'a str,
    /// Words in each frontier bitset.
    pub frontier_words: u32,
    /// Changed array name and word count.
    pub changed: (&'a str, u32),
    /// Converged array name and word count.
    pub converged: (&'a str, u32),
    /// Density array name and word count, when the variant is instrumented.
    pub density_active: Option<(&'a str, u32)>,
}

impl PersistentBfsBuffers<'_> {
    /// Append the bundle in canonical binding order.
    pub(crate) fn push_onto(&self, buffers: &mut Vec<BufferDecl>) {
        let words = self.frontier_words.max(1);
        buffers.push(word_buffer(
            self.frontier_in,
            BINDING_FRONTIER_IN,
            BufferAccess::ReadOnly,
            words,
        ));
        buffers.push(word_buffer(
            self.frontier_out,
            BINDING_FRONTIER_OUT,
            BufferAccess::ReadWrite,
            words,
        ));
        for (name, binding, count) in [
            (self.changed.0, BINDING_CHANGED, self.changed.1),
            (self.converged.0, BINDING_CONVERGED, self.converged.1),
        ] {
            buffers.push(word_buffer(name, binding, BufferAccess::ReadWrite, count));
        }
        if let Some((density, count)) = self.density_active {
            buffers.push(word_buffer(
                density,
                BINDING_DENSITY_ACTIVE,
                BufferAccess::ReadWrite,
                count,
            ));
        }
    }
}

/// Dispatch grid for a single persistent-BFS query.
#[must_use]
pub const fn persistent_bfs_single_dispatch_grid(node_count: u32) -> [u32; 3] {
    [persistent_bfs_grid_x(node_count), 1, 1]
}

/// Dispatch grid for a batched persistent-BFS query set.
#[must_use]
pub const fn persistent_bfs_batch_dispatch_grid(node_count: u32, query_count: u32) -> [u32; 3] {
    if query_count == 0 {
        [1, 1, 1]
    } else {
        [persistent_bfs_grid_x(node_count), query_count, 1]
    }
}

/// Grid X for a resident persistent-BFS launch: one lane per node.
const fn persistent_bfs_grid_x(node_count: u32) -> u32 {
    vyre_primitives::lane_grid(node_count, PERSISTENT_BFS_WORKGROUP_SIZE[0])[0]
}

/// Program graph shape with primitive-owned empty-edge padding.
#[must_use]
pub(super) fn persistent_bfs_program_shape(node_count: u32, edge_count: u32) -> ProgramGraphShape {
    ProgramGraphShape::new(node_count, edge_count.max(1))
}

/// Build a primitive-owned single-query program cache key with explicit layout hash.
#[must_use]
pub(super) const fn persistent_bfs_single_cache_key(
    layout_hash: u64,
    node_count: u32,
    edge_count: u32,
    words_u32: u32,
    allow_mask: u32,
    max_iters: u32,
    device_features: u64,
) -> PersistentBfsPlanCacheKey {
    PersistentBfsPlanCacheKey {
        layout_hash,
        node_count,
        edge_count,
        words_per_query: words_u32,
        query_count: 1,
        allow_mask,
        max_iters,
        device_features,
        kind: PersistentBfsPlanCacheKind::Single,
    }
}

/// Build a shape-only program cache key for a single persistent BFS plan.
#[must_use]
pub(super) fn persistent_bfs_single_program_cache_key(
    node_count: u32,
    edge_count: u32,
    words_u32: u32,
    allow_mask: u32,
    max_iters: u32,
    device_features: u64,
) -> PersistentBfsPlanCacheKey {
    PersistentBfsPlanCacheKey {
        layout_hash: persistent_bfs_program_layout_hash(
            node_count,
            edge_count,
            words_u32,
            1,
            PersistentBfsPlanCacheKind::Single,
        ),
        node_count,
        edge_count,
        words_per_query: words_u32,
        query_count: 1,
        allow_mask,
        max_iters,
        device_features,
        kind: PersistentBfsPlanCacheKind::Single,
    }
}

/// Build a shape-only program cache key for a batched persistent BFS plan.
#[must_use]
pub(super) fn persistent_bfs_batch_program_cache_key(
    node_count: u32,
    edge_count: u32,
    words_per_query: u32,
    query_count: u32,
    allow_mask: u32,
    max_iters: u32,
    device_features: u64,
) -> PersistentBfsPlanCacheKey {
    PersistentBfsPlanCacheKey {
        layout_hash: persistent_bfs_program_layout_hash(
            node_count,
            edge_count,
            words_per_query,
            query_count,
            PersistentBfsPlanCacheKind::Batch,
        ),
        node_count,
        edge_count,
        words_per_query,
        query_count,
        allow_mask,
        max_iters,
        device_features,
        kind: PersistentBfsPlanCacheKind::Batch,
    }
}

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
    /// Content-addressed graph staging should use [`crate::graph::persistent_bfs::persistent_bfs_layout_hash`].
    /// Program caches should prefer [`crate::graph::persistent_bfs::persistent_bfs_program_layout_hash`] so
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
    /// Stable graph-content hash from [`crate::graph::persistent_bfs::persistent_bfs_layout_hash`].
    pub layout_hash: u64,
    /// Number of graph nodes.
    pub node_count: u32,
    /// Number of logical CSR edges.
    pub edge_count: u32,
    /// Number of frontier words.
    pub words: u32,
}
