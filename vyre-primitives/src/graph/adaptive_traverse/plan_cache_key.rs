//! Resident adaptive traversal program identity: the shape-only cache key
//! every dispatch layer keys compiled programs on, plus the in-session content
//! hashes for resident graph uploads.

use std::hash::{Hash, Hasher};

/// Primitive-owned resident adaptive traversal program identity.
///
/// Self-substrate and future CUDA/WGSL/SPIR-V dispatch layers use this as the
/// stable cache-key taxonomy instead of forking per-wrapper enums.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AdaptiveTraversalProgramKind {
    /// Count set bits in the input frontier.
    Popcount,
    /// Clear the output frontier before an OR-writing traversal kernel.
    ClearFrontierOut,
    /// Initialize the active queue length before sparse queue compaction.
    QueueLenInit,
    /// Device-selected CSR/dense reverse-bitmatrix traversal.
    SparseDense,
    /// Compact active source ids from a frontier bitset into a queue.
    FrontierToQueue,
    /// Compute per-word active-node prefix counts for packed-frontier queues.
    FrontierWordCounts,
    /// Convert packed-frontier block totals into exclusive block offsets.
    FrontierWordBlockOffsets,
    /// Scatter packed frontier words into a deterministic active-source queue.
    FrontierWordPrefixQueue,
    /// Scatter packed frontier words using precomputed block offsets.
    FrontierWordBlockOffsetsQueue,
    /// Consume a compacted active-source queue through CSR rows.
    QueueForward,
    /// Consume a compacted active-source queue with lane teams for skewed rows.
    QueueForwardStrided,
    /// Expand low-degree queued rows and compact only high-degree rows.
    QueueSplitLow,
    /// Dense graph traversal through a reusable Four-Russians byte-tile LUT.
    FourRussiansDense,
}

/// Stable cache key for resident adaptive traversal Programs.
///
/// The key deliberately includes program layout identity, frontier width, queue
/// capacity, traversal masks, threshold policy, and backend feature bits so a
/// cached Program cannot be reused across incompatible CUDA/WGSL/SPIR-V shapes.
/// Resident graph contents are represented by dispatch handles, not shader
/// source, so same-shape resident graphs reuse compiled Programs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AdaptiveTraversalPlanCacheKey {
    /// Shape-only hash of the resident Program layout.
    pub layout_hash: u64,
    /// Number of graph nodes.
    pub node_count: u32,
    /// Number of logical CSR edges.
    pub edge_count: u32,
    /// Number of u32 words in one frontier bitset.
    pub words: u32,
    /// Active-source queue capacity for sparse-queue Programs.
    pub queue_capacity: u32,
    /// Allowed edge-kind mask baked into traversal Programs.
    pub allow_mask: u32,
    /// Dense cutover threshold baked into sparse/dense Programs.
    pub dense_threshold_pct: u32,
    /// Backend feature fingerprint from the dispatcher.
    pub device_features: u64,
    /// Resident Program shape represented by this key.
    pub kind: AdaptiveTraversalProgramKind,
}

impl AdaptiveTraversalPlanCacheKey {
    /// Construct a cache key for a resident adaptive traversal Program.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        allow_mask: u32,
        dense_threshold_pct: u32,
        device_features: u64,
        kind: AdaptiveTraversalProgramKind,
    ) -> Self {
        Self {
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            allow_mask,
            dense_threshold_pct,
            device_features,
            kind,
        }
    }

    /// Cache key for the frontier popcount Program.
    #[must_use]
    pub const fn popcount(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            0,
            AdaptiveTraversalProgramKind::Popcount,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            0,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::Popcount,
        )
    }

    /// Cache key for clearing the output frontier.
    #[must_use]
    pub const fn clear_frontier_out(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            0,
            AdaptiveTraversalProgramKind::ClearFrontierOut,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            0,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::ClearFrontierOut,
        )
    }

    /// Cache key for device-selected sparse/dense traversal.
    #[must_use]
    pub const fn sparse_dense(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        allow_mask: u32,
        dense_threshold_pct: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            0,
            AdaptiveTraversalProgramKind::SparseDense,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            0,
            allow_mask,
            dense_threshold_pct,
            device_features,
            AdaptiveTraversalProgramKind::SparseDense,
        )
    }

    /// Cache key for the active-queue length initialization Program.
    #[must_use]
    pub const fn queue_len_init(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::QueueLenInit,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::QueueLenInit,
        )
    }

    /// Cache key for frontier-to-active-queue compaction.
    #[must_use]
    pub const fn frontier_to_queue(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::FrontierToQueue,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FrontierToQueue,
        )
    }

    /// Cache key for packed-frontier word-count scan.
    #[must_use]
    pub const fn frontier_word_counts(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            0,
            AdaptiveTraversalProgramKind::FrontierWordCounts,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            0,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FrontierWordCounts,
        )
    }

    /// Cache key for packed-frontier block-offset scan.
    #[must_use]
    pub const fn frontier_word_block_offsets(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            0,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsets,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            0,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsets,
        )
    }

    /// Cache key for deterministic packed-frontier queue scatter.
    #[must_use]
    pub const fn frontier_word_prefix_queue(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::FrontierWordPrefixQueue,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FrontierWordPrefixQueue,
        )
    }

    /// Cache key for deterministic packed-frontier queue scatter with block offsets.
    #[must_use]
    pub const fn frontier_word_block_offsets_queue(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsetsQueue,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsetsQueue,
        )
    }

    /// Cache key for queue-driven CSR traversal.
    #[must_use]
    pub const fn queue_forward(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        allow_mask: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::QueueForward,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            allow_mask,
            0,
            device_features,
            AdaptiveTraversalProgramKind::QueueForward,
        )
    }

    /// Cache key for row-strided queue-driven CSR traversal.
    #[must_use]
    pub const fn queue_forward_strided(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        allow_mask: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            AdaptiveTraversalProgramKind::QueueForwardStrided,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            allow_mask,
            0,
            device_features,
            AdaptiveTraversalProgramKind::QueueForwardStrided,
        )
    }

    /// Cache key for the low-row half of mixed queue-driven CSR traversal.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn queue_split_low(
        _layout_hash: u64,
        node_count: u32,
        edge_count: u32,
        words: u32,
        queue_capacity: u32,
        high_queue_capacity: u32,
        high_degree_threshold: u32,
        allow_mask: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_split_program_layout_hash(
            node_count,
            edge_count,
            words,
            queue_capacity,
            high_queue_capacity,
            high_degree_threshold,
            AdaptiveTraversalProgramKind::QueueSplitLow,
        );
        Self::new(
            layout_hash,
            node_count,
            edge_count,
            words,
            queue_capacity,
            allow_mask,
            0,
            device_features,
            AdaptiveTraversalProgramKind::QueueSplitLow,
        )
    }

    /// Cache key for dense Four-Russians traversal through a resident LUT.
    #[must_use]
    pub const fn four_russians_dense(
        _layout_hash: u64,
        node_count: u32,
        words: u32,
        device_features: u64,
    ) -> Self {
        let layout_hash = adaptive_traversal_program_layout_hash(
            node_count,
            0,
            words,
            0,
            AdaptiveTraversalProgramKind::FourRussiansDense,
        );
        Self::new(
            layout_hash,
            node_count,
            0,
            words,
            0,
            0,
            0,
            device_features,
            AdaptiveTraversalProgramKind::FourRussiansDense,
        )
    }
}

const fn adaptive_traversal_program_kind_tag(kind: AdaptiveTraversalProgramKind) -> u64 {
    match kind {
        AdaptiveTraversalProgramKind::Popcount => 1,
        AdaptiveTraversalProgramKind::ClearFrontierOut => 2,
        AdaptiveTraversalProgramKind::SparseDense => 3,
        AdaptiveTraversalProgramKind::QueueLenInit => 4,
        AdaptiveTraversalProgramKind::FrontierToQueue => 5,
        AdaptiveTraversalProgramKind::QueueForward => 6,
        AdaptiveTraversalProgramKind::FourRussiansDense => 7,
        AdaptiveTraversalProgramKind::FrontierWordCounts => 8,
        AdaptiveTraversalProgramKind::FrontierWordPrefixQueue => 9,
        AdaptiveTraversalProgramKind::FrontierWordBlockOffsets => 10,
        AdaptiveTraversalProgramKind::FrontierWordBlockOffsetsQueue => 11,
        AdaptiveTraversalProgramKind::QueueForwardStrided => 12,
        AdaptiveTraversalProgramKind::QueueSplitLow => 13,
    }
}

const fn adaptive_traversal_hash_mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x0000_0100_0000_01B3)
}

/// Shape-only hash for resident adaptive traversal program layouts.
///
/// This excludes resident graph contents and dense LUT source rows; those are
/// already bound through resident handles. Including content here fragments the
/// compiled-program cache without changing generated code.
#[must_use]
pub const fn adaptive_traversal_program_layout_hash(
    node_count: u32,
    edge_count: u32,
    words: u32,
    queue_capacity: u32,
    kind: AdaptiveTraversalProgramKind,
) -> u64 {
    let hash = adaptive_traversal_hash_mix(0xcbf2_9ce4_8422_2325, 0x4154_5241_5645_5253);
    let hash = adaptive_traversal_hash_mix(hash, node_count as u64);
    let hash = adaptive_traversal_hash_mix(hash, edge_count as u64);
    let hash = adaptive_traversal_hash_mix(hash, words as u64);
    let hash = adaptive_traversal_hash_mix(hash, queue_capacity as u64);
    adaptive_traversal_hash_mix(hash, adaptive_traversal_program_kind_tag(kind))
}

/// Shape-only hash for mixed queue traversal programs whose low-row half also
/// depends on high-row queue capacity and the high-degree threshold.
#[must_use]
pub const fn adaptive_traversal_split_program_layout_hash(
    node_count: u32,
    edge_count: u32,
    words: u32,
    queue_capacity: u32,
    high_queue_capacity: u32,
    high_degree_threshold: u32,
    kind: AdaptiveTraversalProgramKind,
) -> u64 {
    let hash =
        adaptive_traversal_program_layout_hash(node_count, edge_count, words, queue_capacity, kind);
    let hash = adaptive_traversal_hash_mix(hash, high_queue_capacity as u64);
    adaptive_traversal_hash_mix(hash, high_degree_threshold as u64)
}

/// In-session content hash for resident adaptive CSR+dense graph uploads.
///
/// This hashes graph contents, unlike [`adaptive_traversal_program_layout_hash`],
/// which intentionally hashes only generated-program shape. Resident upload
/// wrappers use this to identify uploaded graph layouts without forking the
/// primitive's graph identity contract.
#[must_use]
pub fn adaptive_traversal_graph_content_hash(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node_count.hash(&mut hasher);
    edge_offsets.hash(&mut hasher);
    edge_targets.hash(&mut hasher);
    edge_kind_mask.hash(&mut hasher);
    adj_rows_dense.hash(&mut hasher);
    hasher.finish()
}

/// In-session content hash for resident adaptive sparse-queue CSR uploads.
#[must_use]
pub fn adaptive_sparse_queue_graph_content_hash(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node_count.hash(&mut hasher);
    edge_offsets.hash(&mut hasher);
    edge_targets.hash(&mut hasher);
    edge_kind_mask.hash(&mut hasher);
    hasher.finish()
}

/// In-session content hash for resident adaptive Four-Russians dense LUT uploads.
#[must_use]
pub fn adaptive_four_russians_graph_content_hash(node_count: u32, adj_rows_dense: &[u32]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    node_count.hash(&mut hasher);
    adj_rows_dense.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod resident_content_hash_tests {
    use super::*;

    #[test]
    fn graph_content_hash_tracks_csr_masks_and_dense_rows() {
        let offsets = [0, 1, 1];
        let targets = [1];
        let masks = [7];
        let dense = [0b10, 0];
        let baseline = adaptive_traversal_graph_content_hash(2, &offsets, &targets, &masks, &dense);
        let changed_mask =
            adaptive_traversal_graph_content_hash(2, &offsets, &targets, &[3], &dense);
        let changed_dense =
            adaptive_traversal_graph_content_hash(2, &offsets, &targets, &masks, &[0, 1]);

        assert_ne!(baseline, changed_mask);
        assert_ne!(baseline, changed_dense);
    }

    #[test]
    fn sparse_queue_content_hash_tracks_csr_without_dense_rows() {
        let offsets = [0, 1, 1];
        let targets = [1];
        let masks = [7];
        let baseline = adaptive_sparse_queue_graph_content_hash(2, &offsets, &targets, &masks);
        let changed_mask = adaptive_sparse_queue_graph_content_hash(2, &offsets, &targets, &[3]);
        let changed_target = adaptive_sparse_queue_graph_content_hash(2, &offsets, &[0], &masks);

        assert_ne!(baseline, changed_mask);
        assert_ne!(baseline, changed_target);
    }

    #[test]
    fn four_russians_content_hash_tracks_lut_source_rows() {
        let baseline = adaptive_four_russians_graph_content_hash(8, &[1, 0, 0, 0, 0, 0, 0, 0]);
        let changed = adaptive_four_russians_graph_content_hash(8, &[2, 0, 0, 0, 0, 0, 0, 0]);

        assert_ne!(baseline, changed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_plan_cache_keys_pin_resident_program_identity() {
        let sparse_dense =
            AdaptiveTraversalPlanCacheKey::sparse_dense(7, 64, 9, 2, 0x55, 25, 0xA11CE);
        assert_eq!(sparse_dense.kind, AdaptiveTraversalProgramKind::SparseDense);
        assert_eq!(
            sparse_dense.layout_hash,
            adaptive_traversal_program_layout_hash(
                64,
                9,
                2,
                0,
                AdaptiveTraversalProgramKind::SparseDense,
            )
        );
        assert_eq!(sparse_dense.queue_capacity, 0);
        assert_eq!(sparse_dense.allow_mask, 0x55);
        assert_eq!(sparse_dense.dense_threshold_pct, 25);
        assert_eq!(
            sparse_dense,
            AdaptiveTraversalPlanCacheKey::sparse_dense(99, 64, 9, 2, 0x55, 25, 0xA11CE),
            "resident graph contents must not fragment adaptive traversal Program caches"
        );

        assert_ne!(
            sparse_dense,
            AdaptiveTraversalPlanCacheKey::sparse_dense(7, 64, 9, 2, 0xAA, 25, 0xA11CE),
            "edge-mask policy must be part of sparse/dense resident Program identity"
        );
        assert_ne!(
            sparse_dense,
            AdaptiveTraversalPlanCacheKey::sparse_dense(7, 64, 9, 2, 0x55, 50, 0xA11CE),
            "dense cutover policy must be part of sparse/dense resident Program identity"
        );
        assert_ne!(
            sparse_dense,
            AdaptiveTraversalPlanCacheKey::sparse_dense(7, 64, 9, 2, 0x55, 25, 0xC0DA),
            "backend feature bits must be part of resident Program identity"
        );

        let queue_forward =
            AdaptiveTraversalPlanCacheKey::queue_forward(7, 64, 9, 2, 64, 0x55, 0xA11CE);
        assert_eq!(
            queue_forward.kind,
            AdaptiveTraversalProgramKind::QueueForward
        );
        assert_eq!(queue_forward.queue_capacity, 64);
        assert_eq!(queue_forward.allow_mask, 0x55);
        let queue_forward_strided =
            AdaptiveTraversalPlanCacheKey::queue_forward_strided(7, 64, 9, 2, 64, 0x55, 0xA11CE);
        assert_eq!(
            queue_forward_strided.kind,
            AdaptiveTraversalProgramKind::QueueForwardStrided
        );
        assert_ne!(
            queue_forward, queue_forward_strided,
            "serial and row-strided queue consumers must not alias in resident Program caches"
        );
        let queue_split_low =
            AdaptiveTraversalPlanCacheKey::queue_split_low(7, 64, 9, 2, 64, 4, 1024, 0x55, 0xA11CE);
        assert_eq!(
            queue_split_low.kind,
            AdaptiveTraversalProgramKind::QueueSplitLow
        );
        assert_eq!(queue_split_low.queue_capacity, 64);
        assert_eq!(queue_split_low.dense_threshold_pct, 0);
        assert_ne!(
            queue_split_low,
            AdaptiveTraversalPlanCacheKey::queue_split_low(7, 64, 9, 2, 64, 8, 1024, 0x55, 0xA11CE,),
            "mixed split queue programs must distinguish high-row queue capacity"
        );
        assert_ne!(
            queue_split_low,
            AdaptiveTraversalPlanCacheKey::queue_split_low(7, 64, 9, 2, 64, 4, 2048, 0x55, 0xA11CE,),
            "mixed split queue programs must distinguish high-degree threshold"
        );
        assert_ne!(
            queue_forward,
            AdaptiveTraversalPlanCacheKey::frontier_to_queue(7, 64, 9, 2, 64, 0xA11CE)
        );
        let word_counts =
            AdaptiveTraversalPlanCacheKey::frontier_word_counts(7, 8_192, 9, 256, 0xA11CE);
        assert_eq!(
            word_counts.kind,
            AdaptiveTraversalProgramKind::FrontierWordCounts
        );
        assert_eq!(word_counts.queue_capacity, 0);
        let block_offsets = AdaptiveTraversalPlanCacheKey::frontier_word_block_offsets(
            7, 32_897, 9, 1_029, 0xA11CE,
        );
        assert_eq!(
            block_offsets.kind,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsets
        );
        assert_eq!(block_offsets.queue_capacity, 0);
        let word_prefix = AdaptiveTraversalPlanCacheKey::frontier_word_prefix_queue(
            7, 8_192, 9, 256, 8_192, 0xA11CE,
        );
        assert_eq!(
            word_prefix.kind,
            AdaptiveTraversalProgramKind::FrontierWordPrefixQueue
        );
        assert_eq!(word_prefix.queue_capacity, 8_192);
        assert_ne!(
            word_prefix,
            AdaptiveTraversalPlanCacheKey::frontier_to_queue(7, 8_192, 9, 256, 8_192, 0xA11CE),
            "deterministic word-prefix queue programs must not alias atomic queue builders"
        );
        let block_offset_queue = AdaptiveTraversalPlanCacheKey::frontier_word_block_offsets_queue(
            7, 32_897, 9, 1_029, 32_897, 0xA11CE,
        );
        assert_eq!(
            block_offset_queue.kind,
            AdaptiveTraversalProgramKind::FrontierWordBlockOffsetsQueue
        );
        assert_eq!(block_offset_queue.queue_capacity, 32_897);
        assert_ne!(
            block_offset_queue, word_prefix,
            "block-offset queue programs must not alias the previous-block-loop scatter"
        );

        let dense = AdaptiveTraversalPlanCacheKey::four_russians_dense(99, 128, 4, 0xA11CE);
        assert_eq!(dense.kind, AdaptiveTraversalProgramKind::FourRussiansDense);
        assert_eq!(dense.edge_count, 0);
        assert_eq!(dense.queue_capacity, 0);
        assert_eq!(
            dense,
            AdaptiveTraversalPlanCacheKey::four_russians_dense(7, 128, 4, 0xA11CE),
            "resident Four-Russians LUT contents must not fragment dense Program caches"
        );
    }
}
