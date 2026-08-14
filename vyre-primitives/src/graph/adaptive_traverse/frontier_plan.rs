//! Validated adaptive traversal layouts and the resident launch plans derived
//! from them: frontier shape, in-domain popcount, queue sizing, and grids.

use super::mode_selection::{select_adaptive_traversal_mode, AdaptiveTraversalMode};
use crate::bitset::{bitset_words, frontier::frontier_tail_mask};

/// Validated adaptive traversal graph layout metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveTraversalLayout {
    /// Number of logical CSR edges.
    pub edge_count: u32,
    /// Largest CSR row degree in the sparse graph.
    pub max_row_degree: u32,
    /// Number of u32 words required by physical edge buffers after padding.
    pub edge_storage_words: usize,
    /// Number of u32 words in one frontier bitset.
    pub words: usize,
    /// Number of u32 words in the dense reverse-adjacency matrix.
    pub dense_words: usize,
}

/// Validated frontier bitset shape for adaptive traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveFrontierLayout {
    /// Number of u32 words in one frontier bitset.
    pub words: usize,
    /// Number of u32 words in one frontier bitset, narrowed for primitive metadata.
    pub words_u32: u32,
}

/// Primitive-owned work classification for a validated adaptive frontier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveFrontierWorkPlan {
    /// Validated frontier layout.
    pub layout: AdaptiveFrontierLayout,
    /// Whether any in-domain frontier bit is active.
    pub has_active_bits: bool,
}

/// In-domain frontier statistics for adaptive traversal planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveFrontierStats {
    /// Validated frontier layout.
    pub layout: AdaptiveFrontierLayout,
    /// Set bits at node ids `< node_count`, excluding padding in the tail word.
    pub popcount: u32,
    /// Packed words with at least one in-domain active bit.
    pub nonzero_words: usize,
}

/// Workgroup lane count used by resident linear adaptive traversal kernels.
pub const ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_LANES: u32 = 256;
/// Workgroup shape for node- and word-linear adaptive traversal kernels.
pub const ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE: [u32; 3] =
    [ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_LANES, 1, 1];
/// Byte length of one resident u32 popcount scalar.
pub const ADAPTIVE_TRAVERSAL_POPCOUNT_BYTES: usize = std::mem::size_of::<u32>();

/// Primitive-owned resident frontier launch and scratch plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveResidentFrontierPlan {
    /// Validated frontier work classification.
    pub work: AdaptiveFrontierWorkPlan,
    /// Number of bytes in one frontier bitset.
    pub frontier_bytes: usize,
    /// Number of bytes in one resident popcount scalar.
    pub popcount_bytes: usize,
    /// Grid for kernels that process frontier words.
    pub frontier_word_grid: [u32; 3],
    /// Grid for kernels that process graph nodes.
    pub node_grid: [u32; 3],
}

/// Primitive-owned resident sparse-queue launch and scratch plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveResidentSparseQueuePlan {
    /// Shared frontier launch and scratch plan.
    pub frontier: AdaptiveResidentFrontierPlan,
    /// Packed frontier words with at least one in-domain active bit.
    pub frontier_nonzero_words: usize,
    /// Active-source queue capacity in u32 node ids.
    pub queue_capacity: u32,
    /// Number of bytes in the resident active-source queue.
    pub queue_bytes: usize,
    /// Grid for kernels that process the active-source queue.
    pub queue_grid: [u32; 3],
}

/// Primitive-owned auto-mode resident traversal plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveResidentAutoStepPlan {
    /// Shared frontier launch and scratch plan.
    pub frontier: AdaptiveResidentFrontierPlan,
    /// Host-visible frontier popcount used only for mode selection.
    pub frontier_popcount: u32,
    /// Selected traversal mode.
    pub mode: AdaptiveTraversalMode,
}

/// Validate CSR plus dense reverse-adjacency rows for adaptive traversal.
///
/// # Errors
///
/// Returns an actionable diagnostic when the layout is empty, malformed,
/// exceeds u32 edge-count indexing, has non-monotonic offsets, contains
/// out-of-range CSR targets, or has the wrong dense matrix length.
pub fn validate_adaptive_traversal_layout(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
) -> Result<AdaptiveTraversalLayout, String> {
    if node_count == 0 {
        return Err("Fix: adaptive traversal requires node_count > 0.".to_string());
    }
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!(
            "Fix: adaptive traversal node_count + 1 overflows usize for node_count={node_count}."
        )
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: adaptive traversal expected {expected_offsets} CSR offsets for {node_count} nodes, got {}.",
            edge_offsets.len()
        ));
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: adaptive traversal target/mask length mismatch: {} targets, {} masks.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    let edge_count = u32::try_from(edge_targets.len()).map_err(|_| {
        format!(
            "Fix: adaptive traversal edge count {} exceeds u32 index space.",
            edge_targets.len()
        )
    })?;
    let final_offset = edge_offsets[expected_offsets - 1] as usize;
    if final_offset != edge_targets.len() {
        return Err(format!(
            "Fix: adaptive traversal final CSR offset {final_offset} must equal edge_count {}.",
            edge_targets.len()
        ));
    }
    let mut max_row_degree = 0u32;
    for (row, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: adaptive traversal CSR offsets are non-monotonic at row {row}: {} > {}.",
                pair[0], pair[1]
            ));
        }
        max_row_degree = max_row_degree.max(pair[1] - pair[0]);
    }
    for (idx, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: adaptive traversal CSR target[{idx}]={target} is outside node_count {node_count}."
            ));
        }
    }

    let words = bitset_words(node_count) as usize;
    let dense_words = (node_count as usize).checked_mul(words).ok_or_else(|| {
        format!(
            "Fix: adaptive traversal dense adjacency word count overflows usize for {node_count} nodes and {words} words."
        )
    })?;
    if adj_rows_dense.len() != dense_words {
        return Err(format!(
            "Fix: adaptive traversal expected {dense_words} dense adjacency words, got {}.",
            adj_rows_dense.len()
        ));
    }

    Ok(AdaptiveTraversalLayout {
        edge_count,
        max_row_degree,
        edge_storage_words: edge_targets.len().max(1),
        words,
        dense_words,
    })
}

/// Validate a packed frontier bitset for adaptive traversal.
///
/// # Errors
///
/// Returns an actionable diagnostic when `node_count` is zero or the frontier
/// slice length does not match `bitset_words(node_count)`.
pub fn validate_adaptive_frontier(
    node_count: u32,
    frontier_in: &[u32],
) -> Result<AdaptiveFrontierLayout, String> {
    if node_count == 0 {
        return Err("Fix: adaptive traversal frontier requires node_count > 0.".to_string());
    }
    let words_u32 = bitset_words(node_count);
    let words = words_u32 as usize;
    if frontier_in.len() != words {
        return Err(format!(
            "Fix: adaptive traversal frontier expected {words} word(s) for node_count={node_count}, got {}.",
            frontier_in.len()
        ));
    }
    Ok(AdaptiveFrontierLayout { words, words_u32 })
}

/// Validate and classify an adaptive traversal frontier.
///
/// The all-zero frontier is a primitive identity case: every adaptive
/// traversal variant produces an all-zero output and does not need a resident
/// popcount, queue compaction, dense traversal, or readback kernel.
///
/// # Errors
///
/// Returns the same frontier-shape diagnostics as [`validate_adaptive_frontier`].
pub fn plan_adaptive_frontier_work(
    node_count: u32,
    frontier_in: &[u32],
) -> Result<AdaptiveFrontierWorkPlan, String> {
    let stats =
        adaptive_frontier_stats(node_count, frontier_in, "adaptive traversal frontier work")?;
    Ok(AdaptiveFrontierWorkPlan {
        layout: stats.layout,
        has_active_bits: stats.popcount != 0,
    })
}

/// Checked physical-word popcount for an adaptive traversal frontier.
///
/// # Errors
///
/// Returns an actionable diagnostic if the frontier contains more set bits than
/// can be represented by the primitive's u32 resident popcount scalar.
pub fn adaptive_frontier_popcount(frontier_in: &[u32], context: &str) -> Result<u32, String> {
    let mut popcount = 0u32;
    for &word in frontier_in {
        popcount = popcount.checked_add(word.count_ones()).ok_or_else(|| {
            format!(
                "Fix: {context} frontier popcount exceeds u32::MAX for {} frontier words.",
                frontier_in.len()
            )
        })?;
    }
    Ok(popcount)
}

/// Checked in-domain popcount for an adaptive traversal frontier.
///
/// # Errors
///
/// Returns frontier-shape diagnostics or an actionable diagnostic if the
/// in-domain frontier contains more set bits than fit in a u32 scalar.
pub fn adaptive_frontier_popcount_in_domain(
    node_count: u32,
    frontier_in: &[u32],
    context: &str,
) -> Result<u32, String> {
    adaptive_frontier_stats(node_count, frontier_in, context).map(|stats| stats.popcount)
}

/// Validate and count only frontier bits whose node ids are in domain.
///
/// # Errors
///
/// Returns frontier-shape diagnostics or an actionable diagnostic if the
/// in-domain frontier contains more set bits than fit in a u32 scalar.
pub fn adaptive_frontier_stats(
    node_count: u32,
    frontier_in: &[u32],
    context: &str,
) -> Result<AdaptiveFrontierStats, String> {
    let layout = validate_adaptive_frontier(node_count, frontier_in)?;
    let final_word_mask = frontier_tail_mask(node_count);
    let mut popcount = 0u32;
    let mut nonzero_words = 0usize;
    for (index, &word) in frontier_in.iter().enumerate() {
        let in_domain_word = if index + 1 == layout.words {
            word & final_word_mask
        } else {
            word
        };
        if in_domain_word != 0 {
            nonzero_words += 1;
        }
        popcount = popcount
            .checked_add(in_domain_word.count_ones())
            .ok_or_else(|| {
                format!(
                    "Fix: {context} frontier popcount exceeds u32::MAX for {} frontier words.",
                    frontier_in.len()
                )
            })?;
    }
    Ok(AdaptiveFrontierStats {
        layout,
        popcount,
        nonzero_words,
    })
}

/// Validate and plan resident frontier scratch plus launch grids.
///
/// # Errors
///
/// Returns frontier-shape diagnostics or byte-size overflow diagnostics.
pub fn plan_adaptive_resident_frontier_step(
    node_count: u32,
    frontier_in: &[u32],
) -> Result<AdaptiveResidentFrontierPlan, String> {
    let work = plan_adaptive_frontier_work(node_count, frontier_in)?;
    adaptive_resident_frontier_plan_from_work(node_count, work)
}

/// Validate and plan a queue-driven resident traversal step.
///
/// # Errors
///
/// Returns frontier-shape diagnostics or queue/frontier byte-size overflow
/// diagnostics. The active queue is sized from the host-visible frontier
/// popcount and rounded to a power-of-two bucket so sparse frontiers do not pay
/// full-graph queue allocation or launch width.
pub fn plan_adaptive_resident_sparse_queue_step(
    node_count: u32,
    frontier_in: &[u32],
) -> Result<AdaptiveResidentSparseQueuePlan, String> {
    let stats = adaptive_frontier_stats(
        node_count,
        frontier_in,
        "adaptive resident sparse queue step",
    )?;
    let work = AdaptiveFrontierWorkPlan {
        layout: stats.layout,
        has_active_bits: stats.popcount != 0,
    };
    let frontier = adaptive_resident_frontier_plan_from_work(node_count, work)?;
    let queue_capacity = adaptive_sparse_queue_capacity(node_count, stats.popcount);
    let queue_bytes = adaptive_u32_byte_len(
        queue_capacity as usize,
        "adaptive traversal resident active-source queue",
    )?;
    Ok(AdaptiveResidentSparseQueuePlan {
        frontier,
        frontier_nonzero_words: stats.nonzero_words,
        queue_capacity,
        queue_bytes,
        queue_grid: adaptive_linear_grid(queue_capacity),
    })
}

fn adaptive_sparse_queue_capacity(node_count: u32, frontier_popcount: u32) -> u32 {
    let active = frontier_popcount.min(node_count).max(1);
    active
        .checked_next_power_of_two()
        .unwrap_or(u32::MAX)
        .min(node_count.max(1))
}

/// Validate, count, and select resident traversal mode in one primitive-owned plan.
///
/// # Errors
///
/// Returns frontier-shape diagnostics or byte-size overflow diagnostics.
pub fn plan_adaptive_resident_auto_step(
    node_count: u32,
    edge_count: u32,
    frontier_in: &[u32],
    dense_threshold_pct: u32,
) -> Result<AdaptiveResidentAutoStepPlan, String> {
    let stats = adaptive_frontier_stats(node_count, frontier_in, "adaptive resident auto step")?;
    let work = AdaptiveFrontierWorkPlan {
        layout: stats.layout,
        has_active_bits: stats.popcount != 0,
    };
    let frontier = adaptive_resident_frontier_plan_from_work(node_count, work)?;
    let mode =
        select_adaptive_traversal_mode(node_count, edge_count, stats.popcount, dense_threshold_pct);
    Ok(AdaptiveResidentAutoStepPlan {
        frontier,
        frontier_popcount: stats.popcount,
        mode,
    })
}

fn adaptive_resident_frontier_plan_from_work(
    node_count: u32,
    work: AdaptiveFrontierWorkPlan,
) -> Result<AdaptiveResidentFrontierPlan, String> {
    let frontier_bytes =
        adaptive_u32_byte_len(work.layout.words, "adaptive traversal resident frontier")?;
    let frontier_word_grid = adaptive_linear_grid(work.layout.words_u32);
    Ok(AdaptiveResidentFrontierPlan {
        work,
        frontier_bytes,
        popcount_bytes: ADAPTIVE_TRAVERSAL_POPCOUNT_BYTES,
        frontier_word_grid,
        node_grid: adaptive_node_dispatch_grid(node_count),
    })
}

fn adaptive_u32_byte_len(words: usize, context: &str) -> Result<usize, String> {
    words.checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
        format!(
            "Fix: {context} byte length overflows usize for {words} u32 word(s). Shard the graph before resident dispatch."
        )
    })
}

const fn adaptive_linear_grid(items: u32) -> [u32; 3] {
    let groups = items.div_ceil(ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_LANES);
    if groups == 0 {
        [1, 1, 1]
    } else {
        [groups, 1, 1]
    }
}

/// Dispatch grid for adaptive traversal kernels that process one node per lane.
#[must_use]
pub const fn adaptive_node_dispatch_grid(node_count: u32) -> [u32; 3] {
    adaptive_linear_grid(node_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::adaptive_traverse::mode_selection::should_use_dense;
    use crate::graph::adaptive_traverse::test_graphs::build_dense_adj;

    #[test]
    fn adaptive_layout_validation_accepts_valid_csr_and_dense_rows() {
        let layout = validate_adaptive_traversal_layout(
            3,
            &[0, 1, 2, 2],
            &[1, 2],
            &[1, 1],
            &build_dense_adj(&[(0, 1), (1, 2)], 3),
        )
        .unwrap();
        assert_eq!(layout.edge_count, 2);
        assert_eq!(layout.max_row_degree, 1);
        assert_eq!(layout.edge_storage_words, 2);
        assert_eq!(layout.words, 1);
        assert_eq!(layout.dense_words, 3);
    }

    #[test]
    fn adaptive_layout_validation_rejects_malformed_layouts() {
        let dense = build_dense_adj(&[(0, 1)], 2);
        let err =
            validate_adaptive_traversal_layout(2, &[0, 2, 1], &[1], &[1], &dense).unwrap_err();
        assert!(err.contains("final CSR offset") || err.contains("non-monotonic"));

        let err =
            validate_adaptive_traversal_layout(2, &[0, 1, 1], &[2], &[1], &dense).unwrap_err();
        assert!(err.contains("outside node_count"));

        let err = validate_adaptive_traversal_layout(2, &[0, 1, 1], &[1], &[1], &[]).unwrap_err();
        assert!(err.contains("dense adjacency words"));
    }

    #[test]
    fn adaptive_frontier_validation_accepts_canonical_frontier() {
        assert_eq!(
            validate_adaptive_frontier(64, &[1, 0]).unwrap(),
            AdaptiveFrontierLayout {
                words: 2,
                words_u32: 2,
            }
        );
    }

    #[test]
    fn adaptive_frontier_work_plan_classifies_zero_and_nonzero_frontiers() {
        assert_eq!(
            plan_adaptive_frontier_work(64, &[0, 0]).unwrap(),
            AdaptiveFrontierWorkPlan {
                layout: AdaptiveFrontierLayout {
                    words: 2,
                    words_u32: 2,
                },
                has_active_bits: false,
            }
        );

        assert!(
            plan_adaptive_frontier_work(64, &[0, 1])
                .unwrap()
                .has_active_bits
        );
    }

    #[test]
    fn adaptive_frontier_stats_ignore_tail_padding_bits() {
        let stats = adaptive_frontier_stats(35, &[0b101, u32::MAX & !0b111], "tail stats")
            .expect("Fix: tail-padded frontier should be valid");

        assert_eq!(stats.popcount, 2);
        assert_eq!(stats.nonzero_words, 1);
        assert_eq!(
            adaptive_frontier_popcount_in_domain(35, &[0b101, u32::MAX & !0b111], "tail popcount")
                .expect("Fix: tail-padded frontier should count"),
            2
        );
        assert!(
            !plan_adaptive_frontier_work(35, &[0, u32::MAX & !0b111])
                .expect("Fix: tail-only padding frontier should be valid")
                .has_active_bits,
            "tail padding bits beyond node_count must not trigger resident traversal work"
        );
        assert!(
            !should_use_dense(&[0, u32::MAX & !0b111], 35),
            "tail padding bits must not push adaptive mode selection toward dense traversal"
        );
    }

    #[test]
    fn adaptive_frontier_validation_rejects_zero_nodes_and_wrong_width() {
        let err = validate_adaptive_frontier(0, &[]).unwrap_err();
        assert!(err.contains("node_count > 0"));

        let err = validate_adaptive_frontier(64, &[1]).unwrap_err();
        assert!(err.contains("expected 2 word"));
    }

    #[test]
    fn resident_frontier_plan_centralizes_bytes_and_grids() {
        let plan = plan_adaptive_resident_frontier_step(8_193, &[1; 257])
            .expect("Fix: resident frontier plan should accept a correctly shaped frontier");

        assert!(plan.work.has_active_bits);
        assert_eq!(plan.work.layout.words_u32, 257);
        assert_eq!(plan.frontier_bytes, 257 * std::mem::size_of::<u32>());
        assert_eq!(plan.popcount_bytes, std::mem::size_of::<u32>());
        assert_eq!(plan.frontier_word_grid, [2, 1, 1]);
        assert_eq!(plan.node_grid, [33, 1, 1]);
    }

    #[test]
    fn adaptive_node_dispatch_grid_packs_node_lanes_into_blocks() {
        assert_eq!(adaptive_node_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(adaptive_node_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(adaptive_node_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(adaptive_node_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(adaptive_node_dispatch_grid(513), [3, 1, 1]);
    }

    #[test]
    fn generated_adaptive_node_dispatch_grid_covers_all_shapes_to_8192() {
        for node_count in 0..=8_192 {
            let grid = adaptive_node_dispatch_grid(node_count);
            assert_eq!(
                grid[1], 1,
                "Fix: adaptive node grid y dimension drifted at node_count={node_count}"
            );
            assert_eq!(
                grid[2], 1,
                "Fix: adaptive node grid z dimension drifted at node_count={node_count}"
            );
            assert!(
                grid[0] >= 1,
                "Fix: adaptive node grid must keep empty traversal launchable"
            );
            assert!(
                grid[0] * ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_LANES >= node_count.max(1),
                "Fix: adaptive node grid under-covers node_count={node_count}"
            );
            assert!(
                grid[0] == 1
                    || (grid[0] - 1) * ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_LANES
                        < node_count.max(1),
                "Fix: adaptive node grid over-launches an avoidable extra block at node_count={node_count}"
            );
        }
    }

    #[test]
    fn resident_sparse_queue_plan_centralizes_queue_shape() {
        let plan = plan_adaptive_resident_sparse_queue_step(513, &[1; 17])
            .expect("Fix: resident sparse-queue plan should accept a correctly shaped frontier");

        assert_eq!(plan.frontier.work.layout.words, 17);
        assert_eq!(plan.frontier_nonzero_words, 17);
        assert_eq!(plan.queue_capacity, 32);
        assert_eq!(plan.queue_bytes, 32 * std::mem::size_of::<u32>());
        assert_eq!(plan.queue_grid, [1, 1, 1]);
    }

    #[test]
    fn resident_sparse_queue_plan_sizes_queue_from_active_frontier() {
        let node_count = 1_000_000u32;
        let mut frontier = vec![0u32; bitset_words(node_count) as usize];
        frontier[0] = 1;

        let single = plan_adaptive_resident_sparse_queue_step(node_count, &frontier)
            .expect("Fix: resident sparse-queue plan should accept a single active source");

        assert_eq!(single.queue_capacity, 1);
        assert_eq!(single.frontier_nonzero_words, 1);
        assert_eq!(single.queue_bytes, std::mem::size_of::<u32>());
        assert_eq!(single.queue_grid, [1, 1, 1]);

        for node in 1..257u32 {
            frontier[(node / 32) as usize] |= 1 << (node % 32);
        }
        let bucketed = plan_adaptive_resident_sparse_queue_step(node_count, &frontier)
            .expect("Fix: resident sparse-queue plan should accept a sparse active frontier");

        assert_eq!(bucketed.queue_capacity, 512);
        assert_eq!(bucketed.frontier_nonzero_words, 9);
        assert_eq!(bucketed.queue_bytes, 512 * std::mem::size_of::<u32>());
        assert_eq!(bucketed.queue_grid, [2, 1, 1]);
    }

    #[test]
    fn generated_sparse_queue_capacity_covers_active_count_without_graph_sized_overlaunch() {
        for seed in 0..10_000u32 {
            let node_count = 1 + (mix32(seed) % 1_000_000);
            let frontier_popcount = mix32(seed ^ 0xA57A_5A7A);
            let active = frontier_popcount.min(node_count);
            let capacity = adaptive_sparse_queue_capacity(node_count, frontier_popcount);

            assert!(capacity >= active.max(1));
            assert!(capacity <= node_count);
            if active <= node_count / 2 && active > 0 {
                assert!(
                    capacity <= active.saturating_mul(2),
                    "Fix: sparse queue capacity should bucket active_count={active} tightly, got {capacity}"
                );
            }
        }
    }

    #[test]
    fn resident_auto_plan_selects_mode_from_primitive_popcount() {
        let mut frontier = vec![0u32; bitset_words(1_000) as usize];
        for node in 0..260u32 {
            frontier[(node / 32) as usize] |= 1 << (node % 32);
        }

        let plan = plan_adaptive_resident_auto_step(1_000, 10_000, &frontier, 25)
            .expect("Fix: resident auto plan should accept a correctly shaped frontier");

        assert_eq!(plan.frontier_popcount, 260);
        assert_eq!(plan.mode, AdaptiveTraversalMode::SparseDense);
        assert!(plan.frontier.work.has_active_bits);
    }

    #[test]
    fn resident_auto_plan_zero_frontier_keeps_sparse_queue_identity_case() {
        let plan = plan_adaptive_resident_auto_step(64, 128, &[0, 0], 25)
            .expect("Fix: zero frontier still has a valid resident auto plan");

        assert_eq!(plan.frontier_popcount, 0);
        assert_eq!(plan.mode, AdaptiveTraversalMode::SparseQueue);
        assert!(!plan.frontier.work.has_active_bits);
    }

    fn mix32(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    }
}
