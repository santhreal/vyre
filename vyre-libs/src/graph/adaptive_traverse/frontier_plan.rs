//! Validated adaptive traversal layouts and frontier statistics.
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
#[cfg(test)]
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
#[cfg(test)]
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
}
