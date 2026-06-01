//! Shared scratch planning for resident CSR frontier queues.

use vyre_primitives::bitset::frontier::frontier_tail_mask;
use vyre_primitives::graph::csr_frontier_queue::FRONTIER_WORD_SCAN_BLOCK_LANES;
use vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_dispatch_grid;

const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Packed-frontier width where resident sparse CSR switches from node scanning
/// to deterministic word-prefix queue materialization.
pub(crate) const WORD_PREFIX_MIN_FRONTIER_WORDS: usize = 256;

/// Maximum word-prefix scan blocks whose offsets are summed inside the scatter
/// pass instead of paying a separate block-offset scan launch.
pub(crate) const WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS: u32 = 8;

/// CSR rows at or above this degree use the row-strided queue consumer.
pub(crate) const STRIDED_FORWARD_MIN_ROW_DEGREE: u32 = 1024;

/// Queue materializer selected for a resident CSR frontier query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentCsrQueueMaterializer {
    /// Single cooperative workgroup scans source nodes and atomically appends.
    AtomicNodeScan,
    /// Packed words are popcount-scanned, then scattered into queue order.
    DeterministicWordPrefix,
}

/// Queue consumer selected for resident CSR traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentCsrQueueTraverseKind {
    /// One lane consumes an entire queued source row.
    RowSerial,
    /// A fixed lane team consumes each queued source row.
    RowStrided,
}

/// Scratch dimensions for deterministic word-prefix queue materialization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FrontierWordPrefixScratch {
    pub(crate) block_count: u32,
    pub(crate) partial_words: usize,
    pub(crate) block_total_words: usize,
}

pub(crate) fn resident_csr_queue_materializer(
    frontier_words: usize,
) -> ResidentCsrQueueMaterializer {
    if frontier_words >= WORD_PREFIX_MIN_FRONTIER_WORDS {
        ResidentCsrQueueMaterializer::DeterministicWordPrefix
    } else {
        ResidentCsrQueueMaterializer::AtomicNodeScan
    }
}

pub(crate) const fn resident_csr_queue_traverse_kind(
    max_row_degree: u32,
) -> ResidentCsrQueueTraverseKind {
    if max_row_degree >= STRIDED_FORWARD_MIN_ROW_DEGREE {
        ResidentCsrQueueTraverseKind::RowStrided
    } else {
        ResidentCsrQueueTraverseKind::RowSerial
    }
}

pub(crate) const fn resident_csr_queue_traverse_grid(
    queue_capacity: u32,
    kind: ResidentCsrQueueTraverseKind,
) -> [u32; 3] {
    match kind {
        ResidentCsrQueueTraverseKind::RowSerial => {
            let blocks = queue_capacity.div_ceil(256);
            [if blocks == 0 { 1 } else { blocks }, 1, 1]
        }
        ResidentCsrQueueTraverseKind::RowStrided => {
            csr_queue_strided_forward_dispatch_grid(queue_capacity)
        }
    }
}

pub(crate) fn resident_csr_queue_effective_capacity(
    node_count: u32,
    frontiers: &[&[u32]],
    requested_capacity: u32,
) -> Result<u32, String> {
    if node_count == 0 {
        return Err(
            "Fix: resident CSR queue effective capacity requires node_count > 0.".to_string(),
        );
    }
    if frontiers.is_empty() {
        return Err(
            "Fix: resident CSR queue effective capacity requires at least one frontier."
                .to_string(),
        );
    }
    if requested_capacity == 0 {
        return Err(
            "Fix: resident CSR queue effective capacity requires requested_capacity > 0."
                .to_string(),
        );
    }

    let final_word_mask = frontier_tail_mask(node_count);
    let mut max_active = 0u32;
    for (query_index, frontier) in frontiers.iter().enumerate() {
        let active = capped_frontier_popcount(
            node_count,
            final_word_mask,
            frontier,
            requested_capacity,
            query_index,
        )?;
        if active >= requested_capacity {
            return Ok(requested_capacity);
        }
        max_active = max_active.max(active);
    }

    let active_floor = max_active.max(1);
    let bucketed_active = active_floor.checked_next_power_of_two().unwrap_or(u32::MAX);
    Ok(requested_capacity.min(bucketed_active).max(1))
}

fn capped_frontier_popcount(
    node_count: u32,
    final_word_mask: u32,
    frontier: &[u32],
    requested_capacity: u32,
    query_index: usize,
) -> Result<u32, String> {
    let expected_words = vyre_primitives::bitset::bitset_words(node_count) as usize;
    let mut active = 0u32;
    for (word_index, &word) in frontier.iter().take(expected_words).enumerate() {
        let in_domain_word = if word_index + 1 == expected_words {
            word & final_word_mask
        } else {
            word
        };
        let remaining = requested_capacity.saturating_sub(active);
        if remaining == 0 {
            return Ok(requested_capacity);
        }
        let word_active = in_domain_word.count_ones();
        if word_active >= remaining {
            return Ok(requested_capacity);
        }
        active = active.checked_add(word_active).ok_or_else(|| {
            format!(
                "Fix: resident CSR queue query {query_index} frontier popcount overflowed u32 while sizing the active queue."
            )
        })?;
    }
    Ok(active)
}

pub(crate) fn frontier_word_prefix_scratch(
    frontier_words: usize,
) -> Result<FrontierWordPrefixScratch, String> {
    let lanes = FRONTIER_WORD_SCAN_BLOCK_LANES as usize;
    let padded = frontier_words.checked_add(lanes - 1).ok_or_else(|| {
        format!(
            "Fix: resident CSR queue frontier_words={frontier_words} overflows word-prefix block rounding."
        )
    })?;
    let block_total_words = (padded / lanes).max(1);
    let partial_words = block_total_words.checked_mul(lanes).ok_or_else(|| {
        format!(
            "Fix: resident CSR queue word-prefix scratch overflows partial word count for frontier_words={frontier_words}."
        )
    })?;
    let block_count = u32::try_from(block_total_words).map_err(|_| {
        format!(
            "Fix: resident CSR queue word-prefix block count {block_total_words} exceeds u32 launch space."
        )
    })?;
    Ok(FrontierWordPrefixScratch {
        block_count,
        partial_words,
        block_total_words,
    })
}

pub(crate) fn frontier_word_prefix_uses_precomputed_offsets(block_count: u32) -> bool {
    block_count > WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS
}

pub(crate) fn frontier_word_dispatch_grid(frontier_words: usize) -> Result<[u32; 3], String> {
    let words = u32::try_from(frontier_words).map_err(|_| {
        format!(
            "Fix: resident CSR queue frontier word count {frontier_words} exceeds u32 launch space."
        )
    })?;
    Ok([words.div_ceil(256).max(1), 1, 1])
}

pub(crate) fn resident_csr_queue_scratch_bytes_per_query(
    frontier_words: usize,
    queue_capacity: u32,
) -> Result<usize, String> {
    let frontier_bytes = words_to_bytes(frontier_words, "frontier")?;
    let queue_bytes = words_to_bytes(queue_capacity as usize, "active_queue")?;
    let mut bytes = frontier_bytes;
    bytes = checked_add(bytes, queue_bytes, "active_queue")?;
    bytes = checked_add(bytes, U32_BYTES, "queue_len")?;
    bytes = checked_add(bytes, frontier_bytes, "frontier_out")?;
    if resident_csr_queue_materializer(frontier_words)
        == ResidentCsrQueueMaterializer::DeterministicWordPrefix
    {
        let word_prefix = frontier_word_prefix_scratch(frontier_words)?;
        bytes = checked_add(
            bytes,
            words_to_bytes(word_prefix.partial_words, "word_partials")?,
            "word_partials",
        )?;
        bytes = checked_add(
            bytes,
            words_to_bytes(word_prefix.block_total_words, "block_totals")?,
            "block_totals",
        )?;
    }
    Ok(bytes)
}

fn words_to_bytes(words: usize, label: &str) -> Result<usize, String> {
    words.checked_mul(U32_BYTES).ok_or_else(|| {
        format!("Fix: resident CSR queue {label} word count {words} overflows byte count.")
    })
}

fn checked_add(base: usize, extra: usize, label: &str) -> Result<usize, String> {
    base.checked_add(extra).ok_or_else(|| {
        format!("Fix: resident CSR queue scratch byte count overflowed while adding {label}.")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn materializer_switches_at_word_prefix_threshold() {
        assert_eq!(
            resident_csr_queue_materializer(WORD_PREFIX_MIN_FRONTIER_WORDS - 1),
            ResidentCsrQueueMaterializer::AtomicNodeScan
        );
        assert_eq!(
            resident_csr_queue_materializer(WORD_PREFIX_MIN_FRONTIER_WORDS),
            ResidentCsrQueueMaterializer::DeterministicWordPrefix
        );
    }

    #[test]
    fn generated_word_prefix_scratch_covers_threshold_and_block_edges() {
        for words in 0..4096usize {
            let scratch = frontier_word_prefix_scratch(words)
                .expect("Fix: generated word-prefix scratch should fit");
            assert!(scratch.block_count >= 1);
            assert!(scratch.partial_words >= FRONTIER_WORD_SCAN_BLOCK_LANES as usize);
            assert_eq!(
                scratch.partial_words,
                scratch.block_total_words * FRONTIER_WORD_SCAN_BLOCK_LANES as usize
            );
            assert!(
                scratch.partial_words >= words,
                "partial scratch must cover every packed frontier word"
            );
        }
    }

    #[test]
    fn small_word_prefix_block_counts_inline_offsets() {
        for block_count in 1..=WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS {
            assert!(
                !frontier_word_prefix_uses_precomputed_offsets(block_count),
                "block_count={block_count} should use in-scatter offsets"
            );
        }
        assert!(frontier_word_prefix_uses_precomputed_offsets(
            WORD_PREFIX_INLINE_BLOCK_OFFSET_MAX_BLOCKS + 1
        ));
    }

    #[test]
    fn high_degree_rows_select_strided_queue_consumer() {
        assert_eq!(
            resident_csr_queue_traverse_kind(STRIDED_FORWARD_MIN_ROW_DEGREE - 1),
            ResidentCsrQueueTraverseKind::RowSerial
        );
        assert_eq!(
            resident_csr_queue_traverse_kind(STRIDED_FORWARD_MIN_ROW_DEGREE),
            ResidentCsrQueueTraverseKind::RowStrided
        );
        assert_eq!(
            resident_csr_queue_traverse_grid(9, ResidentCsrQueueTraverseKind::RowSerial),
            [1, 1, 1]
        );
        assert_eq!(
            resident_csr_queue_traverse_grid(9, ResidentCsrQueueTraverseKind::RowStrided),
            [2, 1, 1]
        );
    }

    #[test]
    fn effective_queue_capacity_buckets_active_frontiers_and_ignores_tail_bits() {
        let first = [0b11_u32, u32::MAX & !0b111_u32];
        let second = [0_u32, 0b101_u32];
        let frontiers: [&[u32]; 2] = [&first, &second];

        assert_eq!(
            resident_csr_queue_effective_capacity(35, &frontiers, 1_024)
                .expect("Fix: valid resident CSR queue frontiers should size"),
            2,
            "tail bits outside node_count must not inflate resident queue capacity"
        );
        let mut single = vec![0u32; vyre_primitives::bitset::bitset_words(1_000) as usize];
        single[0] = 1;
        assert_eq!(
            resident_csr_queue_effective_capacity(1_000, &[&single], 1_024)
                .expect("Fix: single active source should size"),
            1
        );
        let dense = [u32::MAX; 9];
        assert_eq!(
            resident_csr_queue_effective_capacity(288, &[&dense], 257)
                .expect("Fix: requested capacity remains a hard traversal cap"),
            257
        );
    }

    #[test]
    fn effective_queue_capacity_caps_dense_frontiers_to_requested_capacity() {
        let node_count = 1_000_000_u32;
        let frontier = vec![u32::MAX; vyre_primitives::bitset::bitset_words(node_count) as usize];

        assert_eq!(
            resident_csr_queue_effective_capacity(node_count, &[&frontier], 17)
                .expect("Fix: dense resident CSR frontier should size to requested cap"),
            17
        );

        let mut overpadded = vec![0u32; vyre_primitives::bitset::bitset_words(33) as usize + 128];
        overpadded[0] = 1;
        overpadded[2..].fill(u32::MAX);
        assert_eq!(
            resident_csr_queue_effective_capacity(33, &[&overpadded], 1_024)
                .expect("Fix: resident CSR frontier sizing should ignore out-of-domain padding"),
            1,
            "out-of-domain padding must not inflate resident queue capacity"
        );
    }

    #[test]
    fn capped_frontier_popcount_masks_tail_and_saturates_at_capacity() {
        let node_count = 65_u32;
        let final_word_mask = frontier_tail_mask(node_count);
        let frontier = [u32::MAX, u32::MAX, u32::MAX];

        assert_eq!(
            capped_frontier_popcount(node_count, final_word_mask, &frontier, 40, 0)
                .expect("Fix: capped popcount should stop at requested capacity"),
            40
        );
        assert_eq!(
            capped_frontier_popcount(node_count, final_word_mask, &frontier, 100, 0)
                .expect("Fix: capped popcount should mask bits past node_count"),
            65
        );
    }

    #[test]
    fn generated_effective_queue_capacity_bounds_overlaunch() {
        for seed in 0..10_000u32 {
            let node_count = 1 + (mix32(seed) % 4_096);
            let words = vyre_primitives::bitset::bitset_words(node_count) as usize;
            let mut first = vec![0u32; words];
            let mut second = vec![0u32; words];
            for word_index in 0..words {
                first[word_index] = mix32(seed ^ word_index as u32);
                second[word_index] = mix32(seed.rotate_left(7) ^ word_index as u32);
            }
            let frontiers: [&[u32]; 2] = [&first, &second];
            let requested_capacity = 1 + (mix32(seed ^ 0x7a5a_51ce_u32) % 8_192);
            let effective =
                resident_csr_queue_effective_capacity(node_count, &frontiers, requested_capacity)
                    .expect("Fix: generated resident CSR queue frontiers should size");
            let max_active = frontiers
                .iter()
                .map(|frontier| in_domain_popcount(node_count, frontier))
                .max()
                .unwrap_or(0);

            assert!(effective >= 1);
            assert!(effective <= requested_capacity);
            if max_active == 0 {
                assert_eq!(effective, 1);
            } else if max_active > requested_capacity {
                assert_eq!(effective, requested_capacity);
            } else {
                assert!(effective >= max_active);
                assert!(
                    effective <= max_active.next_power_of_two(),
                    "capacity should only round active_count={max_active} to its bucket, got {effective}"
                );
                if max_active <= requested_capacity / 2 {
                    assert!(
                        effective <= max_active * 2,
                        "uncapped sparse frontier should not overlaunch by more than one bucket: active={max_active} effective={effective}"
                    );
                }
            }
        }
    }

    fn in_domain_popcount(node_count: u32, frontier: &[u32]) -> u32 {
        let final_word_mask = frontier_tail_mask(node_count);
        frontier
            .iter()
            .enumerate()
            .map(|(index, &word)| {
                if index + 1 == frontier.len() {
                    word & final_word_mask
                } else {
                    word
                }
                .count_ones()
            })
            .sum()
    }

    fn mix32(mut value: u32) -> u32 {
        value ^= value >> 16;
        value = value.wrapping_mul(0x7feb_352d);
        value ^= value >> 15;
        value = value.wrapping_mul(0x846c_a68b);
        value ^ (value >> 16)
    }
}
