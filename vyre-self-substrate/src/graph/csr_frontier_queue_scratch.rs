//! Shared scratch planning for resident CSR frontier queues.

use vyre_primitives::graph::csr_frontier_queue::FRONTIER_WORD_SCAN_BLOCK_LANES;

const U32_BYTES: usize = std::mem::size_of::<u32>();

/// Packed-frontier width where resident sparse CSR switches from node scanning
/// to deterministic word-prefix queue materialization.
pub(crate) const WORD_PREFIX_MIN_FRONTIER_WORDS: usize = 256;

/// Queue materializer selected for a resident CSR frontier query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResidentCsrQueueMaterializer {
    /// Single cooperative workgroup scans source nodes and atomically appends.
    AtomicNodeScan,
    /// Packed words are popcount-scanned, then scattered into queue order.
    DeterministicWordPrefix,
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
}
