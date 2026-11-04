//! Sequential mathematical witnesses for packed bitset operations.

/// Sequential mathematical witness for bitwise AND on packed bitsets.
#[must_use]
pub fn bitset_and_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len]
        .iter()
        .zip(&rhs[..len])
        .map(|(&a, &b)| a & b)
        .collect()
}

/// Sequential mathematical witness for bitwise AND on packed bitsets writing into caller storage.
pub fn bitset_and_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & b));
}

/// Sequential mathematical witness for bitwise OR on packed bitsets writing into caller storage.
pub fn bitset_or_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a | b));
}

/// Sequential mathematical witness for bitwise XOR on packed bitsets writing into caller storage.
pub fn bitset_xor_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a ^ b));
}

/// Sequential mathematical witness for bitwise NOT on packed bitsets writing into caller storage.
pub fn bitset_not_witness_into(input: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.extend(input.iter().map(|&w| !w));
}

/// Sequential mathematical witness for bitwise AND-NOT on packed bitsets writing into caller storage.
pub fn bitset_and_not_witness_into(lhs: &[u32], rhs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let len = lhs.len().min(rhs.len());
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & !b));
}

/// Sequential mathematical witness for bitset popcount writing into caller storage.
pub fn bitset_popcount_witness_into(input: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.extend(input.iter().map(|&w| w.count_ones()));
}

/// Sequential mathematical witness for bitwise OR on packed bitsets.
#[must_use]
pub fn bitset_or_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len]
        .iter()
        .zip(&rhs[..len])
        .map(|(&a, &b)| a | b)
        .collect()
}

/// Sequential mathematical witness for bitwise XOR on packed bitsets.
#[must_use]
pub fn bitset_xor_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len]
        .iter()
        .zip(&rhs[..len])
        .map(|(&a, &b)| a ^ b)
        .collect()
}

/// Sequential mathematical witness for bitwise NOT on packed bitsets.
#[must_use]
pub fn bitset_not_witness(input: &[u32]) -> Vec<u32> {
    input.iter().map(|&w| !w).collect()
}

/// Sequential mathematical witness for bitwise AND-NOT on packed bitsets (`lhs & !rhs`).
#[must_use]
pub fn bitset_and_not_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len]
        .iter()
        .zip(&rhs[..len])
        .map(|(&a, &b)| a & !b)
        .collect()
}

/// Sequential mathematical witness for bitset equality.
#[must_use]
pub fn bitset_equal_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    lhs == rhs
}

/// Sequential mathematical witness for bitset subset relation (`lhs ⊆ rhs`).
///
/// Missing words in either operand are treated as zero. Extra bits set in `rhs`
/// do not invalidate a subset; any non-zero bit in `lhs` not present in `rhs`
/// (including trailing non-zero words in `lhs` beyond `rhs`) causes the subset
/// relation to return `false`.
#[must_use]
pub fn bitset_subset_of_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    let min_len = lhs.len().min(rhs.len());
    lhs[..min_len]
        .iter()
        .zip(&rhs[..min_len])
        .all(|(&a, &b)| (a & !b) == 0)
        && lhs[min_len..].iter().all(|&a| a == 0)
}

/// Sequential mathematical witness for bitset membership test.
#[must_use]
pub fn bitset_contains_witness(input: &[u32], bit_idx: u32) -> bool {
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word < input.len() {
        (input[word] & (1 << bit)) != 0
    } else {
        false
    }
}

/// Return one addressed bit as the scalar ABI value `0` or `1`.
#[must_use]
pub fn bitset_test_bit_witness(input: &[u32], bit_idx: u32) -> u32 {
    u32::from(bitset_contains_witness(input, bit_idx))
}

/// Sequential mathematical witness for setting a bit in a bitset.
#[must_use]
pub fn bitset_set_bit_witness(target: &[u32], bit_idx: u32) -> Vec<u32> {
    let mut out = target.to_vec();
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word >= out.len() {
        out.resize(word + 1, 0);
    }
    out[word] |= 1 << bit;
    out
}

/// Sequential mathematical witness for clearing a bit in a bitset.
#[must_use]
pub fn bitset_clear_bit_witness(target: &[u32], bit_idx: u32) -> Vec<u32> {
    let mut out = target.to_vec();
    let word = (bit_idx / 32) as usize;
    let bit = bit_idx % 32;
    if word < out.len() {
        out[word] &= !(1 << bit);
    }
    out
}

/// Sequential mathematical witness for bitset popcount (number of set bits per word).
#[must_use]
pub fn bitset_popcount_witness(input: &[u32]) -> Vec<u32> {
    input.iter().map(|&w| w.count_ones()).collect()
}

/// Sequential witness for clearing every packed bitset word.
#[must_use]
pub fn bitset_zero_witness(input: &[u32]) -> Vec<u32> {
    vec![0; input.len()]
}

/// Return `1` when two packed bitsets differ byte-for-byte, otherwise `0`.
#[must_use]
pub fn bitset_difference_flag_witness(current: &[u32], next: &[u32]) -> u32 {
    u32::from(current != next)
}

/// Apply a packed warm-start seed and report change against the pre-seed state.
#[must_use]
pub fn bitset_warm_start_witness(current: &[u32], next: &[u32], seed: &[u32]) -> (Vec<u32>, u32) {
    assert_eq!(
        current.len(),
        next.len(),
        "matching current and next widths"
    );
    assert_eq!(
        current.len(),
        seed.len(),
        "matching current and seed widths"
    );
    let updated = current
        .iter()
        .zip(seed)
        .map(|(&current, &seed)| current | seed)
        .collect();
    (updated, bitset_difference_flag_witness(current, next))
}

/// Apply a 256-by-256 byte lookup table independently to every word byte into caller storage.
pub fn four_russians_binary_witness_into(
    lhs: &[u32],
    rhs: &[u32],
    lut: &[u32],
    out: &mut Vec<u32>,
) {
    let len = lhs.len().min(rhs.len());
    if out.capacity() < len {
        out.reserve(len.saturating_sub(out.len()));
    }
    out.clear();
    out.extend(lhs[..len].iter().zip(&rhs[..len]).map(|(&left, &right)| {
        (0..4).fold(0_u32, |word, byte| {
            let left_byte = left >> (byte * 8) & 0xFF;
            let right_byte = right >> (byte * 8) & 0xFF;
            let index = (left_byte * 256 + right_byte) as usize;
            word | (lut.get(index).copied().unwrap_or(0) & 0xFF) << (byte * 8)
        })
    }));
}

/// Apply a 256-by-256 byte lookup table independently to every word byte.
#[must_use]
pub fn four_russians_binary_witness(lhs: &[u32], rhs: &[u32], lut: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(lhs.len().min(rhs.len()));
    four_russians_binary_witness_into(lhs, rhs, lut, &mut out);
    out
}

/// Sequential dense Boolean matvec over byte-tile Four-Russians lookup tables into caller storage.
///
/// # Panics
///
/// Panics if `frontier` is shorter than required, if `tile_lut` dimensions overflow `usize`,
/// or if `tile_lut` length does not match expected LUT words.
pub fn four_russians_dense_matvec_witness_into(
    frontier: &[u32],
    tile_lut: &[u32],
    tile_count: u32,
    destination_words: u32,
    out: &mut Vec<u32>,
) {
    let expected_frontier_words = tile_count.div_ceil(4) as usize;
    assert!(
        frontier.len() >= expected_frontier_words,
        "complete packed frontier words"
    );
    let expected_lut_words = tile_count
        .checked_mul(256)
        .and_then(|words| words.checked_mul(destination_words))
        .expect("Fix: keep tile_count * 256 * destination_words within u32 bounds to avoid LUT overflow")
        as usize;
    assert_eq!(
        tile_lut.len(),
        expected_lut_words,
        "complete dense Four-Russians LUT"
    );
    let dest_len = destination_words as usize;
    if out.capacity() < dest_len {
        out.reserve(dest_len.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(dest_len, 0_u32);
    for (destination, value) in out.iter_mut().enumerate() {
        for tile in 0..tile_count as usize {
            let byte = frontier[tile / 4] >> ((tile % 4) * 8) & 0xFF;
            let index = ((tile * 256 + byte as usize) * dest_len) + destination;
            *value |= tile_lut[index];
        }
    }
}

/// Sequential dense Boolean matvec over byte-tile Four-Russians lookup tables.
#[must_use]
pub fn four_russians_dense_matvec_witness(
    frontier: &[u32],
    tile_lut: &[u32],
    tile_count: u32,
    destination_words: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(destination_words as usize);
    four_russians_dense_matvec_witness_into(
        frontier,
        tile_lut,
        tile_count,
        destination_words,
        &mut out,
    );
    out
}

/// Encode a probability as a deterministic packed stochastic bitstream.
///
/// # Panics
///
/// Panics if the packed word count overflows or storage cannot be reserved.
#[must_use]
pub fn stochastic_encode_witness(p: f64, len_bits: usize, seed: u32) -> Vec<u32> {
    let mut out = Vec::new();
    try_stochastic_encode_witness_into(p, len_bits, seed, &mut out).expect(
        "Fix: provide a bitstream length within usize bounds and allocate required storage",
    );
    out
}

/// Encode a probability into caller-owned packed stochastic bitstream storage.
///
/// # Errors
///
/// Returns an error when the packed word count overflows or storage cannot be reserved.
pub fn try_stochastic_encode_witness_into(
    p: f64,
    len_bits: usize,
    seed: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let word_count = len_bits
        .checked_add(31)
        .ok_or_else(|| format!("stochastic bitstream length {len_bits} overflows word count"))?
        / 32;
    out.clear();
    out.try_reserve_exact(word_count).map_err(|error| {
        format!("stochastic bitstream witness could not reserve {word_count} words: {error}")
    })?;
    out.resize(word_count, 0);

    let mut state = seed.max(1);
    let threshold = (p.clamp(0.0, 1.0) * f64::from(u32::MAX)) as u32;
    for bit in 0..len_bits {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        if state < threshold {
            out[bit / 32] |= 1 << (bit % 32);
        }
    }
    Ok(())
}

/// Decode a packed stochastic bitstream as a probability estimate.
#[must_use]
pub fn stochastic_decode_witness(bitstream: &[u32], len_bits: usize) -> f64 {
    let set_bits: u32 = bitstream.iter().map(|word| word.count_ones()).sum();
    let bounded = set_bits.min(len_bits as u32);
    f64::from(bounded) / len_bits as f64
}

/// Sequential mathematical witness for bitset copy.
pub fn bitset_copy_witness(target: &mut [u32], source: &[u32]) {
    let n = target.len().min(source.len());
    target[..n].copy_from_slice(&source[..n]);
}

/// In-place bitwise AND on packed bitsets.
pub fn bitset_and_inplace_witness(target: &mut [u32], operand: &[u32]) {
    for (to, from) in target.iter_mut().zip(operand) {
        *to &= *from;
    }
}

/// In-place bitwise OR on packed bitsets.
pub fn bitset_or_inplace_witness(target: &mut [u32], operand: &[u32]) {
    for (to, from) in target.iter_mut().zip(operand) {
        *to |= *from;
    }
}

/// In-place bitwise XOR on packed bitsets.
pub fn bitset_xor_inplace_witness(target: &mut [u32], operand: &[u32]) {
    for (to, from) in target.iter_mut().zip(operand) {
        *to ^= *from;
    }
}

/// In-place bitwise AND-NOT on packed bitsets.
pub fn bitset_and_not_inplace_witness(target: &mut [u32], operand: &[u32]) {
    for (to, from) in target.iter_mut().zip(operand) {
        *to &= !*from;
    }
}

/// In-place set bit on packed bitset.
pub fn bitset_set_bit_inplace_witness(target: &mut [u32], index: u32) {
    if let Some(word) = target.get_mut((index / 32) as usize) {
        *word |= 1_u32 << (index % 32);
    }
}

/// In-place clear bit on packed bitset.
pub fn bitset_clear_bit_inplace_witness(target: &mut [u32], index: u32) {
    if let Some(word) = target.get_mut((index / 32) as usize) {
        *word &= !(1_u32 << (index % 32));
    }
}

/// In-place clear every word in packed bitset.
pub fn bitset_zero_inplace_witness(target: &mut [u32]) {
    target.fill(0);
}

/// Sequential mathematical witness for counting all set bits in a packed frontier with checked overflow.
pub fn frontier_popcount_witness(frontier: &[u32]) -> Result<u32, String> {
    let mut popcount = 0u32;
    for &word in frontier {
        popcount = popcount.checked_add(word.count_ones()).ok_or_else(|| {
            format!(
                "frontier popcount exceeds u32::MAX for {} words",
                frontier.len()
            )
        })?;
    }
    Ok(popcount)
}

/// Sequential mathematical witness for counting in-domain set bits in a packed frontier.
pub fn frontier_domain_popcount_witness(frontier: &[u32], node_count: u32) -> Result<u32, String> {
    let expected_words = (node_count as usize).div_ceil(32);
    if frontier.len() != expected_words {
        return Err(format!(
            "frontier for {node_count} nodes requires {expected_words} u32 words, got {}",
            frontier.len()
        ));
    }
    let tail_bits = node_count % 32;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        (1u32 << tail_bits) - 1
    };
    let final_word_index = expected_words.saturating_sub(1);
    let mut popcount = 0u32;
    for (word_index, &word) in frontier.iter().enumerate() {
        let in_domain_word = if word_index == final_word_index {
            word & tail_mask
        } else {
            word
        };
        popcount = popcount
            .checked_add(in_domain_word.count_ones())
            .ok_or_else(|| {
                format!("frontier popcount exceeds u32::MAX for {expected_words} words")
            })?;
    }
    Ok(popcount)
}

/// Sequential mathematical witness for absorbing new frontier bits into visited set.
pub fn try_frontier_absorb_witness_into(
    visited: &mut [u32],
    neighbors: &[u32],
    node_count: u32,
    next_wave: &mut Vec<u32>,
) -> Result<(bool, u32), String> {
    let expected_words = (node_count as usize).div_ceil(32);
    if visited.len() != expected_words {
        return Err(format!(
            "visited frontier for {node_count} nodes requires {expected_words} u32 words, got {}",
            visited.len()
        ));
    }
    if neighbors.len() != expected_words {
        return Err(format!(
            "neighbors frontier for {node_count} nodes requires {expected_words} u32 words, got {}",
            neighbors.len()
        ));
    }
    let additional = expected_words.saturating_sub(next_wave.len());
    next_wave
        .try_reserve_exact(additional)
        .map_err(|err| format!("failed to reserve next_wave buffer: {err}"))?;
    next_wave.clear();
    next_wave.resize(expected_words, 0);

    let tail_bits = node_count % 32;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        (1u32 << tail_bits) - 1
    };
    let last_word_index = expected_words.saturating_sub(1);
    let mut added_any = false;
    let mut added_popcount = 0u32;

    for (word_index, (visited_word, neighbor_word)) in visited
        .iter_mut()
        .zip(neighbors.iter().copied())
        .enumerate()
    {
        let in_domain_neighbors = if word_index == last_word_index {
            neighbor_word & tail_mask
        } else {
            neighbor_word
        };
        let new_bits = in_domain_neighbors & !*visited_word;
        next_wave[word_index] = new_bits;
        *visited_word |= new_bits;
        added_any |= new_bits != 0;
        added_popcount = added_popcount
            .checked_add(new_bits.count_ones())
            .ok_or_else(|| {
                format!("absorb popcount exceeds u32::MAX for {expected_words} words")
            })?;
    }
    Ok((added_any, added_popcount))
}

/// Sequential mathematical witness for absorbing new frontier bits.
///
/// # Panics
///
/// Panics if buffer lengths do not match expected words for `node_count`,
/// if trailing unused bits are non-zero, or if popcount exceeds `u32::MAX`.
pub fn frontier_absorb_witness(
    visited: &mut [u32],
    neighbors: &[u32],
    node_count: u32,
    next_wave: &mut Vec<u32>,
) -> (bool, u32) {
    try_frontier_absorb_witness_into(visited, neighbors, node_count, next_wave)
        .expect("Fix: pass visited and neighbors buffers sized to ceil(node_count / 32) words")
}

/// Sequential mathematical witness for bitset saturation ratio (set bits / total bits).
#[must_use]
pub fn bitset_saturation_ratio_witness(words: &[u32]) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let total_bits = (words.len() as f64) * 32.0;
    let total_ones: u64 = words.iter().map(|&w| u64::from(w.count_ones())).sum();
    (total_ones as f64) / total_bits
}
