//! Sequential mathematical witnesses for packed bitset operations.

/// Sequential mathematical witness for bitwise AND on packed bitsets.
#[must_use]
pub fn bitset_and_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & b).collect()
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
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a | b).collect()
}

/// Sequential mathematical witness for bitwise XOR on packed bitsets.
#[must_use]
pub fn bitset_xor_witness(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let len = lhs.len().min(rhs.len());
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a ^ b).collect()
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
    lhs[..len].iter().zip(&rhs[..len]).map(|(&a, &b)| a & !b).collect()
}

/// Sequential mathematical witness for bitset equality.
#[must_use]
pub fn bitset_equal_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    lhs == rhs
}

/// Sequential mathematical witness for bitset subset relation (`lhs ⊆ rhs`).
#[must_use]
pub fn bitset_subset_of_witness(lhs: &[u32], rhs: &[u32]) -> bool {
    lhs.iter().zip(rhs.iter()).all(|(&a, &b)| (a & !b) == 0)
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
