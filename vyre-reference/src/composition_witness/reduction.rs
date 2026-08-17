//! Sequential mathematical witnesses for prefix scans, reductions, histogramming, and data movement.

/// Sequential mathematical witness for inclusive and exclusive prefix scans.
#[must_use]
pub fn prefix_scan_witness(
    input: &[u32],
    inclusive: bool,
    combine_op: impl Fn(u32, u32) -> u32,
    identity: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = identity;

    for &val in input {
        if inclusive {
            acc = combine_op(acc, val);
            out.push(acc);
        } else {
            out.push(acc);
            acc = combine_op(acc, val);
        }
    }

    out
}

/// Sequential wrapping sum witness for an unsigned reduction.
#[must_use]
pub fn wrapping_sum_witness(values: &[u32]) -> u32 {
    values.iter().copied().fold(0, u32::wrapping_add)
}

/// Sequential boolean-any reduction witness.
#[must_use]
pub fn reduce_any_witness(input: &[u32]) -> u32 {
    u32::from(input.iter().any(|&value| value != 0))
}

/// Sequential boolean-all reduction witness.
#[must_use]
pub fn reduce_all_witness(input: &[u32]) -> u32 {
    u32::from(input.iter().all(|&value| value != 0))
}

/// Sequential minimum reduction witness.
#[must_use]
pub fn reduce_min_witness(input: &[u32]) -> u32 {
    input.iter().copied().min().unwrap_or(u32::MAX)
}

/// Sequential maximum reduction witness.
#[must_use]
pub fn reduce_max_witness(input: &[u32]) -> u32 {
    input.iter().copied().max().unwrap_or(0)
}

/// Sequential total-set-bit reduction witness.
#[must_use]
pub fn reduce_count_witness(input: &[u32]) -> u32 {
    input.iter().fold(0_u32, |count, value| count.wrapping_add(value.count_ones()))
}

/// Sequential nonzero-word reduction witness.
#[must_use]
pub fn reduce_count_non_zero_witness(input: &[u32]) -> u32 {
    input.iter().filter(|&&value| value != 0).count() as u32
}

/// Sequential bitwise-OR workgroup reduction witness.
#[must_use]
pub fn reduce_workgroup_any_witness(input: &[u32]) -> u32 {
    input.iter().copied().fold(0, |combined, value| combined | value)
}

/// Sequential inclusive wrapping prefix sum witness.
#[must_use]
pub fn inclusive_prefix_sum_witness(input: &[u32]) -> Vec<u32> {
    let mut output = Vec::with_capacity(input.len());
    inclusive_prefix_sum_witness_into(input, &mut output);
    output
}

/// Sequential inclusive wrapping prefix sum into caller-owned storage.
pub fn inclusive_prefix_sum_witness_into(input: &[u32], output: &mut Vec<u32>) {
    output.clear();
    let mut sum = 0_u32;
    output.extend(input.iter().map(|&value| {
        sum = sum.wrapping_add(value);
        sum
    }));
}

/// Sequential exclusive wrapping prefix sum witness.
#[must_use]
pub fn exclusive_prefix_sum_witness(input: &[u32]) -> Vec<u32> {
    let mut sum = 0_u32;
    input
        .iter()
        .map(|&value| {
            let carried = sum;
            sum = sum.wrapping_add(value);
            carried
        })
        .collect()
}

/// Sequential gather witness; out-of-bounds indices produce zero.
#[must_use]
pub fn gather_witness(source: &[u32], indices: &[u32]) -> Vec<u32> {
    indices
        .iter()
        .map(|&index| source.get(index as usize).copied().unwrap_or(0))
        .collect()
}

/// Sequential gather witness into caller-owned storage.
pub fn gather_witness_into(source: &[u32], indices: &[u32], output: &mut Vec<u32>) {
    output.clear();
    output.extend(
        indices
            .iter()
            .map(|&index| source.get(index as usize).copied().unwrap_or(0)),
    );
}

/// Sequential last-writer-wins scatter witness.
#[must_use]
pub fn scatter_witness(source: &[u32], indices: &[u32], destination_len: usize) -> Vec<u32> {
    let mut output = Vec::new();
    scatter_witness_into(source, indices, destination_len, &mut output);
    output
}

/// Sequential last-writer-wins scatter witness into caller-owned storage.
pub fn scatter_witness_into(
    source: &[u32],
    indices: &[u32],
    destination_len: usize,
    output: &mut Vec<u32>,
) {
    output.clear();
    output.resize(destination_len, 0);
    for (&value, &index) in source.iter().zip(indices) {
        if let Some(destination) = output.get_mut(index as usize) {
            *destination = value;
        }
    }
}

/// Sequential bounded histogram witness.
#[must_use]
pub fn histogram_witness(input: &[u32], bin_count: u32) -> Vec<u32> {
    let mut output = Vec::new();
    histogram_witness_into(input, bin_count, &mut output);
    output
}

/// Sequential bounded histogram witness into caller-owned storage.
pub fn histogram_witness_into(input: &[u32], bin_count: u32, output: &mut Vec<u32>) {
    output.clear();
    output.resize(bin_count as usize, 0);
    for &value in input {
        if let Some(bin) = output.get_mut(value as usize) {
            *bin = bin.wrapping_add(1);
        }
    }
}

/// Sequential wrapping sum over a clamped half-open range.
#[must_use]
pub fn range_counts_witness(input: &[u32], start: u32, end: u32) -> u32 {
    let start = start as usize;
    let end = (end as usize).min(input.len());
    input
        .get(start..end)
        .unwrap_or_default()
        .iter()
        .copied()
        .fold(0, u32::wrapping_add)
}
