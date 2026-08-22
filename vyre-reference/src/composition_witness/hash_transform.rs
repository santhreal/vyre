//! Sequential mathematical witnesses for sparse FFT bin hashing, count-sketches, hypervectors, and NTT.

/// Sequential sparse FFT bin-hash witness.
///
/// # Panics
///
/// Panics if `b == 0`, if `b` or signal length exceed host bounds, or if allocation fails.
#[must_use]
pub fn sparse_fft_bin_hash_witness(signal: &[u32], a: u32, c: u32, b: u32) -> Vec<u32> {
    let mut bins = Vec::new();
    match try_sparse_fft_bin_hash_into_witness(signal, a, c, b, &mut bins) {
        Ok(()) => bins,
        Err(error) => panic!("sparse FFT bin-hash witness failed: {error}"),
    }
}

/// Sequential sparse FFT bin-hash witness into caller-owned storage.
///
/// # Panics
///
/// Panics if `b == 0`, if `b` or signal length exceed host bounds, or if allocation fails.
pub fn sparse_fft_bin_hash_into_witness(
    signal: &[u32],
    a: u32,
    c: u32,
    b: u32,
    bins: &mut Vec<u32>,
) {
    if let Err(error) = try_sparse_fft_bin_hash_into_witness(signal, a, c, b, bins) {
        panic!("sparse FFT bin-hash witness failed: {error}");
    }
}

/// Fallible sequential sparse FFT bin-hash witness into caller-owned storage.
pub fn try_sparse_fft_bin_hash_into_witness(
    signal: &[u32],
    a: u32,
    c: u32,
    b: u32,
    bins: &mut Vec<u32>,
) -> Result<(), String> {
    if b == 0 {
        return Err("sparse FFT bin-hash witness requires b > 0.".to_string());
    }
    let b_len = usize::try_from(b)
        .map_err(|_| format!("sparse FFT bin count {b} does not fit host usize."))?;
    vyre_foundation::allocation::reserve_exact_cleared(bins, b_len).map_err(|err| {
        format!("sparse FFT bin-hash witness could not reserve {b_len} bins: {err}")
    })?;
    bins.resize(b_len, 0);
    for (f, &v) in signal.iter().enumerate() {
        let f = u32::try_from(f)
            .map_err(|_| "sparse FFT signal length exceeds u32 frequency ABI.".to_string())?;
        let bin = a.wrapping_mul(f).wrapping_add(c) % b;
        let bin = usize::try_from(bin)
            .map_err(|_| "sparse FFT bin index does not fit host usize.".to_string())?;
        bins[bin] = bins[bin].wrapping_add(v);
    }
    Ok(())
}

/// Sequential sparse FFT voting recovery witness: given `m` binnings under different (a, c) pairs,
/// find the indices most consistently mapped to the same bin.
///
/// # Panics
///
/// Panics if `b == 0`, if `b` or `n` exceed host bounds, or if binning data is malformed.
#[must_use]
pub fn sparse_fft_voting_recovery_witness(
    binnings: &[(u32, u32, Vec<u32>)],
    threshold: u32,
    n: u32,
    b: u32,
) -> Vec<u32> {
    let mut votes = Vec::new();
    let mut out = Vec::new();
    match try_sparse_fft_voting_recovery_into_witness(
        binnings, threshold, n, b, &mut votes, &mut out,
    ) {
        Ok(()) => out,
        Err(error) => panic!("sparse FFT voting recovery witness failed: {error}"),
    }
}

/// Sequential sparse FFT voting recovery witness using caller-owned vote scratch and output storage.
///
/// # Panics
///
/// Panics if `b == 0`, if `b` or `n` exceed host bounds, or if binning data is malformed.
pub fn sparse_fft_voting_recovery_into_witness(
    binnings: &[(u32, u32, Vec<u32>)],
    threshold: u32,
    n: u32,
    b: u32,
    votes: &mut Vec<u32>,
    out: &mut Vec<u32>,
) {
    if let Err(error) =
        try_sparse_fft_voting_recovery_into_witness(binnings, threshold, n, b, votes, out)
    {
        panic!("sparse FFT voting recovery witness failed: {error}");
    }
}

/// Fallible sequential sparse FFT voting recovery witness using caller-owned vote scratch and output storage.
pub fn try_sparse_fft_voting_recovery_into_witness(
    binnings: &[(u32, u32, Vec<u32>)],
    threshold: u32,
    n: u32,
    b: u32,
    votes: &mut Vec<u32>,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    if b == 0 {
        return Err("sparse FFT voting recovery requires b > 0.".to_string());
    }
    let n_len = usize::try_from(n)
        .map_err(|_| format!("sparse FFT signal length {n} does not fit host usize."))?;
    let b_len = usize::try_from(b)
        .map_err(|_| format!("sparse FFT bin count {b} does not fit host usize."))?;
    for (idx, (_, _, bins)) in binnings.iter().enumerate() {
        if bins.len() < b_len {
            return Err(format!(
                "sparse FFT voting binning {idx} has {} bins, expected at least {b_len}.",
                bins.len()
            ));
        }
    }
    vyre_foundation::allocation::reserve_exact_cleared(votes, n_len).map_err(|err| {
        format!("sparse FFT voting recovery could not reserve {n_len} vote slots: {err}")
    })?;
    votes.resize(n_len, 0);
    for (a, c, bins) in binnings {
        for (f, vote) in votes.iter_mut().enumerate() {
            let f = u32::try_from(f)
                .map_err(|_| "sparse FFT frequency index exceeds u32 ABI.".to_string())?;
            let bin = a.wrapping_mul(f).wrapping_add(*c) % b;
            let bin = usize::try_from(bin)
                .map_err(|_| "sparse FFT voting bin index does not fit host usize.".to_string())?;
            if bins[bin] > 0 {
                *vote = vote.wrapping_add(1);
            }
        }
    }
    vyre_foundation::allocation::reserve_exact_cleared(out, n_len).map_err(|err| {
        format!("sparse FFT voting recovery could not reserve {n_len} output slots: {err}")
    })?;
    for (f, &vote) in votes.iter().enumerate() {
        if vote >= threshold {
            let f = u32::try_from(f)
                .map_err(|_| "sparse FFT recovered index exceeds u32 ABI.".to_string())?;
            out.push(f);
        }
    }
    Ok(())
}

/// Count-sketch table length calculation.
pub fn count_sketch_table_len(d: u32, w: u32) -> Result<usize, String> {
    if d == 0 || w == 0 {
        return Err(format!(
            "count-sketch dimensions must be non-zero, got d={d}, w={w}"
        ));
    }
    let cells = d
        .checked_mul(w)
        .ok_or_else(|| format!("count sketch table size overflowed for d={d}, w={w}"))?;
    usize::try_from(cells)
        .map_err(|_| format!("count sketch table size overflowed for d={d}, w={w}"))
}

/// Apply `(hashes, signs)` for one item to a (d × w) count-sketch.
pub fn count_sketch_update_witness(
    table: &mut [u32],
    hashes: &[u32],
    signs: &[i32],
    d: u32,
    w: u32,
) {
    let Ok(expected_len) = count_sketch_table_len(d, w) else {
        return;
    };
    let Ok(d_len) = usize::try_from(d) else {
        return;
    };
    let Ok(w_len) = usize::try_from(w) else {
        return;
    };
    if table.len() != expected_len || hashes.len() < d_len || signs.len() < d_len {
        return;
    }
    for r in 0..d_len {
        let Ok(col) = usize::try_from(hashes[r]) else {
            continue;
        };
        if col >= w_len {
            continue;
        }
        let addr = r * w_len + col;
        let cell = table[addr] as i32;
        table[addr] = cell.wrapping_add(signs[r]) as u32;
    }
}

/// Estimate item frequency from count-sketch.
#[must_use]
pub fn count_sketch_query_witness(
    table: &[u32],
    hashes: &[u32],
    signs: &[i32],
    d: u32,
    w: u32,
) -> i32 {
    let mut estimates = Vec::new();
    count_sketch_query_into_witness(table, hashes, signs, d, w, &mut estimates)
}

/// Caller-owned variant of count_sketch_query.
pub fn count_sketch_query_into_witness(
    table: &[u32],
    hashes: &[u32],
    signs: &[i32],
    d: u32,
    w: u32,
    estimates: &mut Vec<i32>,
) -> i32 {
    try_count_sketch_query_into_witness(table, hashes, signs, d, w, estimates).unwrap_or(0)
}

/// Fallible caller-owned variant of count_sketch_query.
pub fn try_count_sketch_query_into_witness(
    table: &[u32],
    hashes: &[u32],
    signs: &[i32],
    d: u32,
    w: u32,
    estimates: &mut Vec<i32>,
) -> Result<i32, String> {
    let table_len = count_sketch_table_len(d, w)?;
    if table.len() != table_len {
        return Err(format!(
            "bad table shape: expected {table_len}, actual {}",
            table.len()
        ));
    }
    let d_len = usize::try_from(d).map_err(|_| format!("d={d} exceeds host usize"))?;
    let w_len = usize::try_from(w).map_err(|_| format!("w={w} exceeds host usize"))?;
    if hashes.len() < d_len || signs.len() < d_len {
        return Err(format!(
            "bad query shape: d={d_len}, hashes={}, signs={}",
            hashes.len(),
            signs.len()
        ));
    }
    for (row, &col) in hashes.iter().take(d_len).enumerate() {
        if col >= w {
            return Err(format!("hash out of range at row {row}: col={col}, w={w}"));
        }
    }
    vyre_foundation::allocation::reserve_exact_cleared(estimates, d_len)
        .map_err(|err| format!("allocation failed: {err}"))?;
    for r in 0..d_len {
        let col =
            usize::try_from(hashes[r]).map_err(|_| format!("hash out of range at row {r}"))?;
        let cell = table[r * w_len + col] as i32;
        estimates.push(cell.wrapping_mul(signs[r]));
    }
    estimates.sort_unstable();
    Ok(estimates[estimates.len() / 2])
}

/// Sequential hypervector XOR bind witness.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output storage.
#[must_use]
pub fn hypervector_xor_bind_witness(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    match try_hypervector_xor_bind_into_witness(a, b, &mut out) {
        Ok(()) => out,
        Err(error) => panic!("hypervector XOR bind witness failed: {error}"),
    }
}

/// Sequential hypervector XOR bind witness using caller-owned buffer.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output storage.
pub fn hypervector_xor_bind_into_witness(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    if let Err(error) = try_hypervector_xor_bind_into_witness(a, b, out) {
        panic!("hypervector XOR bind witness failed: {error}");
    }
}

/// Fallible sequential hypervector XOR bind witness using caller-owned buffer.
pub fn try_hypervector_xor_bind_into_witness(
    a: &[u32],
    b: &[u32],
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let dim_words = a.len().min(b.len());
    vyre_foundation::allocation::reserve_exact_cleared(out, dim_words).map_err(|err| {
        format!("hypervector XOR bind could not reserve {dim_words} output words: {err}")
    })?;
    out.extend(a.iter().zip(b.iter()).take(dim_words).map(|(&x, &y)| x ^ y));
    Ok(())
}

/// Sequential hypervector majority bundle witness.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output storage.
#[must_use]
pub fn hypervector_majority_bundle_witness(hvs: &[Vec<u32>]) -> Vec<u32> {
    let mut out = Vec::new();
    match try_hypervector_majority_bundle_into_witness(hvs, &mut out) {
        Ok(()) => out,
        Err(error) => panic!("hypervector majority bundle witness failed: {error}"),
    }
}

/// Sequential hypervector majority bundle witness using caller-owned buffer.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output storage.
pub fn hypervector_majority_bundle_into_witness(hvs: &[Vec<u32>], out: &mut Vec<u32>) {
    if let Err(error) = try_hypervector_majority_bundle_into_witness(hvs, out) {
        panic!("hypervector majority bundle witness failed: {error}");
    }
}

/// Fallible sequential hypervector majority bundle witness using caller-owned buffer.
pub fn try_hypervector_majority_bundle_into_witness(
    hvs: &[Vec<u32>],
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let Some(dim_words) = hvs.iter().map(Vec::len).min() else {
        out.clear();
        return Ok(());
    };
    if dim_words == 0 {
        out.clear();
        return Ok(());
    }
    let k = hvs.len();
    let threshold = k / 2;

    vyre_foundation::allocation::reserve_exact_cleared(out, dim_words).map_err(|err| {
        format!("hypervector majority bundle could not reserve {dim_words} output words: {err}")
    })?;
    out.resize(dim_words, 0);
    for w in 0..dim_words {
        for bit in 0..32 {
            let mut count = 0;
            for hv in hvs {
                count += (hv[w] >> bit) & 1;
            }
            if count as usize > threshold {
                out[w] |= 1 << bit;
            }
        }
    }
    Ok(())
}

/// Cosine-style similarity over BSC hypervectors witness.
#[must_use]
pub fn hamming_similarity_witness(a: &[u32], b: &[u32]) -> f32 {
    let dim_words = a.len().min(b.len());
    if dim_words == 0 {
        return 1.0;
    }
    let dim_bits = (dim_words * 32) as f32;
    let hamming: u32 = a
        .iter()
        .zip(b.iter())
        .take(dim_words)
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum();
    1.0 - 2.0 * (hamming as f32) / dim_bits
}

/// Sequential NTT modular addition witness.
#[inline]
#[must_use]
pub fn ntt_mod_add_witness(a: u32, b: u32) -> u32 {
    let s = (a as u64) + (b as u64);
    (if s >= NTT_PRIME_P as u64 {
        s - NTT_PRIME_P as u64
    } else {
        s
    }) as u32
}

/// Sequential NTT modular subtraction witness.
#[inline]
#[must_use]
pub fn ntt_mod_sub_witness(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        NTT_PRIME_P - (b - a)
    }
}

/// Sequential NTT modular multiplication witness.
#[inline]
#[must_use]
pub fn ntt_mod_mul_witness(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % NTT_PRIME_P as u64) as u32
}

/// Sequential NTT modular exponentiation witness.
#[must_use]
pub fn ntt_mod_pow_witness(mut base: u32, mut exp: u32) -> u32 {
    let mut result: u32 = 1;
    base %= NTT_PRIME_P;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ntt_mod_mul_witness(result, base);
        }
        exp >>= 1;
        base = ntt_mod_mul_witness(base, base);
    }
    result
}

/// Sequential bit-reversal permutation witness.
pub fn ntt_bit_reverse_witness<T: Copy>(a: &mut [T]) {
    ntt_bit_reverse(a);
}

/// NTT prime p = 998244353.
const NTT_PRIME_P: u32 = 998_244_353;
/// Generator g = 3.
const NTT_GENERATOR_G: u32 = 3;
/// Maximum NTT length (2^23).
const NTT_MAX_LEN: u32 = 1 << 23;

#[inline]
fn ntt_mod_add(a: u32, b: u32) -> u32 {
    let s = (a as u64) + (b as u64);
    (if s >= NTT_PRIME_P as u64 {
        s - NTT_PRIME_P as u64
    } else {
        s
    }) as u32
}

#[inline]
fn ntt_mod_sub(a: u32, b: u32) -> u32 {
    if a >= b {
        a - b
    } else {
        NTT_PRIME_P - (b - a)
    }
}

#[inline]
fn ntt_mod_mul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) % NTT_PRIME_P as u64) as u32
}

fn ntt_mod_pow(mut base: u32, mut exp: u32) -> u32 {
    let mut result: u32 = 1;
    base %= NTT_PRIME_P;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ntt_mod_mul(result, base);
        }
        exp >>= 1;
        base = ntt_mod_mul(base, base);
    }
    result
}

fn ntt_bit_reverse<T: Copy>(a: &mut [T]) {
    let n = a.len();
    let shift = usize::BITS - n.trailing_zeros();
    for i in 0..n {
        let rev = i.reverse_bits() >> shift;
        if i < rev {
            a.swap(i, rev);
        }
    }
}

/// Sequential in-place forward NTT witness.
pub fn ntt_forward_witness(a: &mut [u32]) {
    let n = a.len() as u32;
    assert!(
        n.is_power_of_two() && n <= NTT_MAX_LEN,
        "ntt_forward_witness requires a power-of-two length <= MAX_LEN={NTT_MAX_LEN}; got {n}"
    );

    ntt_bit_reverse(a);

    let mut len = 2u32;
    while len <= n {
        let w_n = ntt_mod_pow(NTT_GENERATOR_G, (NTT_PRIME_P - 1) / len);
        let half = len / 2;
        let mut i = 0;
        while i < n as usize {
            let mut w: u32 = 1;
            for j in 0..half as usize {
                let u = a[i + j];
                let v = ntt_mod_mul(a[i + j + half as usize], w);
                a[i + j] = ntt_mod_add(u, v);
                a[i + j + half as usize] = ntt_mod_sub(u, v);
                w = ntt_mod_mul(w, w_n);
            }
            i += len as usize;
        }
        len <<= 1;
    }
}

/// Sequential in-place inverse NTT witness.
pub fn ntt_inverse_witness(a: &mut [u32]) {
    let n = a.len() as u32;
    assert!(
        n.is_power_of_two() && n <= NTT_MAX_LEN,
        "ntt_inverse_witness requires a power-of-two length <= MAX_LEN={NTT_MAX_LEN}; got {n}"
    );

    ntt_bit_reverse(a);

    let mut len = 2u32;
    while len <= n {
        let w_n_inv = ntt_mod_pow(
            ntt_mod_pow(NTT_GENERATOR_G, (NTT_PRIME_P - 1) / len),
            NTT_PRIME_P - 2,
        );
        let half = len / 2;
        let mut i = 0;
        while i < n as usize {
            let mut w: u32 = 1;
            for j in 0..half as usize {
                let u = a[i + j];
                let v = ntt_mod_mul(a[i + j + half as usize], w);
                a[i + j] = ntt_mod_add(u, v);
                a[i + j + half as usize] = ntt_mod_sub(u, v);
                w = ntt_mod_mul(w, w_n_inv);
            }
            i += len as usize;
        }
        len <<= 1;
    }

    let n_inv = ntt_mod_pow(n, NTT_PRIME_P - 2);
    for x in a.iter_mut() {
        *x = ntt_mod_mul(*x, n_inv);
    }
}
