//! Sequential hash-compression witnesses.

/// Apply one BLAKE3 G mixing function to four selected state lanes.
pub fn blake3_g_witness(
    state: &mut [u32; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    x: u32,
    y: u32,
) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(x);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(y);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// Apply one BLAKE3 compression round under the supplied message permutation.
pub fn blake3_round_witness(state: &mut [u32; 16], message: &[u32; 16], permutation: &[usize; 16]) {
    let mut scheduled = [0_u32; 16];
    for (destination, &source) in permutation.iter().enumerate() {
        scheduled[destination] = message[source];
    }
    blake3_g_witness(state, 0, 4, 8, 12, scheduled[0], scheduled[1]);
    blake3_g_witness(state, 1, 5, 9, 13, scheduled[2], scheduled[3]);
    blake3_g_witness(state, 2, 6, 10, 14, scheduled[4], scheduled[5]);
    blake3_g_witness(state, 3, 7, 11, 15, scheduled[6], scheduled[7]);
    blake3_g_witness(state, 0, 5, 10, 15, scheduled[8], scheduled[9]);
    blake3_g_witness(state, 1, 6, 11, 12, scheduled[10], scheduled[11]);
    blake3_g_witness(state, 2, 7, 8, 13, scheduled[12], scheduled[13]);
    blake3_g_witness(state, 3, 4, 9, 14, scheduled[14], scheduled[15]);
}

/// Sequential reflected IEEE CRC-32 witness.
#[must_use]
pub fn crc32_witness(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let polynomial = 0_u32.wrapping_sub(crc & 1) & 0xEDB8_8320;
            crc = (crc >> 1) ^ polynomial;
        }
    }
    !crc
}

/// Canonical CRC-32 initial value.
pub const CRC32_INIT_WITNESS: u32 = 0xFFFF_FFFF;
/// Reflected IEEE 802.3 polynomial.
pub const CRC32_POLY_WITNESS: u32 = 0xEDB8_8320;

/// Sequential CRC-32 table builder witness.
#[must_use]
pub fn crc32_table_witness() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                (c >> 1) ^ CRC32_POLY_WITNESS
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    table
}

/// Sequential CRC-32 initial state witness.
#[must_use]
pub const fn crc32_initial_state_witness() -> u32 {
    CRC32_INIT_WITNESS
}

/// Sequential CRC-32 single-byte update witness.
#[must_use]
pub fn crc32_update_byte_witness(crc: u32, table: &[u32; 256], byte: u8) -> u32 {
    let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
    (crc >> 8) ^ table[idx]
}

/// Sequential CRC-32 state finalization witness.
#[must_use]
pub const fn crc32_finalize_witness(crc: u32) -> u32 {
    crc ^ CRC32_INIT_WITNESS
}

fn gf2_matrix_times_witness(matrix: &[u32; 32], vector: u32) -> u32 {
    let mut sum = 0u32;
    for (i, &row) in matrix.iter().enumerate() {
        if (vector >> i) & 1 != 0 {
            sum ^= row;
        }
    }
    sum
}

fn gf2_matrix_square_witness(square: &mut [u32; 32], matrix: &[u32; 32]) {
    for index in 0..32 {
        square[index] = gf2_matrix_times_witness(matrix, matrix[index]);
    }
}

/// Sequential CRC-32 combination witness for two finalized CRC-32 values.
#[must_use]
pub fn crc32_combine_witness(left_crc: u32, right_crc: u32, right_len: u64) -> u32 {
    if right_len == 0 {
        return left_crc;
    }

    let mut odd = [0u32; 32];
    let mut even = [0u32; 32];

    odd[0] = CRC32_POLY_WITNESS;
    let mut row = 1u32;
    for slot in odd.iter_mut().skip(1) {
        *slot = row;
        row <<= 1;
    }

    gf2_matrix_square_witness(&mut even, &odd);
    gf2_matrix_square_witness(&mut odd, &even);

    let mut len = right_len;
    let mut crc = left_crc;
    loop {
        gf2_matrix_square_witness(&mut even, &odd);
        if (len & 1) != 0 {
            crc = gf2_matrix_times_witness(&even, crc);
        }
        len >>= 1;
        if len == 0 {
            break;
        }

        gf2_matrix_square_witness(&mut odd, &even);
        if (len & 1) != 0 {
            crc = gf2_matrix_times_witness(&odd, crc);
        }
        len >>= 1;
        if len == 0 {
            break;
        }
    }

    crc ^ right_crc
}

/// Self-contained CRC-32 chunk witness.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crc32ChunkWitness {
    /// Number of bytes represented by this chunk.
    pub len: u64,
    /// CRC-32 of the represented bytes.
    pub crc: u32,
}

/// Combine two adjacent CRC-32 chunk summaries.
#[must_use]
pub fn crc32_combine_chunks_witness(
    left: Crc32ChunkWitness,
    right: Crc32ChunkWitness,
) -> Option<Crc32ChunkWitness> {
    Some(Crc32ChunkWitness {
        len: left.len.checked_add(right.len)?,
        crc: crc32_combine_witness(left.crc, right.crc, right.len),
    })
}

/// Pair-reduce adjacent CRC-32 chunk summaries, preserving an odd tail.
#[must_use]
pub fn crc32_pair_reduce_chunks_witness(
    chunks: &[Crc32ChunkWitness],
) -> Option<Vec<Crc32ChunkWitness>> {
    let mut reduced = Vec::with_capacity(chunks.len().div_ceil(2));
    for pair in chunks.chunks(2) {
        let chunk = match pair {
            [left, right] => crc32_combine_chunks_witness(*left, *right)?,
            [tail] => *tail,
            [] => continue,
            _ => unreachable!(),
        };
        reduced.push(chunk);
    }
    Some(reduced)
}

/// Pack CRC-32 chunk summaries as interleaved `[crc, length]` words.
#[must_use]
pub fn crc32_pack_chunks_witness(chunks: &[Crc32ChunkWitness]) -> Option<Vec<u32>> {
    let mut words = Vec::with_capacity(chunks.len().checked_mul(2)?);
    for chunk in chunks {
        words.push(chunk.crc);
        words.push(u32::try_from(chunk.len).ok()?);
    }
    Some(words)
}

/// Decode interleaved `[crc, length]` words into chunk summaries.
#[must_use]
pub fn crc32_unpack_chunks_witness(words: &[u32]) -> Option<Vec<Crc32ChunkWitness>> {
    let pairs = words.chunks_exact(2);
    if !pairs.remainder().is_empty() {
        return None;
    }
    Some(
        pairs
            .map(|pair| Crc32ChunkWitness {
                crc: pair[0],
                len: u64::from(pair[1]),
            })
            .collect(),
    )
}

/// Pair-reduce CRC-32 summaries encoded as interleaved words.
#[must_use]
pub fn crc32_pair_reduce_chunk_words_witness(words: &[u32]) -> Option<Vec<u32>> {
    let chunks = crc32_unpack_chunks_witness(words)?;
    let reduced = crc32_pair_reduce_chunks_witness(&chunks)?;
    crc32_pack_chunks_witness(&reduced)
}

/// Kind of one CRC-32 map-reduce dispatch step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Crc32MapReduceStepKindWitness {
    /// Produce one CRC summary for each input chunk.
    ChunkSummary,
    /// Combine adjacent CRC summaries.
    PairReduce,
}

/// Shape of one CRC-32 map-reduce dispatch step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Crc32MapReduceStepWitness {
    /// Operation performed by this step.
    pub kind: Crc32MapReduceStepKindWitness,
    /// Logical input item count.
    pub input_items: u32,
    /// Number of summary pairs produced.
    pub output_pairs: u32,
    /// Number of input words consumed.
    pub input_words: u32,
    /// Number of output words produced.
    pub output_words: u32,
    /// Dispatch grid for the step.
    pub grid: [u32; 3],
}

/// Complete CRC-32 map-reduce dispatch plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Crc32MapReducePlanWitness {
    /// Input byte count.
    pub input_len: u32,
    /// Bytes summarized by each map invocation.
    pub chunk_size: std::num::NonZeroU32,
    /// Ordered map and pair-reduction steps.
    pub steps: Vec<Crc32MapReduceStepWitness>,
}

/// Build the checked map-reduce plan for an input length and chunk size.
#[must_use]
pub fn crc32_map_reduce_plan_witness(
    input_len: u32,
    chunk_size: std::num::NonZeroU32,
) -> Option<Crc32MapReducePlanWitness> {
    let c_size = chunk_size.get();
    let chunks = if input_len == 0 {
        1
    } else {
        input_len.div_ceil(c_size)
    };
    let mut steps = vec![Crc32MapReduceStepWitness {
        kind: Crc32MapReduceStepKindWitness::ChunkSummary,
        input_items: input_len,
        output_pairs: chunks,
        input_words: input_len.max(1),
        output_words: chunks.checked_mul(2)?,
        grid: [chunks, 1, 1],
    }];
    let mut curr_pairs = chunks;
    while curr_pairs > 1 {
        let next_pairs = curr_pairs.div_ceil(2);
        steps.push(Crc32MapReduceStepWitness {
            kind: Crc32MapReduceStepKindWitness::PairReduce,
            input_items: curr_pairs,
            output_pairs: next_pairs,
            input_words: curr_pairs.checked_mul(2)?,
            output_words: next_pairs.checked_mul(2)?,
            grid: [next_pairs, 1, 1],
        });
        curr_pairs = next_pairs;
    }
    Some(Crc32MapReducePlanWitness {
        input_len,
        chunk_size,
        steps,
    })
}

/// Initial FNV-1a32 witness state (offset basis).
pub const FNV1A32_OFFSET_WITNESS: u32 = 0x811C_9DC5;
/// FNV-1a32 witness prime.
pub const FNV1A32_PRIME_WITNESS: u32 = 0x0100_0193;

/// Sequential FNV-1a32 initial state witness.
#[must_use]
pub const fn fnv1a32_initial_state_witness() -> u32 {
    FNV1A32_OFFSET_WITNESS
}

/// Sequential FNV-1a32 single-byte update witness.
#[must_use]
pub const fn fnv1a32_update_byte_witness(hash: u32, byte: u8) -> u32 {
    (hash ^ byte as u32).wrapping_mul(FNV1A32_PRIME_WITNESS)
}

/// Sequential FNV-1a32 structural mix witness (mul-xor).
#[must_use]
pub const fn fnv1a32_mul_xor_word_witness(hash: u32, word: u32) -> u32 {
    hash.wrapping_mul(FNV1A32_PRIME_WITNESS) ^ word
}

/// Initial FNV-1a64 witness state (offset basis).
pub const FNV1A64_OFFSET_WITNESS: u64 = 0xCBF2_9CE4_8422_2325;
/// FNV-1a64 witness prime.
pub const FNV1A64_PRIME_WITNESS: u64 = 0x0000_0100_0000_01B3;

/// Sequential FNV-1a64 initial state witness.
#[must_use]
pub const fn fnv1a64_initial_state_witness() -> u64 {
    FNV1A64_OFFSET_WITNESS
}

/// Sequential FNV-1a64 single-byte update witness.
#[must_use]
pub const fn fnv1a64_update_byte_witness(hash: u64, byte: u8) -> u64 {
    (hash ^ byte as u64).wrapping_mul(FNV1A64_PRIME_WITNESS)
}

/// Sequential FNV-1a32 witness.
#[must_use]
pub fn fnv1a32_witness(bytes: &[u8]) -> u32 {
    bytes.iter().fold(0x811C_9DC5_u32, |hash, &byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    })
}

/// Sequential FNV-1a64 witness.
#[must_use]
pub fn fnv1a64_witness(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xCBF2_9CE4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// Canonical Adler-32 prime modulus witness.
pub const ADLER32_MOD_WITNESS: u32 = 65_521;

/// Sequential Adler-32 initial A state witness.
#[must_use]
pub const fn adler32_initial_a_witness() -> u32 {
    1
}

/// Sequential Adler-32 initial B state witness.
#[must_use]
pub const fn adler32_initial_b_witness() -> u32 {
    0
}

/// Sequential Adler-32 single-byte update witness.
#[must_use]
pub const fn adler32_update_byte_witness(a: u32, b: u32, byte: u8) -> (u32, u32) {
    let a = (a + byte as u32) % ADLER32_MOD_WITNESS;
    let b = (b + a) % ADLER32_MOD_WITNESS;
    (a, b)
}

/// Sequential Adler-32 state finalization witness.
#[must_use]
pub const fn adler32_finalize_witness(a: u32, b: u32) -> u32 {
    (b << 16) | a
}

fn adler32_mod_u64_witness(value: u64) -> u32 {
    (value % u64::from(ADLER32_MOD_WITNESS)) as u32
}

/// Self-contained Adler-32 chunk summary witness for tree reductions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Adler32ChunkWitness {
    /// Chunk length modulo [`ADLER32_MOD_WITNESS`].
    pub len_mod: u32,
    /// Adler-32 A state after the chunk from canonical initialization.
    pub a: u32,
    /// Adler-32 B state after the chunk from canonical initialization.
    pub b: u32,
}

/// Summarize a byte slice as an independently-combinable Adler-32 chunk witness.
#[must_use]
pub fn adler32_chunk_witness(bytes: &[u8]) -> Adler32ChunkWitness {
    let mut a = adler32_initial_a_witness();
    let mut b = adler32_initial_b_witness();
    for &byte in bytes {
        let next = adler32_update_byte_witness(a, b, byte);
        a = next.0;
        b = next.1;
    }
    Adler32ChunkWitness {
        len_mod: (bytes.len() % ADLER32_MOD_WITNESS as usize) as u32,
        a,
        b,
    }
}

/// Apply a precomputed chunk summary to an existing Adler-32 state witness.
#[must_use]
pub fn adler32_combine_state_witness(a: u32, b: u32, chunk: Adler32ChunkWitness) -> (u32, u32) {
    let modulus = u64::from(ADLER32_MOD_WITNESS);
    let a_minus_one = (u64::from(a) + modulus - 1) % modulus;
    let combined_a = adler32_mod_u64_witness(u64::from(a) + u64::from(chunk.a) + modulus - 1);
    let combined_b = adler32_mod_u64_witness(
        u64::from(b) + u64::from(chunk.b) + u64::from(chunk.len_mod) * a_minus_one,
    );
    (combined_a, combined_b)
}

/// Combine adjacent Adler-32 chunk summaries without reading source bytes witness.
#[must_use]
pub fn adler32_combine_chunks_witness(
    left: Adler32ChunkWitness,
    right: Adler32ChunkWitness,
) -> Adler32ChunkWitness {
    let (a, b) = adler32_combine_state_witness(left.a, left.b, right);
    Adler32ChunkWitness {
        len_mod: adler32_mod_u64_witness(u64::from(left.len_mod) + u64::from(right.len_mod)),
        a,
        b,
    }
}

/// Sequential Adler-32 witness.
#[must_use]
pub fn adler32_witness(bytes: &[u8]) -> u32 {
    let mut a = adler32_initial_a_witness();
    let mut b = adler32_initial_b_witness();
    for &byte in bytes {
        let next = adler32_update_byte_witness(a, b, byte);
        a = next.0;
        b = next.1;
    }
    adler32_finalize_witness(a, b)
}

/// Sequential CRC-32, FNV-1a32, and Adler-32 witness over one byte stream.
#[must_use]
pub fn multi_hash_witness(bytes: &[u8]) -> (u32, u32, u32) {
    (
        crc32_witness(bytes),
        fnv1a32_witness(bytes),
        adler32_witness(bytes),
    )
}

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
