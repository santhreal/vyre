//! Sequential mathematical witnesses for Base64, RLE, and LZ4/Ziftsieve decoding.

use std::fmt;

const INVALID: u32 = 0xFF;

const fn build_standard_decode_table() -> [u32; 256] {
    let mut table = [INVALID; 256];
    let mut b = b'A';
    while b <= b'Z' {
        table[b as usize] = (b - b'A') as u32;
        b += 1;
    }
    let mut b = b'a';
    while b <= b'z' {
        table[b as usize] = (b - b'a' + 26) as u32;
        b += 1;
    }
    let mut b = b'0';
    while b <= b'9' {
        table[b as usize] = (b - b'0' + 52) as u32;
        b += 1;
    }
    table[b'+' as usize] = 62;
    table[b'/' as usize] = 63;
    table[b'=' as usize] = 0;
    table
}

const STANDARD_DECODE_TABLE: [u32; 256] = build_standard_decode_table();

/// Failure returned by the RFC 4648 packed base64 witness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Base64DecodeWitnessError {
    /// Base64 input must contain complete four-byte quads.
    InvalidLength {
        /// Input byte length.
        len: usize,
    },
    /// Decoded fixed-capacity word count overflowed host `usize`.
    CapacityOverflow {
        /// Number of four-byte quads.
        blocks: usize,
    },
    /// Decoded fixed-capacity word count cannot fit the public `u32` ABI.
    DecodedLengthOverflow {
        /// Decoded capacity in `u32` slots.
        decoded_words: usize,
    },
    /// Host witness output reservation failed.
    Allocation {
        /// Requested `u32` slots.
        requested: usize,
        /// Allocator detail.
        source: String,
    },
}

impl fmt::Display for Base64DecodeWitnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { len } => write!(
                formatter,
                "base64 witness input length {len} is not a multiple of 4. Fix: pad with '=' or reject the payload before decode."
            ),
            Self::CapacityOverflow { blocks } => write!(
                formatter,
                "base64 witness decoded capacity overflowed for {blocks} input quads. Fix: shard the payload before parity decode."
            ),
            Self::DecodedLengthOverflow { decoded_words } => write!(
                formatter,
                "base64 witness decoded capacity {decoded_words} cannot fit u32. Fix: shard the payload before dispatch."
            ),
            Self::Allocation { requested, source } => write!(
                formatter,
                "base64 witness could not reserve {requested} decoded u32 slots: {source}. Fix: shard the payload before parity decode."
            ),
        }
    }
}

impl std::error::Error for Base64DecodeWitnessError {}

/// Decode RFC 4648 input into the fixed-capacity packed GPU ABI witness.
///
/// # Panics
///
/// Panics if `input` is not valid RFC 4648 base64 or if decoded capacity overflows.
#[must_use]
pub fn base64_decode_packed_witness(input: &[u8]) -> (Vec<u32>, u32) {
    try_base64_decode_packed_witness(input)
        .unwrap_or_else(|error| panic!("base64 decode witness failed: {error}"))
}

/// Decode RFC 4648 input into caller-owned fixed-capacity packed storage.
///
/// # Panics
///
/// Panics if `input` is not valid RFC 4648 base64 or if decoded capacity overflows.
pub fn base64_decode_packed_witness_into(input: &[u8], out: &mut Vec<u32>) -> u32 {
    try_base64_decode_packed_witness_into(input, out)
        .unwrap_or_else(|error| panic!("base64 decode witness failed: {error}"))
}

/// Fallible RFC 4648 packed base64 witness.
pub fn try_base64_decode_packed_witness(
    input: &[u8],
) -> Result<(Vec<u32>, u32), Base64DecodeWitnessError> {
    let mut out = Vec::new();
    let decoded_len = try_base64_decode_packed_witness_into(input, &mut out)?;
    Ok((out, decoded_len))
}

/// Fallible RFC 4648 packed base64 witness into caller-owned storage.
///
/// Validation and reservation failures leave `out` unchanged.
pub fn try_base64_decode_packed_witness_into(
    input: &[u8],
    out: &mut Vec<u32>,
) -> Result<u32, Base64DecodeWitnessError> {
    if input.len() % 4 != 0 {
        return Err(Base64DecodeWitnessError::InvalidLength { len: input.len() });
    }
    let blocks = input.len() / 4;
    let decoded_words = blocks
        .checked_mul(3)
        .ok_or(Base64DecodeWitnessError::CapacityOverflow { blocks })?;
    let decoded_len_abi = u32::try_from(decoded_words)
        .map_err(|_| Base64DecodeWitnessError::DecodedLengthOverflow { decoded_words })?;
    vyre_foundation::allocation::reserve_exact_cleared(out, decoded_words).map_err(|source| {
        Base64DecodeWitnessError::Allocation {
            requested: decoded_words,
            source: source.to_string(),
        }
    })?;
    out.resize(decoded_words, 0);

    let table = &STANDARD_DECODE_TABLE;
    let invalid = INVALID;
    for block in 0..blocks {
        let base = block * 4;
        let values = [
            table[usize::from(input[base])],
            table[usize::from(input[base + 1])],
            table[usize::from(input[base + 2])],
            table[usize::from(input[base + 3])],
        ]
        .map(|value| if value == invalid { 0 } else { value });
        let out_base = block * 3;
        out[out_base] = (values[0] << 2) | (values[1] >> 4);
        if input[base + 2] != b'=' {
            out[out_base + 1] = ((values[1] & 0x0f) << 4) | (values[2] >> 2);
        }
        if input[base + 3] != b'=' {
            out[out_base + 2] = ((values[2] & 0x03) << 6) | values[3];
        }
    }

    let mut padding = 0u32;
    if input.len() >= 2 {
        if input[input.len() - 1] == b'=' {
            padding = padding.saturating_add(1);
        }
        if input[input.len() - 2] == b'=' {
            padding = padding.saturating_add(1);
        }
    }
    Ok(decoded_len_abi.saturating_sub(padding))
}

/// Decode RFC 4648 input into logical bytes, excluding fixed-capacity padding.
#[must_use]
pub fn base64_decode_bytes_witness(input: &[u8]) -> Vec<u8> {
    let (decoded, decoded_len) = base64_decode_packed_witness(input);
    decoded
        .into_iter()
        .take(decoded_len as usize)
        .map(|word| word as u8)
        .collect()
}

/// Unpack canonical `(length << 8) | value` RLE segments.
pub fn try_rle_segment_lengths_witness_into(
    packed: &[u32],
    lengths: &mut Vec<u32>,
    values: &mut Vec<u32>,
) -> Result<(), String> {
    lengths
        .try_reserve(packed.len().saturating_sub(lengths.len()))
        .map_err(|error| error.to_string())?;
    values
        .try_reserve(packed.len().saturating_sub(values.len()))
        .map_err(|error| error.to_string())?;
    lengths.clear();
    values.clear();
    lengths.extend(packed.iter().map(|segment| segment >> 8));
    values.extend(packed.iter().map(|segment| segment & 0xff));
    Ok(())
}

/// Return unpacked canonical RLE segment lengths and values.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output buffers.
#[must_use]
pub fn rle_segment_lengths_witness(packed: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut lengths = Vec::new();
    let mut values = Vec::new();
    try_rle_segment_lengths_witness_into(packed, &mut lengths, &mut values)
        .unwrap_or_else(|error| panic!("RLE segment witness failed: {error}"));
    (lengths, values)
}

/// Unpack canonical RLE segment lengths and values into caller-supplied vectors.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output buffers.
pub fn rle_segment_lengths_witness_into(
    packed: &[u32],
    lengths: &mut Vec<u32>,
    values: &mut Vec<u32>,
) {
    try_rle_segment_lengths_witness_into(packed, lengths, values)
        .unwrap_or_else(|error| panic!("RLE segment witness failed: {error}"));
}

/// Compute exclusive RLE segment offsets and return the saturated total length.
pub fn try_rle_segment_start_offsets_witness_into(
    lengths: &[u32],
    offsets: &mut Vec<u32>,
) -> Result<u32, String> {
    offsets
        .try_reserve(lengths.len().saturating_sub(offsets.len()))
        .map_err(|error| error.to_string())?;
    offsets.clear();
    let mut total = 0_u32;
    for &length in lengths {
        offsets.push(total);
        total = total.saturating_add(length);
    }
    Ok(total)
}

/// Return exclusive RLE segment offsets and the saturated total length.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output buffers.
#[must_use]
pub fn rle_segment_start_offsets_witness(lengths: &[u32]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::new();
    let total = try_rle_segment_start_offsets_witness_into(lengths, &mut offsets)
        .unwrap_or_else(|error| panic!("RLE offset witness failed: {error}"));
    (offsets, total)
}

/// Compute exclusive RLE segment offsets into caller-supplied vector and return the saturated total length.
///
/// # Panics
///
/// Panics if memory allocation fails when reserving output buffers.
pub fn rle_segment_start_offsets_witness_into(lengths: &[u32], offsets: &mut Vec<u32>) -> u32 {
    try_rle_segment_start_offsets_witness_into(lengths, offsets)
        .unwrap_or_else(|error| panic!("RLE offset witness failed: {error}"))
}

/// Expand canonical packed RLE segments into caller-owned bytes.
pub fn try_rle_decode_witness_into(packed: &[u32], decoded: &mut Vec<u8>) -> Result<(), String> {
    let decoded_len = packed
        .iter()
        .map(|segment| (segment >> 8) as usize)
        .try_fold(0_usize, usize::checked_add)
        .ok_or_else(|| "RLE decoded length overflowed usize".to_string())?;
    decoded
        .try_reserve(decoded_len.saturating_sub(decoded.len()))
        .map_err(|error| error.to_string())?;
    decoded.clear();
    for &segment in packed {
        decoded.extend(std::iter::repeat_n(
            (segment & 0xff) as u8,
            (segment >> 8) as usize,
        ));
    }
    Ok(())
}

/// Expand canonical packed RLE segments into caller-owned bytes.
///
/// # Panics
///
/// Panics if decoded length overflows `usize` or if memory allocation fails.
pub fn rle_decode_witness_into(packed: &[u32], decoded: &mut Vec<u8>) {
    try_rle_decode_witness_into(packed, decoded)
        .unwrap_or_else(|error| panic!("RLE decode witness failed: {error}"));
}

/// Expand canonical packed RLE segments.
#[must_use]
pub fn rle_decode_witness(packed: &[u32]) -> Vec<u8> {
    let mut decoded = Vec::new();
    rle_decode_witness_into(packed, &mut decoded);
    decoded
}

/// Extracted literals from an LZ4-style ziftsieve block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ZiftsieveLiteralWitness {
    /// Literal bytes retained up to the caller's output limit.
    pub literals: Vec<u8>,
    /// Total literal byte count encoded by the block.
    pub decoded_len: usize,
}

impl ZiftsieveLiteralWitness {
    /// Whether the caller's output limit truncated encoded literals.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.literals.len() < self.decoded_len
    }
}

/// Parse LZ4 sequence headers and extract literal bytes independently of the IR.
pub fn ziftsieve_extract_literals_witness(
    input: &[u8],
    max_output: usize,
) -> Result<ZiftsieveLiteralWitness, String> {
    const MAX_SEQUENCES_PER_BLOCK: usize = 100_000;
    const MAX_BLOCK_SIZE: usize = 4 * 1024 * 1024;
    let mut cursor = 0_usize;
    let mut decoded_len = 0_usize;
    let mut sequence_count = 0_usize;
    let mut literals = Vec::with_capacity(max_output.min(input.len()));
    while cursor < input.len() {
        sequence_count += 1;
        if sequence_count > MAX_SEQUENCES_PER_BLOCK {
            return Err(format!(
                "too many LZ4 sequences (max {MAX_SEQUENCES_PER_BLOCK})"
            ));
        }
        let token = input[cursor];
        cursor += 1;
        let mut literal_len = usize::from(token >> 4);
        if literal_len == 15 {
            loop {
                let extension = *input.get(cursor).ok_or_else(|| {
                    "truncated length encoding for LZ4 literal. Fix: provide the complete \
                     extended literal length."
                        .to_owned()
                })?;
                literal_len = literal_len
                    .checked_add(usize::from(extension))
                    .ok_or_else(|| "literal length overflow".to_owned())?;
                if literal_len > MAX_BLOCK_SIZE {
                    return Err(format!(
                        "literal length {literal_len} exceeds MAX_BLOCK_SIZE"
                    ));
                }
                if extension != u8::MAX {
                    break;
                }
            }
        }
        let literal_end = cursor
            .checked_add(literal_len)
            .ok_or_else(|| "literal range overflow".to_owned())?;
        let sequence_literals = input
            .get(cursor..literal_end)
            .ok_or_else(|| "truncated literal payload".to_owned())?;
        decoded_len = decoded_len
            .checked_add(literal_len)
            .ok_or_else(|| "decoded literal length overflow".to_owned())?;
        let remaining = max_output.saturating_sub(literals.len());
        literals.extend_from_slice(&sequence_literals[..sequence_literals.len().min(remaining)]);
        cursor = literal_end;
        if cursor == input.len() {
            break;
        }
        let offset_end = cursor
            .checked_add(2)
            .ok_or_else(|| "match offset range overflow".to_owned())?;
        if offset_end > input.len() {
            return Err("truncated match offset".to_owned());
        }
        cursor = offset_end;
    }
    Ok(ZiftsieveLiteralWitness {
        literals,
        decoded_len,
    })
}

/// Decode ASCII hex pairs into one byte value per `u32` output slot.
#[must_use]
pub fn hex_decode_packed_witness(input: &[u8]) -> Vec<u32> {
    fn nibble(byte: u8) -> u32 {
        match byte {
            b'0'..=b'9' => u32::from(byte - b'0'),
            b'A'..=b'F' => u32::from(byte - b'A' + 10),
            b'a'..=b'f' => u32::from(byte - b'a' + 10),
            _ => 0,
        }
    }
    input
        .chunks_exact(2)
        .map(|pair| (nibble(pair[0]) << 4) | nibble(pair[1]))
        .collect()
}

/// Decoded payload and byte length of one DEFLATE stored block.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InflateStoredWitness {
    /// One decoded byte per low-order `u32` lane.
    pub data: Vec<u32>,
    /// Number of decoded bytes.
    pub inflated_len: u32,
}

/// Decode one byte-per-word DEFLATE stored block.
pub fn inflate_stored_witness(input: &[u32]) -> Result<InflateStoredWitness, String> {
    if input.len() < 5 {
        return Err("stored block header requires five words".to_owned());
    }
    if input[0] >> 1 & 0x3 != 0 {
        return Err("DEFLATE block is not stored".to_owned());
    }
    let length = (input[1] & 0xFF) | ((input[2] & 0xFF) << 8);
    let complement = (input[3] & 0xFF) | ((input[4] & 0xFF) << 8);
    if complement != (!length & 0xFFFF) {
        return Err("stored block LEN/NLEN mismatch".to_owned());
    }
    let end = 5_usize
        .checked_add(length as usize)
        .ok_or_else(|| "stored block length overflow".to_owned())?;
    let payload = input
        .get(5..end)
        .ok_or_else(|| "truncated stored block payload".to_owned())?;
    Ok(InflateStoredWitness {
        data: payload.iter().map(|word| word & 0xFF).collect(),
        inflated_len: length,
    })
}

/// Sequential mathematical witness for VSA hypervector fingerprinting by 3-way XOR binding.
#[must_use]
pub fn vsa_fingerprint_witness(
    kind_hv: &[u32],
    signature_hv: &[u32],
    region_hv: &[u32],
) -> Vec<u32> {
    let len = kind_hv.len().min(signature_hv.len()).min(region_hv.len());
    (0..len)
        .map(|i| kind_hv[i] ^ signature_hv[i] ^ region_hv[i])
        .collect()
}
