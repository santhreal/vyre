//! Base64 decode: one module, one op id.
//!
//! The kernel body, the CPU reference oracle, the lookup table, the public
//! builder and the fused decode-then-scan builder all live here. The op id is
//! `vyre-libs::decode::base64`; the program is one region carrying that
//! generator, so a caller reads one name for one kernel.

use std::error::Error as StdError;
use std::fmt;
use std::sync::OnceLock;
use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::GeneratorRef;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(test)]
use crate::buffer_names::fixed_name;
use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decoded_output_buffer};
use crate::decode::scan::linear_aho_scan_body;
use vyre_primitives::wire::pack_u32_slice as pack_words;

/// Canonical op id for base64 decode.
pub const OP_ID: &str = "vyre-libs::decode::base64";
const FUSED_SCAN_OP_ID: &str = "vyre-libs::decode::base64_then_aho_corasick";
const FAMILY_PREFIX: &str = "decode_base64";

/// Fixed buffer name carrying the base64 decode lookup table.
///
/// The buffer contains 256 `u32` entries; each entry is the six-bit value for
/// the corresponding ASCII byte, or `0xFF` for invalid input.
pub const BASE64_DECODE_TABLE_BUFFER: &str = "__vyre_decode_base64_table";
const DECODED_LEN_BUFFER: &str = "__vyre_decode_base64_decoded_len";

/// Base64 padding byte.
pub const PAD: u32 = b'=' as u32;
/// Invalid table entry sentinel.
pub const INVALID: u32 = 0xFF;
/// Number of words in the standard decode lookup table.
pub const BASE64_DECODE_TABLE_WORDS: u32 = 256;
/// Canonical base64 decode workgroup size.
pub const BASE64_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

static STANDARD_DECODE_TABLE: OnceLock<[u32; 256]> = OnceLock::new();

/// CPU-reference base64 decode failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Base64DecodeReferenceError {
    /// Base64 input must be padded to full 4-byte quads.
    InvalidLength {
        /// Input byte length.
        len: usize,
    },
    /// Decoded fixed-capacity word count overflowed host `usize`.
    CapacityOverflow {
        /// Number of four-byte quads.
        blocks: usize,
    },
    /// Decoded fixed-capacity word count cannot fit the public u32 length ABI.
    DecodedLengthOverflow {
        /// Decoded capacity in u32 slots.
        decoded_words: usize,
    },
    /// Host output staging reservation failed.
    Allocation {
        /// Requested u32 slots.
        requested: usize,
        /// Allocator detail.
        source: String,
    },
}

impl fmt::Display for Base64DecodeReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { len } => write!(
                formatter,
                "base64 reference input length {len} is not a multiple of 4. Fix: pad with '=' or reject the payload before decode."
            ),
            Self::CapacityOverflow { blocks } => write!(
                formatter,
                "base64 reference decoded capacity overflowed for {blocks} input quads. Fix: shard the payload before CPU/GPU parity decode."
            ),
            Self::DecodedLengthOverflow { decoded_words } => write!(
                formatter,
                "base64 reference decoded capacity {decoded_words} cannot fit u32. Fix: shard the payload before dispatch."
            ),
            Self::Allocation { requested, source } => write!(
                formatter,
                "base64 reference could not reserve {requested} decoded u32 slots: {source}. Fix: shard the payload before CPU/GPU parity decode."
            ),
        }
    }
}

impl StdError for Base64DecodeReferenceError {}

fn blocks_for_len(input_len: u32) -> u32 {
    input_len / 4
}

/// Return the standard base64 decode table (RFC 4648) by value.
#[must_use]
pub fn standard_decode_table() -> [u32; 256] {
    *standard_decode_table_ref()
}

/// Process-wide standard base64 decode table (RFC 4648).
///
/// The table is immutable after construction. Dispatch setup and CPU oracles
/// should use this reference when they do not need an owned copy.
#[must_use]
pub fn standard_decode_table_ref() -> &'static [u32; 256] {
    STANDARD_DECODE_TABLE.get_or_init(build_standard_decode_table)
}

fn build_standard_decode_table() -> [u32; 256] {
    let mut table = [INVALID; 256];
    for byte in b'A'..=b'Z' {
        table[usize::from(byte)] = u32::from(byte - b'A');
    }
    for byte in b'a'..=b'z' {
        table[usize::from(byte)] = u32::from(byte - b'a' + 26);
    }
    for byte in b'0'..=b'9' {
        table[usize::from(byte)] = u32::from(byte - b'0' + 52);
    }
    table[usize::from(b'+')] = 62;
    table[usize::from(b'/')] = 63;
    table[usize::from(b'=')] = 0;
    table
}

/// Decoded capacity for a padded base64 input.
#[must_use]
pub fn decoded_capacity(input_len: u32) -> u32 {
    blocks_for_len(input_len) * 3
}

/// CPU oracle for the standard RFC 4648 decode table used by the primitive.
///
/// The output mirrors the GPU contract: one decoded byte per `u32` slot, with
/// padded bytes left as zero in the fixed decoded capacity. Invalid input
/// characters are clamped to zero, matching [`base64_decode_body`].
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn decode_standard_packed_reference(input: &[u8]) -> (Vec<u32>, u32) {
    match try_decode_standard_packed_reference(input) {
        Ok(decoded) => decoded,
        // A decode oracle that returns empty on failure makes the GPU-vs-CPU
        // assertion pass on empty==empty, silently masking a divergence
        // (Law 10 / Law 6). Fail loud; callers use the try_ variant.
        Err(error) => panic!("vyre-primitives base64 decode reference failed: {error}"),
    }
}

/// CPU oracle for the standard RFC 4648 decode table into caller-owned storage.
///
/// Returns the decoded logical byte length while `out` holds the fixed-capacity
/// GPU ABI representation: one decoded byte per `u32` slot, including zeroed
/// padding slots.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn decode_standard_packed_reference_into(input: &[u8], out: &mut Vec<u32>) -> u32 {
    match try_decode_standard_packed_reference_into(input, out) {
        Ok(decoded_len) => decoded_len,
        // Clearing and returning 0 on failure silently masks a parity
        // divergence (Law 10 / Law 6). Fail loud; callers use the try_ variant.
        Err(error) => panic!("vyre-primitives base64 decode reference failed: {error}"),
    }
}

/// Fallible CPU oracle for the standard RFC 4648 decode table.
///
/// This variant is suitable for fuzzing and hostile-input parity tests because
/// malformed lengths and output staging failures are reported as typed errors
/// instead of panics.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_decode_standard_packed_reference(
    input: &[u8],
) -> Result<(Vec<u32>, u32), Base64DecodeReferenceError> {
    let mut out = Vec::new();
    let decoded_len = try_decode_standard_packed_reference_into(input, &mut out)?;
    Ok((out, decoded_len))
}

/// Fallible CPU oracle for the standard RFC 4648 decode table into caller-owned storage.
///
/// On validation or reservation failure, the caller-owned output buffer is left
/// unchanged so fuzzers can assert transactional decode behavior.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_decode_standard_packed_reference_into(
    input: &[u8],
    out: &mut Vec<u32>,
) -> Result<u32, Base64DecodeReferenceError> {
    if input.len() % 4 != 0 {
        return Err(Base64DecodeReferenceError::InvalidLength { len: input.len() });
    }
    let table = standard_decode_table_ref();
    let blocks = input.len() / 4;
    let decoded_words = blocks
        .checked_mul(3)
        .ok_or(Base64DecodeReferenceError::CapacityOverflow { blocks })?;
    vyre_foundation::allocation::reserve_exact_cleared(out, decoded_words).map_err(|source| {
        Base64DecodeReferenceError::Allocation {
            requested: decoded_words,
            source: source.to_string(),
        }
    })?;
    out.resize(decoded_words, 0);
    for block in 0..blocks {
        let base = block * 4;
        let vals = [
            table[usize::from(input[base])],
            table[usize::from(input[base + 1])],
            table[usize::from(input[base + 2])],
            table[usize::from(input[base + 3])],
        ]
        .map(|value| if value == INVALID { 0 } else { value });
        let out_base = block * 3;
        out[out_base] = (vals[0] << 2) | (vals[1] >> 4);
        if input[base + 2] != b'=' {
            out[out_base + 1] = ((vals[1] & 0x0F) << 4) | (vals[2] >> 2);
        }
        if input[base + 3] != b'=' {
            out[out_base + 2] = ((vals[2] & 0x03) << 6) | vals[3];
        }
    }
    let mut decoded_len = u32::try_from(out.len()).map_err(|_| {
        Base64DecodeReferenceError::DecodedLengthOverflow {
            decoded_words: out.len(),
        }
    })?;
    if input.len() >= 2 {
        if input[input.len() - 1] == b'=' {
            decoded_len = decoded_len.saturating_sub(1);
        }
        if input[input.len() - 2] == b'=' {
            decoded_len = decoded_len.saturating_sub(1);
        }
    }
    Ok(decoded_len)
}

fn clamp_lookup(name: &str, table: &str) -> Vec<Node> {
    let raw = format!("{name}_raw");
    let value = format!("{name}_v");
    vec![
        // Masked 256-table lookup via the canonical ONE-PLACE helper: a >255 input
        // element (the input buffer is U32 and unvalidated) folds to `c & 0xFF`
        // instead of reading past the 256-entry decode table (a raw OOB read is
        // undefined behaviour on a real GPU). Transparent for valid bytes; an out-of-range value lands on a
        // non-base64 slot (INVALID → clamped to 0 below). Using the shared helper
        // is what keeps this mask from being forgotten again.
        Node::let_bind(
            raw.as_str(),
            vyre_primitives::ir_safe::byte_table_lookup(table, Expr::var(name)),
        ),
        Node::let_bind(
            value.as_str(),
            Expr::select(
                Expr::eq(Expr::var(raw.as_str()), Expr::u32(INVALID)),
                Expr::u32(0),
                Expr::var(raw.as_str()),
            ),
        ),
    ]
}

/// Build the reusable base64 decode body.
#[must_use]
pub fn base64_decode_body(
    input: &str,
    table: &str,
    output: &str,
    decoded_len_buffer: &str,
    input_len: u32,
) -> Vec<Node> {
    if input_len % 4 != 0 {
        return vec![Node::trap(
            Expr::u32(input_len),
            "Fix: base64_decode requires input_len to be a multiple of 4; pad with '=' or reject the truncated payload upstream",
        )];
    }
    let decoded_len = decoded_capacity(input_len);
    let mut body = vec![Node::let_bind("j", Expr::InvocationId { axis: 0 })];
    if input_len >= 2 {
        body.push(Node::if_then(
            Expr::eq(Expr::var("j"), Expr::u32(0)),
            vec![
                Node::let_bind(
                    "tail_pad_1",
                    Expr::select(
                        Expr::eq(Expr::load(input, Expr::u32(input_len - 1)), Expr::u32(PAD)),
                        Expr::u32(1),
                        Expr::u32(0),
                    ),
                ),
                Node::let_bind(
                    "tail_pad_2",
                    Expr::select(
                        Expr::eq(Expr::load(input, Expr::u32(input_len - 2)), Expr::u32(PAD)),
                        Expr::u32(1),
                        Expr::u32(0),
                    ),
                ),
                Node::store(
                    decoded_len_buffer,
                    Expr::u32(0),
                    Expr::sub(
                        Expr::sub(Expr::u32(decoded_len), Expr::var("tail_pad_1")),
                        Expr::var("tail_pad_2"),
                    ),
                ),
            ],
        ));
    } else {
        body.push(Node::if_then(
            Expr::eq(Expr::var("j"), Expr::u32(0)),
            vec![Node::store(decoded_len_buffer, Expr::u32(0), Expr::u32(0))],
        ));
    }
    body.push(Node::if_then(
        Expr::lt(Expr::var("j"), Expr::u32(decoded_len)),
        {
            let mut per_byte = vec![
                Node::let_bind("quad", Expr::div(Expr::var("j"), Expr::u32(3))),
                Node::let_bind("in_base", Expr::mul(Expr::var("quad"), Expr::u32(4))),
                Node::let_bind(
                    "pos",
                    Expr::sub(Expr::var("j"), Expr::mul(Expr::var("quad"), Expr::u32(3))),
                ),
                Node::let_bind("c0", Expr::load(input, Expr::var("in_base"))),
                Node::let_bind(
                    "c1",
                    Expr::load(input, Expr::add(Expr::var("in_base"), Expr::u32(1))),
                ),
                Node::let_bind(
                    "c2",
                    Expr::load(input, Expr::add(Expr::var("in_base"), Expr::u32(2))),
                ),
                Node::let_bind(
                    "c3",
                    Expr::load(input, Expr::add(Expr::var("in_base"), Expr::u32(3))),
                ),
                Node::let_bind("pad2", Expr::eq(Expr::var("c2"), Expr::u32(PAD))),
                Node::let_bind("pad1", Expr::eq(Expr::var("c3"), Expr::u32(PAD))),
            ];
            per_byte.extend(clamp_lookup("c0", table));
            per_byte.extend(clamp_lookup("c1", table));
            per_byte.extend(clamp_lookup("c2", table));
            per_byte.extend(clamp_lookup("c3", table));
            per_byte.extend([
                Node::let_bind(
                    "b0",
                    Expr::bitor(
                        Expr::shl(Expr::var("c0_v"), Expr::u32(2)),
                        Expr::shr(Expr::var("c1_v"), Expr::u32(4)),
                    ),
                ),
                Node::let_bind(
                    "b1",
                    Expr::bitor(
                        Expr::shl(
                            Expr::bitand(Expr::var("c1_v"), Expr::u32(0x0F)),
                            Expr::u32(4),
                        ),
                        Expr::shr(Expr::var("c2_v"), Expr::u32(2)),
                    ),
                ),
                Node::let_bind(
                    "b2",
                    Expr::bitor(
                        Expr::shl(
                            Expr::bitand(Expr::var("c2_v"), Expr::u32(0x03)),
                            Expr::u32(6),
                        ),
                        Expr::var("c3_v"),
                    ),
                ),
                Node::if_then(
                    Expr::eq(Expr::var("pos"), Expr::u32(0)),
                    vec![Node::store(output, Expr::var("j"), Expr::var("b0"))],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("pos"), Expr::u32(1)),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("pad2"), Expr::bool(false)),
                        vec![Node::store(output, Expr::var("j"), Expr::var("b1"))],
                    )],
                ),
                Node::if_then(
                    Expr::eq(Expr::var("pos"), Expr::u32(2)),
                    vec![Node::if_then(
                        Expr::eq(Expr::var("pad1"), Expr::bool(false)),
                        vec![Node::store(output, Expr::var("j"), Expr::var("b2"))],
                    )],
                ),
            ]);
            per_byte
        },
    ));
    body
}

/// Wrap the base64 decode body as a child of `parent_op_id`.
#[must_use]
pub fn base64_decode_child(
    parent_op_id: &str,
    input: &str,
    table: &str,
    output: &str,
    decoded_len_buffer: &str,
    input_len: u32,
) -> Node {
    wrap_child_region(
        OP_ID,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        base64_decode_body(input, table, output, decoded_len_buffer, input_len),
    )
}
/// Base64 decode over explicitly named buffers.
#[must_use]
fn base64_decode_program(
    input: &str,
    table: &str,
    output: &str,
    decoded_len_buffer: &str,
    input_len: u32,
) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::storage(table, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(BASE64_DECODE_TABLE_WORDS),
            BufferDecl::output(output, 2, DataType::U32).with_count(decoded_capacity(input_len)),
            BufferDecl::read_write(decoded_len_buffer, 3, DataType::U32).with_count(1),
        ],
        BASE64_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            base64_decode_body(input, table, output, decoded_len_buffer, input_len),
        )],
    )
}

/// Build a Program that decodes base64-encoded ASCII bytes from `input` into
/// `output`, storing one decoded byte per `u32` slot.
///
/// The input buffer carries one ASCII byte per `u32` element so the decode
/// output can chain directly into Aho-Corasick transition-table programs.
///
/// ```ignore
/// use vyre_libs::decode::base64_decode;
///
/// let program = base64_decode("encoded", "decoded", 8);
/// assert_eq!(program.workgroup_size(), [64, 1, 1]);
/// ```
#[must_use]
pub fn base64_decode(input: &str, output: &str, input_len: u32) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let output = scoped_decoded_output_buffer(FAMILY_PREFIX, output);
    base64_decode_program(
        &input,
        BASE64_DECODE_TABLE_BUFFER,
        &output,
        DECODED_LEN_BUFFER,
        input_len,
    )
}

/// Base64 decode fused with an Aho-Corasick scan of the decoded bytes.
///
/// One program, so the decoded payload never leaves device memory between the
/// two stages. `matches` receives one flag per decoded byte position.
#[must_use]
pub fn base64_decode_then_aho_corasick(
    input: &str,
    decoded: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    input_len: u32,
    state_count: u32,
) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let decoded = scoped_decoded_output_buffer(FAMILY_PREFIX, decoded);
    let decoded_capacity = decoded_capacity(input_len);
    let mut entry = vec![base64_decode_child(
        FUSED_SCAN_OP_ID,
        &input,
        BASE64_DECODE_TABLE_BUFFER,
        &decoded,
        DECODED_LEN_BUFFER,
        input_len,
    )];
    entry.extend(linear_aho_scan_body(
        &decoded,
        transitions,
        accept,
        matches,
        Expr::load(DECODED_LEN_BUFFER, Expr::u32(0)),
    ));
    Program::wrapped(
        vec![
            BufferDecl::storage(&input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::storage(
                BASE64_DECODE_TABLE_BUFFER,
                1,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(BASE64_DECODE_TABLE_WORDS),
            BufferDecl::read_write(&decoded, 2, DataType::U32).with_count(decoded_capacity),
            BufferDecl::storage(transitions, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(accept, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count),
            BufferDecl::output(matches, 5, DataType::U32).with_count(decoded_capacity),
            BufferDecl::read_write(DECODED_LEN_BUFFER, 6, DataType::U32).with_count(1),
        ],
        BASE64_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(FUSED_SCAN_OP_ID, entry)],
    )
}
#[cfg(test)]
fn cpu_ref(input: &[u8]) -> (Vec<u32>, u32) {
    decode_standard_packed_reference(input)
}

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![
        vec![
            pack_words(&[
                u32::from(b'T'),
                u32::from(b'W'),
                u32::from(b'F'),
                u32::from(b'u'),
                u32::from(b'T'),
                u32::from(b'W'),
                u32::from(b'F'),
                u32::from(b'u'),
            ]),
            pack_words(standard_decode_table_ref()),
            vec![0u8; 4],
        ],
        vec![
            pack_words(&[
                u32::from(b'T'),
                u32::from(b'W'),
                u32::from(b'E'),
                u32::from(b'='),
                u32::from(b'T'),
                u32::from(b'W'),
                u32::from(b'E'),
                u32::from(b'='),
            ]),
            pack_words(standard_decode_table_ref()),
            vec![0u8; 4],
        ],
        vec![
            pack_words(&[
                u32::from(b'S'),
                u32::from(b'G'),
                u32::from(b'V'),
                u32::from(b's'),
                u32::from(b'b'),
                u32::from(b'G'),
                u32::from(b'8'),
                u32::from(b'*'),
            ]),
            pack_words(standard_decode_table_ref()),
            vec![0u8; 4],
        ],
    ]
}

fn fixture_outputs() -> Vec<Vec<Vec<u8>>> {
    vec![
        vec![pack_words(&[77, 97, 110, 77, 97, 110]), pack_words(&[6])],
        vec![pack_words(&[77, 97, 0, 77, 97, 0]), pack_words(&[5])],
        vec![pack_words(&[72, 101, 108, 108, 111, 0]), pack_words(&[6])],
    ]
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || base64_decode("input", "output", 8),
        Some(fixture_inputs),
        Some(fixture_outputs),
    )
}
// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Build the standard base64 decode table (RFC 4648).
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_base64_table() -> [u32; 256] {
    standard_decode_table()
}

/// CPU reference: decode a base64-encoded byte slice (standard alphabet,
/// `=`-padded, length must be a multiple of 4). Returns decoded bytes.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_base64_decode(input: &[u8]) -> Vec<u8> {
    let (words, decoded_len) = decode_standard_packed_reference(input);
    let decoded_len = usize::try_from(decoded_len).unwrap_or(words.len());
    words
        .into_iter()
        .take(decoded_len)
        .map(|word| (word & 0xFF) as u8)
        .collect()
}
#[cfg(test)]
mod primitive_tests {
    use super::*;

    #[test]
    fn decode_man() {
        assert_eq!(cpu_base64_decode(b"TWFu"), b"Man");
    }

    #[test]
    fn cpu_table_is_the_standard_primitive_table() {
        assert_eq!(cpu_base64_table(), standard_decode_table());
        assert_eq!(standard_decode_table()[b'/' as usize], 63);
        assert_eq!(standard_decode_table()[b'*' as usize], INVALID);
    }

    #[test]
    fn standard_decode_table_ref_matches_value_api_and_reuses_allocation() {
        let first = standard_decode_table_ref();
        let second = standard_decode_table_ref();
        assert!(
            std::ptr::eq(first, second),
            "Fix: base64 decode setup must reuse the immutable primitive table instead of rebuilding it per dispatch."
        );
        assert_eq!(*first, standard_decode_table());
    }

    #[test]
    fn try_decode_reference_rejects_unaligned_input_without_panic() {
        let err = try_decode_standard_packed_reference(b"abc")
            .expect_err("unaligned base64 input must be rejected");
        assert_eq!(err, Base64DecodeReferenceError::InvalidLength { len: 3 });
    }

    #[test]
    fn try_decode_reference_matches_infallible_wrapper() {
        let fallible = try_decode_standard_packed_reference(b"Zm9vYmFy")
            .expect("Fix: unit-test oracle precondition - valid base64 must decode");
        let infallible = decode_standard_packed_reference(b"Zm9vYmFy");
        assert_eq!(fallible, infallible);
        assert_eq!(fallible.1, 6);
    }

    #[test]
    fn try_decode_reference_into_reuses_output_and_clears_stale_tail() {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(&[u32::MAX; 16]);
        let ptr = out.as_ptr();

        let decoded_len = try_decode_standard_packed_reference_into(b"TWE=", &mut out)
            .expect("Fix: unit-test oracle precondition - valid padded base64 must decode into caller-owned storage");

        assert_eq!(decoded_len, 2);
        assert_eq!(out, vec![u32::from(b'M'), u32::from(b'a'), 0]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn try_decode_reference_into_is_transactional_on_invalid_length() {
        let mut out = vec![0x1234_5678, 0x9abc_def0];
        let before = out.clone();

        let err = try_decode_standard_packed_reference_into(b"abc", &mut out)
            .expect_err("unaligned base64 input must be rejected");

        assert_eq!(err, Base64DecodeReferenceError::InvalidLength { len: 3 });
        assert_eq!(out, before);
    }

    // The infallible base64 oracle wrappers must FAIL LOUD on malformed input,
    // not silently return empty. An empty result would let a GPU-vs-CPU parity
    // assertion pass on empty==empty and hide a real divergence (Law 10 / Law 6).
    // Each wrapper gets its own #[should_panic] behavioral proof; callers that
    // need to tolerate bad input use the try_ fallible variants instead.
    #[test]
    #[should_panic(expected = "vyre-primitives base64 decode reference failed")]
    fn decode_reference_fails_loud_on_invalid_length() {
        let _ = decode_standard_packed_reference(b"abc");
    }

    #[test]
    #[should_panic(expected = "vyre-primitives base64 decode reference failed")]
    fn decode_reference_into_fails_loud_on_invalid_length() {
        let mut out = vec![1, 2, 3];
        let _ = decode_standard_packed_reference_into(b"abc", &mut out);
    }

    #[test]
    #[should_panic(expected = "vyre-primitives base64 decode reference failed")]
    fn cpu_base64_decode_fails_loud_on_invalid_length() {
        let _ = cpu_base64_decode(b"abc");
    }

    #[test]
    fn decode_padded_1() {
        assert_eq!(cpu_base64_decode(b"TWE="), b"Ma");
    }

    #[test]
    fn decode_padded_2() {
        assert_eq!(cpu_base64_decode(b"TQ=="), b"M");
    }

    #[test]
    fn decode_empty() {
        assert_eq!(cpu_base64_decode(b""), b"");
    }

    #[test]
    fn decode_hello_world() {
        assert_eq!(cpu_base64_decode(b"SGVsbG8gV29ybGQ="), b"Hello World");
    }

    #[test]
    fn decode_roundtrip_rfc4648_vectors() {
        // RFC 4648 test vectors
        assert_eq!(cpu_base64_decode(b"Zg=="), b"f");
        assert_eq!(cpu_base64_decode(b"Zm8="), b"fo");
        assert_eq!(cpu_base64_decode(b"Zm9v"), b"foo");
        assert_eq!(cpu_base64_decode(b"Zm9vYg=="), b"foob");
        assert_eq!(cpu_base64_decode(b"Zm9vYmE="), b"fooba");
        assert_eq!(cpu_base64_decode(b"Zm9vYmFy"), b"foobar");
    }

    #[test]
    fn table_index_is_masked_so_high_bit_input_cannot_read_out_of_bounds() {
        use vyre_reference::value::Value;
        // "TWFu" decodes to "Man". The U32 input buffer can carry a >255 element
        // (it is unvalidated); here the first char is `0x100 | 'T'`. The `& 0xFF`
        // index mask must fold it back to 'T' so the decode is IDENTICAL to the
        // clean input, and must never read past the 256-entry decode table (a raw
        // OOB read is undefined behaviour on a real GPU). This is a regression LOCK: the OLD unmasked
        // `load(table, c)` OOB-indexes the table (zero-filled to 0 by the reference
        // interpreter), decoding a wrong first byte instead of 'M'.
        let input_len = 4u32;
        let dirty = [
            0x0100u32 | u32::from(b'T'),
            u32::from(b'W'),
            u32::from(b'F'),
            u32::from(b'u'),
        ];
        let program = base64_decode_program("input", "table", "output", "decoded_len", input_len);
        let inputs = vec![
            Value::from(vyre_primitives::wire::pack_u32_slice(&dirty)),
            Value::from(vyre_primitives::wire::pack_u32_slice(standard_decode_table_ref())),
            Value::from(vec![0u8; decoded_capacity(input_len) as usize * 4]),
            Value::from(vyre_primitives::wire::pack_u32_slice(&[0])),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: base64 decode with a >255 input element must not fault the interpreter");
        // Two outputs (output + decoded_len): locate each by name via the
        // interpreter's own output ABI, never by fixed position.
        let out_idx = vyre_reference::output_index(&program, "output")
            .expect("Fix: base64 output buffer must be a reference output");
        let len_idx = vyre_reference::output_index(&program, "decoded_len")
            .expect("Fix: base64 decoded_len buffer must be a reference output");
        let words = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[out_idx].to_bytes());
        let decoded_len =
            vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[len_idx].to_bytes())[0] as usize;
        let bytes: Vec<u8> = words
            .into_iter()
            .take(decoded_len)
            .map(|word| (word & 0xFF) as u8)
            .collect();
        assert_eq!(
            bytes, b"Man",
            "Fix: masked table index must decode the high-bit-dirty input identically to the clean 'TWFu'"
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::CompiledDfa;
    use vyre_reference::value::Value;

    fn run(input: &[u8]) -> (Vec<u32>, u32) {
        let program = base64_decode("input", "output", input.len() as u32);
        let decoded_capacity = decoded_capacity(input.len() as u32);
        let inputs = vec![
            Value::from(pack_words(
                &input
                    .iter()
                    .map(|&byte| u32::from(byte))
                    .collect::<Vec<_>>(),
            )),
            Value::from(pack_words(standard_decode_table_ref())),
            Value::from(vec![0u8; decoded_capacity as usize * 4]),
            Value::from(vec![0u8; 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: base64 decode must run; restore this invariant before continuing.");
        let decoded = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        let len_bytes = outputs[1].to_bytes();
        let decoded_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]);
        (decoded, decoded_len)
    }

    #[test]
    fn aligned_input_decodes_three_bytes() {
        let (decoded, decoded_len) = run(b"TWFu");
        assert_eq!(&decoded[..3], &[77, 97, 110]);
        assert_eq!(decoded_len, 3);
    }

    #[test]
    fn padded_input_reports_real_length() {
        let (decoded, decoded_len) = run(b"TQ==");
        assert_eq!(decoded[0], 77);
        assert_eq!(decoded_len, 1);
    }

    #[test]
    fn invalid_character_clamps_without_panicking() {
        let (decoded, decoded_len) = run(b"SGVsbG8*");
        assert_eq!(&decoded[..6], &[72, 101, 108, 108, 111, 0]);
        assert_eq!(decoded_len, 6);
    }

    #[test]
    fn malformed_length_lowers_to_ir_trap_not_host_panic() {
        let program = base64_decode("input", "output", 3);
        assert!(program.stats().trap());
    }

    #[test]
    fn fused_program_reuses_decoded_buffer_for_scan() {
        let dfa = CompiledDfa {
            transitions: vec![0; 256],
            accept: vec![0],
            state_count: 1,
            max_pattern_len: 0,
            output_offsets: vec![0, 0],
            output_records: vec![],
        };
        let program = base64_decode_then_aho_corasick(
            "input",
            "decoded",
            "transitions",
            "accept",
            "matches",
            8,
            dfa.state_count,
        );
        assert_eq!(
            program.buffers()[2].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[5].name(), "matches");
        assert_eq!(program.buffers()[6].name(), DECODED_LEN_BUFFER);
    }

    #[test]
    fn generic_default_names_are_family_scoped() {
        let program = base64_decode("input", "decoded", 8);
        assert_eq!(
            program.buffers()[0].name(),
            fixed_name(FAMILY_PREFIX, "input")
        );
        assert_eq!(
            program.buffers()[2].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[3].name(), DECODED_LEN_BUFFER);
    }

    #[test]
    fn twelve_byte_input_decodes_nine_bytes_in_linear_time() {
        let (decoded, decoded_len) = run(b"TWFuTWFuTWFu");
        assert_eq!(&decoded[..9], &[77, 97, 110, 77, 97, 110, 77, 97, 110]);
        assert_eq!(decoded_len, 9);
    }

    #[test]
    fn generated_quads_match_cpu_reference_for_invalid_padding_and_symbols() {
        const ALPHABET: &[u8] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=*#\n";

        for seed in 0u32..4096 {
            let quads = 1 + (seed % 6);
            let mut state = seed ^ 0xB64D_EC0D;
            let mut input = Vec::with_capacity(quads as usize * 4);
            for _ in 0..(quads * 4) {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                input.push(ALPHABET[(state as usize) % ALPHABET.len()]);
            }

            let (actual, actual_len) = run(&input);
            let (expected, expected_len) = cpu_ref(&input);
            assert_eq!(actual_len, expected_len, "decoded length seed {seed}");
            assert_eq!(actual, expected, "decoded bytes seed {seed}");
        }
    }
}
