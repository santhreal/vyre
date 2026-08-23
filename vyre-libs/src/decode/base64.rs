//! Base64 decode `Program` construction and RFC 4648 lookup-table ownership.
//!
//! Independent decode witnesses live in `vyre-reference`.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(test)]
use crate::buffer_names::fixed_name;
use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decoded_output_buffer};
use crate::decode::scan::linear_aho_scan_body;
use vyre_primitives::wire::pack_u32_slice as pack_words;
#[cfg(test)]
use vyre_reference::composition_witness::{
    base64_decode_bytes_witness as decode_standard_bytes_reference,
    base64_decode_packed_witness as decode_standard_packed_reference,
    base64_decode_packed_witness_into as decode_standard_packed_reference_into,
    try_base64_decode_packed_witness as try_decode_standard_packed_reference,
    try_base64_decode_packed_witness_into as try_decode_standard_packed_reference_into,
    Base64DecodeWitnessError as Base64DecodeReferenceError,
};

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

static STANDARD_DECODE_TABLE: [u32; 256] = [
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 62, 255, 255, 255, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 255,
    255, 255, 0, 255, 255, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 255, 255, 255, 255, 255, 255, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
];

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
    &STANDARD_DECODE_TABLE
}

/// Decoded capacity for a padded base64 input.
#[must_use]
pub fn decoded_capacity(input_len: u32) -> u32 {
    blocks_for_len(input_len) * 3
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
            crate::builder::state_machine::TableStateMachineComposer::byte_table_lookup(
                table,
                Expr::var(name),
            ),
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
        Ident::from(parent_op_id),
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
/// use vyre_libs::decode::base64::base64_decode;
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
fn reference_base64_decode_packed(input: &[u8]) -> (Vec<u32>, u32) {
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

const EXPECTED_BASE64_CASE0_BYTES: [u8; 24] = [
    77, 0, 0, 0, 97, 0, 0, 0, 110, 0, 0, 0, 77, 0, 0, 0, 97, 0, 0, 0, 110, 0, 0, 0,
];
const EXPECTED_BASE64_CASE0_LEN: [u8; 4] = [6, 0, 0, 0];
const EXPECTED_BASE64_CASE1_BYTES: [u8; 24] = [
    77, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 97, 0, 0, 0, 0, 0, 0, 0,
];
const EXPECTED_BASE64_CASE1_LEN: [u8; 4] = [5, 0, 0, 0];
const EXPECTED_BASE64_CASE2_BYTES: [u8; 24] = [
    72, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0,
];
const EXPECTED_BASE64_CASE2_LEN: [u8; 4] = [6, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || base64_decode("input", "output", 8),
        Some(fixture_inputs),
        Some(|| {
            vec![
                vec![EXPECTED_BASE64_CASE0_BYTES.to_vec(), EXPECTED_BASE64_CASE0_LEN.to_vec()],
                vec![EXPECTED_BASE64_CASE1_BYTES.to_vec(), EXPECTED_BASE64_CASE1_LEN.to_vec()],
                vec![EXPECTED_BASE64_CASE2_BYTES.to_vec(), EXPECTED_BASE64_CASE2_LEN.to_vec()],
            ]
        }),
    )
}

#[cfg(test)]
mod primitive_tests {
    use super::*;
    use crate::fixture_bytes::eval_bytes;
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

    #[test]
    fn decode_man() {
        assert_eq!(decode_standard_bytes_reference(b"TWFu"), b"Man");
    }

    #[test]
    fn standard_table_is_the_primitive_table() {
        assert_eq!(*standard_decode_table_ref(), standard_decode_table());
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
    #[should_panic(expected = "base64 decode witness failed")]
    fn decode_reference_fails_loud_on_invalid_length() {
        let _ = decode_standard_packed_reference(b"abc");
    }

    #[test]
    #[should_panic(expected = "base64 decode witness failed")]
    fn decode_reference_into_fails_loud_on_invalid_length() {
        let mut out = vec![1, 2, 3];
        let _ = decode_standard_packed_reference_into(b"abc", &mut out);
    }

    #[test]
    #[should_panic(expected = "base64 decode witness failed")]
    fn decode_standard_bytes_reference_fails_loud_on_invalid_length() {
        let _ = decode_standard_bytes_reference(b"abc");
    }

    #[test]
    fn decode_padded_1() {
        assert_eq!(decode_standard_bytes_reference(b"TWE="), b"Ma");
    }

    #[test]
    fn decode_padded_2() {
        assert_eq!(decode_standard_bytes_reference(b"TQ=="), b"M");
    }

    #[test]
    fn decode_empty() {
        assert_eq!(decode_standard_bytes_reference(b""), b"");
    }

    #[test]
    fn decode_hello_world() {
        assert_eq!(
            decode_standard_bytes_reference(b"SGVsbG8gV29ybGQ="),
            b"Hello World"
        );
    }

    #[test]
    fn decode_roundtrip_rfc4648_vectors() {
        // RFC 4648 test vectors
        assert_eq!(decode_standard_bytes_reference(b"Zg=="), b"f");
        assert_eq!(decode_standard_bytes_reference(b"Zm8="), b"fo");
        assert_eq!(decode_standard_bytes_reference(b"Zm9v"), b"foo");
        assert_eq!(decode_standard_bytes_reference(b"Zm9vYg=="), b"foob");
        assert_eq!(decode_standard_bytes_reference(b"Zm9vYmE="), b"fooba");
        assert_eq!(decode_standard_bytes_reference(b"Zm9vYmFy"), b"foobar");
    }

    #[test]
    fn table_index_is_masked_so_high_bit_input_cannot_read_out_of_bounds() {
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
            vyre_primitives::wire::pack_u32_slice(&dirty),
            vyre_primitives::wire::pack_u32_slice(standard_decode_table_ref()),
            vec![0u8; decoded_capacity(input_len) as usize * 4],
            vyre_primitives::wire::pack_u32_slice(&[0]),
        ];
        let outputs = eval_bytes("base64", &program, inputs);
        // Two outputs (output + decoded_len): locate each by name via the
        // interpreter's own output ABI, never by fixed position.
        let out_idx = vyre_reference::output_index(&program, "output")
            .expect("Fix: base64 output buffer must be a reference output");
        let len_idx = vyre_reference::output_index(&program, "decoded_len")
            .expect("Fix: base64 decoded_len buffer must be a reference output");
        let words = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[out_idx]);
        let decoded_len =
            vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[len_idx])[0] as usize;
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
    use crate::decode::scan::dummy_compiled_dfa;
    use crate::fixture_bytes::{bytes_to_u32, decode_u32_one, eval_bytes};

    fn decoded(input: &[u8]) -> (Vec<u32>, u32) {
        let program = base64_decode("input", "output", input.len() as u32);
        let capacity = decoded_capacity(input.len() as u32);
        let widened: Vec<u32> = input.iter().map(|&byte| u32::from(byte)).collect();
        let outputs = eval_bytes(
            "base64_decode",
            &program,
            vec![
                pack_words(&widened),
                pack_words(standard_decode_table_ref()),
                vec![0u8; capacity as usize * 4],
                vec![0u8; 4],
            ],
        );
        (bytes_to_u32(&outputs[0]), decode_u32_one(&outputs[1]))
    }

    #[test]
    fn aligned_input_decodes_three_bytes() {
        let (decoded, decoded_len) = decoded(b"TWFu");
        assert_eq!(&decoded[..3], &[77, 97, 110]);
        assert_eq!(decoded_len, 3);
    }

    #[test]
    fn padded_input_reports_real_length() {
        let (decoded, decoded_len) = decoded(b"TQ==");
        assert_eq!(decoded[0], 77);
        assert_eq!(decoded_len, 1);
    }

    #[test]
    fn invalid_character_clamps_without_panicking() {
        let (decoded, decoded_len) = decoded(b"SGVsbG8*");
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
        let dfa = dummy_compiled_dfa();
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
        let (decoded, decoded_len) = decoded(b"TWFuTWFuTWFu");
        assert_eq!(&decoded[..9], &[77, 97, 110, 77, 97, 110, 77, 97, 110]);
        assert_eq!(decoded_len, 9);
    }

    #[test]
    fn generated_quads_match_reference_for_invalid_padding_and_symbols() {
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

            let (actual, actual_len) = decoded(&input);
            let (expected, expected_len) = reference_base64_decode_packed(&input);
            assert_eq!(actual_len, expected_len, "decoded length seed {seed}");
            assert_eq!(actual, expected, "decoded bytes seed {seed}");
        }
    }
}
