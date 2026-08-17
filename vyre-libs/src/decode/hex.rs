//! ASCII hex decode: one module, one op id.
//!
//! The kernel body, the CPU reference oracle, the lookup table, the public
//! builder and the fused decode-then-scan builder all live here. The op id is
//! `vyre-libs::decode::hex`.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(test)]
use crate::buffer_names::fixed_name;
use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decoded_output_buffer};
use crate::decode::scan::tiled_decode_aho_scan_body;
use vyre_primitives::wire::pack_u32_slice as pack_words;

/// Canonical op id for ASCII hex decode.
pub const OP_ID: &str = "vyre-libs::decode::hex";
const FUSED_SCAN_OP_ID: &str = "vyre-libs::decode::hex_then_aho_corasick";
const FAMILY_PREFIX: &str = "decode_hex";
/// Fixed buffer name carrying the ASCII hex decode lookup table.
pub const HEX_DECODE_TABLE_BUFFER: &str = "__vyre_decode_hex_table";

/// Number of words in the ASCII hex decode lookup table.
pub const HEX_DECODE_TABLE_WORDS: u32 = 256;
/// Canonical hex decode workgroup size.
pub const HEX_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

static HEX_DECODE_TABLE: [u32; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0,
    0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];

/// Return the canonical 256-entry ASCII hex decode table by value.
#[must_use]
pub fn hex_decode_table() -> [u32; 256] {
    *hex_decode_table_ref()
}

/// Process-wide canonical ASCII hex decode table.
///
/// The table is immutable after construction. Dispatch setup and CPU oracles
/// should use this reference when they do not need an owned copy.
#[must_use]
pub fn hex_decode_table_ref() -> &'static [u32; 256] {
    &HEX_DECODE_TABLE
}

/// Number of decoded byte slots produced by an even-length hex input.
#[must_use]
pub const fn hex_decoded_capacity(input_len: u32) -> u32 {
    input_len / 2
}

fn nibble_expr(byte: Expr, table: &str) -> Expr {
    crate::builder::TableStateMachineComposer::byte_table_lookup(table, byte)
}

/// Decode one hex byte pair into a single u32 byte value.
#[must_use]
pub fn hex_decode_pair_expr(input: &str, table: &str, pair: Expr) -> Expr {
    let in_base = Expr::mul(pair, Expr::u32(2));
    let hi = nibble_expr(Expr::load(input, in_base.clone()), table);
    let lo = nibble_expr(Expr::load(input, Expr::add(in_base, Expr::u32(1))), table);
    Expr::bitor(Expr::shl(hi, Expr::u32(4)), lo)
}

/// Build the reusable hex decode body.
#[must_use]
pub fn hex_decode_body(input: &str, output: &str, table: &str, input_len: u32) -> Vec<Node> {
    if input_len % 2 != 0 {
        return vec![Node::trap(
            Expr::u32(input_len),
            "Fix: hex_decode requires an even input_len; reject the dangling nibble upstream",
        )];
    }
    let output_len = hex_decoded_capacity(input_len);
    vec![
        Node::let_bind("pair", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("pair"), Expr::u32(output_len)),
            vec![Node::store(
                output,
                Expr::var("pair"),
                hex_decode_pair_expr(input, table, Expr::var("pair")),
            )],
        ),
    ]
}

/// Wrap the hex decode body as a child of `parent_op_id`.
#[must_use]
pub fn hex_decode_child(
    parent_op_id: &str,
    input: &str,
    output: &str,
    table: &str,
    input_len: u32,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        hex_decode_body(input, output, table, input_len),
    )
}
/// Hex decode over explicitly named buffers.
#[must_use]
fn hex_decode_program(input: &str, output: &str, table: &str, input_len: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::output(output, 1, DataType::U32)
                .with_count(hex_decoded_capacity(input_len)),
            BufferDecl::storage(table, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(HEX_DECODE_TABLE_WORDS),
        ],
        HEX_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            hex_decode_body(input, output, table, input_len),
        )],
    )
}

/// Build a Program that decodes ASCII hex bytes from `input` into `output`,
/// storing one decoded byte per `u32` slot.
///
/// ```ignore
/// use vyre_libs::decode::hex::hex_decode;
///
/// let program = hex_decode("encoded", "decoded", 6);
/// assert_eq!(program.workgroup_size(), [64, 1, 1]);
/// ```
#[must_use]
pub fn hex_decode(input: &str, output: &str, input_len: u32) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let output = scoped_decoded_output_buffer(FAMILY_PREFIX, output);
    hex_decode_program(&input, &output, HEX_DECODE_TABLE_BUFFER, input_len)
}

/// Build one GPU program that hex-decodes and then scans the decoded bytes
/// with the Aho-Corasick transition table, without a host readback between
/// stages.
///
/// ```ignore
/// use vyre_libs::decode::hex::hex_decode_then_aho_corasick;
///
/// let program = hex_decode_then_aho_corasick(
///     "encoded",
///     "decoded",
///     "transitions",
///     "accept",
///     "matches",
///     8,
///     4,
/// );
/// assert_eq!(program.output_buffer_indices().len(), 1);
/// ```
#[must_use]
pub fn hex_decode_then_aho_corasick(
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
    let decoded_len = hex_decoded_capacity(input_len);
    let body = tiled_decode_aho_scan_body(
        transitions,
        accept,
        matches,
        Expr::u32(decoded_len),
        64,
        |pair| hex_decode_pair_expr(&input, HEX_DECODE_TABLE_BUFFER, pair),
        |pair, byte| Some(Node::store(&decoded, pair, byte)),
    );
    Program::wrapped(
        vec![
            BufferDecl::storage(&input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::read_write(&decoded, 1, DataType::U32).with_count(decoded_len),
            BufferDecl::storage(transitions, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(accept, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count),
            BufferDecl::output(matches, 4, DataType::U32).with_count(decoded_len),
            BufferDecl::storage(
                HEX_DECODE_TABLE_BUFFER,
                5,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(HEX_DECODE_TABLE_WORDS),
        ],
        HEX_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(FUSED_SCAN_OP_ID, body)],
    )
}
#[cfg(test)]
fn reference_hex_decode_packed(input: &[u8]) -> Vec<u32> {
    vyre_reference::composition_witness::hex_decode_packed_witness(input)
}

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![
        vec![
            pack_words(&[
                u32::from(b'4'),
                u32::from(b'D'),
                u32::from(b'6'),
                u32::from(b'1'),
                u32::from(b'6'),
                u32::from(b'E'),
            ]),
            pack_words(&[0, 0, 0]),
            pack_words(hex_decode_table_ref()),
        ],
        vec![
            pack_words(&[
                u32::from(b'6'),
                u32::from(b'8'),
                u32::from(b'4'),
                u32::from(b'9'),
                u32::from(b'4'),
                u32::from(b'A'),
            ]),
            pack_words(&[0, 0, 0]),
            pack_words(hex_decode_table_ref()),
        ],
        vec![
            pack_words(&[
                u32::from(b'7'),
                u32::from(b'a'),
                u32::from(b'Z'),
                u32::from(b'1'),
                u32::from(b'0'),
                u32::from(b'0'),
            ]),
            pack_words(&[0, 0, 0]),
            pack_words(hex_decode_table_ref()),
        ],
    ]
}

const EXPECTED_HEX_CASE0_BYTES: [u8; 12] = [0x4D, 0, 0, 0, 0x61, 0, 0, 0, 0x6E, 0, 0, 0];
const EXPECTED_HEX_CASE1_BYTES: [u8; 12] = [0x68, 0, 0, 0, 0x49, 0, 0, 0, 0x4A, 0, 0, 0];
const EXPECTED_HEX_CASE2_BYTES: [u8; 12] = [0x7A, 0, 0, 0, 0x01, 0, 0, 0, 0x00, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || hex_decode("input", "output", 6),
        Some(fixture_inputs),
        Some(|| {
            vec![
                vec![EXPECTED_HEX_CASE0_BYTES.to_vec()],
                vec![EXPECTED_HEX_CASE1_BYTES.to_vec()],
                vec![EXPECTED_HEX_CASE2_BYTES.to_vec()],
            ]
        }),
    )
}
#[cfg(test)]
mod primitive_tests {
    use super::*;
    fn build_hex_decode_table() -> [u32; 256] {
        let mut table = [0u32; 256];
        let mut byte = b'0';
        while byte <= b'9' {
            table[byte as usize] = u32::from(byte - b'0');
            byte += 1;
        }
        byte = b'A';
        while byte <= b'F' {
            table[byte as usize] = u32::from(byte - b'A' + 10);
            byte += 1;
        }
        byte = b'a';
        while byte <= b'f' {
            table[byte as usize] = u32::from(byte - b'a' + 10);
            byte += 1;
        }
        table
    }

    #[test]
    fn reference_decodes_upper_lower_and_invalid_nibbles() {
        assert_eq!(
            reference_hex_decode_packed(b"4D6aZ1"),
            vec![0x4D, 0x6A, 0x01]
        );
    }

    #[test]
    fn hex_decode_table_ref_matches_value_api_and_reuses_allocation() {
        let first = hex_decode_table_ref();
        let second = hex_decode_table_ref();
        assert!(
            std::ptr::eq(first, second),
            "Fix: hex decode setup must reuse the immutable primitive table instead of rebuilding it per dispatch."
        );
        assert_eq!(*first, hex_decode_table());
    }

    #[test]
    fn odd_length_lowers_to_trap_not_silent_truncation() {
        let body = hex_decode_body("input", "output", "table", 3);
        assert!(matches!(body.as_slice(), [Node::Trap { .. }]));
    }

    #[test]
    fn standalone_program_is_single_primitive_region() {
        let program = hex_decode_program("input", "output", "table", 6);
        let [Node::Region { generator, .. }] = program.entry() else {
            panic!("expected one primitive hex decode region");
        };
        assert_eq!(generator.as_str(), OP_ID);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::CompiledDfa;
    use vyre_reference::value::Value;

    fn run(input: &[u8]) -> Vec<u32> {
        let program = hex_decode("input", "output", input.len() as u32);
        let inputs = vec![
            Value::from(pack_words(
                &input
                    .iter()
                    .map(|&byte| u32::from(byte))
                    .collect::<Vec<_>>(),
            )),
            Value::from(vec![0u8; (input.len() / 2) * 4]),
            Value::from(pack_words(hex_decode_table_ref())),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: hex decode must run; restore this invariant before continuing.");
        vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
    }

    #[test]
    fn decodes_uppercase_hex() {
        assert_eq!(run(b"4D616E"), vec![77, 97, 110]);
    }

    #[test]
    fn decodes_lowercase_hex() {
        assert_eq!(run(b"68494a"), vec![104, 73, 74]);
    }

    #[test]
    fn decodes_sixteen_char_hex() {
        // 16-character input → 8 output bytes. Regression guard against
        // any O(n²) path that re-walks the input per output byte.
        assert_eq!(
            run(b"4D616E6973657321"),
            vec![77, 97, 110, 105, 115, 101, 115, 33]
        );
    }

    #[test]
    fn invalid_nibble_clamps_to_zero() {
        assert_eq!(run(b"7aZ100"), vec![122, 1, 0]);
    }

    #[test]
    fn generated_pairs_match_primitive_reference_for_invalid_and_mixed_case() {
        const ALPHABET: &[u8] = b"0123456789abcdefABCDEFXz*#\n";

        for seed in 0u32..4096 {
            let pairs = 1 + (seed % 16);
            let mut state = seed ^ 0x48EC_DECD;
            let mut input = Vec::with_capacity(pairs as usize * 2);
            for _ in 0..(pairs * 2) {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                input.push(ALPHABET[(state as usize) % ALPHABET.len()]);
            }

            assert_eq!(
                run(&input),
                reference_hex_decode_packed(&input),
                "hex wrapper seed {seed}"
            );
        }
    }

    #[test]
    fn generic_default_names_are_family_scoped() {
        let program = hex_decode("input", "decoded", 6);
        assert_eq!(
            program.buffers()[0].name(),
            fixed_name(FAMILY_PREFIX, "input")
        );
        assert_eq!(
            program.buffers()[1].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[2].name(), HEX_DECODE_TABLE_BUFFER);
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
        let program = hex_decode_then_aho_corasick(
            "input",
            "decoded",
            "transitions",
            "accept",
            "matches",
            8,
            dfa.state_count,
        );
        assert_eq!(
            program.buffers()[1].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[4].name(), "matches");
        assert_eq!(program.buffers()[5].name(), HEX_DECODE_TABLE_BUFFER);
    }
}
