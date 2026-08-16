//! DEFLATE stored-block inflate: one module, one op id.
//!
//! The kernel body, the header decode nodes, the CPU reference oracle, the
//! public builder and the fused inflate-then-scan builder all live here. The op
//! id is `vyre-libs::decode::inflate_stored_block`.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::GeneratorRef;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

#[cfg(test)]
use crate::buffer_names::fixed_name;
use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decoded_output_buffer};
use crate::decode::scan::tiled_decode_aho_scan_body;
use vyre_primitives::wire::pack_u32_slice as pack_words;

/// Canonical op id for stored-block inflate.
pub const OP_ID: &str = "vyre-libs::decode::inflate_stored_block";
const FUSED_SCAN_OP_ID: &str = "vyre-libs::decode::inflate_stored_block_then_aho_corasick";
const FAMILY_PREFIX: &str = "decode_inflate";
const INFLATED_LEN_BUFFER: &str = "__vyre_decode_inflate_inflated_len";
const DEFAULT_DECODE_SCAN_TILE: u32 = 64;

/// Fixed-Huffman block diagnostic.
pub const FIXED_HUFFMAN_REJECT: &str = "Fix: vyre-libs::decode::inflate_stored_block accepts raw DEFLATE stored blocks only; route BTYPE=1 input to a compressed-block decoder.";
/// Dynamic-Huffman block diagnostic.
pub const DYNAMIC_HUFFMAN_REJECT: &str = "Fix: vyre-libs::decode::inflate_stored_block accepts raw DEFLATE stored blocks only; route BTYPE=2 input to a dynamic-Huffman decoder.";
/// Reserved BTYPE diagnostic.
pub const RESERVED_BTYPE_FIX: &str =
    "Fix: reject reserved DEFLATE BTYPE=3 inputs before dispatching vyre-libs::decode::inflate_stored_block.";
/// Stored block LEN/NLEN diagnostic.
pub const STORED_HEADER_FIX: &str =
    "Fix: validate LEN/NLEN before copying a stored DEFLATE block in vyre-libs::decode::inflate_stored_block.";
/// Number of u32 byte lanes occupied by the stored-block header.
pub const INFLATE_STORED_HEADER_WORDS: u32 = 5;
/// Canonical workgroup shape for stored-block inflate compositions.
pub const INFLATE_STORED_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// Emit canonical stored-block header decode nodes.
///
/// Defines `header`, `btype`, `len`, and `nlen` in the caller's region.
#[must_use]
pub fn inflate_stored_header_nodes(input: &str) -> Vec<Node> {
    vec![
        Node::let_bind("header", Expr::load(input, Expr::u32(0))),
        Node::let_bind(
            "btype",
            Expr::bitand(Expr::shr(Expr::var("header"), Expr::u32(1)), Expr::u32(0x3)),
        ),
        Node::let_bind(
            "len",
            Expr::bitor(
                Expr::load(input, Expr::u32(1)),
                Expr::shl(Expr::load(input, Expr::u32(2)), Expr::u32(8)),
            ),
        ),
        Node::let_bind(
            "nlen",
            Expr::bitor(
                Expr::load(input, Expr::u32(3)),
                Expr::shl(Expr::load(input, Expr::u32(4)), Expr::u32(8)),
            ),
        ),
    ]
}

/// Expression asserting the stored-block LEN/NLEN complement contract.
#[must_use]
pub fn inflate_stored_len_is_valid_expr() -> Expr {
    Expr::eq(
        Expr::var("nlen"),
        Expr::bitxor(Expr::var("len"), Expr::u32(0xFFFF)),
    )
}

/// Expression loading payload byte lane `index` after the stored-block header.
#[must_use]
pub fn inflate_stored_payload_expr(input: &str, index: Expr) -> Expr {
    Expr::load(
        input,
        Expr::add(Expr::u32(INFLATE_STORED_HEADER_WORDS), index),
    )
}

/// Trap node for a BTYPE=0 block whose LEN/NLEN header is invalid.
#[must_use]
pub fn inflate_stored_invalid_len_trap_node() -> Node {
    Node::if_then(
        Expr::ne(
            Expr::var("nlen"),
            Expr::bitxor(Expr::var("len"), Expr::u32(0xFFFF)),
        ),
        vec![Node::trap(Expr::u32(0), STORED_HEADER_FIX)],
    )
}

/// Trap nodes for non-stored DEFLATE BTYPE values.
#[must_use]
pub fn inflate_stored_non_stored_trap_nodes() -> [Node; 3] {
    [
        Node::if_then(
            Expr::eq(Expr::var("btype"), Expr::u32(1)),
            vec![Node::trap(Expr::u32(1), FIXED_HUFFMAN_REJECT)],
        ),
        Node::if_then(
            Expr::eq(Expr::var("btype"), Expr::u32(2)),
            vec![Node::trap(Expr::u32(2), DYNAMIC_HUFFMAN_REJECT)],
        ),
        Node::if_then(
            Expr::eq(Expr::var("btype"), Expr::u32(3)),
            vec![Node::trap(Expr::u32(3), RESERVED_BTYPE_FIX)],
        ),
    ]
}

/// Build the reusable stored-block inflate body.
#[must_use]
pub fn inflate_stored_body(input: &str, output: &str, inflated_len_buffer: &str) -> Vec<Node> {
    let mut body = vec![
        Node::let_bind("lane", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::eq(Expr::var("lane"), Expr::u32(0)),
            vec![Node::store(inflated_len_buffer, Expr::u32(0), Expr::u32(0))],
        ),
    ];
    body.extend(inflate_stored_header_nodes(input));
    body.extend([Node::if_then(
        Expr::eq(Expr::var("btype"), Expr::u32(0)),
        vec![
            Node::if_then(
                inflate_stored_len_is_valid_expr(),
                vec![
                    Node::if_then(
                        Expr::eq(Expr::var("lane"), Expr::u32(0)),
                        vec![Node::store(
                            inflated_len_buffer,
                            Expr::u32(0),
                            Expr::var("len"),
                        )],
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("lane"), Expr::var("len")),
                        vec![Node::store(
                            output,
                            Expr::var("lane"),
                            inflate_stored_payload_expr(input, Expr::var("lane")),
                        )],
                    ),
                ],
            ),
            inflate_stored_invalid_len_trap_node(),
        ],
    )]);
    body.extend(inflate_stored_non_stored_trap_nodes());
    body
}

/// Wrap the stored-block inflate body as a child of `parent_op_id`.
#[must_use]
pub fn inflate_stored_child(
    parent_op_id: &str,
    input: &str,
    output: &str,
    inflated_len_buffer: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        inflate_stored_body(input, output, inflated_len_buffer),
    )
}
/// Stored-block inflate over explicitly named buffers.
#[must_use]
fn inflate_stored_program(
    input: &str,
    output: &str,
    inflated_len_buffer: &str,
    input_len: u32,
) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::output(output, 1, DataType::U32).with_count(input_len),
            BufferDecl::read_write(inflated_len_buffer, 2, DataType::U32).with_count(1),
        ],
        INFLATE_STORED_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            inflate_stored_body(input, output, inflated_len_buffer),
        )],
    )
}

/// Build a Program that inflates a single DEFLATE stored block from `input`
/// into `output`, storing one inflated byte per `u32` slot.
///
/// ```ignore
/// use vyre_libs::decode::inflate::inflate_stored_block;
///
/// let program = inflate_stored_block("deflated", "inflated", 10);
/// assert_eq!(program.workgroup_size(), [64, 1, 1]);
/// ```
#[must_use]
pub fn inflate_stored_block(input: &str, output: &str, input_len: u32) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let output = scoped_decoded_output_buffer(FAMILY_PREFIX, output);
    inflate_stored_program(&input, &output, INFLATED_LEN_BUFFER, input_len)
}

/// Build one GPU program that inflates a stored DEFLATE block and then scans
/// the inflated bytes with the Aho-Corasick transition table, without a host
/// readback between stages.
///
/// Only BTYPE=0 (stored) blocks are accepted by this builder.
///
/// ```ignore
/// use vyre_libs::decode::inflate::inflate_stored_block_then_aho_corasick;
///
/// let program = inflate_stored_block_then_aho_corasick(
///     "input",
///     "decoded",
///     "transitions",
///     "accept",
///     "matches",
///     10,
///     4,
/// );
/// assert_eq!(program.output_buffer_indices().len(), 1);
/// ```
#[must_use]
pub fn inflate_stored_block_then_aho_corasick(
    input: &str,
    decoded: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    input_len: u32,
    state_count: u32,
) -> Program {
    fused_scan_program(
        input,
        decoded,
        transitions,
        accept,
        matches,
        input_len,
        state_count,
        DEFAULT_DECODE_SCAN_TILE,
    )
}

/// Scan bytes as they are copied from the stored block payload, at an explicit
/// tile width.
///
/// Stored DEFLATE blocks have no entropy decode dependency, so the fused path
/// can keep DFA state in registers and avoid a second pass over the decoded
/// global buffer. The decoded buffer remains populated to preserve the existing
/// output contract. `inflate_stored_block_then_aho_corasick` is the published
/// entry and picks the tile width, so this form takes one.
fn fused_scan_program(
    input: &str,
    decoded: &str,
    transitions: &str,
    accept: &str,
    matches: &str,
    input_len: u32,
    state_count: u32,
    tile_width: u32,
) -> Program {
    let input = scoped_decode_input_buffer(FAMILY_PREFIX, input);
    let decoded = scoped_decoded_output_buffer(FAMILY_PREFIX, decoded);
    let scan = tiled_decode_aho_scan_body(
        transitions,
        accept,
        matches,
        Expr::var("len"),
        tile_width,
        |index| inflate_stored_payload_expr(&input, index),
        |index, value| Some(Node::store(&decoded, index, value)),
    );
    let mut entry = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        vec![Node::store(INFLATED_LEN_BUFFER, Expr::u32(0), Expr::u32(0))],
    )];
    entry.extend(inflate_stored_header_nodes(&input));
    entry.extend([Node::if_then(
        Expr::eq(Expr::var("btype"), Expr::u32(0)),
        vec![
            Node::if_then(inflate_stored_len_is_valid_expr(), {
                let mut body = vec![Node::if_then(
                    Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                    vec![Node::store(
                        INFLATED_LEN_BUFFER,
                        Expr::u32(0),
                        Expr::var("len"),
                    )],
                )];
                body.extend(scan);
                body
            }),
            inflate_stored_invalid_len_trap_node(),
        ],
    )]);
    entry.extend(inflate_stored_non_stored_trap_nodes());
    Program::wrapped(
        vec![
            BufferDecl::storage(&input, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(input_len),
            BufferDecl::read_write(&decoded, 1, DataType::U32).with_count(input_len),
            BufferDecl::storage(transitions, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count.saturating_mul(256)),
            BufferDecl::storage(accept, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(state_count),
            BufferDecl::output(matches, 4, DataType::U32).with_count(input_len),
            BufferDecl::read_write(INFLATED_LEN_BUFFER, 5, DataType::U32).with_count(1),
        ],
        INFLATE_STORED_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(FUSED_SCAN_OP_ID, entry)],
    )
}
#[cfg(test)]
fn cpu_ref(input: &[u8]) -> Result<(Vec<u32>, u32), String> {
    inflate_stored_reference_bytes(input)
        .map(|result| (result.data, result.inflated_len))
        .map_err(str::to_string)
}

fn fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        pack_words(&[
            0x01,
            0x05,
            0x00,
            0xFA,
            0xFF,
            u32::from(b'h'),
            u32::from(b'e'),
            u32::from(b'l'),
            u32::from(b'l'),
            u32::from(b'o'),
        ]),
        vec![0u8; 4],
    ]]
}

fn fixture_outputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        pack_words(&[
            u32::from(b'h'),
            u32::from(b'e'),
            u32::from(b'l'),
            u32::from(b'l'),
            u32::from(b'o'),
            0,
            0,
            0,
            0,
            0,
        ]),
        pack_words(&[5]),
    ]]
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || inflate_stored_block("input", "output", 10),
        Some(fixture_inputs),
        Some(fixture_outputs),
    )
}
// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// Result of a CPU stored-block inflate.
#[derive(Debug, PartialEq, Eq)]
pub struct CpuInflateResult {
    /// Inflated data bytes (one per u32 slot, low 8 bits).
    pub data: Vec<u32>,
    /// Number of data bytes inflated.
    pub inflated_len: u32,
}

/// CPU reference: inflate a DEFLATE stored block (BTYPE=0).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn inflate_stored_reference_bytes(input: &[u8]) -> Result<CpuInflateResult, &'static str> {
    if input.len() < 5 {
        return Err(STORED_HEADER_FIX);
    }
    let btype = (input[0] >> 1) & 0x3;
    match btype {
        0 => {
            let len = u16::from_le_bytes([input[1], input[2]]);
            let nlen = u16::from_le_bytes([input[3], input[4]]);
            if nlen != !len {
                return Err(STORED_HEADER_FIX);
            }
            let len_usize = usize::from(len);
            let header_words = INFLATE_STORED_HEADER_WORDS as usize;
            if input.len() < header_words + len_usize {
                return Err(STORED_HEADER_FIX);
            }
            Ok(CpuInflateResult {
                data: input[header_words..][..len_usize]
                    .iter()
                    .map(|&byte| u32::from(byte))
                    .collect(),
                inflated_len: u32::from(len),
            })
        }
        1 => Err(FIXED_HUFFMAN_REJECT),
        2 => Err(DYNAMIC_HUFFMAN_REJECT),
        _ => Err(RESERVED_BTYPE_FIX),
    }
}

/// CPU reference over one-byte-per-u32 packed lanes.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn inflate_stored_reference_words(input: &[u32]) -> Result<CpuInflateResult, &'static str> {
    let bytes = input
        .iter()
        .map(|word| (word & 0xFF) as u8)
        .collect::<Vec<_>>();
    inflate_stored_reference_bytes(&bytes)
}

/// Compatibility CPU reference: inflate a DEFLATE stored block (BTYPE=0).
///
/// Input words: `[header, len_lo, len_hi, nlen_lo, nlen_hi, data0, data1, ...]`.
/// Each word carries one byte in its low 8 bits. Returns the decoded data
/// and the inflated length. Returns `None` if the input is not a valid
/// stored block (wrong BTYPE or LEN/NLEN mismatch).
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_inflate_stored(input: &[u32]) -> Option<CpuInflateResult> {
    inflate_stored_reference_words(input).ok()
}
#[cfg(test)]
mod primitive_tests {
    use super::*;

    #[test]
    fn inflate_stored_hello() {
        // Matches the inventory test case exactly.
        let input = [
            0x01, // BFINAL=1, BTYPE=00
            0x05,
            0x00, // LEN = 5
            0xFA,
            0xFF, // NLEN = 0xFFFA (= 5 ^ 0xFFFF)
            b'h' as u32,
            b'e' as u32,
            b'l' as u32,
            b'l' as u32,
            b'o' as u32,
        ];
        let result = cpu_inflate_stored(&input).unwrap();
        assert_eq!(result.inflated_len, 5);
        assert_eq!(
            result.data,
            vec![
                b'h' as u32,
                b'e' as u32,
                b'l' as u32,
                b'l' as u32,
                b'o' as u32
            ]
        );
    }

    #[test]
    fn inflate_stored_empty_block() {
        let input = [
            0x01, // BFINAL=1, BTYPE=00
            0x00, 0x00, // LEN = 0
            0xFF, 0xFF, // NLEN = 0xFFFF
        ];
        let result = cpu_inflate_stored(&input).unwrap();
        assert_eq!(result.inflated_len, 0);
        assert!(result.data.is_empty());
    }

    #[test]
    fn inflate_stored_rejects_fixed_huffman() {
        let input = [
            0x03, // BFINAL=1, BTYPE=01 (fixed Huffman)
            0x00, 0x00, 0xFF, 0xFF,
        ];
        assert!(cpu_inflate_stored(&input).is_none());
    }

    #[test]
    fn inflate_stored_rejects_len_nlen_mismatch() {
        let input = [
            0x01, // BFINAL=1, BTYPE=00
            0x05,
            0x00, // LEN = 5
            0x00,
            0x00, // NLEN = 0 (wrong!)
            b'x' as u32,
            b'x' as u32,
            b'x' as u32,
            b'x' as u32,
            b'x' as u32,
        ];
        assert!(cpu_inflate_stored(&input).is_none());
    }

    #[test]
    fn inflate_stored_rejects_truncated_payload() {
        let input = [0x01, 0x05, 0x00, 0xFA, 0xFF, b'h' as u32, b'e' as u32];
        assert_eq!(
            inflate_stored_reference_words(&input),
            Err(STORED_HEADER_FIX)
        );
        assert!(cpu_inflate_stored(&input).is_none());
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{dfa_compile, CompiledDfa};
    use vyre_reference::value::Value;

    fn run(input: &[u8]) -> (Vec<u32>, u32) {
        let program = inflate_stored_block("input", "output", input.len() as u32);
        let inputs = vec![
            Value::from(pack_words(
                &input
                    .iter()
                    .map(|&byte| u32::from(byte))
                    .collect::<Vec<_>>(),
            )),
            Value::from(vec![0u8; input.len() * 4]),
            Value::from(vec![0u8; 4]),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: inflate must run; restore this invariant before continuing.");
        let decoded = vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes());
        let len_bytes = outputs[1].to_bytes();
        let decoded_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]);
        (decoded, decoded_len)
    }

    #[test]
    fn stored_block_decodes_without_host_roundtrip() {
        let (decoded, decoded_len) =
            run(&[0x01, 0x05, 0x00, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o']);
        assert_eq!(&decoded[..5], &[104, 101, 108, 108, 111]);
        assert_eq!(decoded_len, 5);
    }

    #[test]
    fn cpu_reference_names_fixed_huffman_gap() {
        let err = cpu_ref(&[0x03, 0, 0, 0, 0]).expect_err("BTYPE=1 must reject");
        assert_eq!(err, FIXED_HUFFMAN_REJECT);
    }

    #[test]
    fn cpu_reference_names_dynamic_huffman_gap() {
        let err = cpu_ref(&[0x05, 0, 0, 0, 0]).expect_err("BTYPE=2 must reject");
        assert_eq!(err, DYNAMIC_HUFFMAN_REJECT);
    }

    #[test]
    #[cfg(feature = "matching-dfa")]
    fn fused_stored_block_matches_parity_with_separate_inflate_then_aho() {
        let patterns: [&[u8]; 1] = [b"ell"];
        let compiled = dfa_compile(&patterns);
        let input_len = 10u32;

        let stored_block = {
            let payload = b"hello";
            let len = payload.len() as u16;
            let nlen = !len;
            [
                &[0x01u8][..],
                &len.to_le_bytes(),
                &nlen.to_le_bytes(),
                payload.as_slice(),
            ]
            .concat()
        };

        // --- Fused run ---
        let fused_program = inflate_stored_block_then_aho_corasick(
            "input",
            "decoded",
            "transitions",
            "accept",
            "matches",
            input_len,
            compiled.state_count,
        );
        let fused_inputs = vec![
            Value::from(pack_words(
                &stored_block
                    .iter()
                    .map(|&b| u32::from(b))
                    .collect::<Vec<_>>(),
            )),
            Value::from(vec![0u8; input_len as usize * 4]),
            Value::from(pack_words(&compiled.transitions)),
            Value::from(pack_words(&compiled.accept)),
            Value::from(vec![0u8; input_len as usize * 4]),
            Value::from(vec![0u8; 4]),
        ];
        let fused_outputs = vyre_reference::reference_eval(&fused_program, &fused_inputs)
            .expect("Fix: fused must run; restore this invariant before continuing.");
        let fused_matches =
            vyre_primitives::wire::decode_u32_le_bytes_all(&fused_outputs[1].to_bytes());

        // --- Separate inflate ---
        let inflate_program = inflate_stored_block("input", "output", input_len);
        let inflate_inputs = vec![
            Value::from(pack_words(
                &stored_block
                    .iter()
                    .map(|&b| u32::from(b))
                    .collect::<Vec<_>>(),
            )),
            Value::from(vec![0u8; input_len as usize * 4]),
            Value::from(vec![0u8; 4]),
        ];
        let inflate_outputs = vyre_reference::reference_eval(&inflate_program, &inflate_inputs)
            .expect("Fix: inflate must run; restore this invariant before continuing.");
        let decoded_bytes = inflate_outputs[0].to_bytes();
        let len_bytes = inflate_outputs[1].to_bytes();
        let decoded_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]);

        // --- Separate aho ---
        let aho_program = crate::scan::aho_corasick(
            "haystack",
            "transitions",
            "accept",
            "matches",
            decoded_len,
            compiled.state_count,
        );
        let aho_inputs = vec![
            Value::from(decoded_bytes[..decoded_len as usize * 4].to_vec()),
            Value::from(pack_words(&compiled.transitions)),
            Value::from(pack_words(&compiled.accept)),
            Value::from(vec![0u8; decoded_len as usize * 4]),
        ];
        let aho_outputs = vyre_reference::reference_eval(&aho_program, &aho_inputs)
            .expect("Fix: aho must run; restore this invariant before continuing.");
        let separate_matches =
            vyre_primitives::wire::decode_u32_le_bytes_all(&aho_outputs[0].to_bytes());

        assert_eq!(
            &fused_matches[..decoded_len as usize],
            &separate_matches[..]
        );
        for &m in &fused_matches[decoded_len as usize..] {
            assert_eq!(m, 0);
        }
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
        let program = inflate_stored_block_then_aho_corasick(
            "input",
            "decoded",
            "transitions",
            "accept",
            "matches",
            10,
            dfa.state_count,
        );
        assert_eq!(
            program.buffers()[1].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[4].name(), "matches");
        assert_eq!(program.buffers()[5].name(), INFLATED_LEN_BUFFER);
    }

    #[test]
    fn generic_default_names_are_family_scoped() {
        let program = inflate_stored_block("input", "decoded", 10);
        assert_eq!(
            program.buffers()[0].name(),
            fixed_name(FAMILY_PREFIX, "input")
        );
        assert_eq!(
            program.buffers()[1].name(),
            fixed_name(FAMILY_PREFIX, "decoded")
        );
        assert_eq!(program.buffers()[2].name(), INFLATED_LEN_BUFFER);
    }

    #[test]
    fn generated_stored_blocks_match_cpu_reference_and_clear_length_once() {
        for seed in 0u32..2048 {
            let len = (seed % 65) as usize;
            let mut state = seed ^ 0x1F1A_7E55;
            let mut payload = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                payload.push((state >> 16) as u8);
            }
            let len16 = len as u16;
            let nlen = !len16;
            let mut input = Vec::with_capacity(5 + payload.len());
            input.push(0x01);
            input.extend_from_slice(&len16.to_le_bytes());
            input.extend_from_slice(&nlen.to_le_bytes());
            input.extend_from_slice(&payload);

            let (actual, actual_len) = run(&input);
            let (expected, expected_len) = cpu_ref(&input).unwrap_or_else(|error| {
                panic!("generated stored block rejected seed {seed}: {error}")
            });
            assert_eq!(actual_len, expected_len, "inflated length seed {seed}");
            assert_eq!(
                &actual[..expected_len as usize],
                expected.as_slice(),
                "payload seed {seed}"
            );
            assert!(
                actual[expected_len as usize..]
                    .iter()
                    .all(|&word| word == 0),
                "stored inflate must not dirty output tail at seed {seed}"
            );
        }
    }

    #[test]
    fn generated_non_stored_and_corrupt_headers_report_canonical_reasons() {
        for seed in 0u32..2048 {
            let mut input = vec![0x01, 0x04, 0x00, 0xFB, 0xFF, b't', b'e', b's', b't'];
            match seed % 4 {
                0 => input[0] = 0x03,
                1 => input[0] = 0x05,
                2 => input[0] = 0x07,
                _ => input[3] ^= 0x5A,
            }
            let err = match cpu_ref(&input) {
                Ok(_) => panic!("generated corrupt block accepted seed {seed}"),
                Err(error) => error,
            };
            let expected = match seed % 4 {
                0 => FIXED_HUFFMAN_REJECT,
                1 => DYNAMIC_HUFFMAN_REJECT,
                2 => RESERVED_BTYPE_FIX,
                _ => STORED_HEADER_FIX,
            };
            assert_eq!(err, expected, "canonical reject reason seed {seed}");
        }
    }
}
