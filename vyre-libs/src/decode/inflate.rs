//! DEFLATE stored-block inflate: one module, one op id.
//!
//! The kernel body, the header decode nodes, the CPU reference oracle, the
//! public builder and the fused inflate-then-scan builder all live here. The op
//! id is `vyre-libs::decode::inflate_stored_block`.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::decode::buffers::{scoped_decode_input_buffer, scoped_decoded_output_buffer};
use crate::decode::scan::tiled_decode_aho_scan_body;
use vyre_primitives::wire::pack_u32_slice as pack_words;

/// Canonical op id for stored-block inflate.
pub const OP_ID: &str = "vyre-libs::decode::inflate_stored_block";
const FUSED_SCAN_OP_ID: &str = "vyre-libs::decode::inflate_stored_block_then_aho_corasick";
pub(super) const FAMILY_PREFIX: &str = "decode_inflate";
pub(super) const INFLATED_LEN_BUFFER: &str = "__vyre_decode_inflate_inflated_len";
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
        Ident::from(parent_op_id),
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
pub(super) fn reference_inflate_stored_bytes(input: &[u8]) -> Result<(Vec<u32>, u32), String> {
    let words: Vec<u32> = input.iter().map(|&b| u32::from(b)).collect();
    vyre_reference::composition_witness::inflate_stored_witness(&words)
        .map(|result| (result.data, result.inflated_len))
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

const EXPECTED_INFLATE_DATA_BYTES: [u8; 40] = [
    104, 0, 0, 0, 101, 0, 0, 0, 108, 0, 0, 0, 108, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const EXPECTED_INFLATE_LEN_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || inflate_stored_block("input", "output", 10),
        Some(fixture_inputs),
        Some(|| {
            vec![vec![
                EXPECTED_INFLATE_DATA_BYTES.to_vec(),
                EXPECTED_INFLATE_LEN_BYTES.to_vec(),
            ]]
        }),
    )
}
