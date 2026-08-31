//! Succinct bitvector metadata primitives.
//!
//! These ops build the rank side of rank/select navigation for compact token,
//! AST, and graph bitvectors. They keep hot navigation state as packed `u32`
//! words plus sparse superblock counters, so GPU kernels trade bandwidth-heavy
//! pointer chasing for popcount math over coalesced words.

use core::fmt;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, PORTABLE_WORKGROUP_INVOCATIONS,
};

use crate::builder::cooperative::chunks;
use crate::reduce::workgroup_scan::blelloch_inclusive_sum_nodes;

const RANK_SUPERBLOCKS_OP_ID: &str = "vyre-libs::math::succinct::rank1_superblocks";
const RANK_QUERY_OP_ID: &str = "vyre-libs::math::succinct::rank1_query";
/// Phase boundary naming the per-block popcount each lane stages for the scan.
/// It is a phase of this operation and not an operation of its own, so it
/// carries the anonymous prefix instead of borrowing an unrelated op id.
const BLOCK_POPCOUNT_OP_ID: &str = "anonymous::vyre-libs::math::succinct::rank1_block_popcount";
/// Per-lane superblock popcount, and the inclusive scan over it.
const RANK_BLOCK_SCRATCH: &str = "__rank1_block_scratch";
/// The staged addend the Blelloch sweep keeps so its result reads inclusive.
const RANK_SCAN_SCRATCH: &str = "__rank1_scan_scratch";
/// Set bits in every chunk of blocks the workgroup has already scanned.
const RANK_CARRY: &str = "__rank1_carry";

/// Build-time errors for succinct bitvector Programs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SuccinctBuildError {
    /// Superblock size must be non-zero.
    ZeroBlockWords,
    /// The derived superblock output length overflowed `u32`.
    SuperblockCountOverflow,
}

impl fmt::Display for SuccinctBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBlockWords => {
                write!(f, "Fix: rank superblock size must be at least one u32 word")
            }
            Self::SuperblockCountOverflow => write!(
                f,
                "Fix: rank superblock count overflowed u32; shard the bitvector"
            ),
        }
    }
}

impl std::error::Error for SuccinctBuildError {}

fn superblock_count(word_count: u32, block_words: u32) -> Result<u32, SuccinctBuildError> {
    if block_words == 0 {
        return Err(SuccinctBuildError::ZeroBlockWords);
    }
    let full_blocks = word_count / block_words;
    let has_partial = u32::from(word_count % block_words != 0);
    full_blocks
        .checked_add(has_partial)
        .and_then(|blocks| blocks.checked_add(1))
        .ok_or(SuccinctBuildError::SuperblockCountOverflow)
}

/// Build sparse rank1 superblocks for a packed u32 bitvector.
///
/// `superblocks[0]` is always zero. Each following entry stores the cumulative
/// count of set bits before that superblock. The final sentinel stores the
/// total popcount for the whole bitvector.
#[must_use]
pub fn rank1_superblocks(
    bits: &str,
    superblocks: &str,
    word_count: u32,
    block_words: u32,
) -> Program {
    try_rank1_superblocks(bits, superblocks, word_count, block_words).unwrap_or_else(|err| {
        trap_program(
            RANK_SUPERBLOCKS_OP_ID,
            Some((superblocks, DataType::U32)),
            format!("{err}"),
        )
    })
}

/// Checked builder for [`rank1_superblocks`].
///
/// # Errors
///
/// Returns [`SuccinctBuildError`] when `block_words` is zero or the derived
/// metadata length overflows `u32`.
pub fn try_rank1_superblocks(
    bits: &str,
    superblocks: &str,
    word_count: u32,
    block_words: u32,
) -> Result<Program, SuccinctBuildError> {
    let out_count = superblock_count(word_count, block_words)?;
    // One lane per superblock, so the prefix scan this metadata is runs across
    // the workgroup. The block count decides the width because the scan is over
    // blocks: a bitvector with four of them has no work for 252 more lanes, and
    // the Blelloch sweep needs a power of two to walk a balanced tree.
    let tile = block_count(out_count)
        .max(1)
        .next_power_of_two()
        .min(PORTABLE_WORKGROUP_INVOCATIONS);
    let blocks = block_count(out_count);
    let local = Expr::var("local");
    let block = Expr::var("rank_block");
    let carry = Expr::load(RANK_CARRY, Expr::u32(0));
    let inclusive = Expr::load(RANK_BLOCK_SCRATCH, local.clone());
    let staged = Expr::load(RANK_SCAN_SCRATCH, local.clone());

    let mut chunk = vec![
        Node::let_bind(
            "rank_block",
            Expr::add(
                Expr::mul(Expr::var("rank_chunk"), Expr::u32(tile)),
                local.clone(),
            ),
        ),
        wrap_child_region(
            BLOCK_POPCOUNT_OP_ID,
            Ident::from(RANK_SUPERBLOCKS_OP_ID),
            vec![
                Node::let_bind("rank_block_pop", Expr::u32(0)),
                Node::if_then(
                    Expr::lt(block.clone(), Expr::u32(blocks)),
                    vec![Node::loop_for(
                        "rank_block_word",
                        Expr::u32(0),
                        Expr::u32(block_words),
                        vec![
                            Node::let_bind(
                                "rank_word",
                                Expr::add(
                                    Expr::mul(block.clone(), Expr::u32(block_words)),
                                    Expr::var("rank_block_word"),
                                ),
                            ),
                            // The last block is partial whenever `block_words`
                            // does not divide the word count.
                            Node::if_then(
                                Expr::lt(Expr::var("rank_word"), Expr::u32(word_count)),
                                vec![Node::assign(
                                    "rank_block_pop",
                                    Expr::add(
                                        Expr::var("rank_block_pop"),
                                        Expr::popcount(Expr::load(bits, Expr::var("rank_word"))),
                                    ),
                                )],
                            ),
                        ],
                    )],
                ),
                Node::store(
                    RANK_BLOCK_SCRATCH,
                    local.clone(),
                    Expr::var("rank_block_pop"),
                ),
            ],
        ),
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
    ];
    chunk.extend(blelloch_inclusive_sum_nodes(
        RANK_BLOCK_SCRATCH,
        RANK_SCAN_SCRATCH,
        &local,
        tile,
    ));
    // A superblock holds the count before its own block, which is the inclusive
    // scan less this lane's own staged popcount, offset by every earlier chunk.
    chunk.push(Node::if_then(
        Expr::and(
            Expr::is_first_logical_tile(),
            Expr::lt(block.clone(), Expr::u32(blocks)),
        ),
        vec![Node::store(
            superblocks,
            block.clone(),
            Expr::add(carry.clone(), Expr::sub(inclusive.clone(), staged)),
        )],
    ));
    // The carry advances only once every lane has read it, and the next chunk
    // stages over the scan scratch only once the carry has read it.
    chunk.push(Node::logical_barrier(
        vyre_foundation::ir::MemoryOrdering::SeqCst,
    ));
    chunk.push(Node::if_then(
        Expr::eq(local.clone(), Expr::u32(tile - 1)),
        vec![Node::store(
            RANK_CARRY,
            Expr::u32(0),
            Expr::add(carry.clone(), inclusive),
        )],
    ));
    chunk.push(Node::logical_barrier(
        vyre_foundation::ir::MemoryOrdering::SeqCst,
    ));

    let body = vec![
        Node::let_bind("local", Expr::LogicalWithinTileId { axis: 0 }),
        Node::if_then(
            Expr::eq(local.clone(), Expr::u32(0)),
            vec![Node::store(RANK_CARRY, Expr::u32(0), Expr::u32(0))],
        ),
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        Node::loop_for(
            "rank_chunk",
            Expr::u32(0),
            Expr::u32(chunks(blocks, tile)),
            chunk,
        ),
        Node::if_then(
            Expr::and(
                Expr::is_first_logical_tile(),
                Expr::eq(local.clone(), Expr::u32(0)),
            ),
            vec![Node::store(superblocks, Expr::u32(out_count - 1), carry)],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(bits, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(word_count.max(1)),
            BufferDecl::output(superblocks, 1, DataType::U32).with_count(out_count),
            BufferDecl::workgroup(RANK_BLOCK_SCRATCH, tile, DataType::U32),
            BufferDecl::workgroup(RANK_SCAN_SCRATCH, tile, DataType::U32),
            BufferDecl::workgroup(RANK_CARRY, 1, DataType::U32),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(RANK_SUPERBLOCKS_OP_ID, body)],
    ))
}

/// Superblocks a bitvector has, from the metadata length its sentinel closes.
const fn block_count(out_count: u32) -> u32 {
    out_count.saturating_sub(1)
}

/// Answer rank1-before-position queries from sparse superblocks.
///
/// Each `bit_indices[q]` is a zero-based bit offset. The output is the number
/// of set bits strictly before that offset. Query offsets must address an
/// existing packed word; use the final superblock sentinel for total popcount.
#[must_use]
pub fn rank1_query(
    bits: &str,
    superblocks: &str,
    bit_indices: &str,
    out: &str,
    word_count: u32,
    query_count: u32,
    block_words: u32,
) -> Program {
    try_rank1_query(
        bits,
        superblocks,
        bit_indices,
        out,
        word_count,
        query_count,
        block_words,
    )
    .unwrap_or_else(|err| {
        trap_program(
            RANK_QUERY_OP_ID,
            Some((out, DataType::U32)),
            format!("{err}"),
        )
    })
}

/// Checked builder for [`rank1_query`].
///
/// # Errors
///
/// Returns [`SuccinctBuildError`] when `block_words` is zero or the derived
/// metadata length overflows `u32`.
pub fn try_rank1_query(
    bits: &str,
    superblocks: &str,
    bit_indices: &str,
    out: &str,
    word_count: u32,
    query_count: u32,
    block_words: u32,
) -> Result<Program, SuccinctBuildError> {
    let sb_count = superblock_count(word_count, block_words)?;
    let q = Expr::LogicalIndex { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(q.clone(), Expr::u32(query_count)),
        vec![
            Node::let_bind("bit_index", Expr::load(bit_indices, q.clone())),
            Node::let_bind(
                "word_index",
                Expr::div(Expr::var("bit_index"), Expr::u32(32)),
            ),
            Node::if_then(
                Expr::ge(Expr::var("word_index"), Expr::u32(word_count)),
                vec![Node::trap(
                    Expr::var("bit_index"),
                    "rank-query-out-of-bounds",
                )],
            ),
            Node::let_bind(
                "block_index",
                Expr::div(Expr::var("word_index"), Expr::u32(block_words)),
            ),
            Node::let_bind(
                "rank_acc",
                Expr::load(superblocks, Expr::var("block_index")),
            ),
            Node::let_bind(
                "block_start_word",
                Expr::mul(Expr::var("block_index"), Expr::u32(block_words)),
            ),
            Node::loop_for(
                "rank_word",
                Expr::var("block_start_word"),
                Expr::var("word_index"),
                vec![Node::assign(
                    "rank_acc",
                    Expr::add(
                        Expr::var("rank_acc"),
                        Expr::popcount(Expr::load(bits, Expr::var("rank_word"))),
                    ),
                )],
            ),
            Node::let_bind(
                "bit_offset",
                Expr::rem(Expr::var("bit_index"), Expr::u32(32)),
            ),
            Node::let_bind(
                "partial_mask",
                Expr::select(
                    Expr::eq(Expr::var("bit_offset"), Expr::u32(0)),
                    Expr::u32(0),
                    Expr::sub(
                        Expr::shl(Expr::u32(1), Expr::var("bit_offset")),
                        Expr::u32(1),
                    ),
                ),
            ),
            Node::assign(
                "rank_acc",
                Expr::add(
                    Expr::var("rank_acc"),
                    Expr::popcount(Expr::bitand(
                        Expr::load(bits, Expr::var("word_index")),
                        Expr::var("partial_mask"),
                    )),
                ),
            ),
            Node::store(out, q, Expr::var("rank_acc")),
        ],
    )];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(bits, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(word_count.max(1)),
            BufferDecl::storage(superblocks, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(sb_count),
            BufferDecl::storage(bit_indices, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(query_count.max(1)),
            BufferDecl::output(out, 3, DataType::U32).with_count(query_count.max(1)),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(RANK_QUERY_OP_ID, body)],
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        RANK_SUPERBLOCKS_OP_ID,
        || rank1_superblocks("bits", "superblocks", 4, 2),
        Some(|| {
            let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![to_bytes(&bits)]]
        }),
        Some(|| {
            // [0, 4, 20]
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x00, // 0
                0x04, 0x00, 0x00, 0x00, // 4
                0x14, 0x00, 0x00, 0x00, // 20
            ]]]
        }),
    )
    .with_category("math")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        RANK_QUERY_OP_ID,
        || rank1_query("bits", "superblocks", "queries", "out", 4, 5, 2),
        Some(|| {
            let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
            let superblocks = [0u32, 4, 20];
            let queries = [0u32, 1, 4, 63, 80];
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![to_bytes(&bits), to_bytes(&superblocks), to_bytes(&queries)]]
        }),
        Some(|| {
            // [0, 1, 3, 3, 4]
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x00, // 0
                0x01, 0x00, 0x00, 0x00, // 1
                0x03, 0x00, 0x00, 0x00, // 3
                0x03, 0x00, 0x00, 0x00, // 3
                0x04, 0x00, 0x00, 0x00, // 4
            ]]]
        }),
    )
    .with_category("math")
}
