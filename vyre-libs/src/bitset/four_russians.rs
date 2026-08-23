//! Method-of-Four-Russians byte-tile lookup for packed boolean words.
//!
//! The primitive maps each `(lhs_byte, rhs_byte)` pair through a 65,536-entry
//! table and assembles four looked-up bytes back into one `u32`. Higher-level
//! boolean-matrix and reachability kernels can specialize the LUT once, then
//! replace branchy byte logic with coalesced table loads.

#[cfg(test)]
use std::sync::LazyLock;
use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::bitset::four_russians_apply_byte_lut";
/// Canonical op id for dense boolean matvec over byte-frontier tiles.
pub const DENSE_MATVEC_OP_ID: &str = "vyre-libs::bitset::four_russians_dense_matvec_byte_lut";
/// Number of possible active-source subsets in one byte tile.
pub const BYTE_TILE_STATES: u32 = 256;
/// Number of source columns summarized by one byte tile.
pub const BYTE_TILE_WIDTH: u32 = 8;

/// Binary boolean operation encoded into a byte-pair LUT.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BooleanTileOp {
    /// `lhs & rhs`
    And,
    /// `lhs | rhs`
    Or,
    /// `lhs ^ rhs`
    Xor,
    /// `lhs & !rhs`
    AndNot,
}

impl BooleanTileOp {
    const fn apply(self, lhs: u8, rhs: u8) -> u8 {
        match self {
            Self::And => lhs & rhs,
            Self::Or => lhs | rhs,
            Self::Xor => lhs ^ rhs,
            Self::AndNot => lhs & !rhs,
        }
    }
}

/// Build a 65,536-entry LUT indexed by `(lhs_byte << 8) | rhs_byte`.
#[must_use]
pub fn binary_byte_lut(op: BooleanTileOp) -> Vec<u32> {
    let mut table = vec![0u32; 256 * 256];
    for lhs in 0u32..=255 {
        for rhs in 0u32..=255 {
            let idx = ((lhs << 8) | rhs) as usize;
            table[idx] = u32::from(op.apply(lhs as u8, rhs as u8));
        }
    }
    table
}

/// Reuse a process-wide LUT for the standard Boolean byte-tile operations.
///
/// The Method-of-Four-Russians table is 65,536 `u32`s. Rebuilding it for every
/// rule batch or graph shard is pure allocator and cache churn, so common
/// operations share one immutable table per process.
#[cfg(test)]
#[must_use]
pub fn cached_binary_byte_lut(op: BooleanTileOp) -> &'static [u32] {
    static AND: LazyLock<Vec<u32>> = LazyLock::new(|| binary_byte_lut(BooleanTileOp::And));
    static OR: LazyLock<Vec<u32>> = LazyLock::new(|| binary_byte_lut(BooleanTileOp::Or));
    static XOR: LazyLock<Vec<u32>> = LazyLock::new(|| binary_byte_lut(BooleanTileOp::Xor));
    static AND_NOT: LazyLock<Vec<u32>> = LazyLock::new(|| binary_byte_lut(BooleanTileOp::AndNot));

    match op {
        BooleanTileOp::And => AND.as_slice(),
        BooleanTileOp::Or => OR.as_slice(),
        BooleanTileOp::Xor => XOR.as_slice(),
        BooleanTileOp::AndNot => AND_NOT.as_slice(),
    }
}

/// Frontier words needed to address `tile_count` byte-tiles.
#[must_use]
pub const fn frontier_words_for_byte_tiles(tile_count: u32) -> u32 {
    tile_count.div_ceil(4)
}

/// Number of `u32` LUT entries for dense byte-tile boolean matvec.
#[must_use]
pub fn dense_matvec_byte_lut_words(tile_count: u32, dst_words: u32) -> u32 {
    tile_count
        .saturating_mul(BYTE_TILE_STATES)
        .saturating_mul(dst_words)
}

/// Build a dense boolean-matvec byte-tile LUT.
///
/// `columns` is column-major by source tile:
/// `columns[((tile * 8 + source_bit) * dst_words) + dst_word]` is the packed
/// destination bitset reached from that source column. The returned LUT is
/// indexed as `((tile * 256 + active_source_byte) * dst_words) + dst_word`.
/// At dispatch time each frontier byte becomes one coalesced LUT lookup per
/// destination word; graph/dataflow closure kernels replace eight branchy
/// source-column tests with one table read and one OR.
#[must_use]
pub fn dense_matvec_byte_lut(columns: &[u32], tile_count: u32, dst_words: u32) -> Vec<u32> {
    let mut lut = Vec::new();
    dense_matvec_byte_lut_into(columns, tile_count, dst_words, &mut lut);
    lut
}

/// Build a dense boolean-matvec byte-tile LUT into caller-owned storage.
///
/// # Panics
/// Panics on an invalid tile shape. Callers that must recover use the checked twin.
pub fn dense_matvec_byte_lut_into(
    columns: &[u32],
    tile_count: u32,
    dst_words: u32,
    lut: &mut Vec<u32>,
) {
    // Clearing the LUT to empty on failure silently produces an all-zero
    // boolean-matvec (every product reads zero) (a silent fallback (Law 10)).
    // Fail loud; callers use try_dense_matvec_byte_lut_into.
    if let Err(error) = try_dense_matvec_byte_lut_into(columns, tile_count, dst_words, lut) {
        panic!("vyre-primitives dense boolean-matvec LUT build failed: {error}");
    }
}

/// Fallibly build a dense boolean-matvec byte-tile LUT into caller-owned storage.
pub fn try_dense_matvec_byte_lut_into(
    columns: &[u32],
    tile_count: u32,
    dst_words: u32,
    lut: &mut Vec<u32>,
) -> Result<(), String> {
    let column_words = try_checked_dense_column_words(tile_count, dst_words)?;
    if columns.len() != column_words {
        return Err(format!(
            "dense Four-Russians LUT builder received {} column words, expected {column_words}. Fix: pass exactly tile_count * 8 * dst_words column-major words.",
            columns.len()
        ));
    }

    let tile_count = usize_from_u32(tile_count, "tile_count");
    let dst_words = usize_from_u32(dst_words, "dst_words");
    let lut_words = try_checked_dense_lut_words_usize(tile_count, dst_words)?;
    vyre_foundation::allocation::reserve_exact_cleared(lut, lut_words).map_err(|err| {
        format!("dense Four-Russians LUT builder could not reserve {lut_words} output words: {err}")
    })?;
    lut.resize(lut_words, 0);

    for tile in 0..tile_count {
        for active_byte in 0..BYTE_TILE_STATES as usize {
            for source_bit in 0..BYTE_TILE_WIDTH as usize {
                if (active_byte & (1usize << source_bit)) == 0 {
                    continue;
                }
                for dst_word in 0..dst_words {
                    let column_idx =
                        ((tile * BYTE_TILE_WIDTH as usize + source_bit) * dst_words) + dst_word;
                    let lut_idx =
                        ((tile * BYTE_TILE_STATES as usize + active_byte) * dst_words) + dst_word;
                    lut[lut_idx] |= columns[column_idx];
                }
            }
        }
    }
    Ok(())
}

/// Build a Program: `out[w] = lut[(lhs_byte << 8) | rhs_byte]` per byte lane.
#[must_use]
pub fn four_russians_apply_byte_lut(
    lhs: &str,
    rhs: &str,
    lut: &str,
    out: &str,
    words: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut body = vec![
        Node::let_bind("fr_lhs_word", Expr::load(lhs, t.clone())),
        Node::let_bind("fr_rhs_word", Expr::load(rhs, t.clone())),
        Node::let_bind("fr_out_word", Expr::u32(0)),
    ];
    body.push(Node::loop_for(
        "fr_byte_lane",
        Expr::u32(0),
        Expr::u32(4),
        vec![
            Node::let_bind(
                "fr_shift",
                Expr::mul(Expr::var("fr_byte_lane"), Expr::u32(8)),
            ),
            Node::let_bind(
                "fr_lhs_byte",
                Expr::bitand(
                    Expr::shr(Expr::var("fr_lhs_word"), Expr::var("fr_shift")),
                    Expr::u32(0xFF),
                ),
            ),
            Node::let_bind(
                "fr_rhs_byte",
                Expr::bitand(
                    Expr::shr(Expr::var("fr_rhs_word"), Expr::var("fr_shift")),
                    Expr::u32(0xFF),
                ),
            ),
            Node::let_bind(
                "fr_lut_idx",
                Expr::bitor(
                    Expr::shl(Expr::var("fr_lhs_byte"), Expr::u32(8)),
                    Expr::var("fr_rhs_byte"),
                ),
            ),
            Node::let_bind(
                "fr_byte_out",
                Expr::bitand(Expr::load(lut, Expr::var("fr_lut_idx")), Expr::u32(0xFF)),
            ),
            Node::assign(
                "fr_out_word",
                Expr::bitor(
                    Expr::var("fr_out_word"),
                    Expr::shl(Expr::var("fr_byte_out"), Expr::var("fr_shift")),
                ),
            ),
        ],
    ));
    body.push(Node::store(out, t.clone(), Expr::var("fr_out_word")));

    Program::wrapped(
        vec![
            BufferDecl::storage(lhs, 0, BufferAccess::ReadOnly, DataType::U32).with_count(words),
            BufferDecl::storage(rhs, 1, BufferAccess::ReadOnly, DataType::U32).with_count(words),
            BufferDecl::storage(lut, 2, BufferAccess::ReadOnly, DataType::U32).with_count(65_536),
            BufferDecl::storage(out, 3, BufferAccess::ReadWrite, DataType::U32).with_count(words),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(t.clone(), Expr::u32(words)), body)],
        )],
    )
}

/// Build a Program for dense boolean matvec using byte-tile Four Russians LUTs.
///
/// The Program computes:
///
/// `out[dst_word] = OR(tile_lut[tile][frontier_byte(tile)][dst_word])`
///
/// for every packed destination word. This is the GPU-oriented primitive used
/// when graph/dataflow frontiers are dense enough that CSR pointer chasing loses
/// to table-driven boolean-semiring matvec.
#[must_use]
pub fn four_russians_dense_matvec_byte_lut(
    frontier: &str,
    tile_lut: &str,
    out: &str,
    tile_count: u32,
    dst_words: u32,
) -> Program {
    let dst_word = Expr::InvocationId { axis: 0 };
    let tile_lut_words = dense_matvec_byte_lut_words(tile_count, dst_words);
    let frontier_words = frontier_words_for_byte_tiles(tile_count);
    let mut body = vec![Node::let_bind("fr_dense_acc", Expr::u32(0))];
    body.push(Node::loop_for(
        "fr_dense_tile",
        Expr::u32(0),
        Expr::u32(tile_count),
        vec![
            Node::let_bind(
                "fr_dense_frontier_word_idx",
                Expr::div(Expr::var("fr_dense_tile"), Expr::u32(4)),
            ),
            Node::let_bind(
                "fr_dense_frontier_shift",
                Expr::mul(
                    Expr::rem(Expr::var("fr_dense_tile"), Expr::u32(4)),
                    Expr::u32(8),
                ),
            ),
            Node::let_bind(
                "fr_dense_frontier_byte",
                Expr::bitand(
                    Expr::shr(
                        Expr::load(frontier, Expr::var("fr_dense_frontier_word_idx")),
                        Expr::var("fr_dense_frontier_shift"),
                    ),
                    Expr::u32(0xFF),
                ),
            ),
            Node::let_bind(
                "fr_dense_lut_idx",
                Expr::add(
                    Expr::mul(
                        Expr::add(
                            Expr::mul(Expr::var("fr_dense_tile"), Expr::u32(BYTE_TILE_STATES)),
                            Expr::var("fr_dense_frontier_byte"),
                        ),
                        Expr::u32(dst_words),
                    ),
                    dst_word.clone(),
                ),
            ),
            Node::assign(
                "fr_dense_acc",
                Expr::bitor(
                    Expr::var("fr_dense_acc"),
                    Expr::load(tile_lut, Expr::var("fr_dense_lut_idx")),
                ),
            ),
        ],
    ));
    body.push(Node::store(
        out,
        dst_word.clone(),
        Expr::var("fr_dense_acc"),
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(frontier, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(frontier_words),
            BufferDecl::storage(tile_lut, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(tile_lut_words),
            BufferDecl::storage(out, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(dst_words),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            DENSE_MATVEC_OP_ID,
            vec![Node::if_then(
                Expr::lt(dst_word.clone(), Expr::u32(dst_words)),
                body,
            )],
        )],
    )
}

fn try_checked_dense_column_words(tile_count: u32, dst_words: u32) -> Result<usize, String> {
    let tile_count = usize_from_u32(tile_count, "tile_count");
    let dst_words = usize_from_u32(dst_words, "dst_words");
    tile_count
        .checked_mul(BYTE_TILE_WIDTH as usize)
        .and_then(|words| words.checked_mul(dst_words))
        .ok_or_else(|| {
            "dense Four-Russians column table size overflowed usize. Fix: split the graph into smaller source/destination shards.".to_string()
        })
}

fn try_checked_dense_lut_words_usize(tile_count: usize, dst_words: usize) -> Result<usize, String> {
    tile_count
        .checked_mul(BYTE_TILE_STATES as usize)
        .and_then(|words| words.checked_mul(dst_words))
        .ok_or_else(|| {
            "dense Four-Russians LUT size overflowed usize. Fix: split the graph into smaller source/destination shards.".to_string()
        })
}

fn usize_from_u32(value: u32, field: &'static str) -> usize {
    let _ = field;
    usize::try_from(value).unwrap_or(usize::MAX)
}

const EXPECTED_FOUR_RUSSIANS_APPLY_OUTPUT_BYTES: [u8; 8] =
    [0x00, 0xF0, 0x00, 0xF0, 0x00, 0x00, 0x0F, 0x0F];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || four_russians_apply_byte_lut("lhs", "rhs", "lut", "out", 2),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0xFF00_FF00, 0x0F0F_0F0F]),
                to_bytes(&[0xF0F0_F0F0, 0xFFFF_0000]),
                to_bytes(&binary_byte_lut(BooleanTileOp::And)),
                to_bytes(&[0, 0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_FOUR_RUSSIANS_APPLY_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

const EXPECTED_FOUR_RUSSIANS_DENSE_MATVEC_OUTPUT_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        DENSE_MATVEC_OP_ID,
        || four_russians_dense_matvec_byte_lut("frontier", "tile_lut", "out", 1, 1),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            let columns = [0b0001_u32, 0b0010, 0b0100, 0b1000, 0b0001, 0b0010, 0b0100, 0b1000];
            let lut = dense_matvec_byte_lut(&columns, 1, 1);
            vec![vec![
                to_bytes(&[0b0000_0101]),
                to_bytes(&lut),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_FOUR_RUSSIANS_DENSE_MATVEC_OUTPUT_BYTES.to_vec()]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::{
        four_russians_binary_witness as reference_four_russians_binary,
        four_russians_binary_witness_into,
        four_russians_dense_matvec_witness as reference_four_russians_dense_matvec,
        four_russians_dense_matvec_witness_into,
    };
    fn try_reference_four_russians_binary_into(
        lhs: &[u32],
        rhs: &[u32],
        lut: &[u32],
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        if lhs.len() != rhs.len() {
            return Err("equal-width lhs and rhs required".to_string());
        }
        if lut.len() != 65_536 {
            return Err("complete 256x256 (65,536-entry) byte-LUT required".to_string());
        }
        four_russians_binary_witness_into(lhs, rhs, lut, out);
        Ok(())
    }

    fn try_reference_four_russians_dense_matvec_into(
        frontier: &[u32],
        tile_lut: &[u32],
        tile_count: u32,
        dst_words: u32,
        out: &mut Vec<u32>,
    ) -> Result<(), String> {
        let expected_frontier_words = tile_count.div_ceil(4) as usize;
        if frontier.len() < expected_frontier_words {
            return Err(format!(
                "insufficient frontier words: got {}, need at least {} (frontier_len)",
                frontier.len(),
                expected_frontier_words
            ));
        }
        let expected_lut_words = (tile_count as usize)
            .checked_mul(256)
            .and_then(|w| w.checked_mul(dst_words as usize))
            .ok_or_else(|| "overflow calculating expected lut words".to_string())?;
        if tile_lut.len() < expected_lut_words {
            return Err(format!(
                "insufficient lut words: got {}, need at least {} (lut_len)",
                tile_lut.len(),
                expected_lut_words
            ));
        }
        four_russians_dense_matvec_witness_into(frontier, tile_lut, tile_count, dst_words, out);
        Ok(())
    }

    #[test]
    fn byte_lut_matches_word_and() {
        let lhs = [0xFF00_FF00u32, 0x0F0F_0F0F];
        let rhs = [0xF0F0_F0F0u32, 0xFFFF_0000];
        let lut = binary_byte_lut(BooleanTileOp::And);
        assert_eq!(
            reference_four_russians_binary(&lhs, &rhs, &lut),
            vec![0xF000_F000, 0x0F0F_0000]
        );
    }

    #[test]
    fn dense_byte_tile_lut_matches_boolean_matvec() {
        let columns = [
            0b0001u32, 0b0010, 0b0100, 0b1000, 0b0001, 0b0010, 0b0100, 0b1000,
        ];
        let lut = dense_matvec_byte_lut(&columns, 1, 1);
        let frontier = [0b0000_0101u32];

        assert_eq!(
            reference_four_russians_dense_matvec(&frontier, &lut, 1, 1),
            vec![0b0101]
        );
    }

    #[test]
    fn dense_byte_tile_lut_into_reuses_output_and_rejects_bad_columns() {
        let columns = [
            0b0001u32, 0b0010, 0b0100, 0b1000, 0b0001, 0b0010, 0b0100, 0b1000,
        ];
        let mut lut = Vec::with_capacity(512);
        lut.extend_from_slice(&[u32::MAX; 512]);
        let ptr = lut.as_ptr();

        try_dense_matvec_byte_lut_into(&columns, 1, 1, &mut lut).unwrap();

        assert_eq!(lut, dense_matvec_byte_lut(&columns, 1, 1));
        assert_eq!(lut.as_ptr(), ptr);
        let before = lut.clone();
        assert!(
            try_dense_matvec_byte_lut_into(&columns[..7], 1, 1, &mut lut)
                .unwrap_err()
                .contains("column words")
        );
        assert_eq!(lut, before);
    }

    #[test]
    fn generated_byte_luts_match_scalar_boolean_ops() {
        for op in [
            BooleanTileOp::And,
            BooleanTileOp::Or,
            BooleanTileOp::Xor,
            BooleanTileOp::AndNot,
        ] {
            let lut = binary_byte_lut(op);
            assert_eq!(lut.len(), 65_536);
            for lhs in 0u32..=255 {
                for rhs in 0u32..=255 {
                    let idx = ((lhs << 8) | rhs) as usize;
                    assert_eq!(
                        lut[idx],
                        u32::from(op.apply(lhs as u8, rhs as u8)),
                        "op {op:?} lhs={lhs:#04x} rhs={rhs:#04x}"
                    );
                }
            }
        }
    }
    #[test]
    fn cached_luts_reuse_allocation_and_match_owned_tables() {
        let owned = binary_byte_lut(BooleanTileOp::And);
        let cached_first = cached_binary_byte_lut(BooleanTileOp::And);
        let cached_second = cached_binary_byte_lut(BooleanTileOp::And);

        assert_eq!(cached_first, owned.as_slice());
        assert_eq!(
            cached_first.as_ptr(),
            cached_second.as_ptr(),
            "standard Four-Russians LUTs must be process-cached instead of rebuilt per batch"
        );
    }

    #[test]
    fn try_reference_binary_into_reuses_output_and_rejects_bad_shapes() {
        let lhs = [0x0123_4567u32, 0x89ab_cdef];
        let rhs = [0xffff_0000u32, 0x1357_9bdf];
        let lut = binary_byte_lut(BooleanTileOp::Xor);
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&[u32::MAX; 8]);
        let ptr = out.as_ptr();

        try_reference_four_russians_binary_into(&lhs, &rhs, &lut, &mut out).unwrap();

        assert_eq!(out, reference_four_russians_binary(&lhs, &rhs, &lut));
        assert_eq!(out.as_ptr(), ptr);
        assert!(
            try_reference_four_russians_binary_into(&lhs, &rhs[..1], &lut, &mut out)
                .unwrap_err()
                .contains("equal-width")
        );
        assert!(
            try_reference_four_russians_binary_into(&lhs, &rhs, &lut[..1024], &mut out)
                .unwrap_err()
                .contains("complete 256x256")
        );
    }

    #[test]
    fn try_dense_matvec_reference_reuses_output_and_rejects_bad_shapes() {
        let tile_count = 2;
        let dst_words = 2;
        let mut columns =
            vec![0u32; try_checked_dense_column_words(tile_count, dst_words).unwrap()];
        for tile in 0..tile_count as usize {
            for source_bit in 0..BYTE_TILE_WIDTH as usize {
                for dst_word in 0..dst_words as usize {
                    let idx = ((tile * BYTE_TILE_WIDTH as usize + source_bit) * dst_words as usize)
                        + dst_word;
                    columns[idx] = ((tile as u32 + 1) << 24)
                        | ((source_bit as u32 + 1) << 8)
                        | dst_word as u32;
                }
            }
        }
        let lut = dense_matvec_byte_lut(&columns, tile_count, dst_words);
        let frontier = [0b0000_0101_0000_0011u32];
        let mut out = Vec::with_capacity(8);
        out.extend_from_slice(&[u32::MAX; 8]);
        let ptr = out.as_ptr();

        try_reference_four_russians_dense_matvec_into(
            &frontier, &lut, tile_count, dst_words, &mut out,
        )
        .unwrap();

        assert_eq!(
            out,
            reference_four_russians_dense_matvec(&frontier, &lut, tile_count, dst_words)
        );
        assert_eq!(out.as_ptr(), ptr);
        assert!(try_reference_four_russians_dense_matvec_into(
            &[],
            &lut,
            tile_count,
            dst_words,
            &mut out
        )
        .unwrap_err()
        .contains("frontier_len"));
        assert!(try_reference_four_russians_dense_matvec_into(
            &frontier,
            &lut[..lut.len() - 1],
            tile_count,
            dst_words,
            &mut out,
        )
        .unwrap_err()
        .contains("lut_len"));
    }
}
