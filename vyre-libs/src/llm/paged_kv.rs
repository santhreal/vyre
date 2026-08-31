//! Paged key-value cache addressing: one block table, two moves.
//!
//! A paged cache stores tokens in fixed-size physical blocks and gives each
//! sequence a table of block ids, so a sequence grows by claiming a block
//! instead of by owning a contiguous reservation the length of its context.
//! Both moves here are the same index map with the direction reversed: a
//! logical token becomes `(block table lookup, slot within the block)`, and the
//! element address on either side of that lookup is the row-major index the
//! attention layout base already owns.
//!
//! The base owns the guard, the buffer table, the region wrapper and the
//! arithmetic. This module supplies only the map, which is what makes a paged
//! read and a contiguous read comparable by reading them.

use thiserror::Error;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use crate::nn::attention::layout::{
    block_index, check_layout_dims, checked_elements, layout_move_program, IndexMap, LayoutMove,
    LayoutReject, RowMajor,
};

/// Canonical op id of the paged gather.
pub const PAGED_KV_GATHER_OP_ID: &str = "vyre-libs::llm::paged_kv_gather";
/// Canonical op id of the paged append.
pub const PAGED_KV_APPEND_OP_ID: &str = "vyre-libs::llm::paged_kv_append";

/// Invalid paged cache dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PagedKvError {
    /// A required dimension is zero.
    #[error(
        "paged KV addressing requires nonzero sequences, heads, blocks, block tokens, blocks per sequence, token count, and head dimension"
    )]
    EmptyShape,
    /// The addressed token range exceeds what the block table can name.
    #[error(
        "paged KV token range end={end} exceeds the {blocks_per_sequence} blocks of {block_tokens} tokens the block table names"
    )]
    Range {
        /// One past the last logical token addressed.
        end: u32,
        /// Block-table width, in blocks.
        blocks_per_sequence: u32,
        /// Tokens per physical block.
        block_tokens: u32,
    },
    /// Flattened element counts exceeded addressable IR indexing.
    #[error("paged KV element count overflows u32; shard the cache")]
    ElementCountOverflow,
    /// The cache dtype cannot represent floating attention activations.
    #[error("paged KV cache requires F16, BF16, or F32 elements; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected cache element dtype.
        dtype: DataType,
    },
}

/// Buffer names and shape of one paged key-value cache.
///
/// The cache is `[blocks, heads, block_tokens, head_dim]` and the block table
/// is `[sequences, blocks_per_sequence]`, holding the physical block id of each
/// logical block. Both moves address the same two tensors, so the shape is one
/// struct rather than two positional argument lists that could disagree.
pub struct PagedKvCache<'a> {
    /// Physical cache buffer.
    pub cache: &'a str,
    /// Block table buffer, `u32` physical block ids.
    pub block_table: &'a str,
    /// Sequences the block table describes.
    pub sequences: u32,
    /// Attention head count.
    pub heads: u32,
    /// Physical block count.
    pub blocks: u32,
    /// Tokens per physical block.
    pub block_tokens: u32,
    /// Block-table width, in blocks per sequence.
    pub blocks_per_sequence: u32,
    /// Per-head feature width.
    pub head_dim: u32,
    /// Element dtype of the cache and of every tensor moved through it.
    pub dtype: DataType,
}

impl PagedKvCache<'_> {
    /// Row-major axis lengths of the physical cache.
    fn physical(&self) -> RowMajor {
        RowMajor {
            mid: self.heads,
            row: self.block_tokens,
            width: self.head_dim,
        }
    }

    /// Row-major axis lengths of a `[sequences, heads, tokens, head_dim]`
    /// tensor on the contiguous side of a move.
    fn contiguous(&self, tokens: u32) -> RowMajor {
        RowMajor {
            mid: self.heads,
            row: tokens,
            width: self.head_dim,
        }
    }

    /// Reject the shapes no paged move can serve, and return the flat element
    /// counts of the cache, the block table, and a contiguous tensor of
    /// `tokens` tokens.
    fn counts(&self, tokens: u32, end: u32) -> Result<(u32, u32, u32), PagedKvError> {
        let dims = [
            self.sequences,
            self.heads,
            self.blocks,
            self.block_tokens,
            self.blocks_per_sequence,
            tokens,
            self.head_dim,
        ];
        check_layout_dims(&dims, &self.dtype).map_err(|reject| match reject {
            LayoutReject::EmptyShape => PagedKvError::EmptyShape,
            LayoutReject::UnsupportedDtype(dtype) => PagedKvError::UnsupportedDtype { dtype },
        })?;
        let addressable = self
            .blocks_per_sequence
            .checked_mul(self.block_tokens)
            .ok_or(PagedKvError::ElementCountOverflow)?;
        if end > addressable {
            return Err(PagedKvError::Range {
                end,
                blocks_per_sequence: self.blocks_per_sequence,
                block_tokens: self.block_tokens,
            });
        }
        let elements =
            |dims: &[u32]| checked_elements(dims).ok_or(PagedKvError::ElementCountOverflow);
        Ok((
            elements(&[self.blocks, self.heads, self.block_tokens, self.head_dim])?,
            elements(&[self.sequences, self.blocks_per_sequence])?,
            self.moved_elements(tokens)?,
        ))
    }

    /// Elements a move of `tokens` tokens per sequence touches.
    ///
    /// This is the guarded domain of both moves, so it is also the launch span
    /// the compiler reads back out of the emitted guard.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the count overflows `u32` indexing.
    fn moved_elements(&self, tokens: u32) -> Result<u32, PagedKvError> {
        checked_elements(&[self.sequences, self.heads, tokens, self.head_dim])
            .ok_or(PagedKvError::ElementCountOverflow)
    }

    /// Flat cache index of the element at logical `token` of `sequence`, head
    /// `head`, feature `column`.
    ///
    /// This is the whole of paging: the logical token splits into a block and a
    /// slot, the block table turns the logical block into a physical one, and
    /// the physical address is the ordinary row-major index from there.
    fn address(&self, sequence: Expr, head: Expr, token: &Expr, column: Expr) -> Expr {
        let logical_block = Expr::div(token.clone(), Expr::u32(self.block_tokens));
        let slot = Expr::rem(token.clone(), Expr::u32(self.block_tokens));
        let physical = Expr::load(
            self.block_table,
            block_index(sequence, self.blocks_per_sequence, logical_block),
        );
        self.physical().index(physical, head, slot, column)
    }
}

/// Gather `window_tokens` logical tokens per sequence out of the paged cache
/// into a contiguous `[sequences, heads, window_tokens, head_dim]` window.
///
/// The window is what an attention operation reads: attention addresses a
/// contiguous key-value tensor, and paging is a property of where those tokens
/// are stored rather than of the attention itself.
///
/// The block table is data, so its entries carry the same range precondition
/// documented on [`paged_kv_append`]: an entry at or past `blocks` reads past
/// the end of the cache buffer, and no guard here can bound it.
///
/// # Errors
///
/// Returns `Err` for a zero dimension, a non-float dtype, a window longer than
/// the block table can name, or a flattened element count that overflows `u32`
/// indexing.
pub fn paged_kv_gather(
    spec: &PagedKvCache<'_>,
    window: &str,
    window_tokens: u32,
) -> Result<Program, PagedKvError> {
    let (cache_count, table_count, window_count) = spec.counts(window_tokens, window_tokens)?;
    let index = Expr::var("index");
    let [sequence, head, token, column] = spec.contiguous(window_tokens).coords(&index);
    let source = spec.address(sequence, head, &token, column);
    Ok(layout_move_program(LayoutMove {
        op_id: PAGED_KV_GATHER_OP_ID,
        buffers: vec![
            BufferDecl::storage(spec.cache, 0, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(cache_count),
            BufferDecl::storage(spec.block_table, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(table_count),
            BufferDecl::output(window, 2, spec.dtype.clone()).with_count(window_count),
        ],
        write: window,
        count: window_count,
        map: IndexMap::Element {
            value: Expr::load(spec.cache, source),
        },
    }))
}

/// Write `chunk_tokens` tokens per sequence into the paged cache, starting at
/// logical token `position`.
///
/// The cache is read-write and is the only buffer this move touches, so a
/// decoded token costs `sequences * heads * chunk_tokens * head_dim` stores
/// rather than a copy of the whole cache generation. That is the difference
/// between the paged append and the contiguous
/// [`kv_cache_append`](crate::nn::attention::kv_cache_append): the contiguous
/// one produces a successor generation and keeps the prior one addressable,
/// which a paged cache does not need because a block is claimed rather than
/// reserved.
///
/// The block table this move reads must be injective over the chunk: the slot
/// one invocation computes belongs to that invocation alone. A paged cache
/// that maps two sequences onto one physical block, which is how a shared
/// prompt prefix is stored, has to copy that block before either sequence
/// appends into it, because both would otherwise store to the same address in
/// one dispatch. The emitted program cannot check this: the table is data, the
/// guard bounds the chunk rather than the cache, and there is no read of the
/// destination to compare against.
///
/// Range is the second precondition and a separate failure. Every entry of the
/// block table must name a physical block below `blocks`. The guard bounds the
/// chunk index, which decides how many invocations store, and the table lookup
/// then decides where; an entry at or past `blocks` addresses past the end of
/// the cache buffer. [`paged_kv_gather`] reads through the same lookup and has
/// the same requirement, with an out-of-range read in place of a store. Both
/// belong to whoever allocates blocks, because the table is an input here.
///
/// # Errors
///
/// Returns `Err` for a zero dimension, a non-float dtype, a chunk that ends
/// past what the block table can name, or a flattened element count that
/// overflows `u32` indexing.
pub fn paged_kv_append(
    spec: &PagedKvCache<'_>,
    chunk: &str,
    chunk_tokens: u32,
    position: u32,
) -> Result<Program, PagedKvError> {
    let end = position
        .checked_add(chunk_tokens)
        .ok_or(PagedKvError::ElementCountOverflow)?;
    let (cache_count, table_count, chunk_count) = spec.counts(chunk_tokens, end)?;
    let index = Expr::var("index");
    let [sequence, head, chunk_token, column] = spec.contiguous(chunk_tokens).coords(&index);
    let token = Expr::add(chunk_token, Expr::u32(position));
    let destination = spec.address(sequence, head, &token, column);
    Ok(layout_move_program(LayoutMove {
        op_id: PAGED_KV_APPEND_OP_ID,
        buffers: vec![
            BufferDecl::read_write(spec.cache, 0, spec.dtype.clone()).with_count(cache_count),
            BufferDecl::storage(chunk, 1, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(chunk_count),
            BufferDecl::storage(spec.block_table, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(table_count),
        ],
        write: spec.cache,
        count: chunk_count,
        map: IndexMap::Scatter {
            read: chunk.into(),
            destination,
        },
    }))
}

/// The two-block, two-token cache both registration fixtures address.
///
/// Sequence zero maps logical block 0 to physical block 1 and logical block 1
/// to physical block 0, so a fixture that ignored the block table would produce
/// the identity gather and pass.
fn fixture_cache() -> PagedKvCache<'static> {
    PagedKvCache {
        cache: "cache",
        block_table: "block_table",
        sequences: 1,
        heads: 1,
        blocks: 2,
        block_tokens: 2,
        blocks_per_sequence: 2,
        head_dim: 2,
        dtype: DataType::F32,
    }
}

fn fixture_f32(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

fn fixture_u32(values: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(values)
}

fn gather_fixture_program() -> Program {
    match paged_kv_gather(&fixture_cache(), "window", 4) {
        Ok(program) => program,
        Err(error) => vyre_foundation::composition::trap_program(
            PAGED_KV_GATHER_OP_ID,
            None,
            format!("Fix: paged_kv_gather fixture must build: {error}"),
        ),
    }
}

fn append_fixture_program() -> Program {
    match paged_kv_append(&fixture_cache(), "chunk", 1, 1) {
        Ok(program) => program,
        Err(error) => vyre_foundation::composition::trap_program(
            PAGED_KV_APPEND_OP_ID,
            None,
            format!("Fix: paged_kv_append fixture must build: {error}"),
        ),
    }
}

const EXPECTED_PAGED_KV_GATHER_OUTPUT_BYTES: [u8; 32] = [
    0, 0, 160, 64, 0, 0, 192, 64, 0, 0, 224, 64, 0, 0, 0, 65, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64,
    64, 0, 0, 128, 64,
];
const EXPECTED_PAGED_KV_APPEND_OUTPUT_BYTES: [u8; 32] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 65, 0, 0, 32,
    65,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        PAGED_KV_GATHER_OP_ID,
        gather_fixture_program,
        Some(|| vec![vec![
            fixture_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            fixture_u32(&[1, 0]),
        ]]),
        // Physical block 1 holds the first two logical tokens and physical
        // block 0 holds the next two.
        Some(|| vec![vec![EXPECTED_PAGED_KV_GATHER_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("llm")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        PAGED_KV_APPEND_OP_ID,
        append_fixture_program,
        Some(|| vec![vec![
            fixture_f32(&[0.0; 8]),
            fixture_f32(&[9.0, 10.0]),
            fixture_u32(&[1, 0]),
        ]]),
        // Logical token 1 is slot 1 of logical block 0, which the table maps to
        // physical block 1: cache elements 6 and 7.
        Some(|| vec![vec![EXPECTED_PAGED_KV_APPEND_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("llm")
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_append_launches_over_the_chunk_it_moves_not_the_cache_it_writes() {
        let spec = fixture_cache();
        let chunk = spec.moved_elements(1).expect("chunk elements");
        assert_eq!(chunk, 2);
        let program = paged_kv_append(&spec, "chunk", 1, 1).expect("append program");
        assert_eq!(
            vyre_foundation::guarded_logical_span(&program),
            Some(chunk),
            "a cache-sized launch would run the whole cache per decoded chunk"
        );

        let overflow = PagedKvCache {
            heads: u32::MAX,
            ..fixture_cache()
        };
        assert_eq!(
            overflow.moved_elements(4),
            Err(PagedKvError::ElementCountOverflow)
        );
    }
}
