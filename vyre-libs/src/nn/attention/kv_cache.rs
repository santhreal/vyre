//! Explicit immutable-prior to next-generation KV cache updates.

use thiserror::Error;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::kv_cache_append";

/// Invalid cache append dimensions.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum KvCacheAppendError {
    /// A required dimension is zero.
    #[error(
        "KV cache append requires nonzero batch, heads, capacity, chunk length, and head dimension"
    )]
    EmptyShape,
    /// The appended chunk exceeds the cache generation.
    #[error(
        "KV cache append range offset={offset}, chunk_len={chunk_len} exceeds capacity={capacity}"
    )]
    Range {
        /// First cache token replaced by the chunk.
        offset: u32,
        /// Logical chunk token count.
        chunk_len: u32,
        /// Cache token capacity.
        capacity: u32,
    },
    /// Flattened element counts exceeded addressable IR indexing.
    #[error("KV cache append element count overflows u32; shard the cache")]
    ElementCountOverflow,
    /// The cache dtype cannot represent floating attention activations.
    #[error("KV cache append requires F16, BF16, or F32 elements; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected cache element dtype.
        dtype: DataType,
    },
}

/// Copy an immutable prior cache and replace one contiguous token range.
///
/// Layout is `[batch, heads, tokens, head_dim]` for both the cache and chunk.
/// `prior` is never modified. `next` contains the complete successor generation.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_append(
    prior: &str,
    chunk: &str,
    next: &str,
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    head_dim: u32,
    offset: u32,
) -> Result<Program, KvCacheAppendError> {
    kv_cache_append_typed(
        prior,
        chunk,
        next,
        batch,
        heads,
        capacity,
        chunk_len,
        head_dim,
        offset,
        DataType::F32,
    )
}

/// Typed immutable-prior to next-generation KV cache update.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_append_typed(
    prior: &str,
    chunk: &str,
    next: &str,
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    head_dim: u32,
    offset: u32,
    dtype: DataType,
) -> Result<Program, KvCacheAppendError> {
    if batch == 0 || heads == 0 || capacity == 0 || chunk_len == 0 || head_dim == 0 {
        return Err(KvCacheAppendError::EmptyShape);
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(KvCacheAppendError::UnsupportedDtype { dtype });
    }
    let end = offset
        .checked_add(chunk_len)
        .filter(|end| *end <= capacity)
        .ok_or(KvCacheAppendError::Range {
            offset,
            chunk_len,
            capacity,
        })?;
    let checked = |values: &[u32]| {
        values.iter().try_fold(1_u32, |product, value| {
            product
                .checked_mul(*value)
                .ok_or(KvCacheAppendError::ElementCountOverflow)
        })
    };
    let cache_count = checked(&[batch, heads, capacity, head_dim])?;
    let chunk_count = checked(&[batch, heads, chunk_len, head_dim])?;
    let cache_head_span = capacity
        .checked_mul(head_dim)
        .ok_or(KvCacheAppendError::ElementCountOverflow)?;
    let chunk_head_span = chunk_len
        .checked_mul(head_dim)
        .ok_or(KvCacheAppendError::ElementCountOverflow)?;
    let index = Expr::var("index");
    let dimension = Expr::rem(index.clone(), Expr::u32(head_dim));
    let head_row = Expr::div(index.clone(), Expr::u32(cache_head_span));
    let token = Expr::rem(
        Expr::div(index.clone(), Expr::u32(head_dim)),
        Expr::u32(capacity),
    );
    let chunk_index = Expr::add(
        Expr::mul(head_row, Expr::u32(chunk_head_span)),
        Expr::add(
            Expr::mul(
                Expr::sub(token.clone(), Expr::u32(offset)),
                Expr::u32(head_dim),
            ),
            dimension,
        ),
    );
    let in_chunk = Expr::and(
        Expr::ge(token.clone(), Expr::u32(offset)),
        Expr::lt(token, Expr::u32(end)),
    );
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(cache_count)),
            vec![Node::if_then_else(
                in_chunk,
                vec![Node::Store {
                    buffer: next.into(),
                    index: index.clone(),
                    value: Expr::load(chunk, chunk_index),
                }],
                vec![Node::Store {
                    buffer: next.into(),
                    index: index.clone(),
                    value: Expr::load(prior, index),
                }],
            )],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(prior, 0, BufferAccess::ReadWrite, dtype.clone())
                .with_count(cache_count),
            BufferDecl::storage(chunk, 1, BufferAccess::ReadOnly, dtype.clone())
                .with_count(chunk_count),
            BufferDecl::output(next, 2, dtype).with_count(cache_count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}
