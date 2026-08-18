//! Paged scaled dot-product attention with block-table indexing.
//!
//! A paged attention program addresses keys and values stored across non-contiguous
//! physical blocks (pages) via an indirection table (`block_table`).
//!
//! Shape conventions:
//! - Queries `q`: `[sequences, q_heads, query_tokens, head_dim]`
//! - K-cache `k_cache`: `[blocks, kv_heads, block_tokens, head_dim]`
//! - V-cache `v_cache`: `[blocks, kv_heads, block_tokens, head_dim]`
//! - Block table `block_table`: `[sequences, blocks_per_sequence]` of `u32` physical block ids
//! - Output `output`: `[sequences, q_heads, query_tokens, head_dim]`
//!
//! The program handles:
//! - Multi-head attention (MHA), multi-query attention (MQA), and grouped-query attention (GQA)
//! - Explicit page size (`block_tokens`) and partial-page boundaries (`context_tokens`)
//! - Causal masking where query token `t` attends to cache entries `0..=cache_offset + t`
//! - `F16`, `BF16`, and `F32` element dtypes with F32 score and value accumulation
//! - Out-of-range protection for block table lookups and sequence bounds.

use thiserror::Error;
use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

use crate::nn::attention_stability::{bounded_score, positive_denominator};

/// Canonical op id for paged attention.
pub const PAGED_ATTENTION_OP_ID: &str = "vyre-libs::nn::paged_attention";
/// Canonical op id for paged attention max score pass.
pub const PAGED_ATTENTION_MAX_PASS_OP_ID: &str = "vyre-libs::nn::paged_attention_max_pass";
/// Canonical op id for paged attention sum normalization pass.
pub const PAGED_ATTENTION_SUM_PASS_OP_ID: &str = "vyre-libs::nn::paged_attention_sum_pass";
/// Canonical op id for paged attention weighted write pass.
pub const PAGED_ATTENTION_WRITE_PASS_OP_ID: &str = "vyre-libs::nn::paged_attention_write_pass";
/// Canonical op id for paged cache append.
pub const PAGED_CACHE_APPEND_OP_ID: &str = "vyre-libs::nn::paged_cache_append";

/// Validation errors for paged attention parameters.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PagedAttentionError {
    /// A dimension is zero.
    #[error(
        "paged attention requires non-zero sequences, heads, tokens, block tokens, and head_dim"
    )]
    EmptyShape,
    /// Query heads is not a multiple of KV heads.
    #[error("query heads ({q_heads}) must be a multiple of KV heads ({kv_heads})")]
    HeadMismatch {
        /// Number of query heads.
        q_heads: u32,
        /// Number of key/value heads.
        kv_heads: u32,
    },
    /// The addressed context tokens exceed the block table capacity.
    #[error(
        "context tokens ({context_tokens}) exceeds the block table capacity ({blocks_per_sequence} * {block_tokens} = {capacity})"
    )]
    BlockTableCapacity {
        /// Number of context tokens.
        context_tokens: u32,
        /// Blocks per sequence.
        blocks_per_sequence: u32,
        /// Tokens per block.
        block_tokens: u32,
        /// Maximum addressable capacity.
        capacity: u32,
    },
    /// Causal range check failed.
    #[error(
        "causal offset ({cache_offset}) + query tokens ({query_tokens}) exceeds context tokens ({context_tokens})"
    )]
    CausalRange {
        /// Logical token offset.
        cache_offset: u32,
        /// Number of query tokens.
        query_tokens: u32,
        /// Number of context tokens.
        context_tokens: u32,
    },
    /// Element count overflowed u32.
    #[error("paged attention element count overflows u32; shard the attention batch/heads")]
    ElementCountOverflow,
    /// Unsupported tensor dtype.
    #[error("paged attention requires F16, BF16, or F32 tensors; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected dtype.
        dtype: DataType,
    },
}

/// Parameters defining a paged attention program.
#[derive(Debug, Clone, PartialEq)]
pub struct PagedAttentionSpec<'a> {
    /// Query tensor buffer name.
    pub q: &'a str,
    /// Paged key cache buffer name.
    pub k_cache: &'a str,
    /// Paged value cache buffer name.
    pub v_cache: &'a str,
    /// Block table buffer name (`u32`).
    pub block_table: &'a str,
    /// Output buffer name.
    pub output: &'a str,
    /// Number of sequences in the batch.
    pub sequences: u32,
    /// Number of query heads.
    pub q_heads: u32,
    /// Number of key/value heads.
    pub kv_heads: u32,
    /// Number of query tokens per sequence.
    pub query_tokens: u32,
    /// Total valid context tokens in cache to attend to per sequence.
    pub context_tokens: u32,
    /// Physical block count in the cache pool.
    pub blocks: u32,
    /// Tokens per physical block (page size).
    pub block_tokens: u32,
    /// Maximum blocks per sequence in the block table.
    pub blocks_per_sequence: u32,
    /// Head dimension width.
    pub head_dim: u32,
    /// Logical token offset of the query window (for causal masking).
    pub cache_offset: u32,
    /// Whether causal masking is enabled.
    pub causal: bool,
    /// Data type of activations and cache buffers.
    pub dtype: DataType,
    /// Custom scale factor (defaults to `1.0 / sqrt(head_dim)` if None).
    pub scale: Option<f32>,
}

impl PagedAttentionSpec<'_> {
    /// Validate dimensions and compute element buffer sizes.
    pub fn validate(&self) -> Result<(u32, u32, u32, u32), PagedAttentionError> {
        if self.sequences == 0
            || self.q_heads == 0
            || self.kv_heads == 0
            || self.query_tokens == 0
            || self.context_tokens == 0
            || self.blocks == 0
            || self.block_tokens == 0
            || self.blocks_per_sequence == 0
            || self.head_dim == 0
        {
            return Err(PagedAttentionError::EmptyShape);
        }
        if !matches!(self.dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(PagedAttentionError::UnsupportedDtype {
                dtype: self.dtype.clone(),
            });
        }
        if self.q_heads % self.kv_heads != 0 {
            return Err(PagedAttentionError::HeadMismatch {
                q_heads: self.q_heads,
                kv_heads: self.kv_heads,
            });
        }

        let addressable = self
            .blocks_per_sequence
            .checked_mul(self.block_tokens)
            .ok_or(PagedAttentionError::ElementCountOverflow)?;
        if self.context_tokens > addressable {
            return Err(PagedAttentionError::BlockTableCapacity {
                context_tokens: self.context_tokens,
                blocks_per_sequence: self.blocks_per_sequence,
                block_tokens: self.block_tokens,
                capacity: addressable,
            });
        }

        if self.causal {
            let max_query_pos = self
                .cache_offset
                .checked_add(self.query_tokens)
                .ok_or(PagedAttentionError::ElementCountOverflow)?;
            if max_query_pos > self.context_tokens {
                return Err(PagedAttentionError::CausalRange {
                    cache_offset: self.cache_offset,
                    query_tokens: self.query_tokens,
                    context_tokens: self.context_tokens,
                });
            }
        }

        let checked = |dims: &[u32]| -> Result<u32, PagedAttentionError> {
            dims.iter().try_fold(1_u32, |acc, &d| {
                acc.checked_mul(d)
                    .ok_or(PagedAttentionError::ElementCountOverflow)
            })
        };

        let q_count = checked(&[
            self.sequences,
            self.q_heads,
            self.query_tokens,
            self.head_dim,
        ])?;
        let kv_cache_count =
            checked(&[self.blocks, self.kv_heads, self.block_tokens, self.head_dim])?;
        let block_table_count = checked(&[self.sequences, self.blocks_per_sequence])?;
        let out_count = q_count;

        Ok((q_count, kv_cache_count, block_table_count, out_count))
    }
}

/// Compute the flat physical cache index expression for `(sequence, kv_head, token_idx, dk)`.
fn paged_cache_element_expr(
    block_table: &str,
    blocks_per_sequence: u32,
    kv_heads: u32,
    block_tokens: u32,
    head_dim: u32,
    seq_expr: Expr,
    kv_head_expr: Expr,
    token_expr: Expr,
    dk_expr: Expr,
) -> Expr {
    let logical_block = Expr::div(token_expr.clone(), Expr::u32(block_tokens));
    let slot_in_block = Expr::rem(token_expr, Expr::u32(block_tokens));

    // block_table_index = seq * blocks_per_sequence + logical_block
    let table_index = Expr::add(
        Expr::mul(seq_expr, Expr::u32(blocks_per_sequence)),
        logical_block,
    );
    let physical_block = Expr::load(block_table, table_index);

    // physical_stride_block = kv_heads * block_tokens * head_dim
    let block_stride = kv_heads * block_tokens * head_dim;
    // physical_stride_head = block_tokens * head_dim
    let head_stride = block_tokens * head_dim;
    // physical_stride_slot = head_dim
    let slot_stride = head_dim;

    // flat_index = physical_block * block_stride + kv_head * head_stride + slot * slot_stride + dk
    Expr::add(
        Expr::add(
            Expr::mul(physical_block, Expr::u32(block_stride)),
            Expr::mul(kv_head_expr, Expr::u32(head_stride)),
        ),
        Expr::add(Expr::mul(slot_in_block, Expr::u32(slot_stride)), dk_expr),
    )
}

/// Convert a loaded expression from `dtype` to `F32` for accumulation.
fn to_f32(expr: Expr, dtype: &DataType) -> Expr {
    match dtype {
        DataType::F32 => expr,
        DataType::F16 => Expr::Cast {
            target: DataType::F32,
            value: Box::new(expr),
        },
        DataType::BF16 => Expr::Cast {
            target: DataType::F32,
            value: Box::new(expr),
        },
        _ => expr,
    }
}

/// Convert an `F32` accumulated expression back to `dtype` for storing.
fn from_f32(expr: Expr, dtype: &DataType) -> Expr {
    match dtype {
        DataType::F32 => expr,
        DataType::F16 => Expr::Cast {
            target: DataType::F16,
            value: Box::new(expr),
        },
        DataType::BF16 => Expr::Cast {
            target: DataType::BF16,
            value: Box::new(expr),
        },
        _ => expr,
    }
}

/// Build a paged attention [`Program`].
///
/// Each invocation computes one query row `(sequence, q_head, q_token)`.
///
/// # Errors
///
/// Returns [`PagedAttentionError`] if parameters fail validation or element counts overflow.
pub fn paged_attention(spec: &PagedAttentionSpec<'_>) -> Result<Program, PagedAttentionError> {
    let (q_count, kv_cache_count, block_table_count, out_count) = spec.validate()?;

    let group_size = spec.q_heads / spec.kv_heads;
    let scale_val = spec
        .scale
        .unwrap_or_else(|| 1.0f32 / (spec.head_dim as f32).sqrt());
    let scale_expr = Expr::f32(scale_val);

    let total_rows = spec
        .sequences
        .checked_mul(spec.q_heads)
        .and_then(|h| h.checked_mul(spec.query_tokens))
        .ok_or(PagedAttentionError::ElementCountOverflow)?;

    let row_idx = Expr::var("row_idx");

    // Decompose row_idx into (seq_idx, q_head_idx, q_tok_idx)
    // row_idx = (seq * q_heads + q_head) * query_tokens + q_tok
    let q_tok_idx = Expr::rem(row_idx.clone(), Expr::u32(spec.query_tokens));
    let head_and_seq = Expr::div(row_idx.clone(), Expr::u32(spec.query_tokens));
    let q_head_idx = Expr::rem(head_and_seq.clone(), Expr::u32(spec.q_heads));
    let seq_idx = Expr::div(head_and_seq, Expr::u32(spec.q_heads));
    let kv_head_idx = Expr::div(q_head_idx.clone(), Expr::u32(group_size));

    // Query base index = row_idx * head_dim
    let query_base = Expr::mul(row_idx.clone(), Expr::u32(spec.head_dim));

    // Key limit for causal masking or non-causal
    let key_limit = if spec.causal {
        // limit = min(cache_offset + q_tok_idx + 1, context_tokens)
        let causal_limit = Expr::add(
            Expr::add(Expr::u32(spec.cache_offset), q_tok_idx.clone()),
            Expr::u32(1),
        );
        Expr::select(
            Expr::lt(causal_limit.clone(), Expr::u32(spec.context_tokens)),
            causal_limit,
            Expr::u32(spec.context_tokens),
        )
    } else {
        Expr::u32(spec.context_tokens)
    };

    // --- Pass 1: Max Score Pass ---
    let max_pass_body = vec![Node::loop_for(
        "j",
        Expr::u32(0),
        key_limit.clone(),
        vec![
            Node::let_bind("dot_val", Expr::f32(0.0)),
            Node::loop_for(
                "dk",
                Expr::u32(0),
                Expr::u32(spec.head_dim),
                vec![
                    Node::let_bind(
                        "q_elem",
                        to_f32(
                            Expr::load(spec.q, Expr::add(query_base.clone(), Expr::var("dk"))),
                            &spec.dtype,
                        ),
                    ),
                    Node::let_bind(
                        "k_addr",
                        paged_cache_element_expr(
                            spec.block_table,
                            spec.blocks_per_sequence,
                            spec.kv_heads,
                            spec.block_tokens,
                            spec.head_dim,
                            seq_idx.clone(),
                            kv_head_idx.clone(),
                            Expr::var("j"),
                            Expr::var("dk"),
                        ),
                    ),
                    Node::let_bind(
                        "k_elem",
                        to_f32(Expr::load(spec.k_cache, Expr::var("k_addr")), &spec.dtype),
                    ),
                    Node::assign(
                        "dot_val",
                        Expr::add(
                            Expr::var("dot_val"),
                            Expr::mul(Expr::var("q_elem"), Expr::var("k_elem")),
                        ),
                    ),
                ],
            ),
            Node::let_bind(
                "score",
                bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr.clone())),
            ),
            Node::assign(
                "max_val",
                Expr::select(
                    Expr::gt(Expr::var("score"), Expr::var("max_val")),
                    Expr::var("score"),
                    Expr::var("max_val"),
                ),
            ),
        ],
    )];

    // --- Pass 2: Sum Normalization Pass ---
    let sum_pass_body = vec![Node::loop_for(
        "j",
        Expr::u32(0),
        key_limit.clone(),
        vec![
            Node::let_bind("dot_val", Expr::f32(0.0)),
            Node::loop_for(
                "dk",
                Expr::u32(0),
                Expr::u32(spec.head_dim),
                vec![
                    Node::let_bind(
                        "q_elem",
                        to_f32(
                            Expr::load(spec.q, Expr::add(query_base.clone(), Expr::var("dk"))),
                            &spec.dtype,
                        ),
                    ),
                    Node::let_bind(
                        "k_addr",
                        paged_cache_element_expr(
                            spec.block_table,
                            spec.blocks_per_sequence,
                            spec.kv_heads,
                            spec.block_tokens,
                            spec.head_dim,
                            seq_idx.clone(),
                            kv_head_idx.clone(),
                            Expr::var("j"),
                            Expr::var("dk"),
                        ),
                    ),
                    Node::let_bind(
                        "k_elem",
                        to_f32(Expr::load(spec.k_cache, Expr::var("k_addr")), &spec.dtype),
                    ),
                    Node::assign(
                        "dot_val",
                        Expr::add(
                            Expr::var("dot_val"),
                            Expr::mul(Expr::var("q_elem"), Expr::var("k_elem")),
                        ),
                    ),
                ],
            ),
            Node::let_bind(
                "score",
                bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr.clone())),
            ),
            Node::let_bind(
                "exp_val",
                Expr::exp(Expr::sub(Expr::var("score"), Expr::var("max_val"))),
            ),
            Node::assign(
                "sum_val",
                Expr::add(Expr::var("sum_val"), Expr::var("exp_val")),
            ),
        ],
    )];

    // --- Pass 3: Weighted Values Write Pass ---
    let write_pass_body = vec![Node::loop_for(
        "dk",
        Expr::u32(0),
        Expr::u32(spec.head_dim),
        vec![
            Node::let_bind("weighted_sum", Expr::f32(0.0)),
            Node::loop_for(
                "j",
                Expr::u32(0),
                key_limit,
                vec![
                    Node::let_bind("dot_val", Expr::f32(0.0)),
                    Node::loop_for(
                        "inner_dk",
                        Expr::u32(0),
                        Expr::u32(spec.head_dim),
                        vec![
                            Node::let_bind(
                                "q_elem",
                                to_f32(
                                    Expr::load(
                                        spec.q,
                                        Expr::add(query_base.clone(), Expr::var("inner_dk")),
                                    ),
                                    &spec.dtype,
                                ),
                            ),
                            Node::let_bind(
                                "k_addr",
                                paged_cache_element_expr(
                                    spec.block_table,
                                    spec.blocks_per_sequence,
                                    spec.kv_heads,
                                    spec.block_tokens,
                                    spec.head_dim,
                                    seq_idx.clone(),
                                    kv_head_idx.clone(),
                                    Expr::var("j"),
                                    Expr::var("inner_dk"),
                                ),
                            ),
                            Node::let_bind(
                                "k_elem",
                                to_f32(Expr::load(spec.k_cache, Expr::var("k_addr")), &spec.dtype),
                            ),
                            Node::assign(
                                "dot_val",
                                Expr::add(
                                    Expr::var("dot_val"),
                                    Expr::mul(Expr::var("q_elem"), Expr::var("k_elem")),
                                ),
                            ),
                        ],
                    ),
                    Node::let_bind(
                        "score",
                        bounded_score(Expr::mul(Expr::var("dot_val"), scale_expr.clone())),
                    ),
                    Node::let_bind(
                        "p_weight",
                        Expr::div(
                            Expr::exp(Expr::sub(Expr::var("score"), Expr::var("max_val"))),
                            Expr::var("denom"),
                        ),
                    ),
                    Node::let_bind(
                        "v_addr",
                        paged_cache_element_expr(
                            spec.block_table,
                            spec.blocks_per_sequence,
                            spec.kv_heads,
                            spec.block_tokens,
                            spec.head_dim,
                            seq_idx.clone(),
                            kv_head_idx.clone(),
                            Expr::var("j"),
                            Expr::var("dk"),
                        ),
                    ),
                    Node::let_bind(
                        "v_elem",
                        to_f32(Expr::load(spec.v_cache, Expr::var("v_addr")), &spec.dtype),
                    ),
                    Node::assign(
                        "weighted_sum",
                        Expr::add(
                            Expr::var("weighted_sum"),
                            Expr::mul(Expr::var("p_weight"), Expr::var("v_elem")),
                        ),
                    ),
                ],
            ),
            Node::Store {
                buffer: spec.output.into(),
                index: Expr::add(query_base.clone(), Expr::var("dk")),
                value: from_f32(Expr::var("weighted_sum"), &spec.dtype),
            },
        ],
    )];

    let parent = Ident::from(PAGED_ATTENTION_OP_ID);

    let body = vec![
        Node::let_bind("row_idx", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(row_idx, Expr::u32(total_rows)),
            vec![
                Node::let_bind("max_val", Expr::f32(f32::MIN)),
                wrap_child_region(
                    PAGED_ATTENTION_MAX_PASS_OP_ID,
                    parent.clone(),
                    max_pass_body,
                ),
                Node::let_bind("sum_val", Expr::f32(0.0)),
                wrap_child_region(
                    PAGED_ATTENTION_SUM_PASS_OP_ID,
                    parent.clone(),
                    sum_pass_body,
                ),
                Node::let_bind("denom", positive_denominator(Expr::var("sum_val"))),
                wrap_child_region(PAGED_ATTENTION_WRITE_PASS_OP_ID, parent, write_pass_body),
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(spec.q, 0, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(q_count),
            BufferDecl::storage(spec.k_cache, 1, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(kv_cache_count),
            BufferDecl::storage(spec.v_cache, 2, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(kv_cache_count),
            BufferDecl::storage(spec.block_table, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(block_table_count),
            BufferDecl::output(spec.output, 4, spec.dtype.clone()).with_count(out_count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(PAGED_ATTENTION_OP_ID, body)],
    ))
}

/// Errors occurring during paged cache append construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PagedCacheAppendError {
    /// A dimension is zero.
    #[error("paged cache append requires non-zero sequences, heads, chunk_tokens, block_tokens, and head_dim")]
    EmptyShape,
    /// Destination range exceeds the block table's capacity.
    #[error(
        "append end position ({end_position}) exceeds block table capacity ({blocks_per_sequence} * {block_tokens} = {capacity})"
    )]
    BlockTableCapacity {
        /// End token position.
        end_position: u32,
        /// Blocks per sequence.
        blocks_per_sequence: u32,
        /// Tokens per block.
        block_tokens: u32,
        /// Total capacity in tokens.
        capacity: u32,
    },
    /// Flattened element count overflows `u32`.
    #[error("paged cache append element count overflows u32")]
    ElementCountOverflow,
    /// Unsupported tensor element dtype.
    #[error("paged cache append requires F16, BF16, or F32 tensors; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected dtype.
        dtype: DataType,
    },
}

/// Specifications for writing a token chunk into a paged KV cache.
#[derive(Debug, Clone, PartialEq)]
pub struct PagedCacheAppendSpec<'a> {
    /// Input chunk buffer name `[sequences, kv_heads, chunk_tokens, head_dim]`.
    pub chunk: &'a str,
    /// Target physical cache buffer name `[blocks, kv_heads, block_tokens, head_dim]`.
    pub cache: &'a str,
    /// Block table buffer name `[sequences, blocks_per_sequence]` (`u32`).
    pub block_table: &'a str,
    /// Number of sequences in batch.
    pub sequences: u32,
    /// Number of KV heads.
    pub kv_heads: u32,
    /// Number of tokens in the new incoming chunk.
    pub chunk_tokens: u32,
    /// Starting logical token index in the sequence where this chunk is written.
    pub start_position: u32,
    /// Total physical blocks in the cache pool.
    pub blocks: u32,
    /// Tokens per physical block (page size).
    pub block_tokens: u32,
    /// Maximum blocks per sequence in the block table.
    pub blocks_per_sequence: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Tensor element data type.
    pub dtype: DataType,
}

impl PagedCacheAppendSpec<'_> {
    /// Validate append parameters.
    pub fn validate(&self) -> Result<(u32, u32, u32), PagedCacheAppendError> {
        if self.sequences == 0
            || self.kv_heads == 0
            || self.chunk_tokens == 0
            || self.blocks == 0
            || self.block_tokens == 0
            || self.blocks_per_sequence == 0
            || self.head_dim == 0
        {
            return Err(PagedCacheAppendError::EmptyShape);
        }
        if !matches!(self.dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(PagedCacheAppendError::UnsupportedDtype {
                dtype: self.dtype.clone(),
            });
        }

        let end_position = self
            .start_position
            .checked_add(self.chunk_tokens)
            .ok_or(PagedCacheAppendError::ElementCountOverflow)?;

        let capacity = self
            .blocks_per_sequence
            .checked_mul(self.block_tokens)
            .ok_or(PagedCacheAppendError::ElementCountOverflow)?;

        if end_position > capacity {
            return Err(PagedCacheAppendError::BlockTableCapacity {
                end_position,
                blocks_per_sequence: self.blocks_per_sequence,
                block_tokens: self.block_tokens,
                capacity,
            });
        }

        let checked = |dims: &[u32]| -> Result<u32, PagedCacheAppendError> {
            dims.iter().try_fold(1_u32, |acc, &d| {
                acc.checked_mul(d)
                    .ok_or(PagedCacheAppendError::ElementCountOverflow)
            })
        };

        let chunk_count = checked(&[
            self.sequences,
            self.kv_heads,
            self.chunk_tokens,
            self.head_dim,
        ])?;
        let cache_count = checked(&[self.blocks, self.kv_heads, self.block_tokens, self.head_dim])?;
        let table_count = checked(&[self.sequences, self.blocks_per_sequence])?;

        Ok((chunk_count, cache_count, table_count))
    }
}

/// Build a program to append an incoming chunk of tokens into a paged KV cache.
///
/// Each invocation writes one element `(sequence, kv_head, chunk_tok, dk)`.
///
/// # Errors
///
/// Returns [`PagedCacheAppendError`] if validation fails.
pub fn paged_cache_append(
    spec: &PagedCacheAppendSpec<'_>,
) -> Result<Program, PagedCacheAppendError> {
    let (chunk_count, cache_count, table_count) = spec.validate()?;

    let flat_idx = Expr::var("flat_idx");

    // Decompose flat_idx into (seq_idx, kv_head_idx, chunk_tok_idx, dk_idx)
    let dk_idx = Expr::rem(flat_idx.clone(), Expr::u32(spec.head_dim));
    let rem_after_dk = Expr::div(flat_idx.clone(), Expr::u32(spec.head_dim));

    let chunk_tok_idx = Expr::rem(rem_after_dk.clone(), Expr::u32(spec.chunk_tokens));
    let rem_after_tok = Expr::div(rem_after_dk, Expr::u32(spec.chunk_tokens));

    let kv_head_idx = Expr::rem(rem_after_tok.clone(), Expr::u32(spec.kv_heads));
    let seq_idx = Expr::div(rem_after_tok, Expr::u32(spec.kv_heads));

    // Logical token position in cache
    let logical_token = Expr::add(Expr::u32(spec.start_position), chunk_tok_idx);

    let dst_addr = paged_cache_element_expr(
        spec.block_table,
        spec.blocks_per_sequence,
        spec.kv_heads,
        spec.block_tokens,
        spec.head_dim,
        seq_idx,
        kv_head_idx,
        logical_token,
        dk_idx,
    );

    let body = vec![
        Node::let_bind("flat_idx", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(flat_idx.clone(), Expr::u32(chunk_count)),
            vec![
                Node::let_bind("val", Expr::load(spec.chunk, flat_idx)),
                Node::Store {
                    buffer: spec.cache.into(),
                    index: dst_addr,
                    value: Expr::var("val"),
                },
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(spec.chunk, 0, BufferAccess::ReadOnly, spec.dtype.clone())
                .with_count(chunk_count),
            BufferDecl::storage(spec.block_table, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(table_count),
            BufferDecl::storage(spec.cache, 2, BufferAccess::ReadWrite, spec.dtype.clone())
                .with_count(cache_count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(PAGED_CACHE_APPEND_OP_ID, body)],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::{
        decode_f32 as bytes_to_f32, f32_bytes as f32_to_bytes, u32_bytes as u32_to_bytes,
    };
    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    #[test]
    fn paged_attention_validates_dimensions() {
        let spec = PagedAttentionSpec {
            q: "q",
            k_cache: "k",
            v_cache: "v",
            block_table: "table",
            output: "out",
            sequences: 1,
            q_heads: 4,
            kv_heads: 2,
            query_tokens: 2,
            context_tokens: 8,
            blocks: 4,
            block_tokens: 4,
            blocks_per_sequence: 2,
            head_dim: 8,
            cache_offset: 6,
            causal: true,
            dtype: DataType::F32,
            scale: None,
        };
        let program = paged_attention(&spec).expect("Fix: valid spec should build program");
        assert_eq!(program.buffers().len(), 5);
    }

    #[test]
    fn paged_attention_rejects_head_mismatch() {
        let spec = PagedAttentionSpec {
            q: "q",
            k_cache: "k",
            v_cache: "v",
            block_table: "table",
            output: "out",
            sequences: 1,
            q_heads: 3,
            kv_heads: 2,
            query_tokens: 1,
            context_tokens: 4,
            blocks: 2,
            block_tokens: 4,
            blocks_per_sequence: 1,
            head_dim: 4,
            cache_offset: 0,
            causal: false,
            dtype: DataType::F32,
            scale: None,
        };
        let err = paged_attention(&spec).unwrap_err();
        assert!(matches!(err, PagedAttentionError::HeadMismatch { .. }));
    }

    #[test]
    fn paged_attention_rejects_capacity_overflow() {
        let spec = PagedAttentionSpec {
            q: "q",
            k_cache: "k",
            v_cache: "v",
            block_table: "table",
            output: "out",
            sequences: 1,
            q_heads: 2,
            kv_heads: 2,
            query_tokens: 1,
            context_tokens: 10,
            blocks: 2,
            block_tokens: 4,
            blocks_per_sequence: 2, // capacity 8 < 10
            head_dim: 4,
            cache_offset: 0,
            causal: false,
            dtype: DataType::F32,
            scale: None,
        };
        let err = paged_attention(&spec).unwrap_err();
        assert!(matches!(
            err,
            PagedAttentionError::BlockTableCapacity { .. }
        ));
    }

    #[test]
    fn paged_attention_eval_parity_single_token() {
        // 1 sequence, 1 Q head, 1 KV head, 1 query token, 4 context tokens
        // 2 physical blocks, 2 tokens per block, 2 blocks per sequence
        let spec = PagedAttentionSpec {
            q: "q",
            k_cache: "k",
            v_cache: "v",
            block_table: "table",
            output: "out",
            sequences: 1,
            q_heads: 1,
            kv_heads: 1,
            query_tokens: 1,
            context_tokens: 4,
            blocks: 2,
            block_tokens: 2,
            blocks_per_sequence: 2,
            head_dim: 2,
            cache_offset: 3,
            causal: false,
            dtype: DataType::F32,
            scale: Some(1.0),
        };
        let program = paged_attention(&spec).expect("program");

        // q: [1, 1, 1, 2] = [1.0, 0.0]
        let q_data = vec![1.0f32, 0.0];

        // block 0: tokens 0, 1 -> K=[[1.0, 0.0], [0.0, 1.0]], V=[[10.0, 20.0], [30.0, 40.0]]
        // block 1: tokens 2, 3 -> K=[[1.0, 0.0], [0.0, 1.0]], V=[[50.0, 60.0], [70.0, 80.0]]
        // physical layout: [blocks(2), kv_heads(1), block_tokens(2), head_dim(2)]
        let k_data = vec![
            1.0f32, 0.0, 0.0, 1.0, // block 0
            1.0, 0.0, 0.0, 1.0, // block 1
        ];
        let v_data = vec![
            10.0f32, 20.0, 30.0, 40.0, // block 0
            50.0, 60.0, 70.0, 80.0, // block 1
        ];
        // block_table: [1 seq, 2 blocks] -> maps logical block 0 to physical 0, logical 1 to physical 1
        let table_data = vec![0u32, 1];
        let out_init = vec![0.0f32; 2];

        let inputs = vec![
            Value::from(f32_to_bytes(&q_data)),
            Value::from(f32_to_bytes(&k_data)),
            Value::from(f32_to_bytes(&v_data)),
            Value::from(u32_to_bytes(&table_data)),
            Value::from(f32_to_bytes(&out_init)),
        ];

        let outputs = reference_eval(&program, &inputs).expect("eval");
        let result = bytes_to_f32(&outputs[0].to_bytes());

        // Token 0 and token 2 have Q.K = 1.0, Token 1 and 3 have Q.K = 0.0
        // Softmax scores: exp(1) for tok 0, exp(0) for tok 1, exp(1) for tok 2, exp(0) for tok 3
        // sum = 2*e + 2*1
        let e = 1.0f32.exp();
        let sum = 2.0 * e + 2.0;
        let w0 = e / sum;
        let w1 = 1.0 / sum;
        let w2 = e / sum;
        let w3 = 1.0 / sum;
        let expected_out_0 = w0 * 10.0 + w1 * 30.0 + w2 * 50.0 + w3 * 70.0;
        let expected_out_1 = w0 * 20.0 + w1 * 40.0 + w2 * 60.0 + w3 * 80.0;

        assert!(
            (result[0] - expected_out_0).abs() < 1e-4,
            "got {}, expected {}",
            result[0],
            expected_out_0
        );
        assert!(
            (result[1] - expected_out_1).abs() < 1e-4,
            "got {}, expected {}",
            result[1],
            expected_out_1
        );
    }

    #[test]
    fn paged_cache_append_eval() {
        let spec = PagedCacheAppendSpec {
            chunk: "chunk",
            cache: "cache",
            block_table: "table",
            sequences: 1,
            kv_heads: 1,
            chunk_tokens: 2,
            start_position: 1, // write at logical tokens 1 and 2
            blocks: 2,
            block_tokens: 2,
            blocks_per_sequence: 2,
            head_dim: 2,
            dtype: DataType::F32,
        };
        let program = paged_cache_append(&spec).expect("program");

        // Chunk has 2 tokens * 2 dim = 4 elements
        let chunk_data = vec![11.0f32, 22.0, 33.0, 44.0];
        // Cache has 2 blocks * 2 tokens * 2 dim = 8 elements, initially zero
        let cache_data = vec![0.0f32; 8];
        // block_table maps logical block 0 -> physical 0, logical block 1 -> physical 1
        let table_data = vec![0u32, 1];

        let inputs = vec![
            Value::from(f32_to_bytes(&chunk_data)),
            Value::from(u32_to_bytes(&table_data)),
            Value::from(f32_to_bytes(&cache_data)),
        ];

        let outputs = reference_eval(&program, &inputs).expect("eval");
        let result = bytes_to_f32(&outputs[0].to_bytes());

        // Logical token 1 is in physical block 0, slot 1 (indices 2, 3)
        // Logical token 2 is in physical block 1, slot 0 (indices 4, 5)
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 11.0);
        assert_eq!(result[3], 22.0);
        assert_eq!(result[4], 33.0);
        assert_eq!(result[5], 44.0);
        assert_eq!(result[6], 0.0);
        assert_eq!(result[7], 0.0);
    }
}
