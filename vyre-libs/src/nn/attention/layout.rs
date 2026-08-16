//! The index map every attention layout move is built from.
//!
//! Head-major to token-major conversion, token-major to head-major conversion,
//! and the KV cache append are one dispatch: one invocation per output element,
//! an `index < count` guard, and a store whose value comes from an index map
//! over the input buffers. They used to be three files with three copies of the
//! guard, the buffer declarations, the region wrapper, the shape and dtype
//! rejection, and the flat-index arithmetic. The map is the only part that
//! differs, so it is the only part a caller supplies.
//!
//! The flat-index arithmetic here is also what the gated delta schedules
//! address their six tensors with, so an index change lands on the layout moves
//! and the recurrences together.

use thiserror::Error;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const HEAD_TO_TOKEN_OP_ID: &str = "vyre-libs::nn::attention_head_to_token";
const TOKEN_TO_HEAD_OP_ID: &str = "vyre-libs::nn::attention_token_to_head";
const KV_CACHE_APPEND_OP_ID: &str = "vyre-libs::nn::kv_cache_append";

/// Lanes per workgroup for every layout move.
pub const ATTENTION_LAYOUT_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// Dispatch grid covering one layout move: one lane per moved element.
///
/// A launch geometry inferred from the declared buffers takes the largest one,
/// which is right for a gather and wrong for a scatter. The paged append
/// guards on the CHUNK and writes into a cache that is deliberately much
/// larger, so an inferred geometry fires a cache-sized dispatch to move one
/// decoded token and lets the guard discard the rest. The element count the
/// move was built from is the only thing that sizes it, so the base that owns
/// the move owns the grid, and a caller launching one passes it.
#[must_use]
pub const fn attention_layout_dispatch_grid(elements: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(elements, ATTENTION_LAYOUT_WORKGROUP_SIZE[0])
}

/// Axis lengths of a row-major `[outer, mid, row, column]` tensor.
///
/// The outer length is absent on purpose: a flat index never multiplies by it,
/// and the guarded element count is what bounds the outer coordinate.
#[derive(Clone, Copy)]
pub(crate) struct RowMajor {
    /// Length of the second-outermost axis.
    pub(crate) mid: u32,
    /// Length of the third axis.
    pub(crate) row: u32,
    /// Length of the innermost axis.
    pub(crate) width: u32,
}

/// `base * span + offset`: the flat index of one element inside a fixed-size
/// block.
///
/// Every flat index in this subtree is a nest of this shape, and keeping it as
/// one function is what makes two index derivations comparable by reading them.
pub(crate) fn block_index(base: Expr, span: u32, offset: Expr) -> Expr {
    Expr::add(Expr::mul(base, Expr::u32(span)), offset)
}

impl RowMajor {
    /// Flat index of the element at `[outer, mid, row, column]`.
    ///
    /// The row plane is addressed as one block so the emitted expression holds
    /// two multiplies by folded constants rather than four separate strides.
    pub(crate) fn index(self, outer: Expr, mid: Expr, row: Expr, column: Expr) -> Expr {
        block_index(
            block_index(outer, self.mid, mid),
            self.row * self.width,
            block_index(row, self.width, column),
        )
    }

    /// Coordinates of `index`, in `[outer, mid, row, column]` order.
    ///
    /// This is the inverse of [`RowMajor::index`] and the reason a layout move
    /// is written once: the guard bounds the OUTPUT index, the output layout
    /// splits it into coordinates, and the input layout puts them back
    /// together. The outer coordinate is a plain division, so an index past the
    /// element count produces an out-of-range outer coordinate rather than
    /// wrapping into a valid one.
    pub(crate) fn coords(self, index: &Expr) -> [Expr; 4] {
        let column = Expr::rem(index.clone(), Expr::u32(self.width));
        let rows = Expr::div(index.clone(), Expr::u32(self.width));
        let row = Expr::rem(rows.clone(), Expr::u32(self.row));
        let mids = Expr::div(rows, Expr::u32(self.row));
        let mid = Expr::rem(mids.clone(), Expr::u32(self.mid));
        let outer = Expr::div(mids, Expr::u32(self.mid));
        [outer, mid, row, column]
    }
}

/// A shape or dtype input no attention layout move can serve.
///
/// The rejection is shared; the reported error type is not, because the cache
/// append answers with a typed error and the conversions answer with a message
/// naming the entry point.
pub(crate) enum LayoutReject {
    /// At least one dimension is zero, so the dispatch would write nothing and
    /// report success.
    EmptyShape,
    /// The element dtype has no floating attention-activation contract.
    UnsupportedDtype(DataType),
}

/// Reject the shape and dtype inputs no attention layout move can serve.
pub(crate) fn check_layout_dims(dims: &[u32], dtype: &DataType) -> Result<(), LayoutReject> {
    if dims.iter().any(|dimension| *dimension == 0) {
        return Err(LayoutReject::EmptyShape);
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(LayoutReject::UnsupportedDtype(dtype.clone()));
    }
    Ok(())
}

/// Flattened element count of `dims`, or `None` when it overflows `u32`
/// indexing.
pub(crate) fn checked_elements(dims: &[u32]) -> Option<u32> {
    dims.iter()
        .try_fold(1_u32, |product, value| product.checked_mul(*value))
}

/// Where the value of one output element comes from.
pub(crate) enum IndexMap {
    /// `write[index] = read[source]`.
    Gather {
        /// Buffer the value is read from.
        read: String,
        /// Flat index into `read`, derived from the `index` binding.
        source: Expr,
    },
    /// `write[index] = if in_patch { patch[patch_index] } else { base[index] }`.
    ///
    /// The two loads stay in the two arms of a branch rather than being folded
    /// into a select: a select evaluates both sides, and the patch index is out
    /// of range for every element the branch does not take.
    Patch {
        /// Buffer supplying the elements the patch does not cover.
        base: String,
        /// Buffer supplying the patched elements.
        patch: String,
        /// Predicate that selects the patch, derived from the `index` binding.
        in_patch: Expr,
        /// Flat index into `patch`, derived from the `index` binding.
        patch_index: Expr,
    },
    /// `write[destination] = read[index]`.
    ///
    /// The inverse direction of [`IndexMap::Gather`], and the only map whose
    /// guard bounds the INPUT. A paged cache write touches one block slot out
    /// of a cache that is deliberately much larger, so bounding the output
    /// would dispatch the whole cache to move one token, which is the cost
    /// paging exists to avoid. The map has to be injective for the move to
    /// write each destination once; every caller here derives the destination
    /// from distinct source coordinates through one block table lookup.
    Scatter {
        /// Buffer the value is read from, indexed by the guarded index.
        read: String,
        /// Flat index into the written buffer, derived from the `index`
        /// binding.
        destination: Expr,
    },
}

/// One guarded element move: `buffers` in binding order, one invocation per
/// element of the dispatch domain, and `map` producing the stored value.
pub(crate) struct LayoutMove<'a> {
    /// Op id of the emitted region.
    pub(crate) op_id: &'static str,
    /// Every storage binding, in binding order, the output included.
    pub(crate) buffers: Vec<BufferDecl>,
    /// Written buffer name.
    pub(crate) write: &'a str,
    /// Guarded element count of the dispatch domain: the output for a gather
    /// or a patch, the input for a scatter.
    pub(crate) count: u32,
    /// Value source for one moved element.
    pub(crate) map: IndexMap,
}

/// Emit a layout move.
///
/// The guard bounds the OUTPUT index for a gather and a patch, which is the
/// invocation id, so every output element is written exactly once and a move
/// cannot leave a hole. A scatter bounds the input index instead and promises
/// the same coverage through an injective destination map rather than through
/// the guard.
pub(crate) fn layout_move_program(spec: LayoutMove<'_>) -> Program {
    let index = Expr::var("index");
    let store = |value: Expr| Node::Store {
        buffer: spec.write.into(),
        index: Expr::var("index"),
        value,
    };
    let moved = match spec.map {
        IndexMap::Gather { read, source } => vec![store(Expr::load(&read, source))],
        IndexMap::Patch {
            base,
            patch,
            in_patch,
            patch_index,
        } => vec![Node::if_then_else(
            in_patch,
            vec![store(Expr::load(&patch, patch_index))],
            vec![store(Expr::load(&base, index.clone()))],
        )],
        IndexMap::Scatter { read, destination } => vec![Node::Store {
            buffer: spec.write.into(),
            index: destination,
            value: Expr::load(&read, index.clone()),
        }],
    };
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(index, Expr::u32(spec.count)), moved),
    ];
    Program::wrapped(
        spec.buffers,
        ATTENTION_LAYOUT_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(spec.op_id, body)],
    )
}

/// Buffer names, shape, and dtype of one attention layout conversion.
///
/// The dimensions are named rather than positional because the two conversions
/// take them in different orders on the wire: head-major is
/// `[batch, heads, sequence, head_dim]` and token-major is
/// `[batch, sequence, heads, head_dim]`. Positional arguments made those two
/// orders look interchangeable at a call site.
pub struct AttentionPermuteSpec<'a> {
    /// Source buffer name.
    pub input: &'a str,
    /// Destination buffer name.
    pub output: &'a str,
    /// Batch count.
    pub batch: u32,
    /// Attention head count.
    pub heads: u32,
    /// Tokens per sequence.
    pub sequence: u32,
    /// Per-head feature width.
    pub head_dim: u32,
    /// Element dtype of both buffers.
    pub dtype: DataType,
}

impl AttentionPermuteSpec<'_> {
    /// Validate the conversion and return its flat element count.
    fn count(&self, label: &str) -> Result<u32, String> {
        let dims = [self.batch, self.heads, self.sequence, self.head_dim];
        check_layout_dims(&dims, &self.dtype).map_err(|reject| match reject {
            LayoutReject::EmptyShape => format!("Fix: {label} requires nonzero dimensions"),
            LayoutReject::UnsupportedDtype(dtype) => {
                format!("Fix: {label} requires a floating dtype; got {dtype:?}")
            }
        })?;
        checked_elements(&dims)
            .ok_or_else(|| format!("Fix: {label} element count overflows u32; shard the tensor"))
    }

    /// Head-major `[batch, heads, sequence, head_dim]` axis lengths.
    fn head_major(&self) -> RowMajor {
        RowMajor {
            mid: self.heads,
            row: self.sequence,
            width: self.head_dim,
        }
    }

    /// Token-major `[batch, sequence, heads, head_dim]` axis lengths.
    fn token_major(&self) -> RowMajor {
        RowMajor {
            mid: self.sequence,
            row: self.heads,
            width: self.head_dim,
        }
    }

    /// Emit the gather that writes `count` elements of `output`, each read from
    /// `source`.
    fn gather(self, op_id: &'static str, count: u32, source: Expr) -> Program {
        layout_move_program(LayoutMove {
            op_id,
            buffers: vec![
                BufferDecl::storage(self.input, 0, BufferAccess::ReadOnly, self.dtype.clone())
                    .with_count(count),
                BufferDecl::output(self.output, 1, self.dtype).with_count(count),
            ],
            write: self.output,
            count,
            map: IndexMap::Gather {
                read: self.input.into(),
                source,
            },
        })
    }
}

/// Convert `[batch, heads, sequence, head_dim]` into
/// `[batch, sequence, heads, head_dim]` without changing element values.
///
/// # Errors
///
/// Returns `Err` for a zero dimension, a non-float dtype, or a flattened
/// element count that overflows `u32` indexing.
pub fn attention_head_to_token(spec: AttentionPermuteSpec<'_>) -> Result<Program, String> {
    let count = spec.count("attention_head_to_token")?;
    let [batch, token, head, column] = spec.token_major().coords(&Expr::var("index"));
    let source = spec.head_major().index(batch, head, token, column);
    Ok(spec.gather(HEAD_TO_TOKEN_OP_ID, count, source))
}

/// Convert `[batch, sequence, heads, head_dim]` into
/// `[batch, heads, sequence, head_dim]` without changing element values.
///
/// # Errors
///
/// Returns `Err` for a zero dimension, a non-float dtype, or a flattened
/// element count that overflows `u32` indexing.
pub fn attention_token_to_head(spec: AttentionPermuteSpec<'_>) -> Result<Program, String> {
    let count = spec.count("attention_token_to_head")?;
    let [batch, head, token, column] = spec.head_major().coords(&Expr::var("index"));
    let source = spec.token_major().index(batch, token, head, column);
    Ok(spec.gather(TOKEN_TO_HEAD_OP_ID, count, source))
}

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

/// Buffer names, shape, and dtype of one KV cache append.
///
/// Cache and chunk are both `[batch, heads, tokens, head_dim]` and differ only
/// in their token count, so the two token counts are named rather than
/// positional.
pub struct KvCacheAppendSpec<'a> {
    /// Immutable prior cache generation.
    pub prior: &'a str,
    /// Chunk written over the prior cache.
    pub chunk: &'a str,
    /// Successor cache generation.
    pub next: &'a str,
    /// Batch count.
    pub batch: u32,
    /// Attention head count.
    pub heads: u32,
    /// Cache token capacity.
    pub capacity: u32,
    /// Chunk token count.
    pub chunk_len: u32,
    /// Per-head feature width.
    pub head_dim: u32,
    /// First cache token the chunk replaces.
    pub offset: u32,
    /// Element dtype of every cache buffer.
    pub dtype: DataType,
}

/// Copy an immutable prior cache and replace one contiguous token range.
///
/// `prior` is never modified. `next` holds the complete successor generation,
/// so a caller can keep the prior generation addressable for as long as it
/// needs it.
///
/// # Errors
///
/// Returns `Err` for a zero dimension, a non-float dtype, a chunk that does not
/// fit at `offset`, or a flattened element count that overflows `u32` indexing.
pub fn kv_cache_append(spec: KvCacheAppendSpec<'_>) -> Result<Program, KvCacheAppendError> {
    let dtype = spec.dtype.clone();
    let dims = [
        spec.batch,
        spec.heads,
        spec.capacity,
        spec.chunk_len,
        spec.head_dim,
    ];
    check_layout_dims(&dims, &dtype).map_err(|reject| match reject {
        LayoutReject::EmptyShape => KvCacheAppendError::EmptyShape,
        LayoutReject::UnsupportedDtype(dtype) => KvCacheAppendError::UnsupportedDtype { dtype },
    })?;
    let end = spec
        .offset
        .checked_add(spec.chunk_len)
        .filter(|end| *end <= spec.capacity)
        .ok_or(KvCacheAppendError::Range {
            offset: spec.offset,
            chunk_len: spec.chunk_len,
            capacity: spec.capacity,
        })?;
    let elements =
        |dims: &[u32]| checked_elements(dims).ok_or(KvCacheAppendError::ElementCountOverflow);
    let cache_count = elements(&[spec.batch, spec.heads, spec.capacity, spec.head_dim])?;
    let chunk_count = elements(&[spec.batch, spec.heads, spec.chunk_len, spec.head_dim])?;
    let cache = RowMajor {
        mid: spec.heads,
        row: spec.capacity,
        width: spec.head_dim,
    };
    let chunk = RowMajor {
        mid: spec.heads,
        row: spec.chunk_len,
        width: spec.head_dim,
    };
    let [batch, head, token, column] = cache.coords(&Expr::var("index"));
    let patch_index = chunk.index(
        batch,
        head,
        Expr::sub(token.clone(), Expr::u32(spec.offset)),
        column,
    );
    let in_patch = Expr::and(
        Expr::ge(token.clone(), Expr::u32(spec.offset)),
        Expr::lt(token, Expr::u32(end)),
    );
    Ok(layout_move_program(LayoutMove {
        op_id: KV_CACHE_APPEND_OP_ID,
        buffers: vec![
            BufferDecl::storage(spec.prior, 0, BufferAccess::ReadWrite, dtype.clone())
                .with_count(cache_count),
            BufferDecl::storage(spec.chunk, 1, BufferAccess::ReadOnly, dtype.clone())
                .with_count(chunk_count),
            BufferDecl::output(spec.next, 2, dtype).with_count(cache_count),
        ],
        write: spec.next,
        count: cache_count,
        map: IndexMap::Patch {
            base: spec.prior.into(),
            patch: spec.chunk.into(),
            in_patch,
            patch_index,
        },
    }))
}
