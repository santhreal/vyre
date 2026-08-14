//! Shape, index, buffer, and per-token numerics shared by the two gated delta
//! schedules.
//!
//! [`recurrent_gated_delta`](super::gated_delta::recurrent_gated_delta) and
//! [`chunked_gated_delta`](super::gated_delta::chunked_gated_delta) address the
//! same six tensors with the same flat layout and validate the same shape
//! contract; only their schedules differ. Both used to carry a private copy of
//! that math. The copies are merged here so a layout change cannot land in one
//! schedule and miss the other.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, UnOp};

use super::gated_delta::RecurrentGatedDeltaError;

/// Buffer names, shape, and numerics for one gated delta build.
///
/// Both schedules take the same sixteen inputs; carrying them as one value
/// keeps the two builders' entry points and their impls from drifting apart
/// argument by argument.
pub(super) struct GatedDeltaSpec<'a> {
    /// Query activations, `[batch, sequence, key_heads, key_dim]`.
    pub(super) query: &'a str,
    /// Key activations, `[batch, sequence, key_heads, key_dim]`.
    pub(super) key: &'a str,
    /// Value activations, `[batch, sequence, value_heads, value_dim]`.
    pub(super) value: &'a str,
    /// Per-token log decay, `[batch, sequence, value_heads]`.
    pub(super) decay_log: &'a str,
    /// Per-token pre-sigmoid gate, `[batch, sequence, value_heads]`.
    pub(super) beta_logits: &'a str,
    /// Incoming matrix state, `[batch, value_heads, key_dim, value_dim]`.
    pub(super) state_input: &'a str,
    /// Attention output, shaped like `value`.
    pub(super) output: &'a str,
    /// Continued matrix state, shaped like `state_input`.
    pub(super) state_output: &'a str,
    /// Batch count.
    pub(super) batch: u32,
    /// Tokens per sequence.
    pub(super) sequence: u32,
    /// Query/key head count.
    pub(super) key_heads: u32,
    /// Value/state head count; a multiple of `key_heads`.
    pub(super) value_heads: u32,
    /// Query/key feature width.
    pub(super) key_dim: u32,
    /// Value/state feature width.
    pub(super) value_dim: u32,
    /// L2-normalization epsilon.
    pub(super) eps: f32,
    /// Activation dtype for every non-state buffer.
    pub(super) dtype: DataType,
}

/// Flattened element counts derived from a validated [`GatedDeltaSpec`].
pub(super) struct GatedDeltaCounts {
    /// Elements in `query` and in `key`.
    pub(super) qk: u32,
    /// Elements in `value` and in `output`.
    pub(super) value: u32,
    /// Elements in `decay_log` and in `beta_logits`.
    pub(super) scalar: u32,
    /// Elements in `state_input` and in `state_output`.
    pub(super) state: u32,
    /// Dispatched `[batch, value_heads]` head slots.
    pub(super) head: u32,
    /// Value heads sharing one key head.
    pub(super) group: u32,
}

impl GatedDeltaSpec<'_> {
    /// Validate the shape contract and derive the flattened element counts.
    ///
    /// # Errors
    ///
    /// Rejects zero dimensions, a `value_heads` that is not a multiple of
    /// `key_heads`, a non-float activation dtype, and any tensor whose flat
    /// element count overflows `u32`.
    pub(super) fn counts(&self) -> Result<GatedDeltaCounts, RecurrentGatedDeltaError> {
        if self.batch == 0
            || self.sequence == 0
            || self.key_heads == 0
            || self.value_heads == 0
            || self.key_dim == 0
            || self.value_dim == 0
        {
            return Err(RecurrentGatedDeltaError::EmptyShape);
        }
        if self.value_heads % self.key_heads != 0 {
            return Err(RecurrentGatedDeltaError::InvalidHeadGrouping {
                key_heads: self.key_heads,
                value_heads: self.value_heads,
            });
        }
        if !matches!(self.dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(RecurrentGatedDeltaError::UnsupportedDtype {
                dtype: self.dtype.clone(),
            });
        }
        Ok(GatedDeltaCounts {
            qk: checked(&[self.batch, self.sequence, self.key_heads, self.key_dim])?,
            value: checked(&[self.batch, self.sequence, self.value_heads, self.value_dim])?,
            scalar: checked(&[self.batch, self.sequence, self.value_heads])?,
            state: checked(&[self.batch, self.value_heads, self.key_dim, self.value_dim])?,
            head: checked(&[self.batch, self.value_heads])?,
            group: self.value_heads / self.key_heads,
        })
    }
}

/// Multiply `values`, rejecting a product that overflows `u32` indexing.
pub(super) fn checked(values: &[u32]) -> Result<u32, RecurrentGatedDeltaError> {
    values.iter().try_fold(1_u32, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(RecurrentGatedDeltaError::ElementCountOverflow)
    })
}

/// Flat index into a `[batch, sequence, heads, dim]` query or key tensor,
/// keyed off the enclosing `batch_index` and `key_head` bindings.
pub(super) fn qk_index(sequence: u32, heads: u32, dim: u32, token: Expr, feature: Expr) -> Expr {
    Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                token,
            ),
            Expr::u32(heads * dim),
        ),
        Expr::add(Expr::mul(Expr::var("key_head"), Expr::u32(dim)), feature),
    )
}

/// Flat index into a `[batch, sequence, heads, dim]` value or output tensor,
/// keyed off the enclosing `batch_index` and `value_head` bindings.
pub(super) fn value_index(sequence: u32, heads: u32, dim: u32, token: Expr, feature: Expr) -> Expr {
    Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                token,
            ),
            Expr::u32(heads * dim),
        ),
        Expr::add(Expr::mul(Expr::var("value_head"), Expr::u32(dim)), feature),
    )
}

/// Flat index into a `[batch, sequence, heads]` per-token scalar tensor.
pub(super) fn scalar_index(sequence: u32, heads: u32, token: Expr) -> Expr {
    Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                token,
            ),
            Expr::u32(heads),
        ),
        Expr::var("value_head"),
    )
}

/// Flat index into the `[head, key_dim, value_dim]` matrix state.
pub(super) fn state_index(key_dim: u32, value_dim: u32, key: Expr, value: Expr) -> Expr {
    Expr::add(
        Expr::mul(Expr::var("head_index"), Expr::u32(key_dim * value_dim)),
        Expr::add(Expr::mul(key, Expr::u32(value_dim)), value),
    )
}

/// The eight storage bindings both schedules declare, in binding order.
///
/// A schedule that needs workgroup scratch extends the returned vector.
pub(super) fn gated_delta_buffers(
    spec: &GatedDeltaSpec<'_>,
    counts: &GatedDeltaCounts,
) -> Vec<BufferDecl> {
    let dtype = &spec.dtype;
    vec![
        BufferDecl::storage(spec.query, 0, BufferAccess::ReadOnly, dtype.clone())
            .with_count(counts.qk),
        BufferDecl::storage(spec.key, 1, BufferAccess::ReadOnly, dtype.clone())
            .with_count(counts.qk),
        BufferDecl::storage(spec.value, 2, BufferAccess::ReadOnly, dtype.clone())
            .with_count(counts.value),
        BufferDecl::storage(spec.decay_log, 3, BufferAccess::ReadOnly, dtype.clone())
            .with_count(counts.scalar),
        BufferDecl::storage(spec.beta_logits, 4, BufferAccess::ReadOnly, dtype.clone())
            .with_count(counts.scalar),
        BufferDecl::storage(spec.state_input, 5, BufferAccess::ReadWrite, DataType::F32)
            .with_count(counts.state),
        BufferDecl::output(spec.output, 6, dtype.clone()).with_count(counts.value),
        BufferDecl::storage(spec.state_output, 7, BufferAccess::ReadWrite, DataType::F32)
            .with_count(counts.state),
    ]
}

/// The delta-rule write strength for the current token:
/// `beta = sigmoid(beta_logit) = 1 / (1 + exp(-beta_logit))`.
///
/// Both schedules bind `beta_logit` from `beta_logits` their own way (the
/// recurrent one per token, the chunked one per triangular row) and then applied
/// this identical sigmoid, so the gate's numeric form lives here once.
///
/// The form is deliberately the direct reciprocal rather than the
/// overflow-shy `exp(x) / (1 + exp(x))` branch: `exp` of a large NEGATIVE logit
/// saturates toward zero rather than overflowing, which is the harmless
/// direction. It does not clamp, because a caller feeding a poisoned logit
/// should see the NaN rather than a silently full-strength state write.
pub(super) fn beta_gate_node() -> Node {
    Node::let_bind(
        "beta",
        Expr::div(
            Expr::f32(1.0),
            Expr::add(
                Expr::f32(1.0),
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(Expr::UnOp {
                        op: UnOp::Negate,
                        operand: Box::new(Expr::var("beta_logit")),
                    }),
                },
            ),
        ),
    )
}

/// `1 / sqrt(sum + eps)`: the L2 normalizer both schedules apply to a
/// squared-magnitude accumulator.
///
/// `sum` names the accumulator binding and `scale` the binding to produce; the
/// chunked schedule prefixes both per triangular position, which is the only
/// position that differs. `eps` is added inside the root, so a zero-magnitude
/// row scales by `1 / sqrt(eps)` instead of dividing by zero.
pub(super) fn l2_scale_node(scale: impl Into<Ident>, sum: &str, eps: f32) -> Node {
    Node::let_bind(
        scale,
        Expr::UnOp {
            op: UnOp::InverseSqrt,
            operand: Box::new(Expr::add(Expr::var(sum), Expr::f32(eps))),
        },
    )
}

/// The query scale: the L2 normalizer over `query_sum`, multiplied by the
/// `1 / sqrt(key_dim)` attention-logit scale.
///
/// Folding the two roots into one binding is why this is an owner and not two
/// calls to [`l2_scale_node`]: the recurrent and chunked schedules must agree on
/// WHICH scale carries the `1 / sqrt(key_dim)` factor, or the same weights
/// produce different logits on the two paths. The key side takes the plain
/// normalizer and the query side carries the head-dimension factor. Both
/// schedules accumulate into `query_sum`, so that binding is fixed here rather
/// than a parameter.
pub(super) fn query_scale_node(eps: f32, key_dim: u32) -> Node {
    Node::let_bind(
        "query_scale",
        Expr::mul(
            Expr::UnOp {
                op: UnOp::InverseSqrt,
                operand: Box::new(Expr::add(Expr::var("query_sum"), Expr::f32(eps))),
            },
            Expr::UnOp {
                op: UnOp::InverseSqrt,
                operand: Box::new(Expr::f32(key_dim as f32)),
            },
        ),
    )
}
