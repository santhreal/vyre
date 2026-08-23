//! Shape, buffer, and per-token numerics shared by the two gated delta
//! schedules.
//!
//! [`recurrent_gated_delta`](super::gated_delta::recurrent_gated_delta) and
//! [`chunked_gated_delta`](super::gated_delta_chunked::chunked_gated_delta)
//! address the same six tensors with the same flat layout and validate the same
//! shape contract; only their schedules differ. Both used to carry a private
//! copy of that math. The copies are merged here so a layout change cannot land
//! in one schedule and miss the other. The flat indices themselves come from
//! [`super::layout`], which is what every layout move in this subtree is built
//! from.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, UnOp};

use super::gated_delta::RecurrentGatedDeltaError;
use super::layout::{block_index, checked_elements, RowMajor};

/// Buffer names, shape, and numerics for one gated delta build.
///
/// Both schedules take the same sixteen inputs; carrying them as one value
/// keeps the two builders' entry points and their impls from drifting apart
/// argument by argument.
pub struct GatedDeltaSpec<'a> {
    /// Query activations, `[batch, sequence, key_heads, key_dim]`.
    pub query: &'a str,
    /// Key activations, `[batch, sequence, key_heads, key_dim]`.
    pub key: &'a str,
    /// Value activations, `[batch, sequence, value_heads, value_dim]`.
    pub value: &'a str,
    /// Per-token log decay, `[batch, sequence, value_heads]`.
    pub decay_log: &'a str,
    /// Per-token pre-sigmoid gate, `[batch, sequence, value_heads]`.
    pub beta_logits: &'a str,
    /// Incoming matrix state, `[batch, value_heads, key_dim, value_dim]`.
    pub state_input: &'a str,
    /// Attention output, shaped like `value`.
    pub output: &'a str,
    /// Continued matrix state, shaped like `state_input`.
    pub state_output: &'a str,
    /// Batch count.
    pub batch: u32,
    /// Tokens per sequence.
    pub sequence: u32,
    /// Query/key head count.
    pub key_heads: u32,
    /// Value/state head count; a multiple of `key_heads`.
    pub value_heads: u32,
    /// Query/key feature width.
    pub key_dim: u32,
    /// Value/state feature width.
    pub value_dim: u32,
    /// L2-normalization epsilon.
    pub eps: f32,
    /// Activation dtype for every non-state buffer.
    pub dtype: DataType,
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
    checked_elements(values).ok_or(RecurrentGatedDeltaError::ElementCountOverflow)
}

/// Flat index into a `[batch, sequence, heads, dim]` activation tensor, keyed
/// off the enclosing `batch_index` binding and the `head` binding named by the
/// caller.
///
/// The head binding is the only position that differed between the query/key
/// copy and the value/output copy of this index: the two tensors carry
/// different head counts and different widths, but one layout.
pub(super) fn activation_index(
    head: &str,
    sequence: u32,
    heads: u32,
    dim: u32,
    token: Expr,
    feature: Expr,
) -> Expr {
    RowMajor {
        mid: sequence,
        row: heads,
        width: dim,
    }
    .index(Expr::var("batch_index"), token, Expr::var(head), feature)
}

/// Flat index into a `[batch, sequence, heads]` per-token scalar tensor.
pub(super) fn scalar_index(sequence: u32, heads: u32, token: Expr) -> Expr {
    block_index(
        block_index(Expr::var("batch_index"), sequence, token),
        heads,
        Expr::var("value_head"),
    )
}

/// Flat index into the `[head, key_dim, value_dim]` matrix state.
pub(super) fn state_index(key_dim: u32, value_dim: u32, key: Expr, value: Expr) -> Expr {
    block_index(
        Expr::var("head_index"),
        key_dim * value_dim,
        block_index(key, value_dim, value),
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

/// The head decomposition both schedules open with, with `body` inside the
/// dispatch guard.
///
/// One invocation owns one `[batch, value_head]` slot. The two schedules used
/// to spell the same decomposition two ways, one dividing and subtracting and
/// the other taking a remainder, so a head-layout change had two places to
/// land. It has one now.
pub(super) fn head_partition(
    counts: &GatedDeltaCounts,
    value_heads: u32,
    body: Vec<Node>,
) -> Vec<Node> {
    let mut guarded = vec![
        Node::let_bind(
            "batch_index",
            Expr::div(Expr::var("head_index"), Expr::u32(value_heads)),
        ),
        Node::let_bind(
            "value_head",
            Expr::rem(Expr::var("head_index"), Expr::u32(value_heads)),
        ),
        Node::let_bind(
            "key_head",
            Expr::div(Expr::var("value_head"), Expr::u32(counts.group)),
        ),
    ];
    guarded.extend(body);
    vec![
        Node::let_bind("head_index", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("head_index"), Expr::u32(counts.head)),
            guarded,
        ),
    ]
}

/// Copy the incoming matrix state into `state_output` so the schedule can
/// update it in place and leave `state_input` readable.
pub(super) fn init_state_copy(
    state_input: &str,
    state_output: &str,
    key_dim: u32,
    value_dim: u32,
) -> Node {
    Node::loop_for(
        "state_key",
        Expr::u32(0),
        Expr::u32(key_dim),
        vec![Node::loop_for(
            "state_value",
            Expr::u32(0),
            Expr::u32(value_dim),
            vec![Node::Store {
                buffer: state_output.into(),
                index: state_index(
                    key_dim,
                    value_dim,
                    Expr::var("state_key"),
                    Expr::var("state_value"),
                ),
                value: Expr::load(
                    state_input,
                    state_index(
                        key_dim,
                        value_dim,
                        Expr::var("state_key"),
                        Expr::var("state_value"),
                    ),
                ),
            }],
        )],
    )
}

/// Accumulate one key row's squared magnitude and bind its L2 normalizer.
///
/// `prefix` names the four bindings this emits, so a schedule that needs the
/// normalizer of several key rows at once keeps them apart. The recurrent
/// schedule needs exactly one row per token and the chunked schedule needs the
/// current, paired, state, and future rows of a tile.
pub(super) fn key_norm_nodes(
    key: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    eps: f32,
    token: Expr,
    prefix: &str,
) -> Vec<Node> {
    let sum = format!("{prefix}_key_sum");
    let component = format!("{prefix}_key_component");
    let dimension = format!("{prefix}_norm_dimension");
    let scale = format!("{prefix}_key_scale");
    vec![
        Node::let_bind(sum.clone(), Expr::f32(0.0)),
        Node::loop_for(
            dimension.clone(),
            Expr::u32(0),
            Expr::u32(key_dim),
            vec![
                Node::let_bind(
                    component.clone(),
                    Expr::cast(
                        DataType::F32,
                        Expr::load(
                            key,
                            activation_index(
                                "key_head",
                                sequence,
                                key_heads,
                                key_dim,
                                token,
                                Expr::var(dimension),
                            ),
                        ),
                    ),
                ),
                Node::assign(
                    sum.clone(),
                    Expr::add(
                        Expr::var(sum.clone()),
                        Expr::mul(Expr::var(component.clone()), Expr::var(component)),
                    ),
                ),
            ],
        ),
        l2_scale_node(scale, &sum, eps),
    ]
}

/// One L2-normalized key component, scaled by the binding `scale` names.
pub(super) fn normalized_key(
    key: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    token: Expr,
    feature: Expr,
    scale: &str,
) -> Expr {
    Expr::mul(
        Expr::cast(
            DataType::F32,
            Expr::load(
                key,
                activation_index("key_head", sequence, key_heads, key_dim, token, feature),
            ),
        ),
        Expr::var(scale),
    )
}

/// Accumulate the query row's squared magnitude and bind `query_scale`.
///
/// The query normalizer carries the `1 / sqrt(key_dim)` logit factor, so it is
/// a separate builder from [`key_norm_nodes`] rather than the same one under a
/// different prefix.
pub(super) fn query_norm_nodes(
    query: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    eps: f32,
    token: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("query_sum", Expr::f32(0.0)),
        Node::loop_for(
            "query_norm_dimension",
            Expr::u32(0),
            Expr::u32(key_dim),
            vec![
                Node::let_bind(
                    "query_component",
                    Expr::cast(
                        DataType::F32,
                        Expr::load(
                            query,
                            activation_index(
                                "key_head",
                                sequence,
                                key_heads,
                                key_dim,
                                token,
                                Expr::var("query_norm_dimension"),
                            ),
                        ),
                    ),
                ),
                Node::assign(
                    "query_sum",
                    Expr::add(
                        Expr::var("query_sum"),
                        Expr::mul(Expr::var("query_component"), Expr::var("query_component")),
                    ),
                ),
            ],
        ),
        query_scale_node(eps, key_dim),
    ]
}

/// One query component, normalized and carrying the attention-logit scale.
pub(super) fn scaled_query(
    query: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    token: Expr,
    feature: Expr,
) -> Expr {
    Expr::mul(
        Expr::cast(
            DataType::F32,
            Expr::load(
                query,
                activation_index("key_head", sequence, key_heads, key_dim, token, feature),
            ),
        ),
        Expr::var("query_scale"),
    )
}
/// One query-state product term for attention accumulation.
pub(super) fn query_state_product(
    query: &str,
    state_buf: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    token: Expr,
    key_index: Expr,
    state_index: Expr,
) -> Expr {
    Expr::mul(
        scaled_query(query, sequence, key_heads, key_dim, token, key_index),
        Expr::load(state_buf, state_index),
    )
}
