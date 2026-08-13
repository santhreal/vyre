//! Recurrent gated delta-rule attention with explicit matrix state.

use thiserror::Error;
use vyre_foundation::ir::{DataType, Expr, Node, Program, UnOp};

use super::gated_delta_layout::{self, GatedDeltaSpec};
use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::recurrent_gated_delta";

/// Invalid recurrent gated delta construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RecurrentGatedDeltaError {
    /// A required dimension is zero.
    #[error(
        "recurrent gated delta requires nonzero batch, sequence, head, key, and value dimensions"
    )]
    EmptyShape,
    /// Value heads cannot evenly repeat key heads.
    #[error("recurrent gated delta value_heads={value_heads} must be divisible by key_heads={key_heads}")]
    InvalidHeadGrouping {
        /// Key/query head count before repetition.
        key_heads: u32,
        /// Value/state head count.
        value_heads: u32,
    },
    /// A flattened tensor size exceeds u32 indexing.
    #[error("recurrent gated delta tensor element count overflows u32; split the tensor")]
    ElementCountOverflow,
    /// Source dtype lacks the required conversion contract.
    #[error("recurrent gated delta supports F16, BF16, or F32 activations; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected activation dtype.
        dtype: DataType,
    },
}

/// Build a token-recurrent gated delta update.
#[allow(clippy::too_many_arguments)]
pub fn recurrent_gated_delta(
    query: &str,
    key: &str,
    value: &str,
    decay_log: &str,
    beta_logits: &str,
    state_input: &str,
    output: &str,
    state_output: &str,
    batch: u32,
    sequence: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    eps: f32,
    dtype: DataType,
) -> Result<Program, RecurrentGatedDeltaError> {
    recurrent_gated_delta_impl(&GatedDeltaSpec {
        query,
        key,
        value,
        decay_log,
        beta_logits,
        state_input,
        output,
        state_output,
        batch,
        sequence,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        eps,
        dtype,
    })
}

/// Build a fixed-size-64 chunk schedule for gated delta prefill.
///
/// The schedule retains the exact recurrent dependency inside each causal
/// lower-triangular tile. Its final tile is padded structurally and guarded,
/// so padding cannot read inputs, modify state, or appear in the output.
#[allow(clippy::too_many_arguments)]
pub fn chunked_gated_delta(
    query: &str,
    key: &str,
    value: &str,
    decay_log: &str,
    beta_logits: &str,
    state_input: &str,
    output: &str,
    state_output: &str,
    batch: u32,
    sequence: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    eps: f32,
    dtype: DataType,
) -> Result<Program, RecurrentGatedDeltaError> {
    super::gated_delta_chunked::chunked_gated_delta_impl(&GatedDeltaSpec {
        query,
        key,
        value,
        decay_log,
        beta_logits,
        state_input,
        output,
        state_output,
        batch,
        sequence,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        eps,
        dtype,
    })
}

/// Token-recurrent gated delta arithmetic.
///
/// Q and K are L2-normalized in F32. `decay_log` is exponentiated and
/// `beta_logits` is passed through sigmoid. Matrix state remains F32; activation
/// output converts once to `dtype`. `state_input` is preserved and
/// `state_output` receives the continued generation.
fn recurrent_gated_delta_impl(
    spec: &GatedDeltaSpec<'_>,
) -> Result<Program, RecurrentGatedDeltaError> {
    let counts = spec.counts()?;
    let GatedDeltaSpec {
        query,
        key,
        value,
        decay_log,
        beta_logits,
        state_input,
        output,
        state_output,
        sequence,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        eps,
        ref dtype,
        ..
    } = *spec;

    let qk_index =
        |dim: Expr| gated_delta_layout::qk_index(sequence, key_heads, key_dim, Expr::var("token"), dim);
    let state_index = |key_index: Expr, value_index: Expr| {
        gated_delta_layout::state_index(key_dim, value_dim, key_index, value_index)
    };
    let value_index = |dim: Expr| {
        gated_delta_layout::value_index(sequence, value_heads, value_dim, Expr::var("token"), dim)
    };
    let scalar_index = gated_delta_layout::scalar_index(sequence, value_heads, Expr::var("token"));
    let output_index = value_index(Expr::var("value_index"));

    let init_state = Node::loop_for(
        "key_index",
        Expr::u32(0),
        Expr::u32(key_dim),
        vec![Node::loop_for(
            "value_index",
            Expr::u32(0),
            Expr::u32(value_dim),
            vec![Node::Store {
                buffer: state_output.into(),
                index: state_index(Expr::var("key_index"), Expr::var("value_index")),
                value: Expr::load(
                    state_input,
                    state_index(Expr::var("key_index"), Expr::var("value_index")),
                ),
            }],
        )],
    );
    let norm_sums = vec![
        Node::let_bind("query_sum", Expr::f32(0.0)),
        Node::let_bind("key_sum", Expr::f32(0.0)),
        Node::loop_for(
            "key_index",
            Expr::u32(0),
            Expr::u32(key_dim),
            vec![
                Node::let_bind(
                    "query_component",
                    Expr::cast(
                        DataType::F32,
                        Expr::load(query, qk_index(Expr::var("key_index"))),
                    ),
                ),
                Node::let_bind(
                    "key_component",
                    Expr::cast(
                        DataType::F32,
                        Expr::load(key, qk_index(Expr::var("key_index"))),
                    ),
                ),
                Node::assign(
                    "query_sum",
                    Expr::add(
                        Expr::var("query_sum"),
                        Expr::mul(Expr::var("query_component"), Expr::var("query_component")),
                    ),
                ),
                Node::assign(
                    "key_sum",
                    Expr::add(
                        Expr::var("key_sum"),
                        Expr::mul(Expr::var("key_component"), Expr::var("key_component")),
                    ),
                ),
            ],
        ),
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
        ),
        Node::let_bind(
            "key_scale",
            Expr::UnOp {
                op: UnOp::InverseSqrt,
                operand: Box::new(Expr::add(Expr::var("key_sum"), Expr::f32(eps))),
            },
        ),
    ];
    let decay_state = Node::loop_for(
        "key_index",
        Expr::u32(0),
        Expr::u32(key_dim),
        vec![Node::loop_for(
            "value_index",
            Expr::u32(0),
            Expr::u32(value_dim),
            vec![Node::Store {
                buffer: state_output.into(),
                index: state_index(Expr::var("key_index"), Expr::var("value_index")),
                value: Expr::mul(
                    Expr::load(
                        state_output,
                        state_index(Expr::var("key_index"), Expr::var("value_index")),
                    ),
                    Expr::var("decay"),
                ),
            }],
        )],
    );
    let value_update = Node::loop_for(
        "value_index",
        Expr::u32(0),
        Expr::u32(value_dim),
        vec![
            Node::let_bind("memory", Expr::f32(0.0)),
            Node::loop_for(
                "key_index",
                Expr::u32(0),
                Expr::u32(key_dim),
                vec![Node::assign(
                    "memory",
                    Expr::add(
                        Expr::var("memory"),
                        Expr::mul(
                            Expr::load(
                                state_output,
                                state_index(Expr::var("key_index"), Expr::var("value_index")),
                            ),
                            Expr::mul(
                                Expr::cast(
                                    DataType::F32,
                                    Expr::load(key, qk_index(Expr::var("key_index"))),
                                ),
                                Expr::var("key_scale"),
                            ),
                        ),
                    ),
                )],
            ),
            Node::let_bind(
                "delta",
                Expr::mul(
                    Expr::sub(
                        Expr::cast(
                            DataType::F32,
                            Expr::load(value, value_index(Expr::var("value_index"))),
                        ),
                        Expr::var("memory"),
                    ),
                    Expr::var("beta"),
                ),
            ),
            Node::loop_for(
                "key_index",
                Expr::u32(0),
                Expr::u32(key_dim),
                vec![Node::Store {
                    buffer: state_output.into(),
                    index: state_index(Expr::var("key_index"), Expr::var("value_index")),
                    value: Expr::add(
                        Expr::load(
                            state_output,
                            state_index(Expr::var("key_index"), Expr::var("value_index")),
                        ),
                        Expr::mul(
                            Expr::mul(
                                Expr::cast(
                                    DataType::F32,
                                    Expr::load(key, qk_index(Expr::var("key_index"))),
                                ),
                                Expr::var("key_scale"),
                            ),
                            Expr::var("delta"),
                        ),
                    ),
                }],
            ),
            Node::let_bind("attention_output", Expr::f32(0.0)),
            Node::loop_for(
                "key_index",
                Expr::u32(0),
                Expr::u32(key_dim),
                vec![Node::assign(
                    "attention_output",
                    Expr::add(
                        Expr::var("attention_output"),
                        Expr::mul(
                            Expr::load(
                                state_output,
                                state_index(Expr::var("key_index"), Expr::var("value_index")),
                            ),
                            Expr::mul(
                                Expr::cast(
                                    DataType::F32,
                                    Expr::load(query, qk_index(Expr::var("key_index"))),
                                ),
                                Expr::var("query_scale"),
                            ),
                        ),
                    ),
                )],
            ),
            Node::Store {
                buffer: output.into(),
                index: output_index,
                value: Expr::cast(dtype.clone(), Expr::var("attention_output")),
            },
        ],
    );
    let mut token_body = norm_sums;
    token_body.extend([
        Node::let_bind(
            "decay",
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(Expr::cast(
                    DataType::F32,
                    Expr::load(decay_log, scalar_index.clone()),
                )),
            },
        ),
        Node::let_bind(
            "beta_logit",
            Expr::cast(DataType::F32, Expr::load(beta_logits, scalar_index)),
        ),
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
        ),
        decay_state,
        value_update,
    ]);
    let token_schedule = Node::loop_for("token", Expr::u32(0), Expr::u32(sequence), token_body);
    let body = vec![
        Node::let_bind("head_index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("head_index"), Expr::u32(counts.head)),
            vec![
                Node::let_bind(
                    "batch_index",
                    Expr::div(Expr::var("head_index"), Expr::u32(value_heads)),
                ),
                Node::let_bind(
                    "value_head",
                    Expr::sub(
                        Expr::var("head_index"),
                        Expr::mul(Expr::var("batch_index"), Expr::u32(value_heads)),
                    ),
                ),
                Node::let_bind(
                    "key_head",
                    Expr::div(Expr::var("value_head"), Expr::u32(counts.group)),
                ),
                init_state,
                token_schedule,
            ],
        ),
    ];

    Ok(Program::wrapped(
        gated_delta_layout::gated_delta_buffers(spec, &counts),
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}
