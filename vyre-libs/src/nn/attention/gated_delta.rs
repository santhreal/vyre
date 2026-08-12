//! Recurrent gated delta-rule attention with explicit matrix state.

use thiserror::Error;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

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
    recurrent_gated_delta_impl(
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
    )
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
    super::gated_delta_chunked::chunked_gated_delta_impl(
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
    )
}

/// Token-recurrent gated delta arithmetic.
///
/// Q and K are L2-normalized in F32. `decay_log` is exponentiated and
/// `beta_logits` is passed through sigmoid. Matrix state remains F32; activation
/// output converts once to `dtype`. `state_input` is preserved and
/// `state_output` receives the continued generation.
#[allow(clippy::too_many_arguments)]
fn recurrent_gated_delta_impl(
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
    if batch == 0
        || sequence == 0
        || key_heads == 0
        || value_heads == 0
        || key_dim == 0
        || value_dim == 0
    {
        return Err(RecurrentGatedDeltaError::EmptyShape);
    }
    if value_heads % key_heads != 0 {
        return Err(RecurrentGatedDeltaError::InvalidHeadGrouping {
            key_heads,
            value_heads,
        });
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(RecurrentGatedDeltaError::UnsupportedDtype { dtype });
    }
    let checked = |values: &[u32]| {
        values.iter().try_fold(1_u32, |product, value| {
            product
                .checked_mul(*value)
                .ok_or(RecurrentGatedDeltaError::ElementCountOverflow)
        })
    };
    let qk_count = checked(&[batch, sequence, key_heads, key_dim])?;
    let value_count = checked(&[batch, sequence, value_heads, value_dim])?;
    let scalar_count = checked(&[batch, sequence, value_heads])?;
    let state_count = checked(&[batch, value_heads, key_dim, value_dim])?;
    let head_count = checked(&[batch, value_heads])?;
    let group = value_heads / key_heads;

    let qk_index = |dim: Expr| {
        Expr::add(
            Expr::mul(
                Expr::add(
                    Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                    Expr::var("token"),
                ),
                Expr::u32(key_heads * key_dim),
            ),
            Expr::add(Expr::mul(Expr::var("key_head"), Expr::u32(key_dim)), dim),
        )
    };
    let state_index = |key_index: Expr, value_index: Expr| {
        Expr::add(
            Expr::mul(Expr::var("head_index"), Expr::u32(key_dim * value_dim)),
            Expr::add(Expr::mul(key_index, Expr::u32(value_dim)), value_index),
        )
    };
    let value_index = |dim: Expr| {
        Expr::add(
            Expr::mul(
                Expr::add(
                    Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                    Expr::var("token"),
                ),
                Expr::u32(value_heads * value_dim),
            ),
            Expr::add(
                Expr::mul(Expr::var("value_head"), Expr::u32(value_dim)),
                dim,
            ),
        )
    };
    let scalar_index = Expr::add(
        Expr::mul(
            Expr::add(
                Expr::mul(Expr::var("batch_index"), Expr::u32(sequence)),
                Expr::var("token"),
            ),
            Expr::u32(value_heads),
        ),
        Expr::var("value_head"),
    );
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
            Expr::lt(Expr::var("head_index"), Expr::u32(head_count)),
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
                    Expr::div(Expr::var("value_head"), Expr::u32(group)),
                ),
                init_state,
                token_schedule,
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(query, 0, BufferAccess::ReadOnly, dtype.clone())
                .with_count(qk_count),
            BufferDecl::storage(key, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(qk_count),
            BufferDecl::storage(value, 2, BufferAccess::ReadOnly, dtype.clone())
                .with_count(value_count),
            BufferDecl::storage(decay_log, 3, BufferAccess::ReadOnly, dtype.clone())
                .with_count(scalar_count),
            BufferDecl::storage(beta_logits, 4, BufferAccess::ReadOnly, dtype.clone())
                .with_count(scalar_count),
            BufferDecl::storage(state_input, 5, BufferAccess::ReadWrite, DataType::F32)
                .with_count(state_count),
            BufferDecl::output(output, 6, dtype).with_count(value_count),
            BufferDecl::storage(state_output, 7, BufferAccess::ReadWrite, DataType::F32)
                .with_count(state_count),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}
