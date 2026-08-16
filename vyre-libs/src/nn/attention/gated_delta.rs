//! Recurrent gated delta-rule attention with explicit matrix state.

use thiserror::Error;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{DataType, Expr, Node, Program, UnOp};

use super::gated_delta_spec::{self, GatedDeltaSpec};

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
///
/// Q and K are L2-normalized in F32. `decay_log` is exponentiated and
/// `beta_logits` is passed through sigmoid. Matrix state remains F32; activation
/// output converts once to `dtype`. `state_input` is preserved and
/// `state_output` receives the continued generation.
///
/// Everything this shares with
/// [`chunked_gated_delta`](super::gated_delta_chunked::chunked_gated_delta) is
/// built by [`super::gated_delta_spec`]: the head partition, the state copy,
/// the key and query normalizers, and the scaled operands. What remains here is
/// the schedule, which is the only thing that differs: this one carries the
/// matrix state forward one token at a time and needs no tile scratch, so it
/// dispatches 64 head slots per workgroup where the chunked prefill dispatches
/// one.
pub fn recurrent_gated_delta(
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

    let state_index = |key_index: Expr, value_index: Expr| {
        gated_delta_spec::state_index(key_dim, value_dim, key_index, value_index)
    };
    let value_index = |dim: Expr| {
        gated_delta_spec::activation_index(
            "value_head",
            sequence,
            value_heads,
            value_dim,
            Expr::var("token"),
            dim,
        )
    };
    let key_row = |dim: Expr| {
        gated_delta_spec::normalized_key(
            key,
            sequence,
            key_heads,
            key_dim,
            Expr::var("token"),
            dim,
            "current_key_scale",
        )
    };
    let scalar_index = gated_delta_spec::scalar_index(sequence, value_heads, Expr::var("token"));

    let init_state =
        gated_delta_spec::init_state_copy(state_input, state_output, key_dim, value_dim);
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
                            key_row(Expr::var("key_index")),
                            Expr::load(
                                state_output,
                                state_index(Expr::var("key_index"), Expr::var("value_index")),
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
                        Expr::mul(key_row(Expr::var("key_index")), Expr::var("delta")),
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
                            gated_delta_spec::scaled_query(
                                query,
                                sequence,
                                key_heads,
                                key_dim,
                                Expr::var("token"),
                                Expr::var("key_index"),
                            ),
                            Expr::load(
                                state_output,
                                state_index(Expr::var("key_index"), Expr::var("value_index")),
                            ),
                        ),
                    ),
                )],
            ),
            Node::Store {
                buffer: output.into(),
                index: value_index(Expr::var("value_index")),
                value: Expr::cast(dtype.clone(), Expr::var("attention_output")),
            },
        ],
    );
    let mut token_body = gated_delta_spec::key_norm_nodes(
        key,
        sequence,
        key_heads,
        key_dim,
        eps,
        Expr::var("token"),
        "current",
    );
    token_body.extend(gated_delta_spec::query_norm_nodes(
        query,
        sequence,
        key_heads,
        key_dim,
        eps,
        Expr::var("token"),
    ));
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
        gated_delta_spec::beta_gate_node(),
        decay_state,
        value_update,
    ]);
    let token_schedule = Node::loop_for("token", Expr::u32(0), Expr::u32(sequence), token_body);
    let body =
        gated_delta_spec::head_partition(&counts, value_heads, vec![init_state, token_schedule]);

    Ok(Program::wrapped(
        gated_delta_spec::gated_delta_buffers(spec, &counts),
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}
