//! Chunk-size-64 cumulative-decay triangular gated delta-rule prefill.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, UnOp};

use super::gated_delta::RecurrentGatedDeltaError;
use super::gated_delta_spec::{
    self, activation_index, key_norm_nodes, normalized_key, scalar_index, state_index,
    GatedDeltaSpec,
};
use super::layout::block_index;

const OP_ID: &str = "vyre-libs::nn::chunked_gated_delta";
const CHUNK_SIZE: u32 = 64;
const CHUNK_DECAY: &str = "chunk_decay";
const CHUNK_VALUE: &str = "chunk_value";

fn pair_dot_nodes(
    key: &str,
    sequence: u32,
    key_heads: u32,
    key_dim: u32,
    eps: f32,
    current_token: Expr,
    other_token: Expr,
) -> Vec<Node> {
    let mut nodes = key_norm_nodes(
        key,
        sequence,
        key_heads,
        key_dim,
        eps,
        other_token.clone(),
        "other",
    );
    nodes.extend([
        Node::let_bind("key_pair_dot", Expr::f32(0.0)),
        Node::loop_for(
            "dot_dimension",
            Expr::u32(0),
            Expr::u32(key_dim),
            vec![Node::assign(
                "key_pair_dot",
                Expr::add(
                    Expr::var("key_pair_dot"),
                    Expr::mul(
                        normalized_key(
                            key,
                            sequence,
                            key_heads,
                            key_dim,
                            current_token,
                            Expr::var("dot_dimension"),
                            "current_key_scale",
                        ),
                        normalized_key(
                            key,
                            sequence,
                            key_heads,
                            key_dim,
                            other_token,
                            Expr::var("dot_dimension"),
                            "other_key_scale",
                        ),
                    ),
                ),
            )],
        ),
    ]);
    nodes
}

/// Build a fixed-size-64 chunk schedule for gated delta prefill.
///
/// The schedule retains the exact recurrent dependency inside each causal
/// lower-triangular tile. Its final tile is padded structurally and guarded,
/// so padding cannot read inputs, modify state, or appear in the output. It
/// builds the authoritative cumulative-decay triangular schedule.
pub fn chunked_gated_delta(spec: &GatedDeltaSpec<'_>) -> Result<Program, RecurrentGatedDeltaError> {
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
    let chunk_value_count = gated_delta_spec::checked(&[CHUNK_SIZE, value_dim])?;
    let chunk_count = sequence.div_ceil(CHUNK_SIZE);

    let init_state =
        gated_delta_spec::init_state_copy(state_input, state_output, key_dim, value_dim);

    let cumulative_decay = vec![
        Node::let_bind("cumulative_decay", Expr::f32(0.0)),
        Node::loop_for(
            "decay_row",
            Expr::u32(0),
            Expr::var("valid_len"),
            vec![
                Node::let_bind(
                    "decay_token",
                    Expr::add(Expr::var("chunk_base"), Expr::var("decay_row")),
                ),
                Node::assign(
                    "cumulative_decay",
                    Expr::add(
                        Expr::var("cumulative_decay"),
                        Expr::cast(
                            DataType::F32,
                            Expr::load(
                                decay_log,
                                scalar_index(sequence, value_heads, Expr::var("decay_token")),
                            ),
                        ),
                    ),
                ),
                Node::Store {
                    buffer: CHUNK_DECAY.into(),
                    index: Expr::var("decay_row"),
                    value: Expr::var("cumulative_decay"),
                },
            ],
        ),
    ];

    let mut triangular_row = vec![Node::let_bind(
        "current_token",
        Expr::add(Expr::var("chunk_base"), Expr::var("chunk_row")),
    )];
    triangular_row.extend(key_norm_nodes(
        key,
        sequence,
        key_heads,
        key_dim,
        eps,
        Expr::var("current_token"),
        "current",
    ));
    triangular_row.extend([
        Node::let_bind(
            "beta_logit",
            Expr::cast(
                DataType::F32,
                Expr::load(
                    beta_logits,
                    scalar_index(sequence, value_heads, Expr::var("current_token")),
                ),
            ),
        ),
        gated_delta_spec::beta_gate_node(),
        Node::loop_for("tri_value", Expr::u32(0), Expr::u32(value_dim), {
            let mut value_nodes = vec![
                Node::let_bind(
                    "transformed_value",
                    Expr::mul(
                        Expr::var("beta"),
                        Expr::cast(
                            DataType::F32,
                            Expr::load(
                                value,
                                activation_index(
                                    "value_head",
                                    sequence,
                                    value_heads,
                                    value_dim,
                                    Expr::var("current_token"),
                                    Expr::var("tri_value"),
                                ),
                            ),
                        ),
                    ),
                ),
                Node::let_bind("initial_memory", Expr::f32(0.0)),
                Node::loop_for(
                    "memory_key",
                    Expr::u32(0),
                    Expr::u32(key_dim),
                    vec![Node::assign(
                        "initial_memory",
                        Expr::add(
                            Expr::var("initial_memory"),
                            Expr::mul(
                                normalized_key(
                                    key,
                                    sequence,
                                    key_heads,
                                    key_dim,
                                    Expr::var("current_token"),
                                    Expr::var("memory_key"),
                                    "current_key_scale",
                                ),
                                Expr::load(
                                    state_output,
                                    state_index(
                                        key_dim,
                                        value_dim,
                                        Expr::var("memory_key"),
                                        Expr::var("tri_value"),
                                    ),
                                ),
                            ),
                        ),
                    )],
                ),
                Node::assign(
                    "transformed_value",
                    Expr::sub(
                        Expr::var("transformed_value"),
                        Expr::mul(
                            Expr::mul(
                                Expr::var("beta"),
                                Expr::UnOp {
                                    op: UnOp::Exp,
                                    operand: Box::new(Expr::load(
                                        CHUNK_DECAY,
                                        Expr::var("chunk_row"),
                                    )),
                                },
                            ),
                            Expr::var("initial_memory"),
                        ),
                    ),
                ),
            ];
            let mut previous_body = vec![Node::let_bind(
                "previous_token",
                Expr::add(Expr::var("chunk_base"), Expr::var("previous_row")),
            )];
            previous_body.extend(pair_dot_nodes(
                key,
                sequence,
                key_heads,
                key_dim,
                eps,
                Expr::var("current_token"),
                Expr::var("previous_token"),
            ));
            previous_body.push(Node::assign(
                "transformed_value",
                Expr::sub(
                    Expr::var("transformed_value"),
                    Expr::mul(
                        Expr::mul(
                            Expr::var("beta"),
                            Expr::mul(
                                Expr::UnOp {
                                    op: UnOp::Exp,
                                    operand: Box::new(Expr::sub(
                                        Expr::load(CHUNK_DECAY, Expr::var("chunk_row")),
                                        Expr::load(CHUNK_DECAY, Expr::var("previous_row")),
                                    )),
                                },
                                Expr::var("key_pair_dot"),
                            ),
                        ),
                        Expr::load(
                            CHUNK_VALUE,
                            block_index(
                                Expr::var("previous_row"),
                                value_dim,
                                Expr::var("tri_value"),
                            ),
                        ),
                    ),
                ),
            ));
            value_nodes.push(Node::loop_for(
                "previous_row",
                Expr::u32(0),
                Expr::var("chunk_row"),
                previous_body,
            ));
            value_nodes.push(Node::Store {
                buffer: CHUNK_VALUE.into(),
                index: block_index(Expr::var("chunk_row"), value_dim, Expr::var("tri_value")),
                value: Expr::var("transformed_value"),
            });
            value_nodes
        }),
    ]);
    let triangular_solve = Node::loop_for(
        "chunk_row",
        Expr::u32(0),
        Expr::var("valid_len"),
        triangular_row,
    );

    let state_update = Node::loop_for(
        "final_key",
        Expr::u32(0),
        Expr::u32(key_dim),
        vec![Node::loop_for(
            "final_value",
            Expr::u32(0),
            Expr::u32(value_dim),
            {
                let mut nodes = vec![Node::let_bind(
                    "final_state",
                    Expr::mul(
                        Expr::UnOp {
                            op: UnOp::Exp,
                            operand: Box::new(Expr::var("last_decay")),
                        },
                        Expr::load(
                            state_output,
                            state_index(
                                key_dim,
                                value_dim,
                                Expr::var("final_key"),
                                Expr::var("final_value"),
                            ),
                        ),
                    ),
                )];
                let mut contribution = vec![Node::let_bind(
                    "state_token",
                    Expr::add(Expr::var("chunk_base"), Expr::var("state_row")),
                )];
                contribution.extend(key_norm_nodes(
                    key,
                    sequence,
                    key_heads,
                    key_dim,
                    eps,
                    Expr::var("state_token"),
                    "state",
                ));
                contribution.push(Node::assign(
                    "final_state",
                    Expr::add(
                        Expr::var("final_state"),
                        Expr::mul(
                            Expr::mul(
                                Expr::UnOp {
                                    op: UnOp::Exp,
                                    operand: Box::new(Expr::sub(
                                        Expr::var("last_decay"),
                                        Expr::load(CHUNK_DECAY, Expr::var("state_row")),
                                    )),
                                },
                                normalized_key(
                                    key,
                                    sequence,
                                    key_heads,
                                    key_dim,
                                    Expr::var("state_token"),
                                    Expr::var("final_key"),
                                    "state_key_scale",
                                ),
                            ),
                            Expr::load(
                                CHUNK_VALUE,
                                block_index(
                                    Expr::var("state_row"),
                                    value_dim,
                                    Expr::var("final_value"),
                                ),
                            ),
                        ),
                    ),
                ));
                nodes.push(Node::loop_for(
                    "state_row",
                    Expr::u32(0),
                    Expr::var("valid_len"),
                    contribution,
                ));
                nodes.push(Node::Store {
                    buffer: state_output.into(),
                    index: state_index(
                        key_dim,
                        value_dim,
                        Expr::var("final_key"),
                        Expr::var("final_value"),
                    ),
                    value: Expr::var("final_state"),
                });
                nodes
            },
        )],
    );

    let mut output_row = vec![Node::let_bind(
        "output_token",
        Expr::add(Expr::var("chunk_base"), Expr::var("output_row")),
    )];
    output_row.extend(gated_delta_spec::query_norm_nodes(
        query,
        sequence,
        key_heads,
        key_dim,
        eps,
        Expr::var("output_token"),
    ));
    output_row.push(Node::loop_for(
        "output_value",
        Expr::u32(0),
        Expr::u32(value_dim),
        vec![
            Node::let_bind("attention_output", Expr::f32(0.0)),
            Node::loop_for("output_key", Expr::u32(0), Expr::u32(key_dim), {
                let mut nodes = vec![Node::let_bind(
                    "state_at_token",
                    Expr::load(
                        state_output,
                        state_index(
                            key_dim,
                            value_dim,
                            Expr::var("output_key"),
                            Expr::var("output_value"),
                        ),
                    ),
                )];
                let mut future = vec![Node::let_bind(
                    "future_token",
                    Expr::add(Expr::var("chunk_base"), Expr::var("future_row")),
                )];
                future.extend(key_norm_nodes(
                    key,
                    sequence,
                    key_heads,
                    key_dim,
                    eps,
                    Expr::var("future_token"),
                    "future",
                ));
                future.push(Node::assign(
                    "state_at_token",
                    Expr::sub(
                        Expr::var("state_at_token"),
                        Expr::mul(
                            Expr::mul(
                                Expr::UnOp {
                                    op: UnOp::Exp,
                                    operand: Box::new(Expr::sub(
                                        Expr::var("last_decay"),
                                        Expr::load(CHUNK_DECAY, Expr::var("future_row")),
                                    )),
                                },
                                normalized_key(
                                    key,
                                    sequence,
                                    key_heads,
                                    key_dim,
                                    Expr::var("future_token"),
                                    Expr::var("output_key"),
                                    "future_key_scale",
                                ),
                            ),
                            Expr::load(
                                CHUNK_VALUE,
                                block_index(
                                    Expr::var("future_row"),
                                    value_dim,
                                    Expr::var("output_value"),
                                ),
                            ),
                        ),
                    ),
                ));
                nodes.push(Node::loop_for(
                    "future_row",
                    Expr::add(Expr::var("output_row"), Expr::u32(1)),
                    Expr::var("valid_len"),
                    future,
                ));
                nodes.extend([
                    Node::assign(
                        "state_at_token",
                        Expr::div(
                            Expr::var("state_at_token"),
                            Expr::UnOp {
                                op: UnOp::Exp,
                                operand: Box::new(Expr::sub(
                                    Expr::var("last_decay"),
                                    Expr::load(CHUNK_DECAY, Expr::var("output_row")),
                                )),
                            },
                        ),
                    ),
                    Node::assign(
                        "attention_output",
                        Expr::add(
                            Expr::var("attention_output"),
                            Expr::mul(
                                gated_delta_spec::scaled_query(
                                    query,
                                    sequence,
                                    key_heads,
                                    key_dim,
                                    Expr::var("output_token"),
                                    Expr::var("output_key"),
                                ),
                                Expr::var("state_at_token"),
                            ),
                        ),
                    ),
                ]);
                nodes
            }),
            Node::Store {
                buffer: output.into(),
                index: activation_index(
                    "value_head",
                    sequence,
                    value_heads,
                    value_dim,
                    Expr::var("output_token"),
                    Expr::var("output_value"),
                ),
                value: Expr::cast(dtype.clone(), Expr::var("attention_output")),
            },
        ],
    ));
    let output_rows = Node::loop_for(
        "output_row",
        Expr::u32(0),
        Expr::var("valid_len"),
        output_row,
    );

    let mut chunk_body = vec![
        Node::let_bind(
            "chunk_base",
            Expr::mul(Expr::var("chunk"), Expr::u32(CHUNK_SIZE)),
        ),
        Node::let_bind(
            "remaining",
            Expr::sub(Expr::u32(sequence), Expr::var("chunk_base")),
        ),
        Node::let_bind(
            "valid_len",
            Expr::select(
                Expr::lt(Expr::var("remaining"), Expr::u32(CHUNK_SIZE)),
                Expr::var("remaining"),
                Expr::u32(CHUNK_SIZE),
            ),
        ),
    ];
    chunk_body.extend(cumulative_decay);
    chunk_body.extend([
        triangular_solve,
        Node::let_bind(
            "last_decay",
            Expr::load(CHUNK_DECAY, Expr::sub(Expr::var("valid_len"), Expr::u32(1))),
        ),
        state_update,
        output_rows,
    ]);

    let body = gated_delta_spec::head_partition(
        &counts,
        value_heads,
        vec![
            init_state,
            Node::loop_for("chunk", Expr::u32(0), Expr::u32(chunk_count), chunk_body),
        ],
    );

    let mut buffers = gated_delta_spec::gated_delta_buffers(spec, &counts);
    buffers.extend([
        BufferDecl::workgroup(CHUNK_DECAY, CHUNK_SIZE, DataType::F32),
        BufferDecl::workgroup(CHUNK_VALUE, chunk_value_count, DataType::F32),
    ]);

    Ok(Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}
