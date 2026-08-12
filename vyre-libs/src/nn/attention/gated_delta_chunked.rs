//! Chunk-size-64 cumulative-decay triangular gated delta-rule prefill.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use super::gated_delta::RecurrentGatedDeltaError;
use crate::region::wrap_anonymous;

const OP_ID: &str = "vyre-libs::nn::chunked_gated_delta";
const CHUNK_SIZE: u32 = 64;
const CHUNK_DECAY: &str = "chunk_decay";
const CHUNK_VALUE: &str = "chunk_value";

fn qk_index(sequence: u32, heads: u32, dim: u32, token: Expr, feature: Expr) -> Expr {
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

fn value_index(sequence: u32, heads: u32, dim: u32, token: Expr, feature: Expr) -> Expr {
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

fn scalar_index(sequence: u32, heads: u32, token: Expr) -> Expr {
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

fn state_index(key_dim: u32, value_dim: u32, key: Expr, value: Expr) -> Expr {
    Expr::add(
        Expr::mul(Expr::var("head_index"), Expr::u32(key_dim * value_dim)),
        Expr::add(Expr::mul(key, Expr::u32(value_dim)), value),
    )
}

fn chunk_value_index(value_dim: u32, row: Expr, value: Expr) -> Expr {
    Expr::add(Expr::mul(row, Expr::u32(value_dim)), value)
}

fn key_norm_nodes(
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
                            qk_index(sequence, key_heads, key_dim, token, Expr::var(dimension)),
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
        Node::let_bind(
            scale,
            Expr::UnOp {
                op: UnOp::InverseSqrt,
                operand: Box::new(Expr::add(Expr::var(sum), Expr::f32(eps))),
            },
        ),
    ]
}

fn normalized_key(
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
            Expr::load(key, qk_index(sequence, key_heads, key_dim, token, feature)),
        ),
        Expr::var(scale),
    )
}

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

/// Build the authoritative cumulative-decay triangular chunk schedule.
#[allow(clippy::too_many_arguments)]
pub(super) fn chunked_gated_delta_impl(
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
    let chunk_value_count = checked(&[CHUNK_SIZE, value_dim])?;
    let group = value_heads / key_heads;
    let chunk_count = sequence.div_ceil(CHUNK_SIZE);

    let init_state = Node::loop_for(
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
    );

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
                                value_index(
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
                            chunk_value_index(
                                value_dim,
                                Expr::var("previous_row"),
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
                index: chunk_value_index(value_dim, Expr::var("chunk_row"), Expr::var("tri_value")),
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
                                chunk_value_index(
                                    value_dim,
                                    Expr::var("state_row"),
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

    let mut output_row = vec![
        Node::let_bind(
            "output_token",
            Expr::add(Expr::var("chunk_base"), Expr::var("output_row")),
        ),
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
                            qk_index(
                                sequence,
                                key_heads,
                                key_dim,
                                Expr::var("output_token"),
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
    ];
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
                                chunk_value_index(
                                    value_dim,
                                    Expr::var("future_row"),
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
                                Expr::mul(
                                    Expr::cast(
                                        DataType::F32,
                                        Expr::load(
                                            query,
                                            qk_index(
                                                sequence,
                                                key_heads,
                                                key_dim,
                                                Expr::var("output_token"),
                                                Expr::var("output_key"),
                                            ),
                                        ),
                                    ),
                                    Expr::var("query_scale"),
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
                index: value_index(
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
                    Expr::rem(Expr::var("head_index"), Expr::u32(value_heads)),
                ),
                Node::let_bind(
                    "key_head",
                    Expr::div(Expr::var("value_head"), Expr::u32(group)),
                ),
                init_state,
                Node::loop_for("chunk", Expr::u32(0), Expr::u32(chunk_count), chunk_body),
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
            BufferDecl::workgroup(CHUNK_DECAY, CHUNK_SIZE, DataType::F32),
            BufferDecl::workgroup(CHUNK_VALUE, chunk_value_count, DataType::F32),
        ],
        [1, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}
