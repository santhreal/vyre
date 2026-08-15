//! Floating depthwise causal convolution for sequence models.

use thiserror::Error;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

const OP_ID: &str = "vyre-libs::nn::depthwise_causal_conv1d";

/// Optional post-convolution activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalConvActivation {
    /// Return the affine convolution result.
    None,
    /// Apply SiLU in F32 before source-dtype conversion.
    Silu,
}

/// Invalid depthwise causal convolution construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DepthwiseCausalConv1dError {
    /// One required tensor dimension is zero.
    #[error("depthwise causal convolution requires nonzero batch, channels, sequence, and kernel dimensions")]
    EmptyShape,
    /// One flattened tensor size exceeds u32 indexing.
    #[error("depthwise causal convolution tensor element count overflows u32; split the tensor")]
    ElementCountOverflow,
    /// Streaming state is unnecessary for a pointwise kernel.
    #[error("causal convolution state update requires kernel >= 2")]
    StateKernelTooShort,
    /// Source dtype lacks the required floating conversion contract.
    #[error("depthwise causal convolution supports F16, BF16, or F32 tensors; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected dtype.
        dtype: DataType,
    },
}

/// Build channel-major left-padded depthwise causal convolution.
///
/// `input` and `output` use `[batch, channels, sequence]`; `weight` uses
/// `[channels, kernel]`; optional `bias` uses `[channels]`; optional `mask`
/// uses U32 `[batch, sequence]`, where zero excludes an input position.
/// Accumulation and optional SiLU execute in F32 before one output conversion.
///
/// # Errors
///
/// Returns [`DepthwiseCausalConv1dError`] for empty/overflowing shapes or an
/// unsupported source dtype.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_causal_conv1d(
    input: &str,
    weight: &str,
    bias: Option<&str>,
    mask: Option<&str>,
    output: &str,
    batch: u32,
    channels: u32,
    sequence: u32,
    kernel: u32,
    activation: CausalConvActivation,
    dtype: DataType,
) -> Result<Program, DepthwiseCausalConv1dError> {
    if batch == 0 || channels == 0 || sequence == 0 || kernel == 0 {
        return Err(DepthwiseCausalConv1dError::EmptyShape);
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(DepthwiseCausalConv1dError::UnsupportedDtype { dtype });
    }
    let channel_sequence = channels
        .checked_mul(sequence)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let total = batch
        .checked_mul(channel_sequence)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let weight_count = channels
        .checked_mul(kernel)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let mask_count = batch
        .checked_mul(sequence)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;

    let index = Expr::var("index");
    let batch_index = Expr::div(index.clone(), Expr::u32(channel_sequence));
    let within_batch = Expr::sub(
        index.clone(),
        Expr::mul(batch_index.clone(), Expr::u32(channel_sequence)),
    );
    let channel = Expr::div(within_batch.clone(), Expr::u32(sequence));
    let time = Expr::sub(
        within_batch.clone(),
        Expr::mul(channel.clone(), Expr::u32(sequence)),
    );
    let initial = bias.map_or_else(
        || Expr::f32(0.0),
        |name| Expr::cast(DataType::F32, Expr::load(name, channel.clone())),
    );
    let lag = Expr::sub(Expr::u32(kernel - 1), Expr::var("kernel_index"));
    let position = Expr::sub(Expr::var("time"), lag.clone());
    let input_index = Expr::add(
        Expr::mul(Expr::var("batch"), Expr::u32(channel_sequence)),
        Expr::add(
            Expr::mul(Expr::var("channel"), Expr::u32(sequence)),
            Expr::var("position"),
        ),
    );
    let product = Expr::mul(
        Expr::cast(DataType::F32, Expr::load(input, input_index)),
        Expr::cast(
            DataType::F32,
            Expr::load(
                weight,
                Expr::add(
                    Expr::mul(Expr::var("channel"), Expr::u32(kernel)),
                    Expr::var("kernel_index"),
                ),
            ),
        ),
    );
    let accumulate = Node::assign("accumulator", Expr::add(Expr::var("accumulator"), product));
    let valid_body = if let Some(mask_name) = mask {
        vec![
            Node::let_bind("position", position),
            Node::if_then(
                Expr::ne(
                    Expr::load(
                        mask_name,
                        Expr::add(
                            Expr::mul(Expr::var("batch"), Expr::u32(sequence)),
                            Expr::var("position"),
                        ),
                    ),
                    Expr::u32(0),
                ),
                vec![accumulate],
            ),
        ]
    } else {
        vec![Node::let_bind("position", position), accumulate]
    };
    let activated = match activation {
        CausalConvActivation::None => Expr::var("accumulator"),
        CausalConvActivation::Silu => {
            let value = Expr::var("accumulator");
            Expr::div(
                value.clone(),
                Expr::add(
                    Expr::f32(1.0),
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(Expr::UnOp {
                            op: UnOp::Negate,
                            operand: Box::new(value),
                        }),
                    },
                ),
            )
        }
    };
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(total)),
            vec![
                Node::let_bind("batch", batch_index),
                Node::let_bind("within_batch", within_batch),
                Node::let_bind("channel", channel),
                Node::let_bind("time", time),
                Node::let_bind("accumulator", initial),
                Node::loop_for(
                    "kernel_index",
                    Expr::u32(0),
                    Expr::u32(kernel),
                    vec![Node::if_then(Expr::ge(Expr::var("time"), lag), valid_body)],
                ),
                Node::Store {
                    buffer: output.into(),
                    index,
                    value: Expr::cast(dtype.clone(), activated),
                },
            ],
        ),
    ];

    let mut buffers = Vec::with_capacity(5);
    buffers.push(
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(total),
    );
    buffers.push(
        BufferDecl::storage(weight, 1, BufferAccess::ReadOnly, dtype.clone())
            .with_count(weight_count),
    );
    let mut binding = 2;
    if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, binding, BufferAccess::ReadOnly, dtype.clone())
                .with_count(channels),
        );
        binding += 1;
    }
    if let Some(name) = mask {
        buffers.push(
            BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32)
                .with_count(mask_count),
        );
        binding += 1;
    }
    buffers.push(BufferDecl::output(output, binding, dtype).with_count(total));
    Ok(Program::wrapped(
        buffers,
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}

/// Build short-chunk convolution and its next explicit loop-carried state.
///
/// State tensors use `[batch, channels, kernel - 1]`. The next state is a
/// separate output, so runtimes may recycle prior-state storage only after all
/// reads finish.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_causal_conv1d_update(
    input: &str,
    weight: &str,
    bias: Option<&str>,
    state_input: &str,
    output: &str,
    state_output: &str,
    batch: u32,
    channels: u32,
    chunk: u32,
    kernel: u32,
    activation: CausalConvActivation,
    dtype: DataType,
) -> Result<Program, DepthwiseCausalConv1dError> {
    if kernel < 2 {
        return Err(DepthwiseCausalConv1dError::StateKernelTooShort);
    }
    if batch == 0 || channels == 0 || chunk == 0 {
        return Err(DepthwiseCausalConv1dError::EmptyShape);
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(DepthwiseCausalConv1dError::UnsupportedDtype { dtype });
    }
    let state_len = kernel - 1;
    let channel_chunk = channels
        .checked_mul(chunk)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let output_count = batch
        .checked_mul(channel_chunk)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let channel_state = channels
        .checked_mul(state_len)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let state_count = batch
        .checked_mul(channel_state)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;
    let weight_count = channels
        .checked_mul(kernel)
        .ok_or(DepthwiseCausalConv1dError::ElementCountOverflow)?;

    let initial = bias.map_or_else(
        || Expr::f32(0.0),
        |name| Expr::cast(DataType::F32, Expr::load(name, Expr::var("output_channel"))),
    );
    let state_sample = Expr::load(
        state_input,
        Expr::add(
            Expr::mul(Expr::var("output_batch"), Expr::u32(channel_state)),
            Expr::add(
                Expr::mul(Expr::var("output_channel"), Expr::u32(state_len)),
                Expr::var("combined"),
            ),
        ),
    );
    let input_sample = Expr::load(
        input,
        Expr::add(
            Expr::mul(Expr::var("output_batch"), Expr::u32(channel_chunk)),
            Expr::add(
                Expr::mul(Expr::var("output_channel"), Expr::u32(chunk)),
                Expr::sub(Expr::var("combined"), Expr::u32(state_len)),
            ),
        ),
    );
    let weight_sample = Expr::load(
        weight,
        Expr::add(
            Expr::mul(Expr::var("output_channel"), Expr::u32(kernel)),
            Expr::var("kernel_index"),
        ),
    );
    let activated = match activation {
        CausalConvActivation::None => Expr::var("accumulator"),
        CausalConvActivation::Silu => {
            let value = Expr::var("accumulator");
            Expr::div(
                value.clone(),
                Expr::add(
                    Expr::f32(1.0),
                    Expr::UnOp {
                        op: UnOp::Exp,
                        operand: Box::new(Expr::UnOp {
                            op: UnOp::Negate,
                            operand: Box::new(value),
                        }),
                    },
                ),
            )
        }
    };
    let output_body = vec![
        Node::let_bind(
            "output_batch",
            Expr::div(Expr::var("dispatch_index"), Expr::u32(channel_chunk)),
        ),
        Node::let_bind(
            "output_remainder",
            Expr::sub(
                Expr::var("dispatch_index"),
                Expr::mul(Expr::var("output_batch"), Expr::u32(channel_chunk)),
            ),
        ),
        Node::let_bind(
            "output_channel",
            Expr::div(Expr::var("output_remainder"), Expr::u32(chunk)),
        ),
        Node::let_bind(
            "output_time",
            Expr::sub(
                Expr::var("output_remainder"),
                Expr::mul(Expr::var("output_channel"), Expr::u32(chunk)),
            ),
        ),
        Node::let_bind("accumulator", initial),
        Node::loop_for(
            "kernel_index",
            Expr::u32(0),
            Expr::u32(kernel),
            vec![
                Node::let_bind(
                    "combined",
                    Expr::add(Expr::var("output_time"), Expr::var("kernel_index")),
                ),
                Node::let_bind("sample", Expr::f32(0.0)),
                Node::if_then(
                    Expr::lt(Expr::var("combined"), Expr::u32(state_len)),
                    vec![Node::assign(
                        "sample",
                        Expr::cast(DataType::F32, state_sample),
                    )],
                ),
                Node::if_then(
                    Expr::ge(Expr::var("combined"), Expr::u32(state_len)),
                    vec![Node::assign(
                        "sample",
                        Expr::cast(DataType::F32, input_sample),
                    )],
                ),
                Node::assign(
                    "accumulator",
                    Expr::add(
                        Expr::var("accumulator"),
                        Expr::mul(
                            Expr::var("sample"),
                            Expr::cast(DataType::F32, weight_sample),
                        ),
                    ),
                ),
            ],
        ),
        Node::Store {
            buffer: output.into(),
            index: Expr::var("dispatch_index"),
            value: Expr::cast(dtype.clone(), activated),
        },
    ];

    let prior_state_tail = Expr::load(
        state_input,
        Expr::add(
            Expr::mul(Expr::var("state_batch"), Expr::u32(channel_state)),
            Expr::add(
                Expr::mul(Expr::var("state_channel"), Expr::u32(state_len)),
                Expr::var("state_combined"),
            ),
        ),
    );
    let input_tail = Expr::load(
        input,
        Expr::add(
            Expr::mul(Expr::var("state_batch"), Expr::u32(channel_chunk)),
            Expr::add(
                Expr::mul(Expr::var("state_channel"), Expr::u32(chunk)),
                Expr::sub(Expr::var("state_combined"), Expr::u32(state_len)),
            ),
        ),
    );
    let state_body = vec![
        Node::let_bind(
            "state_batch",
            Expr::div(Expr::var("dispatch_index"), Expr::u32(channel_state)),
        ),
        Node::let_bind(
            "state_remainder",
            Expr::sub(
                Expr::var("dispatch_index"),
                Expr::mul(Expr::var("state_batch"), Expr::u32(channel_state)),
            ),
        ),
        Node::let_bind(
            "state_channel",
            Expr::div(Expr::var("state_remainder"), Expr::u32(state_len)),
        ),
        Node::let_bind(
            "state_offset",
            Expr::sub(
                Expr::var("state_remainder"),
                Expr::mul(Expr::var("state_channel"), Expr::u32(state_len)),
            ),
        ),
        Node::let_bind(
            "state_combined",
            Expr::add(Expr::u32(chunk), Expr::var("state_offset")),
        ),
        Node::let_bind("next_state_value", Expr::f32(0.0)),
        Node::if_then(
            Expr::lt(Expr::var("state_combined"), Expr::u32(state_len)),
            vec![Node::assign(
                "next_state_value",
                Expr::cast(DataType::F32, prior_state_tail),
            )],
        ),
        Node::if_then(
            Expr::ge(Expr::var("state_combined"), Expr::u32(state_len)),
            vec![Node::assign(
                "next_state_value",
                Expr::cast(DataType::F32, input_tail),
            )],
        ),
        Node::Store {
            buffer: state_output.into(),
            index: Expr::var("dispatch_index"),
            value: Expr::cast(dtype.clone(), Expr::var("next_state_value")),
        },
    ];
    let body = vec![
        Node::let_bind("dispatch_index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("dispatch_index"), Expr::u32(output_count)),
            output_body,
        ),
        Node::if_then(
            Expr::lt(Expr::var("dispatch_index"), Expr::u32(state_count)),
            state_body,
        ),
    ];

    let mut buffers = vec![
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone())
            .with_count(output_count),
        BufferDecl::storage(weight, 1, BufferAccess::ReadOnly, dtype.clone())
            .with_count(weight_count),
        BufferDecl::storage(state_input, 2, BufferAccess::ReadWrite, dtype.clone())
            .with_count(state_count),
    ];
    let mut binding = 3;
    if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, binding, BufferAccess::ReadOnly, dtype.clone())
                .with_count(channels),
        );
        binding += 1;
    }
    buffers.push(BufferDecl::output(output, binding, dtype.clone()).with_count(output_count));
    buffers.push(
        BufferDecl::storage(state_output, binding + 1, BufferAccess::ReadWrite, dtype)
            .with_count(state_count),
    );
    Ok(Program::wrapped(
        buffers,
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::nn::depthwise_causal_conv1d_update",
            body,
        )],
    ))
}
