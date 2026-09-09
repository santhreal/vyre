//! Reusable pre-normalized dense gated-MLP residual block.

use thiserror::Error;
use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, ProgramGraph, ProgramGraphError, ShapeDim,
    ValueContract, ValueLifetime,
};

use crate::nn::{
    activation::{residual_add_typed, swiglu_typed},
    linear::linear_rows_no_bias_out_in_typed,
    norm::learned_rms_norm,
};

/// Dimensions and source dtype for a dense gated MLP residual block.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseGatedMlpSpec {
    /// Independent sequence batch count.
    pub batch: u32,
    /// Token rows per batch.
    pub sequence: u32,
    /// Residual-stream feature width.
    pub hidden_dim: u32,
    /// Gate/up projection width.
    pub intermediate_dim: u32,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// F16, BF16, or F32 activation and weight representation.
    pub dtype: DataType,
}

/// Invalid dense gated MLP composition.
#[derive(Debug, Error)]
pub enum DenseGatedMlpError {
    /// Primitive dimensions or dtype were rejected.
    #[error("dense gated MLP primitive construction failed: {0}")]
    Primitive(String),
    /// Typed graph wiring rejected a port contract.
    #[error("dense gated MLP graph construction failed: {0}")]
    Graph(#[from] ProgramGraphError),
}

fn contract(
    dtype: DataType,
    shape: Vec<ShapeDim>,
    lifetime: ValueLifetime,
    access: BufferAccess,
) -> ValueContract {
    ValueContract {
        dtype,
        shape,
        access,
        lifetime,
    }
}

/// Build learned RMSNorm, bias-free gate/up projections, SwiGLU,
/// bias-free down projection, and residual addition.
pub fn dense_gated_mlp_graph(spec: &DenseGatedMlpSpec) -> Result<ProgramGraph, DenseGatedMlpError> {
    dense_gated_mlp_graph_with_lifetime(spec, ValueLifetime::Output)
}

pub(crate) fn dense_gated_mlp_graph_with_lifetime(
    spec: &DenseGatedMlpSpec,
    output_lifetime: ValueLifetime,
) -> Result<ProgramGraph, DenseGatedMlpError> {
    if spec.batch == 0 || spec.sequence == 0 || spec.hidden_dim == 0 || spec.intermediate_dim == 0 {
        return Err(DenseGatedMlpError::Primitive(
            "batch, sequence, hidden_dim, and intermediate_dim must be nonzero".to_string(),
        ));
    }
    if !matches!(spec.dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(DenseGatedMlpError::Primitive(format!(
            "dtype must be F16, BF16, or F32; got {:?}",
            spec.dtype
        )));
    }
    let rows = spec
        .batch
        .checked_mul(spec.sequence)
        .ok_or_else(|| DenseGatedMlpError::Primitive("batch*sequence overflows u32".to_string()))?;
    let hidden_count = rows.checked_mul(spec.hidden_dim).ok_or_else(|| {
        DenseGatedMlpError::Primitive("token rows*hidden_dim overflows u32".to_string())
    })?;
    let intermediate_count = rows.checked_mul(spec.intermediate_dim).ok_or_else(|| {
        DenseGatedMlpError::Primitive("token rows*intermediate_dim overflows u32".to_string())
    })?;
    spec.hidden_dim
        .checked_mul(spec.intermediate_dim)
        .ok_or_else(|| {
            DenseGatedMlpError::Primitive(
                "hidden_dim*intermediate_dim weight count overflows u32".to_string(),
            )
        })?;

    let hidden_shape = vec![
        ShapeDim::Known(u64::from(spec.batch)),
        ShapeDim::Known(u64::from(spec.sequence)),
        ShapeDim::Known(u64::from(spec.hidden_dim)),
    ];
    let intermediate_shape = vec![
        ShapeDim::Known(u64::from(spec.batch)),
        ShapeDim::Known(u64::from(spec.sequence)),
        ShapeDim::Known(u64::from(spec.intermediate_dim)),
    ];
    let norm_weight_shape = vec![ShapeDim::Known(u64::from(spec.hidden_dim))];
    let up_weight_shape = vec![
        ShapeDim::Known(u64::from(spec.intermediate_dim)),
        ShapeDim::Known(u64::from(spec.hidden_dim)),
    ];
    let down_weight_shape = vec![
        ShapeDim::Known(u64::from(spec.hidden_dim)),
        ShapeDim::Known(u64::from(spec.intermediate_dim)),
    ];
    let read_hidden = contract(
        spec.dtype.clone(),
        hidden_shape.clone(),
        ValueLifetime::Invocation,
        BufferAccess::ReadOnly,
    );
    let write_hidden = contract(
        spec.dtype.clone(),
        hidden_shape.clone(),
        ValueLifetime::Invocation,
        BufferAccess::ReadWrite,
    );
    let read_intermediate = contract(
        spec.dtype.clone(),
        intermediate_shape.clone(),
        ValueLifetime::Invocation,
        BufferAccess::ReadOnly,
    );
    let write_intermediate = contract(
        spec.dtype.clone(),
        intermediate_shape,
        ValueLifetime::Invocation,
        BufferAccess::ReadWrite,
    );
    let norm_weight_contract = contract(
        spec.dtype.clone(),
        norm_weight_shape,
        ValueLifetime::Constant,
        BufferAccess::ReadOnly,
    );
    let up_weight_contract = contract(
        spec.dtype.clone(),
        up_weight_shape,
        ValueLifetime::Constant,
        BufferAccess::ReadOnly,
    );
    let down_weight_contract = contract(
        spec.dtype.clone(),
        down_weight_shape,
        ValueLifetime::Constant,
        BufferAccess::ReadOnly,
    );

    let mut graph = ProgramGraph::new();
    let residual = graph.add_external_value("residual", read_hidden.clone())?;
    let norm_weight = graph.add_external_value(
        "post_attention_layernorm.weight",
        norm_weight_contract.clone(),
    )?;
    let gate_weight =
        graph.add_external_value("mlp.gate_proj.weight", up_weight_contract.clone())?;
    let up_weight = graph.add_external_value("mlp.up_proj.weight", up_weight_contract.clone())?;
    let down_weight =
        graph.add_external_value("mlp.down_proj.weight", down_weight_contract.clone())?;

    let (_, normalized) = graph.add_node(
        "mlp.norm",
        learned_rms_norm(
            "input",
            "weight",
            "output",
            rows,
            spec.hidden_dim,
            spec.norm_eps,
            spec.dtype.clone(),
        )
        .map_err(|error| DenseGatedMlpError::Primitive(error.to_string()))?,
        vec![
            GraphInput {
                buffer: "input".into(),
                value: residual,
                contract: read_hidden.clone(),
            },
            GraphInput {
                buffer: "weight".into(),
                value: norm_weight,
                contract: norm_weight_contract,
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.normalized".into(),
            contract: write_hidden.clone(),
            retained_successor_of: None,
        }],
    )?;
    let (_, gate_projection) = graph.add_node(
        "mlp.gate_proj",
        linear_rows_no_bias_out_in_typed(
            "input",
            "weight",
            "output",
            rows,
            spec.hidden_dim,
            spec.intermediate_dim,
            spec.dtype.clone(),
        )
        .map_err(DenseGatedMlpError::Primitive)?,
        vec![
            GraphInput {
                buffer: "input".into(),
                value: normalized[0],
                contract: read_hidden.clone(),
            },
            GraphInput {
                buffer: "weight".into(),
                value: gate_weight,
                contract: up_weight_contract.clone(),
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.gate".into(),
            contract: write_intermediate.clone(),
            retained_successor_of: None,
        }],
    )?;
    let (_, up_projection) = graph.add_node(
        "mlp.up_proj",
        linear_rows_no_bias_out_in_typed(
            "input",
            "weight",
            "output",
            rows,
            spec.hidden_dim,
            spec.intermediate_dim,
            spec.dtype.clone(),
        )
        .map_err(DenseGatedMlpError::Primitive)?,
        vec![
            GraphInput {
                buffer: "input".into(),
                value: normalized[0],
                contract: read_hidden.clone(),
            },
            GraphInput {
                buffer: "weight".into(),
                value: up_weight,
                contract: up_weight_contract,
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.up".into(),
            contract: write_intermediate.clone(),
            retained_successor_of: None,
        }],
    )?;
    let (_, activated) = graph.add_node(
        "mlp.swiglu",
        swiglu_typed(
            "gate",
            "up",
            "output",
            intermediate_count,
            spec.dtype.clone(),
        )
        .map_err(DenseGatedMlpError::Primitive)?,
        vec![
            GraphInput {
                buffer: "gate".into(),
                value: gate_projection[0],
                contract: read_intermediate.clone(),
            },
            GraphInput {
                buffer: "up".into(),
                value: up_projection[0],
                contract: read_intermediate,
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.activated".into(),
            contract: write_intermediate,
            retained_successor_of: None,
        }],
    )?;
    let (_, down_projection) = graph.add_node(
        "mlp.down_proj",
        linear_rows_no_bias_out_in_typed(
            "input",
            "weight",
            "output",
            rows,
            spec.intermediate_dim,
            spec.hidden_dim,
            spec.dtype.clone(),
        )
        .map_err(DenseGatedMlpError::Primitive)?,
        vec![
            GraphInput {
                buffer: "input".into(),
                value: activated[0],
                contract: contract(
                    spec.dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(spec.batch)),
                        ShapeDim::Known(u64::from(spec.sequence)),
                        ShapeDim::Known(u64::from(spec.intermediate_dim)),
                    ],
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
            },
            GraphInput {
                buffer: "weight".into(),
                value: down_weight,
                contract: down_weight_contract,
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.down".into(),
            contract: write_hidden.clone(),
            retained_successor_of: None,
        }],
    )?;
    graph.add_node(
        "mlp.residual",
        residual_add_typed(
            "residual",
            "branch",
            "output",
            hidden_count,
            spec.dtype.clone(),
        )
        .map_err(DenseGatedMlpError::Primitive)?,
        vec![
            GraphInput {
                buffer: "residual".into(),
                value: residual,
                contract: read_hidden.clone(),
            },
            GraphInput {
                buffer: "branch".into(),
                value: down_projection[0],
                contract: read_hidden,
            },
        ],
        vec![GraphOutput {
            buffer: "output".into(),
            name: "mlp.block_output".into(),
            contract: contract(
                spec.dtype.clone(),
                hidden_shape,
                output_lifetime,
                BufferAccess::ReadWrite,
            ),
            retained_successor_of: None,
        }],
    )?;
    graph
        .analyze()
        .map_err(|error| DenseGatedMlpError::Primitive(error.to_string()))?;
    Ok(graph)
}
