//! One learned RMS normalization node, wired into the graph.
//!
//! A transformer normalizes three times per forward pass with the same node
//! shape: a constant weight vector of the hidden width, an invocation-scoped
//! hidden state in, and a read-write hidden state out. The three sites differ
//! only in the weight tensor name, the node name and the output name, and a
//! copy of the wiring is where one of them declares a lifetime or an access
//! mode the others do not.

use vyre::ir::{BufferAccess, DataType, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime};
use vyre_libs::nn::norm::learned_rms_norm;

use super::{graph_input, graph_output, make_contract, TranslationError};

/// The names one normalization site states.
pub(super) struct RmsNormNames {
    /// External value holding the learned scale.
    pub(super) weight: String,
    /// Graph node the normalization runs as.
    pub(super) node: String,
    /// Graph value the normalized hidden state is published as.
    pub(super) output: String,
}

/// Add the weight external value and the normalization node, and answer with
/// the normalized hidden state.
pub(super) fn add_learned_rms_norm(
    graph: &mut ProgramGraph,
    names: RmsNormNames,
    input_hidden: GraphValueId,
    hidden_shape: &[ShapeDim],
    rows: u32,
    hidden_dim: u32,
    norm_eps: f32,
    dtype: DataType,
) -> Result<GraphValueId, TranslationError> {
    let weight_shape = vec![ShapeDim::Known(u64::from(hidden_dim))];
    let weight = graph.add_external_value(
        names.weight,
        make_contract(
            dtype.clone(),
            weight_shape.clone(),
            ValueLifetime::Constant,
            BufferAccess::ReadOnly,
        ),
    )?;

    let program = learned_rms_norm(
        "input",
        "weight",
        "output",
        rows,
        hidden_dim,
        norm_eps,
        dtype.clone(),
    )
    .map_err(|error| TranslationError::Primitive(error.to_string()))?;

    let (_, outputs) = graph.add_node(
        names.node,
        program,
        vec![
            graph_input(
                "input",
                input_hidden,
                dtype.clone(),
                hidden_shape.to_vec(),
                ValueLifetime::Invocation,
                BufferAccess::ReadOnly,
            ),
            graph_input(
                "weight",
                weight,
                dtype.clone(),
                weight_shape,
                ValueLifetime::Constant,
                BufferAccess::ReadOnly,
            ),
        ],
        vec![graph_output(
            "output",
            names.output,
            dtype,
            hidden_shape.to_vec(),
            ValueLifetime::Invocation,
            BufferAccess::ReadWrite,
        )],
    )?;

    Ok(outputs[0])
}
