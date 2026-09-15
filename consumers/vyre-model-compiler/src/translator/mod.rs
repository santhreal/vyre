//! Translation from model architecture configurations to domain-neutral ProgramGraphs.
//!
//! Translates layers, attention mechanisms, MoE routing, feed-forward networks,
//! and state edges (Key/Value history) into pure `vyre_foundation::ir::ProgramGraph`
//! compositions without target-specific templates or schedule hints.

use thiserror::Error;
use vyre::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, GraphValueId, ProgramGraph, ProgramGraphError,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_libs::nn::{activation::embedding_typed, linear::linear_rows_no_bias_out_in_typed};

mod layer;
mod rms_norm_node;

use rms_norm_node::{add_learned_rms_norm, RmsNormNames};

use crate::config::ModelConfig;
use crate::workload::WorkloadEnvelope;

/// Error during model graph translation.
#[derive(Debug, Error)]
pub enum TranslationError {
    /// Dimension overflow or invalid parameter specification.
    #[error("Fix: invalid model dimensions for '{name}': {reason}")]
    InvalidDimensions {
        /// Model name.
        name: String,
        /// Detail.
        reason: String,
    },
    /// Error emitted by the underlying IR graph builder.
    #[error("Fix: graph construction failed: {0}")]
    Graph(#[from] ProgramGraphError),
    /// Primitive composition build error.
    #[error("Fix: primitive composition failed: {0}")]
    Primitive(String),
}

fn make_contract(
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

/// One node input, bound to a graph value under a stated contract.
///
/// Every input this translator declares states the same six facts in the same
/// order. Written out per site it is a seven-line block, and the sites that
/// carry it disagreed on nothing except those six values.
fn graph_input(
    buffer: &str,
    value: GraphValueId,
    dtype: DataType,
    shape: Vec<ShapeDim>,
    lifetime: ValueLifetime,
    access: BufferAccess,
) -> GraphInput {
    GraphInput {
        buffer: buffer.into(),
        value,
        contract: make_contract(dtype, shape, lifetime, access),
    }
}

/// One node output, published under `name`.
///
/// No output this translator declares succeeds a retained value, so the node
/// writes a fresh graph value each time.
fn graph_output(
    buffer: &str,
    name: impl Into<String>,
    dtype: DataType,
    shape: Vec<ShapeDim>,
    lifetime: ValueLifetime,
    access: BufferAccess,
) -> GraphOutput {
    GraphOutput {
        buffer: buffer.into(),
        name: name.into(),
        contract: make_contract(dtype, shape, lifetime, access),
        retained_successor_of: None,
    }
}

/// Domain-neutral translation engine converting model configurations to ProgramGraphs.
pub struct ModelGraphBuilder<'a> {
    config: &'a ModelConfig,
    workload: &'a WorkloadEnvelope,
}

impl<'a> ModelGraphBuilder<'a> {
    /// Construct a new graph builder for a model and workload envelope.
    #[must_use]
    pub const fn new(config: &'a ModelConfig, workload: &'a WorkloadEnvelope) -> Self {
        Self { config, workload }
    }

    /// Build the complete forward-pass [`ProgramGraph`] covering all layers,
    /// attention blocks, MLPs/MoE routing, and KV state edges.
    pub fn build_graph(&self) -> Result<ProgramGraph, TranslationError> {
        let mut graph = ProgramGraph::new();
        let batch = self.workload.batch_size;
        let seq_len = self.workload.sequence_len;
        let hidden_dim = self.config.hidden_dim;
        let dtype = self.config.dtype.clone();

        let rows =
            batch
                .checked_mul(seq_len)
                .ok_or_else(|| TranslationError::InvalidDimensions {
                    name: self.config.name.clone(),
                    reason: "batch * sequence_len overflows u32".to_string(),
                })?;

        let hidden_shape = vec![
            ShapeDim::Known(u64::from(batch)),
            ShapeDim::Known(u64::from(seq_len)),
            ShapeDim::Known(u64::from(hidden_dim)),
        ];

        let tokens_shape = vec![
            ShapeDim::Known(u64::from(batch)),
            ShapeDim::Known(u64::from(seq_len)),
        ];

        // 1. External Inputs: Tokens or visual patches
        let mut current_hidden = if self.config.vocab_size > 0 {
            let tokens_contract = make_contract(
                DataType::U32,
                tokens_shape,
                ValueLifetime::Invocation,
                BufferAccess::ReadOnly,
            );
            let embed_table_contract = make_contract(
                dtype.clone(),
                vec![
                    ShapeDim::Known(u64::from(self.config.vocab_size)),
                    ShapeDim::Known(u64::from(hidden_dim)),
                ],
                ValueLifetime::Constant,
                BufferAccess::ReadOnly,
            );
            let tokens_val = graph.add_external_value("tokens", tokens_contract.clone())?;
            let embed_table_val = graph
                .add_external_value("model.embed_tokens.weight", embed_table_contract.clone())?;

            let embed_prog = embedding_typed(
                "embed_table",
                "tokens",
                "embed_out",
                rows,
                self.config.vocab_size,
                hidden_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let hidden_contract = make_contract(
                dtype.clone(),
                hidden_shape.clone(),
                ValueLifetime::Invocation,
                BufferAccess::ReadWrite,
            );

            let (_, embed_out) = graph.add_node(
                "embed_tokens",
                embed_prog,
                vec![
                    GraphInput {
                        buffer: "embed_table".into(),
                        value: embed_table_val,
                        contract: embed_table_contract,
                    },
                    GraphInput {
                        buffer: "tokens".into(),
                        value: tokens_val,
                        contract: tokens_contract,
                    },
                ],
                vec![GraphOutput {
                    buffer: "embed_out".into(),
                    name: "hidden_states_0".into(),
                    contract: hidden_contract,
                    retained_successor_of: None,
                }],
            )?;
            embed_out[0]
        } else {
            // Non-text input (e.g. pre-projected vision features)
            let hidden_contract = make_contract(
                dtype.clone(),
                hidden_shape.clone(),
                ValueLifetime::Invocation,
                BufferAccess::ReadOnly,
            );
            graph.add_external_value("input_features", hidden_contract)?
        };

        // 2. Transformer Layer Stack
        for layer_idx in 0..self.config.num_layers {
            let layer_prefix = format!("model.layers.{layer_idx}");
            current_hidden = self.build_layer(
                &mut graph,
                &layer_prefix,
                layer_idx,
                current_hidden,
                rows,
                &hidden_shape,
            )?;
        }

        // 3. Final Normalization
        let final_norm_out = add_learned_rms_norm(
            &mut graph,
            RmsNormNames {
                weight: "model.norm.weight".into(),
                node: "final_norm".into(),
                output: "final_hidden_states".into(),
            },
            current_hidden,
            &hidden_shape,
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
        )?;

        // 4. LM Head Logits Projection (if language model)
        if self.config.vocab_size > 0 {
            let lm_head_weight = graph.add_external_value(
                "lm_head.weight",
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(self.config.vocab_size)),
                        ShapeDim::Known(u64::from(hidden_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;

            let lm_head_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                hidden_dim,
                self.config.vocab_size,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let logits_shape = vec![
                ShapeDim::Known(u64::from(batch)),
                ShapeDim::Known(u64::from(seq_len)),
                ShapeDim::Known(u64::from(self.config.vocab_size)),
            ];

            let (_node_id, _logits_out) = graph.add_node(
                "lm_head",
                lm_head_prog,
                vec![
                    graph_input(
                        "input",
                        final_norm_out,
                        dtype.clone(),
                        hidden_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        lm_head_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(self.config.vocab_size)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    "logits",
                    dtype,
                    logits_shape,
                    ValueLifetime::Output,
                    BufferAccess::ReadWrite,
                )],
            )?;
        }

        Ok(graph)
    }
}
