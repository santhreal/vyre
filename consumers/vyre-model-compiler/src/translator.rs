//! Translation from model architecture configurations to domain-neutral ProgramGraphs.
//!
//! Translates layers, attention mechanisms, MoE routing, feed-forward networks,
//! and state edges (Key/Value history) into pure `vyre_foundation::ir::ProgramGraph`
//! compositions without target-specific templates or schedule hints.

use thiserror::Error;
use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, GraphValueId,
    ProgramGraph, ProgramGraphError, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_libs::nn::{
    activation::{embedding_typed, residual_add_typed, swiglu_typed},
    attention::mla_decode,
    linear::linear_rows_no_bias_out_in_typed,
    moe::moe_layer_route_and_accumulate,
    norm::learned_rms_norm,
};

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

        let rows = batch
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
            let embed_table_val =
                graph.add_external_value("model.embed_tokens.weight", embed_table_contract.clone())?;

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
        let final_norm_weight = graph.add_external_value(
            "model.norm.weight",
            make_contract(
                dtype.clone(),
                vec![ShapeDim::Known(u64::from(hidden_dim))],
                ValueLifetime::Constant,
                BufferAccess::ReadOnly,
            ),
        )?;

        let final_norm_prog = learned_rms_norm(
            "input",
            "weight",
            "output",
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
        )
        .map_err(|e| TranslationError::Primitive(e.to_string()))?;

        let (_, final_norm_out) = graph.add_node(
            "final_norm",
            final_norm_prog,
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: current_hidden,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
                GraphInput {
                    buffer: "weight".into(),
                    value: final_norm_weight,
                    contract: make_contract(
                        dtype.clone(),
                        vec![ShapeDim::Known(u64::from(hidden_dim))],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "output".into(),
                name: "final_hidden_states".into(),
                contract: make_contract(
                    dtype.clone(),
                    hidden_shape.clone(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                ),
                retained_successor_of: None,
            }],
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

            let (_, logits_out) = graph.add_node(
                "lm_head",
                lm_head_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: final_norm_out[0],
                        contract: make_contract(
                            dtype.clone(),
                            hidden_shape.clone(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: lm_head_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(self.config.vocab_size)),
                                ShapeDim::Known(u64::from(hidden_dim)),
                            ],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: "logits".into(),
                    contract: make_contract(
                        dtype,
                        logits_shape,
                        ValueLifetime::Output,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;
            let _ = logits_out;
        }

        Ok(graph)
    }

    fn build_layer(
        &self,
        graph: &mut ProgramGraph,
        prefix: &str,
        layer_idx: u32,
        input_hidden: GraphValueId,
        rows: u32,
        hidden_shape: &[ShapeDim],
    ) -> Result<GraphValueId, TranslationError> {
        let hidden_dim = self.config.hidden_dim;
        let dtype = self.config.dtype.clone();

        // 1. Input RMSNorm / LayerNorm
        let input_norm_weight = graph.add_external_value(
            format!("{prefix}.input_layernorm.weight"),
            make_contract(
                dtype.clone(),
                vec![ShapeDim::Known(u64::from(hidden_dim))],
                ValueLifetime::Constant,
                BufferAccess::ReadOnly,
            ),
        )?;

        let norm_prog = learned_rms_norm(
            "input",
            "weight",
            "output",
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
        )
        .map_err(|e| TranslationError::Primitive(e.to_string()))?;

        let (_, norm_out) = graph.add_node(
            format!("{prefix}.input_norm"),
            norm_prog,
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: input_hidden,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
                GraphInput {
                    buffer: "weight".into(),
                    value: input_norm_weight,
                    contract: make_contract(
                        dtype.clone(),
                        vec![ShapeDim::Known(u64::from(hidden_dim))],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "output".into(),
                name: format!("{prefix}.normalized_attn_in"),
                contract: make_contract(
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                ),
                retained_successor_of: None,
            }],
        )?;

        // 2. Attention Projections and Core Attention (MLA or GQA/MHA)
        let attn_out = if let Some(mla) = &self.config.mla {
            // DeepSeek MLA: compressed latent attention
            let q_dim = self.config.num_heads * self.config.head_dim;
            let _w_uk = graph.add_external_value(
                format!("{prefix}.self_attn.w_uk"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(mla.kv_lora_rank)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let _w_uv = graph.add_external_value(
                format!("{prefix}.self_attn.w_uv"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(mla.kv_lora_rank)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;

            // State Edge: Prior KV Cache
            let _kv_cache_in = graph.add_external_value(
                format!("kv_cache_in_layer_{layer_idx}"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(self.workload.batch_size)),
                        ShapeDim::Known(u64::from(self.config.max_seq_len)),
                        ShapeDim::Known(u64::from(mla.kv_lora_rank)),
                    ],
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
            )?;

            let q_weight = graph.add_external_value(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![ShapeDim::Known(u64::from(q_dim)), ShapeDim::Known(u64::from(hidden_dim))],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let o_weight = graph.add_external_value(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![ShapeDim::Known(u64::from(hidden_dim)), ShapeDim::Known(u64::from(q_dim))],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;

            let q_proj_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                hidden_dim,
                q_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, q_outs) = graph.add_node(
                format!("{prefix}.q_proj"),
                q_proj_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: norm_out[0],
                        contract: make_contract(
                            dtype.clone(),
                            hidden_shape.to_vec(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: q_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![ShapeDim::Known(u64::from(q_dim)), ShapeDim::Known(u64::from(hidden_dim))],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.q_states"),
                    contract: make_contract(
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(self.workload.batch_size)),
                            ShapeDim::Known(u64::from(self.workload.sequence_len)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;

            // Output projection
            let o_proj_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                q_dim,
                hidden_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, o_outs) = graph.add_node(
                format!("{prefix}.o_proj"),
                o_proj_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: q_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(self.workload.batch_size)),
                                ShapeDim::Known(u64::from(self.workload.sequence_len)),
                                ShapeDim::Known(u64::from(q_dim)),
                            ],
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: o_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![ShapeDim::Known(u64::from(hidden_dim)), ShapeDim::Known(u64::from(q_dim))],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.attn_out"),
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;
            o_outs[0]
        } else {
            // Standard Multi-Head / Grouped-Query Attention Projections
            let q_dim = self.config.num_heads * self.config.head_dim;

            let q_weight = graph.add_external_value(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![ShapeDim::Known(u64::from(q_dim)), ShapeDim::Known(u64::from(hidden_dim))],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let o_weight = graph.add_external_value(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![ShapeDim::Known(u64::from(hidden_dim)), ShapeDim::Known(u64::from(q_dim))],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;

            let q_proj_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                hidden_dim,
                q_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, q_outs) = graph.add_node(
                format!("{prefix}.q_proj"),
                q_proj_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: norm_out[0],
                        contract: make_contract(
                            dtype.clone(),
                            hidden_shape.to_vec(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: q_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![ShapeDim::Known(u64::from(q_dim)), ShapeDim::Known(u64::from(hidden_dim))],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.q_states"),
                    contract: make_contract(
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(self.workload.batch_size)),
                            ShapeDim::Known(u64::from(self.workload.sequence_len)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;

            // Output projection
            let o_proj_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                q_dim,
                hidden_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, o_outs) = graph.add_node(
                format!("{prefix}.o_proj"),
                o_proj_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: q_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(self.workload.batch_size)),
                                ShapeDim::Known(u64::from(self.workload.sequence_len)),
                                ShapeDim::Known(u64::from(q_dim)),
                            ],
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: o_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![ShapeDim::Known(u64::from(hidden_dim)), ShapeDim::Known(u64::from(q_dim))],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.attn_out"),
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;
            o_outs[0]
        };

        // 3. Attention Residual Addition: input_hidden + attn_out
        let attn_residual_prog = residual_add_typed(
            "a",
            "b",
            "out",
            rows * hidden_dim,
            dtype.clone(),
        )
        .map_err(TranslationError::Primitive)?;

        let (_, attn_res_outs) = graph.add_node(
            format!("{prefix}.attn_residual_add"),
            attn_residual_prog,
            vec![
                GraphInput {
                    buffer: "a".into(),
                    value: input_hidden,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
                GraphInput {
                    buffer: "b".into(),
                    value: attn_out,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: format!("{prefix}.hidden_after_attn"),
                contract: make_contract(
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                ),
                retained_successor_of: None,
            }],
        )?;
        let hidden_after_attn = attn_res_outs[0];

        // 4. Post-Attention Normalization
        let post_norm_weight = graph.add_external_value(
            format!("{prefix}.post_attention_layernorm.weight"),
            make_contract(
                dtype.clone(),
                vec![ShapeDim::Known(u64::from(hidden_dim))],
                ValueLifetime::Constant,
                BufferAccess::ReadOnly,
            ),
        )?;

        let post_norm_prog = learned_rms_norm(
            "input",
            "weight",
            "output",
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
        )
        .map_err(|e| TranslationError::Primitive(e.to_string()))?;

        let (_, post_norm_outs) = graph.add_node(
            format!("{prefix}.post_attention_norm"),
            post_norm_prog,
            vec![
                GraphInput {
                    buffer: "input".into(),
                    value: hidden_after_attn,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
                GraphInput {
                    buffer: "weight".into(),
                    value: post_norm_weight,
                    contract: make_contract(
                        dtype.clone(),
                        vec![ShapeDim::Known(u64::from(hidden_dim))],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "output".into(),
                name: format!("{prefix}.normalized_mlp_in"),
                contract: make_contract(
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                ),
                retained_successor_of: None,
            }],
        )?;

        // 5. MLP or MoE Layer
        let mlp_out = {
            let inter_dim = if let Some(moe) = &self.config.moe {
                let _router_weight = graph.add_external_value(
                    format!("{prefix}.mlp.gate.weight"),
                    make_contract(
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(moe.num_experts)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                )?;
                moe.expert_hidden_dim
            } else {
                self.config.intermediate_dim
            };

            let gate_weight = graph.add_external_value(
                format!("{prefix}.mlp.gate_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(inter_dim)),
                        ShapeDim::Known(u64::from(hidden_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let up_weight = graph.add_external_value(
                format!("{prefix}.mlp.up_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(inter_dim)),
                        ShapeDim::Known(u64::from(hidden_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let down_weight = graph.add_external_value(
                format!("{prefix}.mlp.down_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(hidden_dim)),
                        ShapeDim::Known(u64::from(inter_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;

            // Gate projection
            let gate_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                hidden_dim,
                inter_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let intermediate_shape = vec![
                ShapeDim::Known(u64::from(self.workload.batch_size)),
                ShapeDim::Known(u64::from(self.workload.sequence_len)),
                ShapeDim::Known(u64::from(inter_dim)),
            ];

            let (_, gate_outs) = graph.add_node(
                format!("{prefix}.gate_proj"),
                gate_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: post_norm_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            hidden_shape.to_vec(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: gate_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(inter_dim)),
                                ShapeDim::Known(u64::from(hidden_dim)),
                            ],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.gate_out"),
                    contract: make_contract(
                        dtype.clone(),
                        intermediate_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;

            // Up projection
            let up_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                hidden_dim,
                inter_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, up_outs) = graph.add_node(
                format!("{prefix}.up_proj"),
                up_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: post_norm_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            hidden_shape.to_vec(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: up_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(inter_dim)),
                                ShapeDim::Known(u64::from(hidden_dim)),
                            ],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.up_out"),
                    contract: make_contract(
                        dtype.clone(),
                        intermediate_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;

            // SwiGLU activation
            let swiglu_prog = swiglu_typed(
                "gate",
                "up",
                "out",
                rows * inter_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, swiglu_outs) = graph.add_node(
                format!("{prefix}.swiglu"),
                swiglu_prog,
                vec![
                    GraphInput {
                        buffer: "gate".into(),
                        value: gate_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            intermediate_shape.clone(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "up".into(),
                        value: up_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            intermediate_shape.clone(),
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "out".into(),
                    name: format!("{prefix}.swiglu_out"),
                    contract: make_contract(
                        dtype.clone(),
                        intermediate_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;

            // Down projection
            let down_prog = linear_rows_no_bias_out_in_typed(
                "input",
                "weight",
                "output",
                rows,
                inter_dim,
                hidden_dim,
                dtype.clone(),
            )
            .map_err(TranslationError::Primitive)?;

            let (_, down_outs) = graph.add_node(
                format!("{prefix}.down_proj"),
                down_prog,
                vec![
                    GraphInput {
                        buffer: "input".into(),
                        value: swiglu_outs[0],
                        contract: make_contract(
                            dtype.clone(),
                            intermediate_shape,
                            ValueLifetime::Invocation,
                            BufferAccess::ReadOnly,
                        ),
                    },
                    GraphInput {
                        buffer: "weight".into(),
                        value: down_weight,
                        contract: make_contract(
                            dtype.clone(),
                            vec![
                                ShapeDim::Known(u64::from(hidden_dim)),
                                ShapeDim::Known(u64::from(inter_dim)),
                            ],
                            ValueLifetime::Constant,
                            BufferAccess::ReadOnly,
                        ),
                    },
                ],
                vec![GraphOutput {
                    buffer: "output".into(),
                    name: format!("{prefix}.mlp_out"),
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadWrite,
                    ),
                    retained_successor_of: None,
                }],
            )?;
            down_outs[0]
        };

        // 6. MLP Residual Addition: hidden_after_attn + mlp_out
        let mlp_residual_prog = residual_add_typed(
            "a",
            "b",
            "out",
            rows * hidden_dim,
            dtype.clone(),
        )
        .map_err(TranslationError::Primitive)?;

        let (_, mlp_res_outs) = graph.add_node(
            format!("{prefix}.mlp_residual_add"),
            mlp_residual_prog,
            vec![
                GraphInput {
                    buffer: "a".into(),
                    value: hidden_after_attn,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
                GraphInput {
                    buffer: "b".into(),
                    value: mlp_out,
                    contract: make_contract(
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: format!("{prefix}.layer_output"),
                contract: make_contract(
                    dtype,
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                ),
                retained_successor_of: None,
            }],
        )?;

        Ok(mlp_res_outs[0])
    }
}
