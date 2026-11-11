//! One transformer layer, translated into graph nodes.
//!
//! Normalization, attention with its key and value state edges, and the
//! feed-forward or routed expert block, in the order the forward pass runs
//! them.

use vyre::ir::{BufferAccess, GraphValueId, ProgramGraph, ShapeDim, ValueLifetime};
use vyre_libs::nn::{
    activation::{residual_add_typed, swiglu_typed},
    linear::linear_rows_no_bias_out_in_typed,
};

use super::rms_norm_node::{add_learned_rms_norm, RmsNormNames};
use super::{graph_input, graph_output, make_contract, ModelGraphBuilder, TranslationError};

impl ModelGraphBuilder<'_> {
    pub(super) fn build_layer(
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
        let norm_out = add_learned_rms_norm(
            graph,
            RmsNormNames {
                weight: format!("{prefix}.input_layernorm.weight"),
                node: format!("{prefix}.input_norm"),
                output: format!("{prefix}.normalized_attn_in"),
            },
            input_hidden,
            hidden_shape,
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
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
                    vec![
                        ShapeDim::Known(u64::from(q_dim)),
                        ShapeDim::Known(u64::from(hidden_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let o_weight = graph.add_external_value(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(hidden_dim)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
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
                    graph_input(
                        "input",
                        norm_out,
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        q_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(q_dim)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.q_states"),
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(self.workload.batch_size)),
                        ShapeDim::Known(u64::from(self.workload.sequence_len)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
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
                    graph_input(
                        "input",
                        q_outs[0],
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(self.workload.batch_size)),
                            ShapeDim::Known(u64::from(self.workload.sequence_len)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        o_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(hidden_dim)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.attn_out"),
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
            )?;
            o_outs[0]
        } else {
            // Standard Multi-Head / Grouped-Query Attention Projections
            let q_dim = self.config.num_heads * self.config.head_dim;

            let q_weight = graph.add_external_value(
                format!("{prefix}.self_attn.q_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(q_dim)),
                        ShapeDim::Known(u64::from(hidden_dim)),
                    ],
                    ValueLifetime::Constant,
                    BufferAccess::ReadOnly,
                ),
            )?;
            let o_weight = graph.add_external_value(
                format!("{prefix}.self_attn.o_proj.weight"),
                make_contract(
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(hidden_dim)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
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
                    graph_input(
                        "input",
                        norm_out,
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        q_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(q_dim)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.q_states"),
                    dtype.clone(),
                    vec![
                        ShapeDim::Known(u64::from(self.workload.batch_size)),
                        ShapeDim::Known(u64::from(self.workload.sequence_len)),
                        ShapeDim::Known(u64::from(q_dim)),
                    ],
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
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
                    graph_input(
                        "input",
                        q_outs[0],
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(self.workload.batch_size)),
                            ShapeDim::Known(u64::from(self.workload.sequence_len)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        o_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(hidden_dim)),
                            ShapeDim::Known(u64::from(q_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.attn_out"),
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
            )?;
            o_outs[0]
        };

        // 3. Attention Residual Addition: input_hidden + attn_out
        let attn_residual_prog =
            residual_add_typed("a", "b", "out", rows * hidden_dim, dtype.clone())
                .map_err(TranslationError::Primitive)?;

        let (_, attn_res_outs) = graph.add_node(
            format!("{prefix}.attn_residual_add"),
            attn_residual_prog,
            vec![
                graph_input(
                    "a",
                    input_hidden,
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
                graph_input(
                    "b",
                    attn_out,
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
            ],
            vec![graph_output(
                "out",
                format!("{prefix}.hidden_after_attn"),
                dtype.clone(),
                hidden_shape.to_vec(),
                ValueLifetime::Invocation,
                BufferAccess::ReadWrite,
            )],
        )?;
        let hidden_after_attn = attn_res_outs[0];

        // 4. Post-Attention Normalization
        let post_norm_outs = add_learned_rms_norm(
            graph,
            RmsNormNames {
                weight: format!("{prefix}.post_attention_layernorm.weight"),
                node: format!("{prefix}.post_attention_norm"),
                output: format!("{prefix}.normalized_mlp_in"),
            },
            hidden_after_attn,
            hidden_shape,
            rows,
            hidden_dim,
            self.config.norm_eps,
            dtype.clone(),
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
                    graph_input(
                        "input",
                        post_norm_outs,
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        gate_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(inter_dim)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.gate_out"),
                    dtype.clone(),
                    intermediate_shape.clone(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
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
                    graph_input(
                        "input",
                        post_norm_outs,
                        dtype.clone(),
                        hidden_shape.to_vec(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        up_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(inter_dim)),
                            ShapeDim::Known(u64::from(hidden_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.up_out"),
                    dtype.clone(),
                    intermediate_shape.clone(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
            )?;

            // SwiGLU activation
            let swiglu_prog = swiglu_typed("gate", "up", "out", rows * inter_dim, dtype.clone())
                .map_err(TranslationError::Primitive)?;

            let (_, swiglu_outs) = graph.add_node(
                format!("{prefix}.swiglu"),
                swiglu_prog,
                vec![
                    graph_input(
                        "gate",
                        gate_outs[0],
                        dtype.clone(),
                        intermediate_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "up",
                        up_outs[0],
                        dtype.clone(),
                        intermediate_shape.clone(),
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "out",
                    format!("{prefix}.swiglu_out"),
                    dtype.clone(),
                    intermediate_shape.clone(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
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
                    graph_input(
                        "input",
                        swiglu_outs[0],
                        dtype.clone(),
                        intermediate_shape,
                        ValueLifetime::Invocation,
                        BufferAccess::ReadOnly,
                    ),
                    graph_input(
                        "weight",
                        down_weight,
                        dtype.clone(),
                        vec![
                            ShapeDim::Known(u64::from(hidden_dim)),
                            ShapeDim::Known(u64::from(inter_dim)),
                        ],
                        ValueLifetime::Constant,
                        BufferAccess::ReadOnly,
                    ),
                ],
                vec![graph_output(
                    "output",
                    format!("{prefix}.mlp_out"),
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadWrite,
                )],
            )?;
            down_outs[0]
        };

        // 6. MLP Residual Addition: hidden_after_attn + mlp_out
        let mlp_residual_prog =
            residual_add_typed("a", "b", "out", rows * hidden_dim, dtype.clone())
                .map_err(TranslationError::Primitive)?;

        let (_, mlp_res_outs) = graph.add_node(
            format!("{prefix}.mlp_residual_add"),
            mlp_residual_prog,
            vec![
                graph_input(
                    "a",
                    hidden_after_attn,
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
                graph_input(
                    "b",
                    mlp_out,
                    dtype.clone(),
                    hidden_shape.to_vec(),
                    ValueLifetime::Invocation,
                    BufferAccess::ReadOnly,
                ),
            ],
            vec![graph_output(
                "out",
                format!("{prefix}.layer_output"),
                dtype,
                hidden_shape.to_vec(),
                ValueLifetime::Invocation,
                BufferAccess::ReadWrite,
            )],
        )?;

        Ok(mlp_res_outs[0])
    }
}
