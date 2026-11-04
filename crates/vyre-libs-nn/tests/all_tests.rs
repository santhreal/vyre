//! One binary for every integration test in this crate.

extern crate vyre_foundation as vyre;

#[path = "harness/mod.rs"]
pub mod harness;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

#[path = "attention_head_to_token_contract.rs"]
pub mod attention_head_to_token_contract;

#[path = "attention_layout_launch_domain.rs"]
pub mod attention_layout_launch_domain;

#[path = "causal_conv_state_transition_contract.rs"]
pub mod causal_conv_state_transition_contract;

#[path = "causal_gqa_contract.rs"]
pub mod causal_gqa_contract;

#[path = "causal_gqa_typed_contract.rs"]
pub mod causal_gqa_typed_contract;

#[path = "chunked_gated_delta_contract.rs"]
pub mod chunked_gated_delta_contract;

#[path = "dense_gated_mlp_graph_contract.rs"]
pub mod dense_gated_mlp_graph_contract;

#[path = "depthwise_causal_conv1d_contract.rs"]
pub mod depthwise_causal_conv1d_contract;

#[path = "flash_attention_plan_shared_memory.rs"]
pub mod flash_attention_plan_shared_memory;

#[path = "fused_tile_attention_lowering.rs"]
pub mod fused_tile_attention_lowering;

#[path = "gated_rms_norm_contract.rs"]
pub mod gated_rms_norm_contract;

#[path = "gqa_attention_primitive_composition_contracts.rs"]
pub mod gqa_attention_primitive_composition_contracts;

#[path = "head_to_token_typed_contract.rs"]
pub mod head_to_token_typed_contract;

#[path = "indexed_map_composition_contracts.rs"]
pub mod indexed_map_composition_contracts;

#[path = "int4_primitive_composition.rs"]
pub mod int4_primitive_composition;

#[path = "kv_cache_append_contract.rs"]
pub mod kv_cache_append_contract;

#[path = "kv_cache_typed_contract.rs"]
pub mod kv_cache_typed_contract;

#[path = "last_dim_l2_norm_contract.rs"]
pub mod last_dim_l2_norm_contract;

#[path = "linear_rows_contract.rs"]
pub mod linear_rows_contract;

#[path = "llm_fused_sampler_matches_the_unfused_pipeline.rs"]
pub mod llm_fused_sampler_matches_the_unfused_pipeline;

#[path = "llm_sampler_rejects_degenerate_shapes.rs"]
pub mod llm_sampler_rejects_degenerate_shapes;

#[path = "mlp_4x_leaky_sq_multi_workgroup_span.rs"]
pub mod mlp_4x_leaky_sq_multi_workgroup_span;

#[path = "nn_attention_clone_family_ir_invariance.rs"]
pub mod nn_attention_clone_family_ir_invariance;

#[path = "op_boundaries.rs"]
pub mod op_boundaries;

#[path = "optimized_programs.rs"]
pub mod optimized_programs;

#[path = "overflow_guards.rs"]
pub mod overflow_guards;

#[path = "paged_attention_eval.rs"]
pub mod paged_attention_eval;

#[path = "partial_rope_offset_contract.rs"]
pub mod partial_rope_offset_contract;

#[path = "partial_rope_typed_contract.rs"]
pub mod partial_rope_typed_contract;

#[path = "qk_gain_shape_overflow_contracts.rs"]
pub mod qk_gain_shape_overflow_contracts;

#[path = "qk_gain_zero_shape_contracts.rs"]
pub mod qk_gain_zero_shape_contracts;

#[path = "quantized_linear_affine_fma.rs"]
pub mod quantized_linear_affine_fma;

#[path = "quest_paging_extent_contracts.rs"]
pub mod quest_paging_extent_contracts;

#[path = "recurrent_gated_delta_contract.rs"]
pub mod recurrent_gated_delta_contract;

#[path = "sigmoid_gate_typed_contract.rs"]
pub mod sigmoid_gate_typed_contract;

#[path = "workgroup_cooperative_tiling.rs"]
pub mod workgroup_cooperative_tiling;
