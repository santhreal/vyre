//! One binary for every nn-attention integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/harness/mod.rs`.
#[path = "harness/mod.rs"]
pub mod harness;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/attention_head_to_token_contract.rs`.
#[path = "attention_head_to_token_contract.rs"]
pub mod attention_head_to_token_contract;

/// Integration tests from `tests/causal_gqa_contract.rs`.
#[path = "causal_gqa_contract.rs"]
pub mod causal_gqa_contract;

/// Integration tests from `tests/causal_gqa_typed_contract.rs`.
#[path = "causal_gqa_typed_contract.rs"]
pub mod causal_gqa_typed_contract;

/// Integration tests from `tests/chunked_gated_delta_contract.rs`.
#[path = "chunked_gated_delta_contract.rs"]
pub mod chunked_gated_delta_contract;

/// Integration tests from `tests/flash_attention_plan_shared_memory.rs`.
#[path = "flash_attention_plan_shared_memory.rs"]
pub mod flash_attention_plan_shared_memory;

/// Integration tests from `tests/gqa_attention_primitive_composition_contracts.rs`.
#[cfg(feature = "nn-attention")]
#[path = "gqa_attention_primitive_composition_contracts.rs"]
pub mod gqa_attention_primitive_composition_contracts;

/// Integration tests from `tests/head_to_token_typed_contract.rs`.
#[path = "head_to_token_typed_contract.rs"]
pub mod head_to_token_typed_contract;

/// Integration tests from `tests/kv_cache_append_contract.rs`.
#[path = "kv_cache_append_contract.rs"]
pub mod kv_cache_append_contract;

/// Integration tests from `tests/kv_cache_typed_contract.rs`.
#[path = "kv_cache_typed_contract.rs"]
pub mod kv_cache_typed_contract;

/// Integration tests from `tests/nn_attention_clone_family_ir_invariance.rs`.
#[path = "nn_attention_clone_family_ir_invariance.rs"]
pub mod nn_attention_clone_family_ir_invariance;

/// Integration tests from `tests/overflow_guards.rs`.
#[cfg(all(feature = "math-linalg", feature = "nn-attention"))]
#[path = "overflow_guards.rs"]
pub mod overflow_guards;

/// Integration tests from `tests/partial_rope_offset_contract.rs`.
#[path = "partial_rope_offset_contract.rs"]
pub mod partial_rope_offset_contract;

/// Integration tests from `tests/partial_rope_typed_contract.rs`.
#[path = "partial_rope_typed_contract.rs"]
pub mod partial_rope_typed_contract;

/// Integration tests from `tests/qk_gain_shape_overflow_contracts.rs`.
#[cfg(feature = "nn-attention")]
#[path = "qk_gain_shape_overflow_contracts.rs"]
pub mod qk_gain_shape_overflow_contracts;

/// Integration tests from `tests/qk_gain_zero_shape_contracts.rs`.
#[cfg(feature = "nn-attention")]
#[path = "qk_gain_zero_shape_contracts.rs"]
pub mod qk_gain_zero_shape_contracts;

/// Integration tests from `tests/recurrent_gated_delta_contract.rs`.
#[path = "recurrent_gated_delta_contract.rs"]
pub mod recurrent_gated_delta_contract;

/// Integration tests from `tests/quest_paging_extent_contracts.rs`.
#[path = "quest_paging_extent_contracts.rs"]
pub mod quest_paging_extent_contracts;
