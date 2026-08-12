//! Attention sub-dialect: softmax + scaled dot-product + GQA + RoPE + MLA.
mod attention;
pub mod flash_attention;
pub mod flash_attention_2;
mod gated_delta;
mod gated_delta_chunked;
pub mod gqa_attention;
mod head_to_token;
mod kv_cache;
pub mod mla;
pub mod partial_rope;
pub mod planner;
pub mod qk_gain;
pub mod quest;
mod softmax;
mod tiled_online_softmax;
mod token_to_head;
pub mod turboquant;

pub use attention::{attention, attention_reference, try_attention_reference, Attention};
pub use flash_attention::flash_attention;
pub use flash_attention_2::{flash_attention_2, flash_attention_2_reference};
pub use gated_delta::{chunked_gated_delta, recurrent_gated_delta, RecurrentGatedDeltaError};
pub use gqa_attention::{gqa_attention, gqa_attention_causal, gqa_attention_causal_typed};
pub use head_to_token::{attention_head_to_token, attention_head_to_token_typed};
pub use kv_cache::{kv_cache_append, kv_cache_append_typed, KvCacheAppendError};
pub use mla::{mla_compress_kv, mla_decode};
pub use partial_rope::{partial_rope, partial_rope_at_offset, partial_rope_at_offset_typed};
pub use planner::{
    plan_flash_attention_scalar, plan_flash_attention_tiled, FlashAttentionBenchMetrics,
    FlashAttentionKernelKind, FlashAttentionMemoryTraffic, FlashAttentionWorkPlan,
    FLASH_ATTENTION_SEQUENCE_PARALLEL_TARGET_TILES_PER_SPLIT,
};
pub use qk_gain::qk_gain;
pub use quest::quest_paging;
pub use softmax::{softmax, softmax_reference, Softmax};
pub use token_to_head::{attention_token_to_head, attention_token_to_head_typed};
pub use turboquant::turboquant_attention;
