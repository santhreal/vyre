//! Attention sub-dialect: softmax + scaled dot-product + GQA + RoPE + MLA.
mod attention;
pub mod flash_attention;
pub mod flash_attention_2;
mod gated_delta;
mod gated_delta_chunked;
mod gated_delta_layout;
pub mod gqa_attention;
mod head_to_token;
mod kv_cache;
mod layout_permute;
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

/// Test-only owner of the `q`/`k`/`v`/`out` reference-eval harness shared by the
/// attention program tests: it locates the `out` buffer, sizes a zeroed output
/// value from its element count, runs the reference interpreter, and decodes the
/// result as `f32`.
///
/// It owns setup only. No expectation, tolerance, or reference computation lives
/// here, so a differential test still supplies both arms itself and still fails
/// when either arm is wrong.
#[cfg(test)]
pub(crate) fn eval_qkv_program(
    program: &vyre_foundation::ir::Program,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    on_failure: &str,
) -> Vec<f32> {
    use crate::fixture_bytes::{decode_f32, f32_bytes};
    use vyre_reference::value::Value;

    let out_bytes = program
        .buffers()
        .iter()
        .find(|b| b.name() == "out")
        .map(|b| b.count() as usize * core::mem::size_of::<f32>())
        .expect("Fix: output buffer present");
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(f32_bytes(q)),
            Value::from(f32_bytes(k)),
            Value::from(f32_bytes(v)),
            Value::from(vec![0u8; out_bytes]),
        ],
    )
    .unwrap_or_else(|err| panic!("{on_failure} ({err:?})"));
    decode_f32(&outputs[0].to_bytes())
}
