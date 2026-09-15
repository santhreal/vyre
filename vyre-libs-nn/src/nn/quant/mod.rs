//! Quantization sub-dialect for Parameter Golf recipe.
//!
//! Contains int4 dot, int6/int8 pack/unpack, byte shuffle, GPTQ-SDClip,
//! and GGML K-Quants ops.
pub(crate) mod byte_shuffle;
pub(crate) mod ggml;
pub(crate) mod gptq;
pub(crate) mod int4;
pub(crate) mod int6;
pub(crate) mod int8;

pub use byte_shuffle::byte_shuffle;
pub use ggml::{
    q2_k_linear, q2_k_unpack, q4_k_linear, q4_k_unpack, Q2_K_BLOCKS_PER_SUPER, Q2_K_BLOCK_SIZE,
    Q2_K_SUPER_BLOCK_SIZE, Q4_K_BLOCKS_PER_SUPER, Q4_K_BLOCK_SIZE, Q4_K_SUPER_BLOCK_SIZE,
};
pub use gptq::{gptq_round, gptq_sdclip};
pub use int4::{
    int4_batched_matmul_f32_scaled, int4_batched_matmul_top1_f32_scaled,
    int4_batched_matvec_f32_scaled, int4_dot_f32_scaled, int4_dot_i32, int4_matvec_f32_scaled,
};
pub use int4::{
    int4_batched_matmul_scaled_extension_id, int4_batched_matmul_top1_scaled_extension_id,
    int4_batched_matvec_scaled_extension_id, int4_dot_extension_id, int4_dot_scaled_extension_id,
    int4_matvec_scaled_extension_id, INT4_BATCHED_MATMUL_SCALED_EXTENSION_NAME,
    INT4_BATCHED_MATMUL_TOP1_SCALED_EXTENSION_NAME, INT4_BATCHED_MATVEC_SCALED_EXTENSION_NAME,
    INT4_DOT_EXTENSION_NAME, INT4_DOT_SCALED_EXTENSION_NAME, INT4_MATVEC_SCALED_EXTENSION_NAME,
};
pub use int6::{int6_pack, int6_unpack};
pub use int8::{int8_pack, int8_unpack};
