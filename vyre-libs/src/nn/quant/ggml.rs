//! GGML K-Quants dequantization primitives.
//!
//! Supports Q2_K, Q4_K, Q6_K block formats used by llama.cpp/GGUF.
//! These are block-wise quantization formats with per-block (or per-super-block)
//! scales and zero-points.
//!
//! Category A composition.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

// ---------------------------------------------------------------------------
// Q4_K: "type-1" 4-bit quantization
// Super-blocks: 8 blocks per super-block
// Block: 32 weights per block
// Scales/mins: quantized with 6 bits each
// Total: 4.5 bits per weight
// ---------------------------------------------------------------------------

/// Q4_K super-block layout constants.
/** Q4_K super-block size: 8 blocks × 32 weights = 256 weights. */
pub const Q4_K_SUPER_BLOCK_SIZE: u32 = 256;
/** Q4_K block size in weights. */
pub const Q4_K_BLOCK_SIZE: u32 = 32;
/** Q4_K blocks per super-block. */
pub const Q4_K_BLOCKS_PER_SUPER: u32 = 8;

fn k_quant_unpack(
    spec: KQuantLinearSpec,
    packed: &str,
    scales: &str,
    mins: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Err(format!("Fix: {} n=0 is invalid", spec.unpack_op_id));
    }
    let n_blocks = n.div_ceil(spec.block_size);
    let packed_count = n_blocks
        .checked_mul(spec.words_per_block)
        .ok_or_else(|| format!("Fix: {} packed word count overflows u32", spec.unpack_op_id))?;
    let value_mask = (1_u32 << spec.bits_per_value) - 1;
    let row = Expr::var("i");
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(row.clone(), Expr::u32(n)),
            vec![
                Node::let_bind(
                    "block_idx",
                    Expr::div(row.clone(), Expr::u32(spec.block_size)),
                ),
                Node::let_bind(
                    "within_block",
                    Expr::rem(row.clone(), Expr::u32(spec.block_size)),
                ),
                Node::let_bind(
                    "byte_idx",
                    Expr::div(Expr::var("within_block"), Expr::u32(spec.values_per_byte)),
                ),
                Node::let_bind(
                    "value_shift",
                    Expr::mul(
                        Expr::rem(Expr::var("within_block"), Expr::u32(spec.values_per_byte)),
                        Expr::u32(spec.bits_per_value),
                    ),
                ),
                Node::let_bind(
                    "word_idx",
                    Expr::add(
                        Expr::mul(Expr::var("block_idx"), Expr::u32(spec.words_per_block)),
                        Expr::div(Expr::var("byte_idx"), Expr::u32(4)),
                    ),
                ),
                Node::let_bind(
                    "word_shift",
                    Expr::mul(Expr::rem(Expr::var("byte_idx"), Expr::u32(4)), Expr::u32(8)),
                ),
                Node::let_bind("packed_word", Expr::load(packed, Expr::var("word_idx"))),
                Node::let_bind(
                    "quantized",
                    Expr::bitand(
                        Expr::shr(
                            Expr::var("packed_word"),
                            Expr::add(Expr::var("word_shift"), Expr::var("value_shift")),
                        ),
                        Expr::u32(value_mask),
                    ),
                ),
                Node::let_bind("scale", Expr::load(scales, Expr::var("block_idx"))),
                Node::let_bind("min", Expr::load(mins, Expr::var("block_idx"))),
                Node::store(
                    output,
                    row,
                    Expr::add(
                        Expr::mul(
                            Expr::cast(DataType::F32, Expr::var("quantized")),
                            Expr::var("scale"),
                        ),
                        Expr::var("min"),
                    ),
                ),
            ],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(packed, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(packed_count),
            BufferDecl::storage(scales, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(n_blocks),
            BufferDecl::storage(mins, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(n_blocks),
            BufferDecl::output(output, 3, DataType::F32).with_count(n),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(spec.unpack_op_id, body)],
    ))
}

/// Dequantize Q4_K weights.
///
/// Buffer layout (per super-block):
///   - bytes 0..1:   scale_min_low (u16)  -  low 6 bits of 8 scales + 8 mins
///   - bytes 2..3:   scale_min_high (u16)  -  high bits
///   - bytes 4..11:  8 scales (6-bit each, packed)
///   - bytes 12..19: 8 mins (6-bit each, packed)
///   - bytes 20..147: 256 nibbles (128 bytes) = 8 blocks * 32 weights * 4 bits
///
/// For simplicity, this kernel assumes pre-unpacked scales/mins buffers
/// produced by the loader. The `scales` and `mins` buffers are F32
/// with one element per block.
///
/// `packed` contains the 4-bit nibbles, 2 per byte, stored as U32 words
/// for aligned access.
pub fn q4_k_unpack(
    packed: &str,
    scales: &str,
    mins: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    k_quant_unpack(Q4_K_LINEAR_SPEC, packed, scales, mins, output, n)
}

// ---------------------------------------------------------------------------
// Q2_K: "type-1" 2-bit quantization
// Super-blocks: 16 blocks per super-block
// Block: 16 weights per block
// Scales/mins: quantized with 4 bits each
// Total: 2.5625 bits per weight
// ---------------------------------------------------------------------------

/** Q2_K super-block size: 16 blocks × 16 weights = 256 weights. */
pub const Q2_K_SUPER_BLOCK_SIZE: u32 = 256;
/** Q2_K block size in weights. */
pub const Q2_K_BLOCK_SIZE: u32 = 16;
/** Q2_K blocks per super-block. */
pub const Q2_K_BLOCKS_PER_SUPER: u32 = 16;

/// Dequantize Q2_K weights.
///
/// `packed` contains 2-bit values, 4 per byte, stored as U32 words.
/// `scales` and `mins` are per-block F32 values.
pub fn q2_k_unpack(
    packed: &str,
    scales: &str,
    mins: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    k_quant_unpack(Q2_K_LINEAR_SPEC, packed, scales, mins, output, n)
}

// ---------------------------------------------------------------------------
// Fused dequant + matmul for Q4_K and Q2_K
// These avoid materializing the full dequantized buffer.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct KQuantLinearSpec {
    op_id: &'static str,
    unpack_op_id: &'static str,
    block_size: u32,
    values_per_byte: u32,
    bits_per_value: u32,
    words_per_block: u32,
}

const Q4_K_LINEAR_SPEC: KQuantLinearSpec = KQuantLinearSpec {
    unpack_op_id: "vyre-libs::quant::q4_k_unpack",
    op_id: "vyre-libs::quant::q4_k_linear",
    block_size: Q4_K_BLOCK_SIZE,
    values_per_byte: 2,
    bits_per_value: 4,
    words_per_block: 4,
};
const Q2_K_LINEAR_SPEC: KQuantLinearSpec = KQuantLinearSpec {
    unpack_op_id: "vyre-libs::quant::q2_k_unpack",
    op_id: "vyre-libs::quant::q2_k_linear",
    block_size: Q2_K_BLOCK_SIZE,
    values_per_byte: 4,
    bits_per_value: 2,
    words_per_block: 1,
};

#[allow(clippy::too_many_arguments)]
fn k_quant_linear(
    spec: KQuantLinearSpec,
    x: &str,
    w_packed: &str,
    w_scales: &str,
    w_mins: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    if in_dim == 0 || out_dim == 0 {
        return Err(format!("Fix: {} all dims must be > 0", spec.op_id));
    }
    let weight_count = in_dim
        .checked_mul(out_dim)
        .ok_or_else(|| format!("Fix: {} dimensions overflow u32", spec.op_id))?;
    let block_count = weight_count.div_ceil(spec.block_size);
    let packed_word_count = block_count
        .checked_mul(spec.words_per_block)
        .ok_or_else(|| format!("Fix: {} packed word count overflows u32", spec.op_id))?;
    let i = Expr::var("i");
    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(i.clone(), Expr::u32(out_dim)),
            vec![
                Node::let_bind("acc", Expr::load(b, i.clone())),
                Node::loop_for(
                    "k",
                    Expr::u32(0),
                    Expr::u32(in_dim),
                    vec![
                        Node::let_bind(
                            "linear_idx",
                            Expr::add(Expr::mul(Expr::var("k"), Expr::u32(out_dim)), i.clone()),
                        ),
                        Node::let_bind(
                            "block_idx",
                            Expr::div(Expr::var("linear_idx"), Expr::u32(spec.block_size)),
                        ),
                        Node::let_bind(
                            "within_block",
                            Expr::rem(Expr::var("linear_idx"), Expr::u32(spec.block_size)),
                        ),
                        Node::let_bind(
                            "byte_idx",
                            Expr::div(Expr::var("within_block"), Expr::u32(spec.values_per_byte)),
                        ),
                        Node::let_bind(
                            "value_shift",
                            Expr::mul(
                                Expr::rem(
                                    Expr::var("within_block"),
                                    Expr::u32(spec.values_per_byte),
                                ),
                                Expr::u32(spec.bits_per_value),
                            ),
                        ),
                        Node::let_bind(
                            "word_idx",
                            Expr::add(
                                Expr::mul(Expr::var("block_idx"), Expr::u32(spec.words_per_block)),
                                Expr::div(Expr::var("byte_idx"), Expr::u32(4)),
                            ),
                        ),
                        Node::let_bind(
                            "word_shift",
                            Expr::mul(Expr::rem(Expr::var("byte_idx"), Expr::u32(4)), Expr::u32(8)),
                        ),
                        Node::let_bind("packed_word", Expr::load(w_packed, Expr::var("word_idx"))),
                        Node::let_bind(
                            "quantized",
                            Expr::bitand(
                                Expr::shr(
                                    Expr::var("packed_word"),
                                    Expr::add(Expr::var("word_shift"), Expr::var("value_shift")),
                                ),
                                Expr::u32((1_u32 << spec.bits_per_value) - 1),
                            ),
                        ),
                        Node::let_bind("scale", Expr::load(w_scales, Expr::var("block_idx"))),
                        Node::let_bind("min", Expr::load(w_mins, Expr::var("block_idx"))),
                        Node::let_bind(
                            "weight",
                            Expr::add(
                                Expr::mul(
                                    Expr::cast(DataType::F32, Expr::var("quantized")),
                                    Expr::var("scale"),
                                ),
                                Expr::var("min"),
                            ),
                        ),
                        Node::assign(
                            "acc",
                            Expr::add(
                                Expr::var("acc"),
                                Expr::mul(Expr::load(x, Expr::var("k")), Expr::var("weight")),
                            ),
                        ),
                    ],
                ),
                Node::store(out, i, Expr::var("acc")),
            ],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32).with_count(in_dim),
            BufferDecl::storage(w_packed, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(packed_word_count),
            BufferDecl::storage(w_scales, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(block_count),
            BufferDecl::storage(w_mins, 3, BufferAccess::ReadOnly, DataType::F32)
                .with_count(block_count),
            BufferDecl::storage(b, 4, BufferAccess::ReadOnly, DataType::F32).with_count(out_dim),
            BufferDecl::output(out, 5, DataType::F32).with_count(out_dim),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(spec.op_id, body)],
    ))
}

/// Fused Q4_K dequant + linear: `out = x @ dequant(w_q4k) + b`
///
/// `w_packed` is Q4_K packed nibbles (U32 words).
/// `w_scales` and `w_mins` are per-block F32.
/// `x` is F32 input, `b` is F32 bias, `out` is F32 output.
pub fn q4_k_linear(
    x: &str,
    w_packed: &str,
    w_scales: &str,
    w_mins: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    k_quant_linear(
        Q4_K_LINEAR_SPEC,
        x,
        w_packed,
        w_scales,
        w_mins,
        b,
        out,
        in_dim,
        out_dim,
    )
}

/// Fused Q2_K dequant + linear: `out = x @ dequant(w_q2k) + b`
pub fn q2_k_linear(
    x: &str,
    w_packed: &str,
    w_scales: &str,
    w_mins: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    k_quant_linear(
        Q2_K_LINEAR_SPEC,
        x,
        w_packed,
        w_scales,
        w_mins,
        b,
        out,
        in_dim,
        out_dim,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::decode_f32;
    use crate::fixture_bytes::f32_bytes;
    use crate::fixture_bytes::u32_bytes;
    use vyre_reference::value::Value;

    #[test]
    fn q4_k_unpack_simple() {
        // 32 weights, 1 block
        // scales = [1.0], mins = [0.0]
        // packed: 16 bytes = 4 u32 words
        // nibbles: 0,1,2,3,...,15 (first 16 weights), then repeat
        let scales = vec![1.0f32];
        let mins = vec![0.0f32];
        let packed = vec![0x7654_3210u32, 0xFEDC_BA98, 0x0, 0x0];
        let program = q4_k_unpack("packed", "scales", "mins", "out", 16).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(u32_bytes(&packed)),
                Value::from(f32_bytes(&scales)),
                Value::from(f32_bytes(&mins)),
                Value::from(vec![0u8; 64]),
            ],
        )
        .expect("Fix: q4_k_unpack must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[15], 15.0);
    }

    #[test]
    fn q2_k_unpack_simple() {
        // 16 weights, 1 block
        // scales = [1.0], mins = [0.0]
        // packed: 1 u32 word containing 16 2-bit values
        // q2 values: 0,1,2,3,0,1,2,3,... (4 bytes)
        let scales = vec![1.0f32];
        let mins = vec![0.0f32];
        let packed = vec![0xE4E4_E4E4u32]; // 11_10_01_00 repeated
        let program = q2_k_unpack("packed", "scales", "mins", "out", 16).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(u32_bytes(&packed)),
                Value::from(f32_bytes(&scales)),
                Value::from(f32_bytes(&mins)),
                Value::from(vec![0u8; 64]),
            ],
        )
        .expect("Fix: q2_k_unpack must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        // Byte pattern: 0xE4 = 11_10_01_00 -> values 0,1,2,3
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 2.0);
        assert_eq!(out[3], 3.0);
    }

    /// Proves Q4 linear sizes packed storage across every output column, not
    /// only one input row, so accesses in later quantization blocks stay valid.
    #[test]
    fn q4_k_linear_reads_blocks_created_by_output_dimension() {
        let mut x = vec![0.0f32; 32];
        x[31] = 1.0;
        let b = vec![0.5f32, 1.5];
        let packed = vec![0x1111_1111u32; 8];
        let scales = vec![1.0f32, 2.0];
        let mins = vec![0.0f32, 0.0];
        let program = q4_k_linear("x", "packed", "scales", "mins", "b", "out", 32, 2).unwrap();
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&x)),
                Value::from(u32_bytes(&packed)),
                Value::from(f32_bytes(&scales)),
                Value::from(f32_bytes(&mins)),
                Value::from(f32_bytes(&b)),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: q4_k_linear must read the second packed quantization block");
        let out = decode_f32(&outputs[0].to_bytes());
        assert_eq!(out, vec![2.5, 3.5]);
    }
}
