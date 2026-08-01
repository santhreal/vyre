//! Behavioral parity for the fused affine grouped INT4 arithmetic path.

#![cfg(feature = "nn-linear-4bit")]

use vyre_libs::nn::{linear_4bit_affine_grouped_typed, QuantizedLinear4BitSpec};
use vyre_reference::value::Value;

const MAX_ABS_DRIFT: f32 = 1.0e-4;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect()
}

fn pack_weights(weights: &[u32], in_dim: usize, out_dim: usize) -> Vec<u32> {
    let mut packed = vec![0u32; (in_dim / 8) * out_dim];
    for block in 0..(in_dim / 8) {
        for out in 0..out_dim {
            let mut word = 0u32;
            for lane in 0..8 {
                word |= (weights[(block * 8 + lane) * out_dim + out] & 0xF) << (lane * 4);
            }
            packed[block * out_dim + out] = word;
        }
    }
    packed
}

fn scalar_oracle(
    x: &[f32],
    weights: &[u32],
    scales: &[f32],
    zero_points: &[u32],
    bias: &[f32],
    group_size: usize,
) -> Vec<f32> {
    let out_dim = bias.len();
    (0..out_dim)
        .map(|out| {
            let mut acc = bias[out];
            for group_start in (0..x.len()).step_by(group_size) {
                let group = group_start / group_size;
                let scale = scales[group * out_dim + out];
                let zero_offset = -1.0f32 * ((zero_points[group * out_dim + out] as f32) * scale);
                for k in group_start..(group_start + group_size).min(x.len()) {
                    let weight = (weights[k * out_dim + out] as f32).mul_add(scale, zero_offset);
                    acc = x[k].mul_add(weight, acc);
                }
            }
            acc
        })
        .collect()
}

fn execute_case(
    x: &[f32],
    weights: &[u32],
    scales: &[f32],
    zero_points: &[u32],
    bias: &[f32],
    group_size: usize,
) -> Vec<f32> {
    let in_dim = x.len();
    let out_dim = bias.len();
    let packed = pack_weights(weights, in_dim, out_dim);
    let spec =
        QuantizedLinear4BitSpec::affine_grouped(in_dim as u32, out_dim as u32, group_size as u32);
    let program = linear_4bit_affine_grouped_typed(&spec, "x", "w", "scale", "zp", "b", "out")
        .expect("valid grouped INT4 fixture must build");
    let padded_output_len = program.buffers()[5].count() as usize;
    let inputs = vec![
        Value::from(f32_bytes(x)),
        Value::from(u32_bytes(&packed)),
        Value::from(f32_bytes(scales)),
        Value::from(u32_bytes(zero_points)),
        Value::from(f32_bytes(bias)),
        Value::from(vec![0u8; padded_output_len * 4]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("grouped INT4 program must execute under the reference oracle");
    assert_eq!(outputs.len(), 1, "grouped INT4 emits one output buffer");
    decode_f32(&outputs[0].to_bytes())[..out_dim].to_vec()
}

fn assert_parity(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "output cardinality changed");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let drift = (actual - expected).abs();
        assert!(
            drift <= MAX_ABS_DRIFT,
            "output {index} drifted by {drift}: actual={actual}, expected={expected}, limit={MAX_ABS_DRIFT}"
        );
    }
}

/// The release-aligned 64-value group path must preserve exact affine dequantization while using fused multiply-add arithmetic.
#[test]
fn aligned_group_fma_matches_scalar_affine_oracle() {
    let in_dim = 64usize;
    let out_dim = 3usize;
    let x = (0..in_dim)
        .map(|k| ((k % 7) as f32 - 3.0) * 0.25)
        .collect::<Vec<_>>();
    let weights = (0..in_dim * out_dim)
        .map(|index| ((index * 5 + 3) & 0xF) as u32)
        .collect::<Vec<_>>();
    let scales = vec![0.25, 0.5, 1.0];
    let zero_points = vec![7, 8, 2];
    let bias = vec![1.0, -2.0, 0.5];

    let expected = scalar_oracle(&x, &weights, &scales, &zero_points, &bias, 64);
    let actual = execute_case(&x, &weights, &scales, &zero_points, &bias, 64);
    assert_parity(&actual, &expected);
}

/// The fallback chunk path must recompute each group's affine offset instead of leaking metadata across eight-value boundaries.
#[test]
fn chunk_path_fma_respects_every_quantization_group_boundary() {
    let in_dim = 32usize;
    let out_dim = 5usize;
    let x = (0..in_dim)
        .map(|k| ((k % 11) as f32 - 5.0) * 0.125)
        .collect::<Vec<_>>();
    let weights = (0..in_dim * out_dim)
        .map(|index| ((index * 11 + index / out_dim) & 0xF) as u32)
        .collect::<Vec<_>>();
    let scales = (0..(in_dim / 8) * out_dim)
        .map(|index| [0.125, 0.25, 0.5, 1.0][index & 3])
        .collect::<Vec<_>>();
    let zero_points = (0..scales.len())
        .map(|index| ((index * 3 + 1) & 0xF) as u32)
        .collect::<Vec<_>>();
    let bias = vec![-3.0, -1.0, 0.0, 2.0, 4.0];

    let expected = scalar_oracle(&x, &weights, &scales, &zero_points, &bias, 8);
    let actual = execute_case(&x, &weights, &scales, &zero_points, &bias, 8);
    assert_parity(&actual, &expected);
}

/// Nibbles below a large zero point must remain negative after the affine FMA rewrite instead of wrapping as unsigned values.
#[test]
fn affine_fma_preserves_negative_dequantized_weights() {
    let x = vec![1.0; 8];
    let weights = vec![0u32; 16];
    let scales = vec![2.0, 0.5];
    let zero_points = vec![15, 14];
    let bias = vec![0.0, 1.0];

    let expected = vec![-240.0, -55.0];
    let actual = execute_case(&x, &weights, &scales, &zero_points, &bias, 8);
    assert_parity(&actual, &expected);
}
