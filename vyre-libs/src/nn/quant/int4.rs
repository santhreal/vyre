//! Packed INT4 inference primitives.
//!
//! The dot-product wrapper delegates to `math::quantized` so packed
//! weights stay packed through the inner product. This avoids the extra global
//! memory traffic of materializing an unpacked i32 lane buffer before a dot.

use crate::math::quantized::i4x8_batched_matmul_f32_scaled as primitive_i4x8_batched_matmul_f32_scaled;
use crate::math::quantized::i4x8_batched_matmul_top1_f32_scaled as primitive_i4x8_batched_matmul_top1_f32_scaled;
use crate::math::quantized::i4x8_batched_matvec_f32_scaled as primitive_i4x8_batched_matvec_f32_scaled;
use crate::math::quantized::i4x8_dot_f32_scaled as primitive_i4x8_dot_f32_scaled;
use crate::math::quantized::i4x8_dot_i32 as primitive_i4x8_dot_i32;
use crate::math::quantized::i4x8_matvec_f32_scaled as primitive_i4x8_matvec_f32_scaled;
use vyre_foundation::composition::tag_program;
use vyre_foundation::ir::Program;

/// Stable spec-level extension name for packed INT4 dot products.
pub const INT4_DOT_EXTENSION_NAME: &str = "quant.int4.dot";

/// Stable spec-level extension name for fused scaled packed INT4 dot products.
pub const INT4_DOT_SCALED_EXTENSION_NAME: &str = "quant.int4.dot.scaled";

/// Stable spec-level extension name for fused scaled packed INT4 matvec.
pub const INT4_MATVEC_SCALED_EXTENSION_NAME: &str = "quant.int4.matvec.scaled";

/// Stable spec-level extension name for batched fused scaled packed INT4 matvec.
pub const INT4_BATCHED_MATVEC_SCALED_EXTENSION_NAME: &str = "quant.int4.batched_matvec.scaled";

/// Stable spec-level extension name for packed-activation batched INT4 matmul.
pub const INT4_BATCHED_MATMUL_SCALED_EXTENSION_NAME: &str = "quant.int4.batched_matmul.scaled";

/// Stable spec-level extension name for fused packed INT4 matmul top-1 routing.
pub const INT4_BATCHED_MATMUL_TOP1_SCALED_EXTENSION_NAME: &str =
    "quant.int4.batched_matmul.top1.scaled";

const DOT_OP_ID: &str = "vyre-libs::quant::int4_dot_i32";
const DOT_SCALED_OP_ID: &str = "vyre-libs::quant::int4_dot_f32_scaled";
const MATVEC_SCALED_OP_ID: &str = "vyre-libs::quant::int4_matvec_f32_scaled";
const BATCHED_MATVEC_SCALED_OP_ID: &str = "vyre-libs::quant::int4_batched_matvec_f32_scaled";
const BATCHED_MATMUL_SCALED_OP_ID: &str = "vyre-libs::quant::int4_batched_matmul_f32_scaled";
const BATCHED_MATMUL_TOP1_SCALED_OP_ID: &str =
    "vyre-libs::quant::int4_batched_matmul_top1_f32_scaled";

/// Stable extension id for packed INT4 dot products.
#[must_use]
pub fn int4_dot_extension_id() -> vyre_spec::extension::ExtensionBinOpId {
    vyre_spec::extension::ExtensionBinOpId::from_name(INT4_DOT_EXTENSION_NAME)
}

/// Stable extension id for fused scaled packed INT4 dot products.
#[must_use]
pub fn int4_dot_scaled_extension_id() -> vyre_spec::extension::ExtensionBinOpId {
    vyre_spec::extension::ExtensionBinOpId::from_name(INT4_DOT_SCALED_EXTENSION_NAME)
}

/// Stable extension id for fused scaled packed INT4 matvec.
#[must_use]
pub fn int4_matvec_scaled_extension_id() -> vyre_spec::extension::ExtensionTernaryOpId {
    vyre_spec::extension::ExtensionTernaryOpId::from_name(INT4_MATVEC_SCALED_EXTENSION_NAME)
}

/// Stable extension id for batched fused scaled packed INT4 matvec.
#[must_use]
pub fn int4_batched_matvec_scaled_extension_id() -> vyre_spec::extension::ExtensionTernaryOpId {
    vyre_spec::extension::ExtensionTernaryOpId::from_name(INT4_BATCHED_MATVEC_SCALED_EXTENSION_NAME)
}

/// Stable extension id for packed-activation batched INT4 matmul.
#[must_use]
pub fn int4_batched_matmul_scaled_extension_id() -> vyre_spec::extension::ExtensionTernaryOpId {
    vyre_spec::extension::ExtensionTernaryOpId::from_name(INT4_BATCHED_MATMUL_SCALED_EXTENSION_NAME)
}

/// Stable extension id for fused packed INT4 matmul top-1 routing.
#[must_use]
pub fn int4_batched_matmul_top1_scaled_extension_id() -> vyre_spec::extension::ExtensionTernaryOpId
{
    vyre_spec::extension::ExtensionTernaryOpId::from_name(
        INT4_BATCHED_MATMUL_TOP1_SCALED_EXTENSION_NAME,
    )
}

/// Build a packed signed INT4 dot-product Program.
#[must_use]
pub fn int4_dot_i32(lhs_packed: &str, rhs_packed: &str, out: &str, lane_count: u32) -> Program {
    tag_program(
        DOT_OP_ID,
        primitive_i4x8_dot_i32(lhs_packed, rhs_packed, out, lane_count),
    )
}

/// Build a fused packed signed INT4 dot-product Program with symmetric scales.
#[must_use]
pub fn int4_dot_f32_scaled(
    lhs_packed: &str,
    rhs_packed: &str,
    lhs_scale: &str,
    rhs_scale: &str,
    out: &str,
    lane_count: u32,
) -> Program {
    tag_program(
        DOT_SCALED_OP_ID,
        primitive_i4x8_dot_f32_scaled(
            lhs_packed, rhs_packed, lhs_scale, rhs_scale, out, lane_count,
        ),
    )
}

/// Build a fused row-scaled packed signed INT4 matrix-vector Program.
#[must_use]
pub fn int4_matvec_f32_scaled(
    weights_packed: &str,
    x: &str,
    row_scales: &str,
    out: &str,
    rows: u32,
    cols: u32,
) -> Program {
    tag_program(
        MATVEC_SCALED_OP_ID,
        primitive_i4x8_matvec_f32_scaled(weights_packed, x, row_scales, out, rows, cols),
    )
}

/// Build a fused batched row-scaled packed signed INT4 matrix-vector Program.
#[must_use]
pub fn int4_batched_matvec_f32_scaled(
    weights_packed: &str,
    x_batches: &str,
    row_scales: &str,
    out: &str,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Program {
    tag_program(
        BATCHED_MATVEC_SCALED_OP_ID,
        primitive_i4x8_batched_matvec_f32_scaled(
            weights_packed,
            x_batches,
            row_scales,
            out,
            batch,
            rows,
            cols,
        ),
    )
}

/// Build a packed-activation batched signed INT4 matrix-product Program.
#[must_use]
pub fn int4_batched_matmul_f32_scaled(
    weights_packed: &str,
    activation_batches_packed: &str,
    row_scales: &str,
    batch_scales: &str,
    out: &str,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Program {
    tag_program(
        BATCHED_MATMUL_SCALED_OP_ID,
        primitive_i4x8_batched_matmul_f32_scaled(
            weights_packed,
            activation_batches_packed,
            row_scales,
            batch_scales,
            out,
            batch,
            rows,
            cols,
        ),
    )
}

/// Build a fused packed-activation INT4 top-1 routing Program.
#[must_use]
pub fn int4_batched_matmul_top1_f32_scaled(
    weights_packed: &str,
    activation_batches_packed: &str,
    row_scales: &str,
    batch_scales: &str,
    out: &str,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Program {
    tag_program(
        BATCHED_MATMUL_TOP1_SCALED_OP_ID,
        primitive_i4x8_batched_matmul_top1_f32_scaled(
            weights_packed,
            activation_batches_packed,
            row_scales,
            batch_scales,
            out,
            batch,
            rows,
            cols,
        ),
    )
}

const EXPECTED_INT4_DOT_OUTPUT_BYTES: [u8; 4] = [0x28, 0x00, 0x00, 0x00];
const EXPECTED_INT4_DOT_SCALED_OUTPUT_BYTES: [u8; 4] = [0x00, 0x00, 0xA0, 0x40];
const EXPECTED_INT4_MATVEC_SCALED_OUTPUT_BYTES: [u8; 8] = [0u8; 8];
const EXPECTED_INT4_4F32_ZEROS_OUTPUT_BYTES: [u8; 16] = [0u8; 16];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        DOT_OP_ID,
        || int4_dot_i32("lhs", "rhs", "out", 8),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0xCDEF_4321]),
                vyre_primitives::wire::pack_u32_slice(&[0xFEDC_1234]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_DOT_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        DOT_SCALED_OP_ID,
        || int4_dot_f32_scaled("lhs", "rhs", "lhs_scale", "rhs_scale", "out", 8),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0xCDEF_4321]),
                vyre_primitives::wire::pack_u32_slice(&[0xFEDC_1234]),
                vyre_primitives::wire::pack_f32_slice(&[0.5]),
                vyre_primitives::wire::pack_f32_slice(&[0.25]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_DOT_SCALED_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        MATVEC_SCALED_OP_ID,
        || int4_matvec_f32_scaled("weights", "x", "scales", "out", 2, 8),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0xCDEF_4321, 0xFEDC_1234]),
                vyre_primitives::wire::pack_f32_slice(&[1.0; 8]),
                vyre_primitives::wire::pack_f32_slice(&[0.5, 0.25]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_MATVEC_SCALED_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        BATCHED_MATVEC_SCALED_OP_ID,
        || int4_batched_matvec_f32_scaled("weights", "x", "scales", "out", 2, 2, 8),
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0xCDEF_4321, 0xFEDC_1234]),
                vyre_primitives::wire::pack_f32_slice(&[1.0; 16]),
                vyre_primitives::wire::pack_f32_slice(&[0.5, 0.25]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_4F32_ZEROS_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        BATCHED_MATMUL_SCALED_OP_ID,
        || {
            int4_batched_matmul_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "out",
                2,
                2,
                8,
            )
        },
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
                vyre_primitives::wire::pack_f32_slice(&[0.5, 0.25]),
                vyre_primitives::wire::pack_f32_slice(&[0.125, 0.25]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_4F32_ZEROS_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        BATCHED_MATMUL_TOP1_SCALED_OP_ID,
        || {
            int4_batched_matmul_top1_f32_scaled(
                "weights",
                "activations",
                "row_scales",
                "batch_scales",
                "out",
                2,
                2,
                8,
            )
        },
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
                vyre_primitives::wire::pack_u32_slice(&[0, 0]),
                vyre_primitives::wire::pack_f32_slice(&[0.5, 0.25]),
                vyre_primitives::wire::pack_f32_slice(&[0.125, 0.25]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_INT4_4F32_ZEROS_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_bytes;
    use vyre_reference::composition_witness::{
        i4x8_batched_matmul_f32_scaled_witness, i4x8_batched_matmul_top1_f32_scaled_witness,
        i4x8_batched_matvec_f32_scaled_witness, i4x8_dot_f32_scaled_witness, i4x8_dot_i32_witness,
        i4x8_matvec_f32_scaled_witness, pack_i4x8_witness,
    };

    fn dot(lhs: &[i32], rhs: &[i32]) -> i32 {
        let lhs_packed = pack_i4x8_witness(lhs);
        let rhs_packed = pack_i4x8_witness(rhs);
        let program = int4_dot_i32("lhs", "rhs", "out", lhs.len() as u32);
        let raw = crate::fixture_bytes::eval_bytes(
            "int4_dot_i32",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&lhs_packed),
                vyre_primitives::wire::pack_u32_slice(&rhs_packed),
            ],
        )
        .swap_remove(0);
        i32::from_le_bytes(
            raw.get(0..4)
                .expect("Fix: packed INT4 dot must emit one i32.")
                .try_into()
                .expect("Fix: one i32 output must be exactly four bytes."),
        )
    }

    fn run_scaled(lhs: &[i32], rhs: &[i32], lhs_scale: f32, rhs_scale: f32) -> f32 {
        let lhs_packed = pack_i4x8_witness(lhs);
        let rhs_packed = pack_i4x8_witness(rhs);
        let program = int4_dot_f32_scaled(
            "lhs",
            "rhs",
            "lhs_scale",
            "rhs_scale",
            "out",
            lhs.len() as u32,
        );
        let outputs = eval_bytes(
            "int4",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&lhs_packed),
                vyre_primitives::wire::pack_u32_slice(&rhs_packed),
                vyre_primitives::wire::pack_f32_slice(&[lhs_scale]),
                vyre_primitives::wire::pack_f32_slice(&[rhs_scale]),
            ],
        );
        let raw = outputs[0].clone();
        f32::from_le_bytes(
            raw.get(0..4)
                .expect("Fix: scaled packed INT4 dot must emit one f32.")
                .try_into()
                .expect("Fix: one f32 output must be exactly four bytes."),
        )
    }

    fn pack_i4_matrix_rows(rows: &[Vec<i32>]) -> Vec<u32> {
        let cols = rows.first().map_or(0, Vec::len) as u32;
        let words_per_row = crate::math::quantized::i4_packed_words(cols) as usize;
        let mut out = Vec::with_capacity(rows.len() * words_per_row);
        for row in rows {
            let mut packed = pack_i4x8_witness(row);
            packed.resize(words_per_row, 0);
            out.extend_from_slice(&packed);
        }
        out
    }

    fn run_matvec(weights: &[Vec<i32>], x: &[f32], scales: &[f32]) -> Vec<f32> {
        let rows = weights.len() as u32;
        let cols = x.len() as u32;
        let packed = pack_i4_matrix_rows(weights);
        let program = int4_matvec_f32_scaled("weights", "x", "scales", "out", rows, cols);
        let outputs = eval_bytes(
            "int4",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&packed),
                vyre_primitives::wire::pack_f32_slice(x),
                vyre_primitives::wire::pack_f32_slice(scales),
            ],
        );
        vyre_primitives::wire::unpack_f32_slice(&outputs[0], rows as usize, "int4 matvec output")
            .expect("Fix: fused INT4 matvec output must decode as f32 rows.")
    }

    fn run_batched_matvec(
        weights: &[Vec<i32>],
        x_batches: &[f32],
        scales: &[f32],
        batch: u32,
    ) -> Vec<f32> {
        let rows = weights.len() as u32;
        let cols = weights.first().map_or(0, Vec::len) as u32;
        let packed = pack_i4_matrix_rows(weights);
        let program =
            int4_batched_matvec_f32_scaled("weights", "x", "scales", "out", batch, rows, cols);
        let outputs = eval_bytes(
            "int4",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&packed),
                vyre_primitives::wire::pack_f32_slice(x_batches),
                vyre_primitives::wire::pack_f32_slice(scales),
            ],
        );
        vyre_primitives::wire::unpack_f32_slice(
            &outputs[0],
            (batch * rows) as usize,
            "batched int4 matvec output",
        )
        .expect("Fix: batched fused INT4 matvec output must decode as f32 rows.")
    }

    fn run_batched_matmul(
        weights: &[Vec<i32>],
        activation_batches: &[Vec<i32>],
        row_scales: &[f32],
        batch_scales: &[f32],
    ) -> Vec<f32> {
        let batch = activation_batches.len() as u32;
        let rows = weights.len() as u32;
        let cols = weights.first().map_or(0, Vec::len) as u32;

        let weights_packed = pack_i4_matrix_rows(weights);
        let activations_packed = pack_i4_matrix_rows(activation_batches);
        let program = int4_batched_matmul_f32_scaled(
            "weights",
            "activations",
            "row_scales",
            "batch_scales",
            "out",
            batch,
            rows,
            cols,
        );
        let outputs = eval_bytes(
            "int4",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&weights_packed),
                vyre_primitives::wire::pack_u32_slice(&activations_packed),
                vyre_primitives::wire::pack_f32_slice(row_scales),
                vyre_primitives::wire::pack_f32_slice(batch_scales),
            ],
        );
        vyre_primitives::wire::unpack_f32_slice(
            &outputs[0],
            (batch * rows) as usize,
            "batched int4 matmul output",
        )
        .expect("Fix: packed-activation batched INT4 matmul output must decode as f32 rows.")
    }

    fn run_batched_matmul_top1(
        weights: &[Vec<i32>],
        activation_batches: &[Vec<i32>],
        row_scales: &[f32],
        batch_scales: &[f32],
    ) -> (Vec<f32>, Vec<u32>) {
        let batch = activation_batches.len() as u32;
        let rows = weights.len() as u32;
        let cols = weights.first().map_or(0, Vec::len) as u32;
        let weights_packed = pack_i4_matrix_rows(weights);
        let activations_packed = pack_i4_matrix_rows(activation_batches);
        let program = int4_batched_matmul_top1_f32_scaled(
            "weights",
            "activations",
            "row_scales",
            "batch_scales",
            "out",
            batch,
            rows,
            cols,
        );
        let outputs = eval_bytes(
            "int4",
            &program,
            vec![
                vyre_primitives::wire::pack_u32_slice(&weights_packed),
                vyre_primitives::wire::pack_u32_slice(&activations_packed),
                vyre_primitives::wire::pack_f32_slice(row_scales),
                vyre_primitives::wire::pack_f32_slice(batch_scales),
            ],
        );
        let packed = vyre_primitives::wire::unpack_f32_slice(
            &outputs[0],
            (batch * 2) as usize,
            "batched int4 top1 packed output",
        )
        .expect("Fix: packed-activation INT4 top1 output must decode as f32 rows.");
        let scores = packed[..batch as usize].to_vec();
        let indices = packed[batch as usize..]
            .iter()
            .map(|index| *index as u32)
            .collect::<Vec<_>>();
        (scores, indices)
    }

    #[test]
    fn packed_dot_matches_reference_oracle() {
        let lhs = [1, 2, 3, 4, -1, -2, -3, -4];
        let rhs = [4, 3, 2, 1, -4, -3, -2, -1];
        let lhs_packed = pack_i4x8_witness(&lhs);
        let rhs_packed = pack_i4x8_witness(&rhs);

        assert_eq!(dot(&lhs, &rhs), 40);
        assert_eq!(
            dot(&lhs, &rhs),
            i4x8_dot_i32_witness(&lhs_packed, &rhs_packed, 8)
        );
    }

    #[test]
    fn scaled_dot_matches_reference_oracle() {
        let lhs = [1, 2, 3, 4, -1, -2, -3, -4];
        let rhs = [4, 3, 2, 1, -4, -3, -2, -1];
        let lhs_scale = 0.5_f32;
        let rhs_scale = 0.25_f32;
        let lhs_packed = pack_i4x8_witness(&lhs);
        let rhs_packed = pack_i4x8_witness(&rhs);

        assert_eq!(
            run_scaled(&lhs, &rhs, lhs_scale, rhs_scale).to_bits(),
            i4x8_dot_f32_scaled_witness(&lhs_packed, &rhs_packed, lhs_scale, rhs_scale, 8)
                .to_bits()
        );
    }

    #[test]
    fn scaled_matvec_matches_reference_oracle() {
        let weights = vec![
            vec![1, 2, 3, 4, -1, -2, -3, -4, 5],
            vec![4, 3, 2, 1, -4, -3, -2, -1, -5],
            vec![-8, -7, -6, -5, -4, -3, -2, -1, 0],
        ];
        let x = [1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -2.0, 0.5];
        let scales = [0.5_f32, 0.25, 0.125];
        let packed = pack_i4_matrix_rows(&weights);

        let actual = run_matvec(&weights, &x, &scales);
        let expected = i4x8_matvec_f32_scaled_witness(
            &packed,
            &x,
            &scales,
            weights.len() as u32,
            x.len() as u32,
        );

        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batched_scaled_matvec_matches_reference_oracle() {
        let weights = vec![
            vec![1, 2, 3, 4, -1, -2, -3, -4, 5],
            vec![4, 3, 2, 1, -4, -3, -2, -1, -5],
            vec![-8, -7, -6, -5, -4, -3, -2, -1, 0],
        ];
        let x_batches = [
            1.0_f32, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -2.0, 0.5, -1.0, 0.5, -0.25, -2.0, 1.5,
            -0.75, -1.25, 2.0, -0.5,
        ];
        let scales = [0.5_f32, 0.25, 0.125];
        let packed = pack_i4_matrix_rows(&weights);

        let actual = run_batched_matvec(&weights, &x_batches, &scales, 2);
        let expected = i4x8_batched_matvec_f32_scaled_witness(
            &packed,
            &x_batches,
            &scales,
            2,
            weights.len() as u32,
            weights[0].len() as u32,
        );

        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batched_scaled_matmul_matches_reference_oracle() {
        let weights = vec![
            vec![1, 2, 3, 4, -1, -2, -3, -4, 5],
            vec![4, 3, 2, 1, -4, -3, -2, -1, -5],
            vec![-8, -7, -6, -5, -4, -3, -2, -1, 0],
        ];
        let activation_batches = vec![
            vec![1, -1, 2, -2, 3, -3, 4, -4, 5],
            vec![-5, 4, -4, 3, -3, 2, -2, 1, -1],
        ];
        let row_scales = [0.5_f32, 0.25, 0.125];
        let batch_scales = [0.75_f32, 0.375];
        let weights_packed = pack_i4_matrix_rows(&weights);
        let activations_packed = pack_i4_matrix_rows(&activation_batches);

        let actual = run_batched_matmul(&weights, &activation_batches, &row_scales, &batch_scales);
        let expected = i4x8_batched_matmul_f32_scaled_witness(
            &weights_packed,
            &activations_packed,
            &row_scales,
            &batch_scales,
            activation_batches.len() as u32,
            weights.len() as u32,
            weights[0].len() as u32,
        );

        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn batched_scaled_matmul_top1_matches_reference_oracle() {
        let weights = vec![
            vec![1, 2, 3, 4, -1, -2, -3, -4, 5],
            vec![4, 3, 2, 1, -4, -3, -2, -1, -5],
            vec![-8, -7, -6, -5, -4, -3, -2, -1, 0],
        ];
        let activation_batches = vec![
            vec![1, -1, 2, -2, 3, -3, 4, -4, 5],
            vec![-5, 4, -4, 3, -3, 2, -2, 1, -1],
        ];
        let row_scales = [0.5_f32, 0.25, 0.125];
        let batch_scales = [0.75_f32, 0.375];
        let weights_packed = pack_i4_matrix_rows(&weights);
        let activations_packed = pack_i4_matrix_rows(&activation_batches);

        let (actual_scores, actual_indices) =
            run_batched_matmul_top1(&weights, &activation_batches, &row_scales, &batch_scales);
        let (expected_scores, expected_indices) = i4x8_batched_matmul_top1_f32_scaled_witness(
            &weights_packed,
            &activations_packed,
            &row_scales,
            &batch_scales,
            activation_batches.len() as u32,
            weights.len() as u32,
            weights[0].len() as u32,
        );

        assert_eq!(actual_indices, expected_indices);
        assert_eq!(
            actual_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected_scores
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn batched_matmul_extension_id_matches_spec_name_contract() {
        assert_eq!(
            int4_batched_matmul_scaled_extension_id(),
            vyre_spec::extension::ExtensionTernaryOpId::from_name(
                INT4_BATCHED_MATMUL_SCALED_EXTENSION_NAME
            )
        );
    }

    #[test]
    fn batched_matmul_top1_extension_id_matches_spec_name_contract() {
        assert_eq!(
            int4_batched_matmul_top1_scaled_extension_id(),
            vyre_spec::extension::ExtensionTernaryOpId::from_name(
                INT4_BATCHED_MATMUL_TOP1_SCALED_EXTENSION_NAME
            )
        );
    }

    #[test]
    fn extension_id_matches_spec_name_contract() {
        assert_eq!(INT4_DOT_EXTENSION_NAME, "quant.int4.dot");
        assert_eq!(
            int4_dot_extension_id(),
            vyre_spec::extension::ExtensionBinOpId::from_name("quant.int4.dot")
        );
        assert!(int4_dot_extension_id().is_extension());
    }

    #[test]
    fn scaled_extension_id_matches_spec_name_contract() {
        assert_eq!(INT4_DOT_SCALED_EXTENSION_NAME, "quant.int4.dot.scaled");
        assert_eq!(
            int4_dot_scaled_extension_id(),
            vyre_spec::extension::ExtensionBinOpId::from_name("quant.int4.dot.scaled")
        );
        assert!(int4_dot_scaled_extension_id().is_extension());
        assert_ne!(int4_dot_extension_id(), int4_dot_scaled_extension_id());
    }

    #[test]
    fn matvec_extension_id_matches_spec_name_contract() {
        assert_eq!(
            INT4_MATVEC_SCALED_EXTENSION_NAME,
            "quant.int4.matvec.scaled"
        );
        assert_eq!(
            int4_matvec_scaled_extension_id(),
            vyre_spec::extension::ExtensionTernaryOpId::from_name("quant.int4.matvec.scaled")
        );
        assert!(int4_matvec_scaled_extension_id().is_extension());
    }

    #[test]
    fn batched_matvec_extension_id_matches_spec_name_contract() {
        assert_eq!(
            INT4_BATCHED_MATVEC_SCALED_EXTENSION_NAME,
            "quant.int4.batched_matvec.scaled"
        );
        assert_eq!(
            int4_batched_matvec_scaled_extension_id(),
            vyre_spec::extension::ExtensionTernaryOpId::from_name(
                "quant.int4.batched_matvec.scaled"
            )
        );
        assert!(int4_batched_matvec_scaled_extension_id().is_extension());
        assert_ne!(
            int4_batched_matvec_scaled_extension_id(),
            int4_matvec_scaled_extension_id()
        );
    }
}
