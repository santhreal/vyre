//! Tiled linear-layer constructors (`linear_tiled`, `linear_tiled_reference`).

use vyre_foundation::composition::{tag_program, trap_program};
use vyre_foundation::ir::Program;

use crate::builder::gemm::{ContractionComposer, ContractionTiling};
use crate::plumbing::operand::tensor_ref::TensorRef;

use super::builder::linear;

pub(super) const LINEAR_TILED_OP_ID: &str = "vyre-libs::nn::linear_tiled";
pub(super) const LINEAR_TILED_REFERENCE_OP_ID: &str = "vyre-libs::nn::linear_tiled_reference";
pub(super) const LINEAR_TILED_TILE: u32 = 32;
pub(super) const LINEAR_TILED_MIN_WORK: u32 = 1024;

/// Build a tiled linear-layer Program: `out[j] = b[j] + sum_k x[k] * w[k, j]`.
///
/// # Errors
/// Returns `Err` when dimensions are empty, overflow buffer counts, or `tile == 0`.
pub fn linear_tiled(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    tile: u32,
) -> Result<Program, String> {
    if in_dim == 0 {
        return Err("Fix: linear_tiled in_dim=0 is invalid: empty reduction".to_string());
    }
    if out_dim == 0 {
        return Err("Fix: linear_tiled out_dim=0 is invalid: empty output".to_string());
    }
    if tile == 0 {
        return Err("Fix: linear_tiled tile=0 is invalid: tile width must be > 0".to_string());
    }
    in_dim.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_tiled in_dim*out_dim overflows u32; reduce dimensions.".to_string()
    })?;
    let x_ref = TensorRef::u32_2d(x, 1, in_dim);
    let w_ref = TensorRef::u32_2d(w, in_dim, out_dim);
    let b_ref = TensorRef::u32_1d(b, out_dim);
    let out_ref = TensorRef::u32_2d(out, 1, out_dim);
    let program = ContractionComposer::tiled_2d(
        LINEAR_TILED_OP_ID,
        x_ref,
        w_ref,
        Some(b_ref),
        out_ref,
        1,
        in_dim,
        out_dim,
        tile,
    )
    .build()
    .map_err(|error| format!("Fix: linear_tiled matmul_tiled build failed: {error}"))?;
    Ok(tag_program(LINEAR_TILED_OP_ID, program))
}

/// Reference / oracle implementation of tiled linear (hand-rolled IR).
/// Kept for parity testing against the optimized `linear_tiled` path.
#[allow(clippy::too_many_arguments)]
pub fn linear_tiled_reference(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    tile: u32,
) -> Result<Program, String> {
    if in_dim == 0 {
        return Err("Fix: linear_tiled_reference in_dim=0 is invalid: empty reduction".to_string());
    }
    if out_dim == 0 {
        return Err("Fix: linear_tiled_reference out_dim=0 is invalid: empty output".to_string());
    }
    if tile == 0 {
        return Err(
            "Fix: linear_tiled_reference tile=0 is invalid: tile width must be > 0".to_string(),
        );
    }
    in_dim.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_tiled_reference in_dim*out_dim overflows u32; reduce dimensions.".to_string()
    })?;
    let x_ref = TensorRef::u32_2d(x, 1, in_dim);
    let w_ref = TensorRef::u32_2d(w, in_dim, out_dim);
    let b_ref = TensorRef::u32_1d(b, out_dim);
    let out_ref = TensorRef::u32_2d(out, 1, out_dim);
    let mut composer = ContractionComposer::matmul_bias_2d(
        LINEAR_TILED_REFERENCE_OP_ID,
        x_ref,
        w_ref,
        b_ref,
        out_ref,
        1,
        in_dim,
        out_dim,
    );
    composer.tiling = ContractionTiling::Block1D { tile };
    composer
        .build()
        .map_err(|error| format!("Fix: linear_tiled_reference build failed: {error}"))
}

const EXPECTED_LINEAR_OUTPUT_BYTES: [u8; 16] = [
    0x38, 0x00, 0x00, 0x00, 0x3E, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x4A, 0x00, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::nn::linear",
        || {
            linear("x", "w", "b", "out", 4, 4)
                .unwrap_or_else(|error| trap_program("vyre-libs::nn::linear", None, format!("Fix: linear fixture dimensions are invalid: {error}")))
        },
        // V7-TEST-005: deterministic fixture for linear(4, 4).
        // Body indexes `w[k * out_dim + i]` (column-major per out_dim),
        // so for w = [0..16], out_dim = 4:
        //   out[i] = b[i] + sum_k x[k] * w[k*4 + i]
        // With x = [0, 1, 2, 3] and b = [0, 0, 0, 0]:
        //   out[0] = 0*0 + 1*4 + 2*8  + 3*12 =  4 + 16 + 36 = 56
        //   out[1] = 0*1 + 1*5 + 2*9  + 3*13 =  5 + 18 + 39 = 62
        //   out[2] = 0*2 + 1*6 + 2*10 + 3*14 =  6 + 20 + 42 = 68
        //   out[3] = 0*3 + 1*7 + 2*11 + 3*15 =  7 + 22 + 45 = 74
        Some(|| {

            let x = crate::fixture_bytes::u32_bytes(&(0..4).collect::<Vec<_>>());
            let w = crate::fixture_bytes::u32_bytes(&(0..16).collect::<Vec<_>>());
            let bias = crate::fixture_bytes::u32_bytes(&[0, 0, 0, 0]);
            // The output buffer is declared with `with_count(out_dim) = 4`
            // u32s = 16 bytes. The CPU reference and the GPU dispatch both
            // honor that buffer length; an over-allocated input slot would
            // make CPU echo a longer Value than the GPU returns and trip
            // the CPU/GPU length divergence assertion in cat_a_gpu_differential.
            vec![vec![x, w, bias]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LINEAR_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::nn::linear_tiled",
        || {
            linear_tiled("x", "w", "b", "out", 4, 4, 2)
                .unwrap_or_else(|error| trap_program("vyre-libs::nn::linear_tiled", None, format!("Fix: linear_tiled fixture dimensions are invalid: {error}")))
        },
        Some(|| {

            let x = crate::fixture_bytes::u32_bytes(&(0..4).collect::<Vec<_>>());
            let w = crate::fixture_bytes::u32_bytes(&(0..16).collect::<Vec<_>>());
            let bias = crate::fixture_bytes::u32_bytes(&[0, 0, 0, 0]);
            vec![vec![x, w, bias]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LINEAR_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
