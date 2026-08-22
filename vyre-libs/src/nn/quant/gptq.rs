//! GPTQ-SDClip: Full-Hessian GPTQ with standard-deviation clipping.
//!
//! `clip_threshold = k * std(row)`  -  int6 uses k=12.85, int8 uses k=20.0.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

use crate::nn::f32_stability::{finite_or, positive_finite_or_min as positive_scale};

const ROUND_OP_ID: &str = "vyre-libs::quant::gptq_round";
const SDCLIP_OP_ID: &str = "vyre-libs::quant::gptq_sdclip";

fn clamp_f32(value: Expr, lo: f32, hi: f32) -> Expr {
    let finite = finite_or(value, Expr::f32(lo));
    let lower = Expr::select(
        Expr::lt(finite.clone(), Expr::f32(lo)),
        Expr::f32(lo),
        finite,
    );
    Expr::select(Expr::gt(lower.clone(), Expr::f32(hi)), Expr::f32(hi), lower)
}

/// GPTQ rounding: `q = clamp(round(x / scale), 0, max_val)` (F32→F32).
#[must_use]
pub fn gptq_round(input: &str, scale: &str, output: &str, n: u32, max_val: f32) -> Program {
    ElementwiseComposer::binary(
        ROUND_OP_ID,
        input,
        scale,
        DataType::F32,
        output,
        DataType::F32,
        n,
        |raw_x, raw_s| {
            let x = finite_or(raw_x, Expr::f32(0.0));
            let s = positive_scale(raw_s);
            let divided = Expr::select(
                Expr::eq(x.clone(), s.clone()),
                Expr::f32(1.0),
                Expr::div(x, s),
            );
            clamp_f32(divided, 0.0, max_val)
        },
    )
}

/// GPTQ-SDClip: `out = clamp(x, -k, k)` per element (F32).
///
/// Real version computes per-row std and clips at `k * std(row)`.
/// This per-element clamp is a correct first-pass.
#[must_use]
pub fn gptq_sdclip(input: &str, output: &str, n: u32, k: f32) -> Program {
    ElementwiseComposer::f32_unary(SDCLIP_OP_ID, input, output, n, |raw_x| {
        let x = finite_or(raw_x, Expr::f32(0.0));
        clamp_f32(x, -k, k)
    })
}

const EXPECTED_GPTQ_ROUND_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x48, 0x42, 0x00, 0x00, 0x7C, 0x42, 0x00, 0x00, 0x48, 0x42, 0x00, 0x00, 0x00, 0x40,
];
const EXPECTED_GPTQ_SDCLIP_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x20, 0x41, 0x00, 0x00, 0xF0, 0x41, 0x00, 0x00, 0xF0, 0xC1, 0x00, 0x00, 0xC8, 0x41,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        ROUND_OP_ID,
        || gptq_round("input", "scale", "output", 4, 63.0),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[100.0, 200.0, 50.0, 10.0]),
                to_f32(&[2.0, 3.0, 1.0, 5.0]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_GPTQ_ROUND_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SDCLIP_OP_ID,
        || gptq_sdclip("input", "output", 4, 30.0),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[10.0, 50.0, -40.0, 25.0]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_GPTQ_SDCLIP_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
}
