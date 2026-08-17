//! MuonEq-R: Row-normalized Muon optimizer (F32).
//!
//! Muon + `scale = max(1, rows/cols)^0.5` row normalization.

use vyre_foundation::ir::Program;

use crate::nn::optim::muon_step::muon_step_program;

const OP_ID: &str = "vyre-libs::optim::muoneq_r";

/// MuonEq-R step (F32).
///
/// Same as `muon_update` but with row-norm scaling baked in.
#[must_use]
pub fn muoneq_r(
    params: &str,
    grads: &str,
    momentum_buf: &str,
    output: &str,
    n: u32,
    rows: u32,
    cols: u32,
    lr: f32,
    momentum: f32,
) -> Program {
    // scale = max(1, rows/cols)^0.5
    let ratio = (rows as f32) / (cols as f32);
    let scale = ratio.max(1.0).sqrt();
    muon_step_program(
        OP_ID,
        params,
        grads,
        momentum_buf,
        output,
        n,
        scale * lr,
        momentum,
    )
}

const EXPECTED_MUONEQ_R_MOMENTUM_BYTES: [u8; 16] = [
    0xCD, 0xCC, 0xCC, 0x3D, 0xCD, 0xCC, 0x4C, 0x3E, 0x9A, 0x99, 0x99, 0x3E, 0xCD, 0xCC, 0xCC, 0x3E,
];
const EXPECTED_MUONEQ_R_OUTPUT_BYTES: [u8; 16] = [
    0x40, 0xEF, 0x7D, 0x3F, 0x40, 0xEF, 0xFD, 0x3F, 0x70, 0x73, 0x3E, 0x40, 0x40, 0xEF, 0x7D, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || muoneq_r("params", "grads", "momentum", "output", 4, 4, 2, 0.02, 0.95),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0, 3.0, 4.0]),
                to_f32(&[0.1, 0.2, 0.3, 0.4]),
                to_f32(&[0.0, 0.0, 0.0, 0.0]),
            ]]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_MUONEQ_R_MOMENTUM_BYTES.to_vec(),
                EXPECTED_MUONEQ_R_OUTPUT_BYTES.to_vec(),
            ]]
        }),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(8))
}
