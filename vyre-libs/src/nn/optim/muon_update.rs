//! Muon update: momentum + Newton-Schulz orthogonalization (F32).
//!
//! `buf = momentum * buf + grad`
//! `nesterov = grad + momentum * buf`
//! `orthogonal = newton_schulz_5step(nesterov)` (via composition)
//! `param -= lr * orthogonal * scale`

use vyre_foundation::ir::Program;

use crate::nn::optim::muon_step::muon_step_program;

const OP_ID: &str = "vyre-libs::optim::muon_update";

/// Muon optimizer step (F32).
///
/// `params[n]` (RO), `grads[n]` (RO), `momentum_buf[n]` (RW),
/// `output[n]`  -  updated params.
#[must_use]
pub fn muon_update(
    params: &str,
    grads: &str,
    momentum_buf: &str,
    output: &str,
    n: u32,
    lr: f32,
    momentum: f32,
) -> Program {
    muon_step_program(OP_ID, params, grads, momentum_buf, output, n, lr, momentum)
}

const EXPECTED_MUON_UPDATE_MOMENTUM_BYTES: [u8; 8] =
    [0xCD, 0xCC, 0xCC, 0x3D, 0xCD, 0xCC, 0x4C, 0x3E];
const EXPECTED_MUON_UPDATE_OUTPUT_BYTES: [u8; 8] = [0x1E, 0x8A, 0x7E, 0x3F, 0x1E, 0x8A, 0xFE, 0x3F];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || muon_update("params", "grads", "momentum", "output", 2, 0.02, 0.95),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0]),    // params
                to_f32(&[0.1, 0.2]),    // grads
                to_f32(&[0.0, 0.0]),    // momentum (first step)
            ]]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_MUON_UPDATE_MOMENTUM_BYTES.to_vec(),
                EXPECTED_MUON_UPDATE_OUTPUT_BYTES.to_vec(),
            ]]
        }),
    )
    .with_category("nn")
}
