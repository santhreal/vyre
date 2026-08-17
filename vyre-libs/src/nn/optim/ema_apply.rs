//! EMA update: `ema = decay * ema + (1 - decay) * theta`.
//!
//! Category A  -  element-wise weighted average. Recipe decay=0.9965.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use crate::builder::build_indexed_map;
use crate::nn::f32_stability::flush_tiny;

const OP_ID: &str = "vyre-libs::optim::ema_apply";

/// Build a Program for EMA update in-place (F32).
///
/// `ema[n]` (RW)  -  running average.
/// `theta[n]` (RO)  -  current weights.
/// `decay`  -  scalar, baked as constant.
#[must_use]
pub fn ema_apply(ema: &str, theta: &str, n: u32, decay: f32) -> Program {
    let buffers = vec![
        BufferDecl::storage(ema, 0, BufferAccess::ReadWrite, DataType::F32).with_count(n),
        BufferDecl::storage(theta, 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
    ];

    build_indexed_map(OP_ID, buffers, ema, n, [64, 1, 1], |i| {
        // ema = decay * ema + (1 - decay) * theta
        let updated = Expr::add(
            Expr::mul(Expr::f32(decay), Expr::load(ema, i.clone())),
            Expr::mul(Expr::f32(1.0 - decay), Expr::load(theta, i.clone())),
        );
        (i, flush_tiny(updated))
    })
}

const EXPECTED_EMA_APPLY_OUTPUT_BYTES: [u8; 16] = [
    0x9A, 0x99, 0x21, 0x41, 0xCD, 0xCC, 0xA0, 0x41, 0xCD, 0xCC, 0xF0, 0x41, 0x66, 0x66, 0x20, 0x42,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || ema_apply("ema", "theta", 4, 0.9),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[10.0, 20.0, 30.0, 40.0]),  // ema
                to_f32(&[11.0, 21.0, 31.0, 41.0]),  // theta
            ]]
        }),
        Some(|| vec![vec![EXPECTED_EMA_APPLY_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(1))
}
