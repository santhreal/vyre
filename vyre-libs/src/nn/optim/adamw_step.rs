//! AdamW optimizer step (F32).
//!
//! `m = β₁*m + (1-β₁)*g`
//! `v = β₂*v + (1-β₂)*g²`
//! `θ = θ * (1 - lr*wd) - lr * m̂ / (√v̂ + ε)`

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program, UnOp};

use crate::nn::f32_stability::flush_tiny;

const OP_ID: &str = "vyre-libs::optim::adamw_step";

/// Build a single AdamW step (F32).
///
/// `params[n]` (RW), `grads[n]` (RO), `m[n]` (RW), `v[n]` (RW).
/// Hyperparams baked as constants.
#[must_use]
pub fn adamw_step(
    params: &str,
    grads: &str,
    m_buf: &str,
    v_buf: &str,
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
) -> Program {
    ElementwiseComposer::new(OP_ID, n)
        .add_input_storage(params, BufferAccess::ReadWrite, DataType::F32, n)
        .add_input_storage(grads, BufferAccess::ReadOnly, DataType::F32, n)
        .add_input_storage(m_buf, BufferAccess::ReadWrite, DataType::F32, n)
        .add_input_storage(v_buf, BufferAccess::ReadWrite, DataType::F32, n)
        .build_pointwise_multi(&[m_buf, v_buf, params], |i| {
            let g = flush_tiny(Expr::load(grads, i.clone()));
            let m = flush_tiny(Expr::load(m_buf, i.clone()));
            let v = flush_tiny(Expr::load(v_buf, i.clone()));
            let p = flush_tiny(Expr::load(params, i));

            // m = β₁*m + (1-β₁)*g
            let new_m = Expr::add(
                Expr::mul(Expr::f32(beta1), m),
                Expr::mul(Expr::f32(1.0 - beta1), g.clone()),
            );

            // v = β₂*v + (1-β₂)*g²
            let new_v = Expr::add(
                Expr::mul(Expr::f32(beta2), v),
                Expr::mul(Expr::f32(1.0 - beta2), Expr::mul(g.clone(), g)),
            );

            // weight decay + adam update: θ = θ*(1-lr*wd) - lr * m / (√v + ε)
            // (bias correction omitted  -  recipe uses β-schedule instead)
            let decayed = Expr::mul(p, Expr::f32(1.0 - lr * wd));
            let denom = Expr::add(
                Expr::UnOp {
                    op: UnOp::Sqrt,
                    operand: Box::new(new_v.clone()),
                },
                Expr::f32(eps),
            );
            let new_p = Expr::sub(
                decayed,
                Expr::mul(Expr::f32(lr), Expr::div(new_m.clone(), denom)),
            );

            vec![flush_tiny(new_m), flush_tiny(new_v), flush_tiny(new_p)]
        })
}

const EXPECTED_ADAMW_STEP_M_BYTES: [u8; 8] = [0xD7, 0xE8, 0x7E, 0x3F, 0x18, 0x74, 0xFF, 0x3F];
const EXPECTED_ADAMW_STEP_V_BYTES: [u8; 8] = [0x0D, 0xD7, 0x23, 0x3C, 0x0D, 0xD7, 0xA3, 0x3C];
const EXPECTED_ADAMW_STEP_PARAMS_BYTES: [u8; 8] = [0x1F, 0xC5, 0x27, 0x37, 0x1F, 0xC5, 0x27, 0x38];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || adamw_step("params", "grads", "m", "v", 2, 0.001, 0.9, 0.999, 1e-8, 0.01),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[1.0, 2.0]),          // params
                to_f32(&[0.1, 0.2]),          // grads
                to_f32(&[0.0, 0.0]),          // m (first step)
                to_f32(&[0.0, 0.0]),          // v (first step)
            ]]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_ADAMW_STEP_M_BYTES.to_vec(),
                EXPECTED_ADAMW_STEP_V_BYTES.to_vec(),
                EXPECTED_ADAMW_STEP_PARAMS_BYTES.to_vec(),
            ]]
        }),
    )
    .with_category("nn")
}
