//! Fused `rms_norm_linear` constructor.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::builder::{build_linear_bias_store_tail, build_linear_loop_accumulation};
use crate::nn::rms::{inverse_rms_expr, square_expr};
use crate::plumbing::operand::tensor_ref::TensorRefError;

/// Fused RMSNorm + linear: `out = (input / rms(input)) @ W + b`.
///
/// # Errors
/// Returns `Err` when dimensions are zero or buffer counts overflow `u32`.
pub fn rms_norm_linear(
    input: &str,
    w: &str,
    b: &str,
    out: &str,
    n: u32,
    in_dim: u32,
    out_dim: u32,
    eps: f32,
) -> Program {
    try_rms_norm_linear(input, w, b, out, n, in_dim, out_dim, eps).unwrap_or_else(|error| {
        trap_program(
            "vyre-libs::nn::rms_norm_linear",
            Some((out, DataType::F32)),
            format!("Fix: rms_norm_linear build failed: {error}"),
        )
    })
}

/// Fallible fused RMSNorm + linear constructor.
///
/// # Errors
///
/// Returns [`TensorRefError`] when dimensions are incoherent or counts
/// overflow `u32`.
pub fn try_rms_norm_linear(
    input: &str,
    w: &str,
    b: &str,
    out: &str,
    n: u32,
    in_dim: u32,
    out_dim: u32,
    eps: f32,
) -> Result<Program, TensorRefError> {
    if n == 0 || in_dim == 0 || out_dim == 0 || n > in_dim {
        return Err(TensorRefError::ShapeMismatch {
            name: input.to_string(),
            found: vec![n, in_dim, out_dim],
            expected: vec![1, in_dim.max(1), out_dim.max(1)],
            op: "vyre-libs::nn::rms_norm_linear",
        });
    }
    let weight_count =
        in_dim
            .checked_mul(out_dim)
            .ok_or_else(|| TensorRefError::ElementCountOverflow {
                name: w.to_string(),
                shape: vec![in_dim, out_dim],
            })?;

    let global_lane = Expr::var("global_lane");
    let local_lane = Expr::var("local_lane");
    let k = Expr::var("k");

    let mean_sq = vec![
        Node::let_bind("sum_sq", Expr::f32(0.0)),
        Node::loop_for(
            "k",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::assign(
                "sum_sq",
                Expr::add(
                    Expr::var("sum_sq"),
                    square_expr(Expr::load(input, k.clone())),
                ),
            )],
        ),
        Node::Store {
            buffer: "inv_rms".into(),
            index: Expr::u32(0),
            value: inverse_rms_expr(Expr::var("sum_sq"), n, eps),
        },
    ];

    let dot_term = Expr::mul(
        Expr::mul(Expr::load(input, k.clone()), Expr::var("scale")),
        Expr::load(
            w,
            Expr::add(
                Expr::mul(k.clone(), Expr::u32(out_dim)),
                global_lane.clone(),
            ),
        ),
    );
    let output_lane = vec![
        Node::let_bind("acc", Expr::f32(0.0)),
        Node::let_bind("scale", Expr::load("inv_rms", Expr::u32(0))),
        build_linear_loop_accumulation("k", in_dim, "acc", dot_term),
        build_linear_bias_store_tail(out, global_lane.clone(), "acc", b, global_lane.clone()),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(in_dim),
            BufferDecl::storage(w, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(weight_count),
            BufferDecl::storage(b, 2, BufferAccess::ReadOnly, DataType::F32).with_count(out_dim),
            BufferDecl::workgroup("inv_rms", 1, DataType::F32),
            BufferDecl::output(out, 4, DataType::F32).with_count(out_dim),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::nn::rms_norm_linear",
            vec![
                Node::let_bind("local_lane", Expr::LocalId { axis: 0 }),
                Node::let_bind("global_lane", Expr::InvocationId { axis: 0 }),
                Node::if_then(Expr::eq(local_lane.clone(), Expr::u32(0)), mean_sq),
                Node::barrier(),
                Node::if_then(
                    Expr::lt(global_lane.clone(), Expr::u32(out_dim)),
                    output_lane,
                ),
            ],
        )],
    )
    .with_entry_op_id("vyre-libs::nn::rms_norm_linear"))
}

const EXPECTED_RMS_NORM_LINEAR_OUTPUT_BYTES: [u8; 16] = [
    0xDF, 0xB1, 0xE9, 0x41, 0x0E, 0x74, 0x03, 0x42, 0x2C, 0x0F, 0x12, 0x42, 0x4A, 0xAA, 0x20, 0x42,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        "vyre-libs::nn::rms_norm_linear",
        || rms_norm_linear("input", "w", "b", "out", 4, 4, 4, 1e-5),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_f32_slice;
            let input = [1.0_f32, 2.0, 3.0, 4.0];
            let weights = (0u32..16u32).map(|v| v as f32).collect::<Vec<_>>();
            vec![vec![
                to_bytes(&input),
                to_bytes(&weights),
                vec![0u8; 4 * 4],
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_RMS_NORM_LINEAR_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
    .with_tolerance(vyre_foundation::operation::TolerancePolicy::f32_ulp(2))
}
