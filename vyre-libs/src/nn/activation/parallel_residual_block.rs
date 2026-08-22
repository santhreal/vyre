//! Parallel residual block: `out = x + attn_out + mlp_out`.
//!
//! Category A composition  -  residual stream addition.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use crate::builder::build_indexed_map;

const OP_ID: &str = "vyre-libs::nn::parallel_residual_block";

/// Build parallel residual block (F32).
///
/// # Errors
/// Returns `Err` if n is zero.
pub fn parallel_residual_block(
    x: &str,
    attn_out: &str,
    mlp_out: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Err("Fix: n=0".into());
    }
    let buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::storage(attn_out, 1, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::storage(mlp_out, 2, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::output(output, 3, DataType::F32).with_count(n),
    ];
    Ok(build_indexed_map(
        OP_ID,
        buffers,
        output,
        n,
        [64, 1, 1],
        |i| {
            (
                i.clone(),
                Expr::add(
                    Expr::add(Expr::load(x, i.clone()), Expr::load(attn_out, i.clone())),
                    Expr::load(mlp_out, i),
                ),
            )
        },
    ))
}

const EXPECTED_PARALLEL_RESIDUAL_BLOCK_OUTPUT_BYTES: [u8; 16] = [
    0x7B, 0x14, 0x8E, 0x3F, 0x7B, 0x14, 0x0E, 0x40, 0xB8, 0x1E, 0x55, 0x40, 0x7B, 0x14, 0x8E, 0x40,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            parallel_residual_block("x", "attn", "mlp", "out", 4)
                .unwrap_or_else(|error| trap_program(OP_ID, None, format!("Fix: parallel_residual_block fixture must build: {error}")))
        },
        Some(|| {
            let f = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                f(&[1.0, 2.0, 3.0, 4.0]), f(&[0.1, 0.2, 0.3, 0.4]),
                f(&[0.01, 0.02, 0.03, 0.04]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_PARALLEL_RESIDUAL_BLOCK_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
