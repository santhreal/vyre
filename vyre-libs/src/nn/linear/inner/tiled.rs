//! Tiled linear-layer constructors (`linear_tiled`, `linear_tiled_reference`).

use vyre_foundation::composition::{tag_program, trap_program, wrap_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::math::linalg::MatmulBiasTiled;
use crate::tensor_ref::TensorRef;

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
    let program = MatmulBiasTiled::new(
        TensorRef::u32_2d(x, 1, in_dim),
        TensorRef::u32_2d(w, in_dim, out_dim),
        TensorRef::u32_1d(b, out_dim),
        TensorRef::u32_2d(out, 1, out_dim),
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
    let weight_count = in_dim.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_tiled_reference in_dim*out_dim overflows u32; reduce dimensions.".to_string()
    })?;
    let tile_count = in_dim.div_ceil(tile);
    let lane = Expr::var("lane");
    let kk = Expr::var("kk");
    let body = vec![
        Node::let_bind("lane", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(lane.clone(), Expr::u32(out_dim)),
            vec![
                Node::let_bind("acc", Expr::load(b, lane.clone())),
                Node::loop_for(
                    "tile_idx",
                    Expr::u32(0),
                    Expr::u32(tile_count),
                    vec![
                        Node::let_bind(
                            "tile_base",
                            Expr::mul(Expr::var("tile_idx"), Expr::u32(tile)),
                        ),
                        Node::loop_for(
                            "tile_k",
                            Expr::u32(0),
                            Expr::u32(tile),
                            vec![
                                Node::let_bind(
                                    "kk",
                                    Expr::add(Expr::var("tile_base"), Expr::var("tile_k")),
                                ),
                                Node::if_then(
                                    Expr::lt(kk.clone(), Expr::u32(in_dim)),
                                    vec![Node::assign(
                                        "acc",
                                        Expr::add(
                                            Expr::var("acc"),
                                            Expr::mul(
                                                Expr::load(x, kk.clone()),
                                                Expr::load(
                                                    w,
                                                    Expr::add(
                                                        Expr::mul(kk.clone(), Expr::u32(out_dim)),
                                                        lane.clone(),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    )],
                                ),
                            ],
                        ),
                    ],
                ),
                Node::Store {
                    buffer: out.into(),
                    index: lane,
                    value: Expr::var("acc"),
                },
            ],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(x, 0, BufferAccess::ReadOnly, DataType::U32).with_count(in_dim),
            BufferDecl::storage(w, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(weight_count),
            BufferDecl::storage(b, 2, BufferAccess::ReadOnly, DataType::U32).with_count(out_dim),
            BufferDecl::output(out, 3, DataType::U32).with_count(out_dim),
        ],
        [256, 1, 1],
        vec![wrap_region(LINEAR_TILED_REFERENCE_OP_ID, body, None)],
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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

            vec![vec![crate::fixture_bytes::u32_bytes(&[56, 62, 68, 74])]]
        }),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
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

            vec![vec![crate::fixture_bytes::u32_bytes(&[56, 62, 68, 74])]]
        }),
    )
    .with_category("nn")
}
