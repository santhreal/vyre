//! 2D convolution with a 3x3 kernel and unit stride, expressed as an
//! im2col patch row contracted against the kernel:
//!
//! `out[y, x] = sum_{k=0..9} im2col[y*W + x, k] * kernel[k]`
//!
//! which is the canonical
//! `sum_{ky, kx} input[y+ky-1, x+kx-1] * kernel[ky, kx]`.
//!
//! The patch row comes from the stencil owner (`crate::builder::stencil::stencil_3x3_taps`),
//! so the 3x3 neighbour walk lives in exactly one place. The gemm half is the
//! `k` contraction in the body, unrolled at `k = 9` and fused into the same
//! invocation: a lane consumes only the patch row it produced, so no patch
//! matrix is materialized and the registered buffer signature stays
//! `input` / `kernel` / `output`.
//!
//! Boundary handling: zero-padding (samples outside the input
//! bounds are treated as 0). Input + output are length-`H * W` F32
//! buffers in row-major layout; kernel is length-9 F32 in
//! row-major layout (`kernel[ky*3 + kx]`).

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::builder::stencil::stencil_3x3_taps;

const OP_ID: &str = "vyre-libs::math::conv::conv2d_3x3_direct";

/// Build a Program that computes 2D convolution with a 3x3 kernel,
/// unit stride, and zero-padding.
///
/// # Errors
///
/// Returns `Err` when `h * w` overflows `u32`.
pub fn conv2d_3x3_direct(
    input: &str,
    kernel: &str,
    output: &str,
    h: u32,
    w: u32,
) -> Result<Program, String> {
    if h == 0 || w == 0 {
        return Err("Fix: conv2d_3x3_direct requires non-zero height and width.".to_string());
    }
    let elements = h.checked_mul(w).ok_or_else(|| {
        "Fix: conv2d_3x3_direct h*w overflows u32; reduce dimensions.".to_string()
    })?;
    // Per-output body: one invocation per output pixel.
    let (y, x) = crate::builder::stencil::decompose_index(&Expr::var("flat"), w);
    let body = vec![
        Node::let_bind("flat", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("flat"), Expr::u32(elements)),
            vec![
                Node::let_bind("y", y),
                Node::let_bind("x", x),
                Node::let_bind("acc", Expr::f32(0.0)),
                // gemm over the im2col patch row: patch column k times
                // kernel[k], accumulated across the nine taps. A column
                // outside the image is zero-padding, so it contributes
                // exactly 0.0 instead of a product with the kernel tap.
                Node::Block(
                    stencil_3x3_taps(&Expr::var("y"), &Expr::var("x"), h, w)
                        .into_iter()
                        .map(|tap| {
                            Node::assign(
                                "acc",
                                Expr::add(
                                    Expr::var("acc"),
                                    Expr::select(
                                        tap.in_bounds.clone(),
                                        Expr::mul(
                                            Expr::load(input, tap.bounded_input_index()),
                                            Expr::load(kernel, Expr::u32(tap.column)),
                                        ),
                                        Expr::f32(0.0),
                                    ),
                                ),
                            )
                        })
                        .collect(),
                ),
                Node::store(output, Expr::var("flat"), Expr::var("acc")),
            ],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(elements),
            BufferDecl::storage(kernel, 1, BufferAccess::ReadOnly, DataType::F32).with_count(9),
            BufferDecl::output(output, 2, DataType::F32).with_count(elements),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || {
            conv2d_3x3_direct("input", "kernel", "output", 4, 4)
                .unwrap_or_else(|error| super::trap_f32_output_program(OP_ID, "output", error))
        },
        Some(|| {
            // 4x4 input = identity matrix; 3x3 box kernel
            let input = crate::fixture_bytes::f32_bytes(&[
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]);
            let kernel = crate::fixture_bytes::f32_bytes(&[1.0; 9]);
            vec![vec![input, kernel]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x40, 0x40, // 3.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x40, 0x40, // 3.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x40, // 2.0
                0x00, 0x00, 0x00, 0x40, // 2.0
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::f32_bytes;

    fn decode(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn naive_conv2d_3x3(input: &[f32], kernel: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0_f32;
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let ny = (y as i32) + (ky as i32) - 1;
                        let nx = (x as i32) + (kx as i32) - 1;
                        if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                            continue;
                        }
                        let pixel = input[(ny as usize) * w + (nx as usize)];
                        let k = kernel[ky * 3 + kx];
                        acc += pixel * k;
                    }
                }
                out[y * w + x] = acc;
            }
        }
        out
    }

    fn convolved(h: u32, w: u32, input: &[f32], kernel: &[f32]) -> Vec<f32> {
        let program = conv2d_3x3_direct("input", "kernel", "output", h, w).expect("Fix: build");
        decode(
            &crate::fixture_bytes::eval_bytes(
                "conv2d_3x3_direct",
                &program,
                vec![f32_bytes(input), f32_bytes(kernel)],
            )[0],
        )
    }

    /// Direct 3x3 conv on a 4x4 identity matrix with box kernel
    /// matches the naive Rust reference.
    #[test]
    fn conv2d_identity_box_matches_naive() {
        let input = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let kernel = vec![1.0; 9];
        let actual = convolved(4, 4, &input, &kernel);
        let expected = naive_conv2d_3x3(&input, &kernel, 4, 4);
        assert_eq!(actual.len(), 16);
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1.0e-5, "{a} != {e}");
        }
    }

    /// Identity kernel `[[0,0,0],[0,1,0],[0,0,0]]` reproduces the
    /// input.
    #[test]
    fn conv2d_identity_kernel_passes_input_through() {
        let input: Vec<f32> = (0..16).map(|i| i as f32 - 7.5).collect();
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let actual = convolved(4, 4, &input, &kernel);
        for (a, e) in actual.iter().zip(input.iter()) {
            assert!((a - e).abs() <= 1.0e-5, "{a} != {e}");
        }
    }

    /// Random fuzz on a 5x5 input + random 3x3 kernel matches naive
    /// reference within 1.0e-4.
    #[test]
    fn conv2d_matches_naive_on_random_fuzz() {
        let mut state = 0xDEADC0DE_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
        };
        for _ in 0..30 {
            let input: Vec<f32> = (0..25).map(|_| next()).collect();
            let kernel: Vec<f32> = (0..9).map(|_| next()).collect();
            let actual = convolved(5, 5, &input, &kernel);
            let expected = naive_conv2d_3x3(&input, &kernel, 5, 5);
            for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - e).abs() <= 1.0e-4,
                    "lane {i}: direct={a} naive={e} diff={}",
                    (a - e).abs()
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Adversarial fixtures exposing real gaps
    // ------------------------------------------------------------------

    /// 1x1 image with identity kernel  -  only the center tap hits.
    #[test]
    fn conv2d_1x1_image() {
        let input = vec![5.0_f32];
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let actual = convolved(1, 1, &input, &kernel);
        assert_eq!(actual.len(), 1);
        assert!(
            (actual[0] - 5.0).abs() <= 1.0e-5,
            "1x1 conv with identity kernel = 5.0, got {}",
            actual[0]
        );
    }

    /// NaN input must propagate to every output pixel.
    #[test]
    fn conv2d_nan_input_propagates() {
        let input = vec![f32::NAN; 16];
        let kernel = vec![1.0_f32; 9];
        let actual = convolved(4, 4, &input, &kernel);
        for (i, &v) in actual.iter().enumerate() {
            assert!(
                v.is_nan(),
                "conv2d output[{i}] must be NaN when input is NaN"
            );
        }
    }

    /// Inf input must propagate to every output pixel.
    #[test]
    fn conv2d_inf_input_propagates() {
        let input = vec![f32::INFINITY; 16];
        let kernel = vec![1.0_f32; 9];
        let actual = convolved(4, 4, &input, &kernel);
        for (i, &v) in actual.iter().enumerate() {
            assert!(
                v.is_infinite(),
                "conv2d output[{i}] must be Inf when input is Inf"
            );
        }
    }

    #[test]
    fn conv2d_zero_dimensions_should_error() {
        let err = conv2d_3x3_direct("input", "kernel", "output", 0, 0)
            .expect_err("0x0 conv2d must error instead of returning empty program");
        assert!(
            err.contains("non-zero height and width"),
            "0x0 conv2d error must name the dimension contract: {err}"
        );
    }
}
