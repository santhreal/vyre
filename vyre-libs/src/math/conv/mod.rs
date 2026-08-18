//! 2D convolution sub-dialect.
//!
//! ROADMAP H3  -  Im2col/direct-conv decision by shape and memory
//! budget. Both ops are one algorithm: `crate::builder::stencil::stencil_3x3_taps` owns the
//! zero-padded 3x3 patch of a pixel, `im2col_3x3` writes those patches into an
//! `[H*W, 9]` matrix for a caller-supplied gemm, and `conv2d_3x3_direct`
//! contracts the same patch against the kernel in place.
//!
//! ## Why the patch algebra is the owner
//!
//! Convolution's ground truth is the canonical sum
//! `out[y, x] = sum_{ky=0..3, kx=0..3} input[y+ky-1, x+kx-1] * kernel[ky, kx]`.
//! Im2col's contribution is to reshape that sum into a matmul so a tiled /
//! vectorised gemm can carry it, at the cost of materialising the patch
//! matrix. Sharing the patch definition makes the parity gate "im2col output
//! contracted with the kernel equals `conv2d_3x3_direct`" structural rather
//! than a coincidence between two hand-written walks.

pub(crate) mod conv2d;
pub(crate) mod im2col;

pub use conv2d::conv2d_3x3_direct;
pub use im2col::im2col_3x3;

pub(crate) use crate::math::trap_f32_output_program;

/// Decision wrapper: choose the fused patch contraction in
/// `conv2d_3x3_direct` vs a materialised `im2col_3x3` matrix handed to a
/// tiled gemm, based on image area. Crossover threshold derived from a simple
/// memory vs compute tradeoff: im2col materialises an `H·W·9` patch matrix
/// (vs `H·W` for the input), so it pays an extra `8·H·W·sizeof(f32)`
/// of memory traffic. The matmul tile/vectorisation win recovers
/// that cost once the per-pixel work amortises across enough output
/// pixels  -  empirically the crossover is around 64x64 (4096
/// pixels). Below that threshold the fused form wins.
///
/// Returns the same Program as `conv2d_3x3_direct(input, kernel,
/// output, h, w)` regardless of the decision; the choice is
/// expressed via the Region's `generator` ident so a downstream
/// pass can route the dispatch differently if the runtime chooses
/// to honour the hint.
///
/// # Errors
///
/// Returns `Err` when `h * w` overflows `u32`.
pub fn conv2d_3x3_decision(
    input: &str,
    kernel: &str,
    output: &str,
    h: u32,
    w: u32,
) -> Result<vyre_foundation::ir::Program, String> {
    const IM2COL_PIXEL_THRESHOLD: u32 = 4096; // 64x64
    let pixels = h.checked_mul(w).ok_or_else(|| {
        "Fix: conv2d_3x3_decision h*w overflows u32; reduce dimensions.".to_string()
    })?;
    if pixels >= IM2COL_PIXEL_THRESHOLD {
        // Large image: prefer the materialised im2col matrix plus a tiled
        // gemm, which is a two-dispatch sequence the runtime megakernel
        // scheduler can fuse. We ship the fused Program here with the
        // generator id signalling the hint, and the megakernel-side router
        // substitutes the two-dispatch pair when that path is wired.
        let mut prog = conv2d_3x3_direct(input, kernel, output, h, w)?;
        // Best-effort hint: replace the wrapping Region's generator
        // with a name that signals "preferred for im2col routing".
        let entry = prog.entry().to_vec();
        let new_entry: Vec<vyre_foundation::ir::Node> = entry
            .into_iter()
            .map(|node| match node {
                vyre_foundation::ir::Node::Region {
                    body,
                    source_region,
                    ..
                } => vyre_foundation::ir::Node::Region {
                    generator: "vyre-libs::math::conv::conv2d_3x3_im2col_preferred".into(),
                    source_region,
                    body,
                },
                other => other,
            })
            .collect();
        prog = prog.with_rewritten_entry(new_entry);
        Ok(prog)
    } else {
        // Small image: direct conv wins.
        conv2d_3x3_direct(input, kernel, output, h, w)
    }
}
