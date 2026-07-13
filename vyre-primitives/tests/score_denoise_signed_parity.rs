//! GPU-IR parity for `math/score_denoise::score_denoise_step`: the diffusion/flow-matching denoise
//! blend `x_{t-1} = α·x_t + β·score_θ(x_t,t) + σ·noise` in 16.16 fixed point, driven through
//! `vyre_reference::reference_eval` with SIGNED inputs.
//!
//! Why this test exists: the kernel is three `fixed_mul_16_16_expr` products summed with wrapping
//! adds. The score `score_θ = ∇_x log p` is inherently SIGNED (it points toward higher density and
//! is negative wherever density falls off), the latent `x` is signed, and the Gaussian `noise` is
//! signed. Before the signed-multiply fix (BACKLOG `FIXED-amg-fixed-path-unsigned-mul-negatives`)
//! `fixed_mul_16_16_expr` reconstructed the product from the UNSIGNED high word, so a negative score
//! or latent value (a u32 with the top bit set, read as ~2^32) produced a garbage term, silently
//! corrupting every reverse-diffusion step, the single most common ML use of this primitive. This
//! locks the corrected SIGNED multiply at the score_denoise consumer.
//!
//! BIT-EXACT: the kernel is pure integer arithmetic, so the oracle replicates it exactly 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32` summed with
//! `wrapping_add`. Any divergence is a real IR/dispatch defect, not a rounding artifact.
#![cfg(feature = "math")]

use vyre_primitives::math::score_denoise::score_denoise_step;
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

/// Bit-exact replica of the `score_denoise_step` IR term multiply.
fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated. The top
/// bit is set for the negative half (exactly the operands the old unsigned multiply corrupted).
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Exact u32 oracle for the denoise blend: `out[t] = α·x[t] + β·score[t] + σ·noise[t]` in 16.16.
fn denoise_fixed(
    x: &[u32],
    score: &[u32],
    noise: &[u32],
    alpha: u32,
    beta: u32,
    sigma: u32,
) -> Vec<u32> {
    (0..x.len())
        .map(|t| {
            fixed_mul(alpha, x[t])
                .wrapping_add(fixed_mul(beta, score[t]))
                .wrapping_add(fixed_mul(sigma, noise[t]))
        })
        .collect()
}

fn run_via_reference(
    x: &[u32],
    score: &[u32],
    noise: &[u32],
    alpha: u32,
    beta: u32,
    sigma: u32,
) -> Vec<u32> {
    let n = x.len() as u32;
    let program = score_denoise_step("x", "score", "noise", "alpha", "beta", "sigma", "out", n);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(x)),
            Value::from(pack_u32(score)),
            Value::from(pack_u32(noise)),
            Value::from(pack_u32(&[alpha])),
            Value::from(pack_u32(&[beta])),
            Value::from(pack_u32(&[sigma])),
            Value::from(pack_u32(&vec![0u32; x.len()])),
        ],
    )
    .expect("score_denoise_step reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn score_denoise_signed_matches_exact_fixed_point_blend() {
    let mut state = 0x51ED_270Bu32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let n = 1 + (xorshift(&mut state) % 8) as usize; // 1..=8 latent dimensions
        let x: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();
        let score: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();
        let noise: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();
        // Schedule coefficients: α positive (a contraction toward the prior), β/σ signed, the
        // reverse-SDE score term flips sign across parameterizations and σ can be a signed increment.
        let alpha = (xorshift(&mut state) & 0x0000_FFFF) as u32; // [0, 1.0)
        let beta = signed_fixed(&mut state);
        let sigma = signed_fixed(&mut state);

        neg_inputs += x
            .iter()
            .chain(&score)
            .chain(&noise)
            .filter(|&&v| (v as i32) < 0)
            .count() as u32;

        let got = run_via_reference(&x, &score, &noise, alpha, beta, sigma);
        let want = denoise_fixed(&x, &score, &noise, alpha, beta, sigma);
        assert_eq!(
            got, want,
            "case {case} (n={n}): SIGNED denoise blend _via {got:?} != exact signed oracle {want:?} \
             (x={x:?} score={score:?} noise={noise:?} α={alpha} β={beta} σ={sigma})"
        );

        if want.iter().any(|&v| v != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 800,
        "sweep must feed many negative latent/score/noise entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed denoise blends must produce negative denoised entries, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 blends were non-zero, the multiply-add is not being exercised"
    );
}

#[test]
fn score_denoise_hand_checked_signed_step() {
    // α = 0.5, β = -0.25, σ = 2.0; x = [2.0, -1.0], score = [4.0, 8.0], noise = [-1.0, 0.5]:
    //   out[0] = 0.5·2.0 + (-0.25)·4.0 + 2.0·(-1.0) = 1.0 - 1.0 - 2.0 = -2.0
    //   out[1] = 0.5·(-1.0) + (-0.25)·8.0 + 2.0·0.5 = -0.5 - 2.0 + 1.0 = -1.5
    let x = vec![to_fixed(2.0), to_fixed(-1.0)];
    let score = vec![to_fixed(4.0), to_fixed(8.0)];
    let noise = vec![to_fixed(-1.0), to_fixed(0.5)];
    let (alpha, beta, sigma) = (to_fixed(0.5), to_fixed(-0.25), to_fixed(2.0));

    let got = run_via_reference(&x, &score, &noise, alpha, beta, sigma);
    let want = denoise_fixed(&x, &score, &noise, alpha, beta, sigma);
    assert_eq!(
        want,
        vec![to_fixed(-2.0), to_fixed(-1.5)],
        "sanity: signed denoise step = [-2.0, -1.5]"
    );
    assert_eq!(
        got, want,
        "the dispatched denoise step must preserve sign: [-2.0, -1.5]"
    );
}
