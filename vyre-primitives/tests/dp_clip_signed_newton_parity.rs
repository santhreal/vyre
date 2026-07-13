//! GPU-IR parity for `math/dp_clip::dp_clip_per_sample`: DP-SGD per-sample gradient clipping, driven
//! through `vyre_reference::reference_eval` with SIGNED gradients and realistic-magnitude norms.
//!
//! Why this test exists: the kernel enforces `clipped = g · min(1, C/‖g‖)`. The original IR computed it as
//! `Expr::div(Expr::mul(g, min(C,n)), max(n,1))` which was doubly broken (BACKLOG
//! `DP-CLIP-broken-signed-div-AND-overflow`): (1) `Expr::mul` keeps only the low 32 bits so `g·scale`
//! OVERFLOWED for unit-magnitude 16.16, and (2) the unsigned `Expr::div` corrupted the sign of negative
//! gradients. The rewrite computes the 16.16 reciprocal `1/n` by clz-seeded Newton-Raphson (no overflow,
//! all products via the SIGNED `fixed_mul_16_16_expr`), then `factor = min(1, C·(1/n))` and `g·factor`.
//!
//! TOLERANCE parity (Newton is iterative, not exact): the f64 reference `g · min(1, C/‖g‖)` is the oracle;
//! five Newton steps reach ~full 16.16 precision so a small tolerance holds. Two BIT-EXACT anchors are
//! also asserted: (a) the NO-CLIP case `n ≤ C` must return the gradient UNCHANGED (factor clamps to
//! exactly 1.0), and (b) negative gradients keep their sign.
#![cfg(feature = "math")]

use vyre_primitives::math::dp_clip::dp_clip_per_sample;
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

fn from_fixed(v: u32) -> f64 {
    f64::from(v as i32) / FIXED_ONE
}

/// A signed half-integer gradient in {-3, -2.5, …, 3}.
fn signed_grad(state: &mut u32) -> f64 {
    0.5 * f64::from((xorshift(state) % 13) as i32 - 6)
}

fn run_via_reference(grads: &[u32], norms: &[u32], clip: u32, b: u32, d: u32) -> Vec<u32> {
    let program = dp_clip_per_sample("grads", "norms", "clip", "clipped", b, d);
    let cells = (b * d) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(grads)),
            Value::from(pack_u32(norms)),
            Value::from(pack_u32(&[clip])),
            Value::from(pack_u32(&vec![0u32; cells])),
        ],
    )
    .expect("dp_clip_per_sample reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn dp_clip_signed_gradients_match_reference_within_tolerance() {
    let mut state = 0xD9_00_00_01u32;
    let clip_real = 1.0;
    let clip = to_fixed(clip_real);
    // Norms deliberately span both sides of C so both the clip and no-clip branches are exercised.
    let norm_choices = [0.5, 1.0, 1.5, 2.0, 3.0];

    let mut neg_grads = 0u32;
    let mut clipped_cases = 0u32;
    let mut noclip_cases = 0u32;
    let mut neg_out = 0u32;
    for case in 0..300u32 {
        let b = 1 + case % 4; // 1..4 samples
        let d = 1 + case % 3; // 1..3 dims

        let norms_f: Vec<f64> = (0..b)
            .map(|_| norm_choices[(xorshift(&mut state) % 5) as usize])
            .collect();
        let grads_f: Vec<f64> = (0..b * d).map(|_| signed_grad(&mut state)).collect();

        let norms_fx: Vec<u32> = norms_f.iter().map(|&v| to_fixed(v)).collect();
        let grads_fx: Vec<u32> = grads_f.iter().map(|&v| to_fixed(v)).collect();

        let got = run_via_reference(&grads_fx, &norms_fx, clip, b, d);
        assert_eq!(got.len(), (b * d) as usize, "case {case}: output length");

        for i in 0..b as usize {
            let n = norms_f[i];
            let factor = if n > clip_real { clip_real / n } else { 1.0 };
            for j in 0..d as usize {
                let cell = i * d as usize + j;
                let g = grads_f[cell];
                let want = g * factor;
                let got_v = from_fixed(got[cell]);

                if n <= clip_real {
                    // NO-CLIP: factor is exactly 1.0, so the gradient must pass through BIT-EXACT.
                    assert_eq!(
                        got[cell],
                        grads_fx[cell],
                        "case {case} cell {cell}: n={n} ≤ C so gradient must be unchanged; got={got_v} g={g}"
                    );
                    noclip_cases += 1;
                } else {
                    let tol = 0.01 + 0.01 * want.abs();
                    assert!(
                        (got_v - want).abs() <= tol,
                        "case {case} cell {cell}: clipped gradient must match g·(C/n) within tol; \
                         got={got_v} want={want} tol={tol} (g={g} n={n} C={clip_real})"
                    );
                    // Sign preservation: a nonzero gradient keeps its sign after clipping.
                    if g != 0.0 {
                        assert_eq!(
                            got_v.signum(),
                            g.signum(),
                            "case {case} cell {cell}: clipping must preserve the gradient sign; got={got_v} g={g}"
                        );
                    }
                    clipped_cases += 1;
                }
                if g < 0.0 {
                    neg_grads += 1;
                }
                if got_v < 0.0 {
                    neg_out += 1;
                }
            }
        }
    }
    assert!(
        neg_grads > 200,
        "sweep must feed negative gradients (the sign-preserving regime), got {neg_grads}"
    );
    assert!(
        clipped_cases > 100 && noclip_cases > 100,
        "sweep must exercise BOTH clip and no-clip branches: clipped={clipped_cases} noclip={noclip_cases}"
    );
    assert!(
        neg_out > 100,
        "sweep must produce negative clipped gradients, got {neg_out}"
    );
}

#[test]
fn dp_clip_hand_checked_negative_and_boundary() {
    let clip = to_fixed(1.0);

    // norm 2.0 > C=1.0 → factor 0.5. g = [-2.0, 1.0] (one sample, two dims) → [-1.0, 0.5].
    let got = run_via_reference(
        &[to_fixed(-2.0), to_fixed(1.0)],
        &[to_fixed(2.0)],
        clip,
        1,
        2,
    );
    let out: Vec<f64> = got.iter().map(|&v| from_fixed(v)).collect();
    assert!(
        (out[0] - (-1.0)).abs() < 0.01 && (out[1] - 0.5).abs() < 0.01,
        "clip by 0.5 preserving sign: [-2,1] → [-1,0.5], got {out:?}"
    );

    // norm 0.5 ≤ C=1.0 → NO clip, gradients unchanged bit-exact (incl. negative).
    let got = run_via_reference(
        &[to_fixed(-3.0), to_fixed(2.5)],
        &[to_fixed(0.5)],
        clip,
        1,
        2,
    );
    assert_eq!(
        got,
        vec![to_fixed(-3.0), to_fixed(2.5)],
        "norm below the clip bound leaves gradients untouched, bit-exact"
    );

    // norm exactly = C → factor = 1.0 (no clip), unchanged.
    let got = run_via_reference(&[to_fixed(-1.5)], &[to_fixed(1.0)], clip, 1, 1);
    assert_eq!(
        got,
        vec![to_fixed(-1.5)],
        "norm == clip bound is the no-clip boundary (factor exactly 1.0)"
    );
}
