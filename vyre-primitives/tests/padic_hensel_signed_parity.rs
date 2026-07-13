//! GPU-IR parity for `math/padic::hensel_lift_step`: one Newton/Hensel refinement
//! `out = x - f(x)·f'(x)^{-1}` in 16.16 fixed point, driven through `vyre_reference::reference_eval`
//! with SIGNED roots, residuals, and inverse derivatives.
//!
//! Why this test exists: the step is a SIGNED `fixed_mul_16_16_expr` (the `f_x·inv_f_prime` correction)
//! followed by a two's-complement `Expr::sub`. Roots `x`, residuals `f(x)`, and inverse derivatives
//! `f'(x)^{-1}` are all freely SIGNED. Before the signed-multiply fix (BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`) the correction term `fixed_mul(f_x, inv_f_prime)`
//! reconstructed its product from the UNSIGNED high word, so a negative residual or inverse-derivative
//! (a u32 with the top bit set, read as ~2^32) produced a garbage correction, pushing the Newton
//! iterate the wrong way. This locks the signed correction AND its composition with the signed
//! subtraction. (The subtraction itself was always two's-complement-safe; the danger was the multiply.)
//!
//! NOTE on semantics: despite the p-adic framing, the IR and the `*_cpu` reference both compute the
//! REAL-number fixed-point Newton step `x - f_x·inv_f_prime` (not modular mod-p^k arithmetic); the two
//! agree, so this bit-exact test is the correct contract for what the primitive actually implements.
//! (The doc/impl framing mismatch is tracked separately in BACKLOG `PADIC-doc-claims-modular-impl-is-real-fixed`.)
//!
//! BIT-EXACT: pure integer arithmetic 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`, then `wrapping_sub`.
#![cfg(feature = "math")]

use vyre_primitives::math::padic::hensel_lift_step;
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

/// Bit-exact replica of the correction term's SIGNED 16.16 multiply.
fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated.
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Exact u32 16.16 oracle: `out[t] = x[t] - fixed_mul(f_x[t], inv_f_prime[t])`.
fn hensel_fixed(x: &[u32], f_x: &[u32], inv_f_prime: &[u32]) -> Vec<u32> {
    (0..x.len())
        .map(|t| x[t].wrapping_sub(fixed_mul(f_x[t], inv_f_prime[t])))
        .collect()
}

fn run_via_reference(x: &[u32], f_x: &[u32], inv_f_prime: &[u32]) -> Vec<u32> {
    let n = x.len() as u32;
    let program = hensel_lift_step("x", "f_x", "inv_f_prime", "out", n);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(x)),
            Value::from(pack_u32(f_x)),
            Value::from(pack_u32(inv_f_prime)),
            Value::from(pack_u32(&vec![0u32; x.len()])),
        ],
    )
    .expect("hensel_lift_step reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn hensel_lift_signed_matches_exact_fixed_point_step() {
    let mut state = 0xBADD_CAFEu32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let n = 1 + (xorshift(&mut state) % 8) as usize;
        let x: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();
        let f_x: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();
        let inv_f_prime: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += x
            .iter()
            .chain(&f_x)
            .chain(&inv_f_prime)
            .filter(|&&v| (v as i32) < 0)
            .count() as u32;

        let got = run_via_reference(&x, &f_x, &inv_f_prime);
        let want = hensel_fixed(&x, &f_x, &inv_f_prime);
        assert_eq!(
            got, want,
            "case {case} (n={n}): SIGNED Hensel step _via {got:?} != exact signed oracle {want:?} \
             (x={x:?} f_x={f_x:?} inv_f_prime={inv_f_prime:?})"
        );

        if want.iter().any(|&v| v != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 800,
        "sweep must feed many negative root/residual/derivative entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed Hensel steps must produce negative refined iterates, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 steps were non-zero, the step is not being exercised"
    );
}

#[test]
fn hensel_lift_hand_checked_signed() {
    // x = [1.0, 5.0], f_x = [3.0, 1.0], inv_f_prime = [2.0, -1.0]:
    //   out[0] = 1.0 - (3.0)(2.0)  = 1.0 - 6.0 = -5.0
    //   out[1] = 5.0 - (1.0)(-1.0) = 5.0 + 1.0 =  6.0
    let x = vec![to_fixed(1.0), to_fixed(5.0)];
    let f_x = vec![to_fixed(3.0), to_fixed(1.0)];
    let inv_f_prime = vec![to_fixed(2.0), to_fixed(-1.0)];
    let got = run_via_reference(&x, &f_x, &inv_f_prime);
    let want = hensel_fixed(&x, &f_x, &inv_f_prime);
    assert_eq!(
        want,
        vec![to_fixed(-5.0), to_fixed(6.0)],
        "sanity: signed Hensel step = [-5.0, 6.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched Hensel step must preserve sign: [-5.0, 6.0]"
    );
}
