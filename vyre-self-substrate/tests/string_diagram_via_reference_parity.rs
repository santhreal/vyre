//! End-to-end parity for `logic::string_diagram_ir_rewrite::compose_ir_arrows_fixed_via` through the
//! shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap AND locks a real bug the conversion surfaced (see BACKLOG
//! `SWEEP-via-consumer-input-output-contract-audit`): `monoidal_compose` delegates to the shared
//! `fixed_u32_matmul_program` (lhs RO(0) + rhs RO(1) + out plain-ReadWrite(2) = 3 input-consuming),
//! but the consumer used to `ensure_input_slots(2)` and pass only `[f, g]`: MISSING the zero-fill
//! slot for the plain-ReadWrite `out` (InputOutput). That UNDER-FEED would hard-fail the real
//! backend's strict input-count validation ("expected 3, received 2"); pre-fix it surfaces here as
//! the faithful dispatcher's "more input-consuming buffers than dispatch inputs" error.
//!
//! `fixed_u32_matmul_program` computes `out[i*c+j] = Σ_k fixed_mul(f[i*b+k], g[k*c+j])` with
//! `fixed_mul(x,y)` = the SIGNED 16.16 multiply (bits [16..48] of the i64 product, matching the corrected
//! `fixed_mul_16_16_expr`), bit-exactly reproducible in u32, so this is a zero-tolerance oracle (the same
//! exact-fixed-point route mz_project / natural_gradient use).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::logic::string_diagram_ir_rewrite::compose_ir_arrows_fixed_via;

mod common;
use common::fixed_mul;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Exact u32 oracle for the 16.16 fixed-point matrix composition `f(a×b) · g(b×c) = out(a×c)`.
fn compose_fixed(f: &[u32], g: &[u32], a: usize, b: usize, c: usize) -> Vec<u32> {
    let mut out = vec![0u32; a * c];
    for i in 0..a {
        for j in 0..c {
            let mut acc = 0u32;
            for k in 0..b {
                acc = acc.wrapping_add(fixed_mul(f[i * b + k], g[k * c + j]));
            }
            out[i * c + j] = acc;
        }
    }
    out
}

#[test]
fn compose_via_matches_exact_fixed_matmul_over_generated_shapes() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x5D_1A_9001u32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let a = 1 + (case % 4) as usize;
        let b = 1 + ((case / 4) % 4) as usize;
        let c = 1 + ((case / 16) % 4) as usize;

        // Values in [0, 2.0) fixed → products/sums stay well under u32::MAX (b <= 4).
        let f: Vec<u32> = (0..a * b)
            .map(|_| xorshift(&mut state) % (2 * FIXED_ONE))
            .collect();
        let g: Vec<u32> = (0..b * c)
            .map(|_| xorshift(&mut state) % (2 * FIXED_ONE))
            .collect();

        let got = compose_ir_arrows_fixed_via(&dispatcher, &f, &g, a as u32, b as u32, c as u32)
            .expect("compose_ir_arrows_fixed_via must dispatch the fixed-point matmul kernel");
        let want = compose_fixed(&f, &g, a, b, c);
        assert_eq!(
            got, want,
            "case {case}: monoidal composition must match the exact fixed-point matmul; a={a} b={b} c={c}"
        );
        if want.iter().any(|&v| v != 0) {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 300,
        "expected >300 nonzero compositions, got {nontrivial}"
    );
}

/// A signed 16.16 value in roughly `[-2.0, 2.0)`: a 17-bit magnitude, optionally negated (top bit
/// set on the negative half (the operand class the old unsigned high-word multiply corrupted)).
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0001_FFFF) as i32; // [0.0, 2.0) in 16.16
    let signed = if xorshift(state) & 1 == 0 {
        magnitude
    } else {
        -magnitude
    };
    signed as u32
}

fn to_fixed(v: f64) -> u32 {
    (v * 65536.0).round() as i64 as u32
}

#[test]
fn compose_via_matches_signed_fixed_matmul_with_negative_arrow_weights() {
    // A monoidal arrow (a linear map between objects) freely carries NEGATIVE weights, so the
    // composite `out[i,j] = Σ_k f[i,k]·g[k,j]` is genuinely SIGNED. The base sweep draws values from
    // `% (2·FIXED_ONE)` (all non-negative) and never runs a negative arrow weight through the
    // fixed-point matmul. This sweep composes SIGNED arrows and asserts the dispatched kernel
    // bit-exactly matches the SIGNED oracle, locking the signed `fixed_mul_16_16_expr` fix on the
    // full a×b×c matmul path (pre-fix, a negative `f[i,k]·g[k,j]` term diverged: the unsigned high
    // word read a negative operand as ~2^32 and produced garbage).
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x8BAD_F00Du32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut nontrivial = 0u32;
    for case in 0..400u32 {
        let a = 1 + (case % 4) as usize;
        let b = 1 + ((case / 4) % 4) as usize;
        let c = 1 + ((case / 16) % 4) as usize;

        let f: Vec<u32> = (0..a * b).map(|_| signed_fixed(&mut state)).collect();
        let g: Vec<u32> = (0..b * c).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += f.iter().filter(|&&v| (v as i32) < 0).count() as u32;
        neg_inputs += g.iter().filter(|&&v| (v as i32) < 0).count() as u32;

        let got = compose_ir_arrows_fixed_via(&dispatcher, &f, &g, a as u32, b as u32, c as u32)
            .expect("compose_ir_arrows_fixed_via must dispatch the fixed-point matmul kernel");
        let want = compose_fixed(&f, &g, a, b, c);
        assert_eq!(
            got, want,
            "case {case}: SIGNED monoidal composition must match the signed fixed-point matmul; \
             a={a} b={b} c={c} f={f:?} g={g:?}"
        );

        if want.iter().any(|&v| v != 0) {
            nontrivial += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative arrow weights (the sign-corruption regime), got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed compositions must produce negative composite entries, got {neg_outputs}"
    );
    assert!(
        nontrivial > 300,
        "expected >300 nonzero signed compositions, got {nontrivial}"
    );
}

#[test]
fn compose_via_hand_checked_signed_composition() {
    // f(2×2) = [[2.0, -1.0], [0.0, 3.0]], g(2×2) = [[1.0, 0.0], [-2.0, 1.0]]:
    //   out[0,0] = (2.0)(1.0) + (-1.0)(-2.0) = 2.0 + 2.0 =  4.0
    //   out[0,1] = (2.0)(0.0) + (-1.0)( 1.0) = 0.0 - 1.0 = -1.0
    //   out[1,0] = (0.0)(1.0) + ( 3.0)(-2.0) = 0.0 - 6.0 = -6.0
    //   out[1,1] = (0.0)(0.0) + ( 3.0)( 1.0) = 0.0 + 3.0 =  3.0
    let dispatcher = ReferenceEvalDispatcher;
    let f = vec![to_fixed(2.0), to_fixed(-1.0), to_fixed(0.0), to_fixed(3.0)];
    let g = vec![to_fixed(1.0), to_fixed(0.0), to_fixed(-2.0), to_fixed(1.0)];
    let got = compose_ir_arrows_fixed_via(&dispatcher, &f, &g, 2, 2, 2).unwrap();
    let want = compose_fixed(&f, &g, 2, 2, 2);
    assert_eq!(
        want,
        vec![to_fixed(4.0), to_fixed(-1.0), to_fixed(-6.0), to_fixed(3.0)],
        "sanity: signed 2×2 fixed-point composition = [4.0, -1.0, -6.0, 3.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched composition must preserve sign: [4.0, -1.0, -6.0, 3.0]"
    );
}

#[test]
fn compose_via_matches_hand_checked_cases() {
    let dispatcher = ReferenceEvalDispatcher;

    // Identity(2×2) · M(2×2) = M. Identity in 16.16 = [[1,0],[0,1]].
    let id = vec![FIXED_ONE, 0, 0, FIXED_ONE];
    let m = vec![2 * FIXED_ONE, FIXED_ONE, 0, 3 * FIXED_ONE];
    let got = compose_ir_arrows_fixed_via(&dispatcher, &id, &m, 2, 2, 2).unwrap();
    assert_eq!(got, m, "identity composed with M yields M");

    // Row f(1×2)=[2.0, 3.0] composed with column g(2×1)=[1.0, 0.5] → 2*1 + 3*0.5 = 3.5.
    let f = vec![2 * FIXED_ONE, 3 * FIXED_ONE];
    let g = vec![FIXED_ONE, FIXED_ONE / 2];
    let got = compose_ir_arrows_fixed_via(&dispatcher, &f, &g, 1, 2, 1).unwrap();
    assert_eq!(got, vec![FIXED_ONE * 7 / 2], "2·1 + 3·0.5 = 3.5 in 16.16");
}
