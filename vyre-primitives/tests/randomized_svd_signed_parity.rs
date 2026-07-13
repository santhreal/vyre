//! GPU-IR parity for `math/randomized_svd::randomized_projection_step`: the range-finder projection
//! `Y = A·Ω` (data matrix times a random Gaussian test matrix) at the heart of randomized SVD
//! (Halko-Martinsson-Tropp 2011) (driven through `vyre_reference::reference_eval` with SIGNED inputs).
//!
//! Why this test exists: `randomized_projection_step` is a 16.16 fixed-point matmul,
//! `Y[i,j] = Σ_k fixed_mul(A[i,k], Ω[k,j])`. BOTH operands are inherently SIGNED, a real data matrix
//! `A` has negative entries, and the random test matrix `Ω` is drawn i.i.d. Gaussian (mean zero, so
//! ~half its entries are negative). Before the signed-multiply fix (BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`) `fixed_mul_16_16_expr` reconstructed the product
//! from the UNSIGNED high word, so any negative `A[i,k]` or `Ω[k,j]` (a u32 with the top bit set, read
//! as ~2^32) produced a garbage term, corrupting the range sketch and every downstream singular
//! value. The primitive had NO IR-execution parity coverage (only f64 `*_cpu` tests), so this is the
//! FIRST faithful run of the projection kernel and it exercises the signed regime.
//!
//! BIT-EXACT: pure integer arithmetic, so the oracle replicates the kernel exactly 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`, accumulated with
//! wrapping u32 add. Any divergence is a real IR/dispatch defect, not a rounding artifact.
#![cfg(feature = "math")]

use vyre_primitives::math::randomized_svd::randomized_projection_step;
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

/// Bit-exact replica of the kernel's term multiply (the corrected SIGNED 16.16 multiply).
fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated (top bit
/// set on the negative half (the operand class the old unsigned multiply corrupted)).
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Exact u32 16.16 oracle for `Y(m×l) = A(m×n) · Ω(n×l)`.
fn project_fixed(a: &[u32], omega: &[u32], m: usize, n: usize, l: usize) -> Vec<u32> {
    let mut y = vec![0u32; m * l];
    for i in 0..m {
        for j in 0..l {
            let mut acc = 0u32;
            for k in 0..n {
                acc = acc.wrapping_add(fixed_mul(a[i * n + k], omega[k * l + j]));
            }
            y[i * l + j] = acc;
        }
    }
    y
}

fn run_via_reference(a: &[u32], omega: &[u32], m: u32, n: u32, l: u32) -> Vec<u32> {
    let program = randomized_projection_step("a", "omega", "y", m, n, l);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(a)),
            Value::from(pack_u32(omega)),
            Value::from(pack_u32(&vec![0u32; (m * l) as usize])),
        ],
    )
    .expect("randomized_projection_step reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn randomized_projection_signed_matches_exact_fixed_point_matmul() {
    let mut state = 0x0D15_EA5Eu32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let m = 1 + (case % 4) as usize;
        let n = 1 + ((case / 4) % 4) as usize;
        let l = 1 + ((case / 16) % 4) as usize;

        let a: Vec<u32> = (0..m * n).map(|_| signed_fixed(&mut state)).collect();
        let omega: Vec<u32> = (0..n * l).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += a.iter().chain(&omega).filter(|&&v| (v as i32) < 0).count() as u32;

        let got = run_via_reference(&a, &omega, m as u32, n as u32, l as u32);
        let want = project_fixed(&a, &omega, m, n, l);
        assert_eq!(
            got, want,
            "case {case} (m={m} n={n} l={l}): SIGNED range projection _via {got:?} != exact signed \
             oracle {want:?} (a={a:?} omega={omega:?})"
        );

        if want.iter().any(|&v| v != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative data/test-matrix entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed projections must produce negative sketch entries, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 projections were non-zero, the matmul is not being exercised"
    );
}

#[test]
fn randomized_projection_hand_checked_signed() {
    // A(2×2) = [[1.0, -2.0], [0.0, 3.0]], Ω(2×2) = [[2.0, 0.0], [-1.0, 1.0]]:
    //   Y[0,0] = (1.0)(2.0) + (-2.0)(-1.0) = 2.0 + 2.0 =  4.0
    //   Y[0,1] = (1.0)(0.0) + (-2.0)( 1.0) = 0.0 - 2.0 = -2.0
    //   Y[1,0] = (0.0)(2.0) + ( 3.0)(-1.0) = 0.0 - 3.0 = -3.0
    //   Y[1,1] = (0.0)(0.0) + ( 3.0)( 1.0) = 0.0 + 3.0 =  3.0
    let a = vec![to_fixed(1.0), to_fixed(-2.0), to_fixed(0.0), to_fixed(3.0)];
    let omega = vec![to_fixed(2.0), to_fixed(0.0), to_fixed(-1.0), to_fixed(1.0)];
    let got = run_via_reference(&a, &omega, 2, 2, 2);
    let want = project_fixed(&a, &omega, 2, 2, 2);
    assert_eq!(
        want,
        vec![to_fixed(4.0), to_fixed(-2.0), to_fixed(-3.0), to_fixed(3.0)],
        "sanity: signed range projection = [4.0, -2.0, -3.0, 3.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched projection must preserve sign: [4.0, -2.0, -3.0, 3.0]"
    );
}
