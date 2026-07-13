//! End-to-end parity for `math::mori_zwanzig_region_coarsen::coarsen_region_state_fixed_via`.
//!
//! `mz_project_step` (the Mori-Zwanzig Markovian projection matvec `out[i] = Σ_j P[i,j]·f[j]` in
//! 16.16 fixed point) had NO IR-execution coverage: `rg -l mz_project_step vyre-primitives/tests/`
//! = zero files, and its only self-substrate consumer path (`coarsen_region_state_fixed_via`) is
//! exercised solely by a mock dispatcher (so the fixed-point matvec kernel never actually ran).
//! Its only host oracle (`mz_project_step_cpu`) is f64, giving no exact reference for the u32
//! fixed-point dispatch path.
//!
//! This runs the real fixed-point `mz_project_step` Program through the shared
//! `ReferenceEvalDispatcher` and asserts it EXACTLY (no tolerance) reproduces a u32 16.16 matvec
//! oracle. The oracle replicates the IR's arithmetic bit-for-bit: `fixed_mul_16_16(a, b) =
//! ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32` (matching the corrected SIGNED
//! `fixed_mul_16_16_expr` = bits 16..47 of the SIGNED 64-bit product), accumulated with wrapping
//! u32 addition (GPU add semantics). Because both sides use identical integer arithmetic, any
//! divergence is a real IR/dispatch defect, not a rounding artifact.
#![forbid(unsafe_code)]

use vyre_self_substrate::math::mori_zwanzig_region_coarsen::coarsen_region_state_fixed_via;

mod common;
use common::fixed_mul as fixed_mul_16_16;
use common::ReferenceEvalDispatcher;

/// Exact u32 16.16 matvec oracle mirroring the `mz_project_step` kernel: per output row `i`,
/// `acc = 0; for j: acc = acc.wrapping_add(fixed_mul_16_16(P[i*n+j], f[j]))`.
fn mz_project_fixed(p_matrix: &[u32], f_vec: &[u32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let mut acc = 0u32;
            for j in 0..n {
                acc = acc.wrapping_add(fixed_mul_16_16(p_matrix[i * n + j], f_vec[j]));
            }
            acc
        })
        .collect()
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn coarsen_region_state_fixed_via_matches_exact_fixed_point_matvec() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x3141_5926u32;
    let mut moved_cases = 0u32;
    for case in 0..400u32 {
        let n = 1 + xorshift(&mut state) % 8; // 1..=8 resolved modes
        let cells = (n * n) as usize;
        // 16.16 values up to ~64.0 (20-bit magnitude): products stay well within u64, per-term
        // results ~ up to 64*64 = 4096.0 (0x1000_0000), and n-term sums occasionally wrap u32 
        // which BOTH the IR and the oracle do identically, so exact equality still holds.
        let p_matrix: Vec<u32> = (0..cells)
            .map(|_| xorshift(&mut state) & 0x000F_FFFF)
            .collect();
        let f_vec: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 0x000F_FFFF).collect();

        let via = coarsen_region_state_fixed_via(&dispatcher, &p_matrix, &f_vec, n)
            .expect("coarsen_region_state_fixed_via must dispatch the projection kernel");
        let oracle = mz_project_fixed(&p_matrix, &f_vec, n as usize);
        if oracle.iter().any(|&w| w != 0) {
            moved_cases += 1;
        }
        assert_eq!(
            via, oracle,
            "case {case} (n={n}): mz projection _via {via:?} != exact fixed-point oracle {oracle:?} \
             (p_matrix={p_matrix:?}, f_vec={f_vec:?})"
        );
    }
    assert!(
        moved_cases > 380,
        "only {moved_cases}/400 projections were non-zero, the matvec is not being exercised"
    );
}

/// A signed 16.16 value in roughly `[-8.0, 8.0)`: a 19-bit magnitude, optionally negated. The top
/// bit is set for the negative half, so these are exactly the operands the OLD unsigned
/// `fixed_mul_16_16_expr` (bits 16..48 of an UNSIGNED product) corrupted, a negative operand read
/// as ~2^32 produced a garbage high word. The corrected SIGNED multiply is what this exercises.
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0007_FFFF) as i32; // [0.0, 8.0) in 16.16
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
fn coarsen_region_state_fixed_via_matches_signed_projection_with_negative_entries() {
    // A real Mori-Zwanzig Markovian projector coarsens by SUBTRACTING memory contributions, so both
    // the projector `P` and the coarse state `f` routinely carry NEGATIVE 16.16 entries. The base
    // sweep masks every value to `& 0x000F_FFFF` (top bit clear → all non-negative), so it never ran
    // a single negative operand through the fixed-point matvec. This sweep feeds SIGNED P and f and
    // asserts the dispatched kernel bit-exactly matches the SIGNED oracle, locking the signed
    // `fixed_mul_16_16_expr` fix at the mz_project consumer (pre-fix, a negative P[i,j]·f[j] term
    // would diverge because the unsigned high word was wrong).
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x9E37_79B1u32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved_cases = 0u32;
    for case in 0..400u32 {
        let n = 1 + xorshift(&mut state) % 8; // 1..=8 resolved modes
        let cells = (n * n) as usize;
        let p_matrix: Vec<u32> = (0..cells).map(|_| signed_fixed(&mut state)).collect();
        let f_vec: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += p_matrix.iter().filter(|&&v| (v as i32) < 0).count() as u32;
        neg_inputs += f_vec.iter().filter(|&&v| (v as i32) < 0).count() as u32;

        let via = coarsen_region_state_fixed_via(&dispatcher, &p_matrix, &f_vec, n)
            .expect("coarsen_region_state_fixed_via must dispatch the projection kernel");
        let oracle = mz_project_fixed(&p_matrix, &f_vec, n as usize);
        assert_eq!(
            via, oracle,
            "case {case} (n={n}): SIGNED mz projection _via {via:?} != exact signed fixed-point \
             oracle {oracle:?} (p_matrix={p_matrix:?}, f_vec={f_vec:?})"
        );

        if oracle.iter().any(|&w| w != 0) {
            moved_cases += 1;
        }
        neg_outputs += oracle.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative 16.16 operands (the sign-corruption regime), got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed projections must produce negative coarse-state entries, got {neg_outputs}"
    );
    assert!(
        moved_cases > 380,
        "only {moved_cases}/400 projections were non-zero, the matvec is not being exercised"
    );
}

#[test]
fn coarsen_region_state_fixed_via_hand_checked_negative_projection() {
    // P = [[-1.0, 0.5], [0.0, -2.0]], f = [2.0, -3.0]:
    //   out[0] = (-1.0)(2.0) + (0.5)(-3.0) = -2.0 - 1.5 = -3.5
    //   out[1] = ( 0.0)(2.0) + (-2.0)(-3.0) = 0.0 + 6.0  =  6.0
    let dispatcher = ReferenceEvalDispatcher;
    let p_matrix = vec![to_fixed(-1.0), to_fixed(0.5), to_fixed(0.0), to_fixed(-2.0)];
    let f_vec = vec![to_fixed(2.0), to_fixed(-3.0)];
    let via = coarsen_region_state_fixed_via(&dispatcher, &p_matrix, &f_vec, 2)
        .expect("coarsen_region_state_fixed_via must dispatch");
    let oracle = mz_project_fixed(&p_matrix, &f_vec, 2);
    assert_eq!(
        oracle,
        vec![to_fixed(-3.5), to_fixed(6.0)],
        "sanity: signed fixed-point matvec = [-3.5, 6.0]"
    );
    assert_eq!(
        via, oracle,
        "the dispatched projection kernel must preserve sign: [-3.5, 6.0]"
    );
}

#[test]
fn coarsen_region_state_fixed_via_computes_a_known_projection() {
    // 2x2 identity-scaled projector P = [[1.0, 0.0],[0.0, 2.0]] in 16.16; f = [3.0, 4.0].
    // out[0] = 1.0*3.0 + 0.0*4.0 = 3.0; out[1] = 0.0*3.0 + 2.0*4.0 = 8.0.
    let dispatcher = ReferenceEvalDispatcher;
    let one = 1u32 << 16;
    let p_matrix = vec![one, 0, 0, 2 * one];
    let f_vec = vec![3 * one, 4 * one];
    let via = coarsen_region_state_fixed_via(&dispatcher, &p_matrix, &f_vec, 2)
        .expect("coarsen_region_state_fixed_via must dispatch");
    let oracle = mz_project_fixed(&p_matrix, &f_vec, 2);
    assert_eq!(
        oracle,
        vec![3 * one, 8 * one],
        "sanity: exact fixed-point matvec = [3.0, 8.0]"
    );
    assert_eq!(
        via, oracle,
        "the dispatched projection kernel must equal the exact fixed-point matvec"
    );
}
