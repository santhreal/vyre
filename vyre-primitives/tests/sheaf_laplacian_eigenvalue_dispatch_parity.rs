//! Value parity + dispatch-grid invariance for `math::sheaf_laplacian_eigenvalue`.
//!
//! The primitive extracts the dominant eigenpair of the DIAGONAL sheaf Laplacian `diag(r)`: the
//! eigenvalues are exactly the diagonal entries and the eigenvectors are the standard basis
//! vectors, so the dominant eigenpair is the closed form `(lambda, v) = (max_i r[i], e_argmax)`
//! (first arg-max on ties). This is exact (no power iteration or square root).
//!
//! Two properties are locked:
//! 1. VALUE PARITY, the IR must equal the exact 16.16 fixed-point closed form (`lambda = max r`,
//!    `v[j] = 1.0 iff j == argmax`). Because the answer is a pure max/select over integers, this is
//!    an EXACT equality, not a fixed-point tolerance.
//! 2. GRID INVARIANCE, the kernel is single-threaded (the reference/GPU infers grid = cells from
//!    buffer shapes, spawning `cells` invocations that each run the whole scan). The running-max
//!    scratch is a plain accumulator guarded to `InvocationId == 0`, so the output must be identical
//!    regardless of the dispatch grid.
#![cfg(feature = "all-lego")]

use vyre_primitives::math::sheaf_laplacian_eigenvalue::sheaf_laplacian_eigenvalue;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

const ONE_FP: u32 = 1 << 16; // 16.16 fixed-point 1.0

/// Run the IR at an explicit dispatch-grid FLOOR and return `(lambda, v)`.
fn run(n_nodes: u32, d: u32, r: &[u32], floor: u32) -> (Vec<u32>, Vec<u32>) {
    let program = sheaf_laplacian_eigenvalue("r", "v", "lambda", n_nodes, d, 4);
    let cells = (n_nodes * d) as usize;
    // Buffer order: restriction_diag(0), v(1), lambda(2), one_fp_buf(3). The v/lambda outputs are
    // seeded to zero (the backend zero-allocates them); one_fp_buf carries the 16.16 unit written
    // into the eigenvector's arg-max slot. The running max/arg-max are loop-carried locals, so
    // there are no scratch buffers to seed.
    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(r)),
            Value::from(pack(&vec![0u32; cells])),
            Value::from(pack(&[0u32])),
            Value::from(pack(&[ONE_FP])),
        ],
        floor,
    )
    .expect("sheaf_laplacian_eigenvalue reference evaluation must succeed");
    let lam = unpack(
        &outputs[vyre_reference::output_index(&program, "lambda").expect("lambda output")]
            .to_bytes(),
    );
    let vout =
        unpack(&outputs[vyre_reference::output_index(&program, "v").expect("v output")].to_bytes());
    (lam, vout[..cells].to_vec())
}

/// Exact fixed-point closed form: `(max r, e_argmax)` with a 0 running-max floor and first-arg-max
/// tie-break (the same rule the IR encodes).
fn exact(r: &[u32]) -> (u32, Vec<u32>) {
    let mut max_r = 0u32;
    let mut argmax = 0usize;
    for (i, &ri) in r.iter().enumerate() {
        if ri > max_r {
            max_r = ri;
            argmax = i;
        }
    }
    let v: Vec<u32> = (0..r.len())
        .map(|j| if j == argmax { ONE_FP } else { 0 })
        .collect();
    (max_r, v)
}

#[test]
fn sheaf_eigenvalue_matches_exact_closed_form_over_generated_diagonals() {
    let mut state = 0x51EA_F00Du32;
    let next = |s: &mut u32| {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    };
    let mut distinct_max_cases = 0u32;
    for case in 0..400u32 {
        let n = 1 + next(&mut state) % 8; // 1..=8 diagonal entries
                                          // 16.16 magnitudes up to ~64.0; a fresh distribution so ties and distinct maxima both occur.
        let r: Vec<u32> = (0..n).map(|_| next(&mut state) & 0x000F_FFFF).collect();

        let (lam, v) = run(n, 1, &r, 0);
        let (exp_lam, exp_v) = exact(&r);
        // Count the non-degenerate cases (a unique strict maximum) so the arg-max/select path is
        // genuinely exercised, not dominated by all-equal or all-zero diagonals.
        let max_count = r.iter().filter(|&&x| x == exp_lam).count();
        if exp_lam > 0 && max_count == 1 {
            distinct_max_cases += 1;
        }
        assert_eq!(
            lam,
            vec![exp_lam],
            "case {case} (n={n}): lambda {lam:?} != max r {exp_lam} (r={r:?})"
        );
        assert_eq!(
            v, exp_v,
            "case {case} (n={n}): eigenvector {v:?} != e_argmax {exp_v:?} (r={r:?})"
        );
    }
    assert!(
        distinct_max_cases > 200,
        "only {distinct_max_cases}/400 cases had a unique strict maximum, strengthen the diagonal \
         distribution so the arg-max eigenvector is exercised"
    );
}

#[test]
fn sheaf_eigenvalue_is_invariant_to_dispatch_grid_size() {
    // 16.16 fixed point: r = [0.5, 2.0, 1.0]; the dominant eigenpair is (2.0, e_1).
    let r = vec![ONE_FP >> 1, ONE_FP * 2, ONE_FP];
    let (lam0, v0) = run(3, 1, &r, 0);
    for floor in [1u32, 2, 4, 16, 64, 256] {
        let (lam, vv) = run(3, 1, &r, floor);
        assert_eq!(
            (lam.as_slice(), vv.as_slice()),
            (lam0.as_slice(), v0.as_slice()),
            "dispatch floor {floor}: sheaf eigenvalue output must not depend on the dispatch grid \
             size (lambda {lam:?} v {vv:?} vs base lambda {lam0:?} v {v0:?})"
        );
    }
    // Concrete truth: max(r) = 2.0 = 131072 (16.16) at index 1; eigenvector = e_1 = [0, 1.0, 0].
    assert_eq!(
        lam0,
        vec![ONE_FP * 2],
        "lambda must equal the max diagonal entry 2.0"
    );
    assert_eq!(
        v0,
        vec![0, ONE_FP, 0],
        "eigenvector must be the unit indicator at the arg-max index"
    );
}
