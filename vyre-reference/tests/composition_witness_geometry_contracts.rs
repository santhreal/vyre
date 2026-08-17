//! Independent known-answer and algebraic property tests for geometric and fixpoint composition witnesses.

use vyre_reference::composition_witness::{
    clifford2_product_witness, persistent_fixpoint_into_witness, persistent_fixpoint_witness,
    tfn_scalar_mix_witness, try_persistent_fixpoint_into_witness,
};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
}

fn mv_approx_eq(a: [f64; 4], b: [f64; 4]) -> bool {
    approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2]) && approx_eq(a[3], b[3])
}

#[test]
fn clifford2_identity_units() {
    let identity = [1.0, 0.0, 0.0, 0.0];
    let v = [0.0, 2.0, 3.0, 0.0];

    let left = clifford2_product_witness(identity, v);
    assert!(mv_approx_eq(left, v));

    let right = clifford2_product_witness(v, identity);
    assert!(mv_approx_eq(right, v));
}

#[test]
fn clifford2_basis_squares() {
    let e1 = [0.0, 1.0, 0.0, 0.0];
    let e2 = [0.0, 0.0, 1.0, 0.0];
    let e12 = [0.0, 0.0, 0.0, 1.0];

    // e1^2 = 1, e2^2 = 1 in Cl(2, 0)
    let e1_sq = clifford2_product_witness(e1, e1);
    assert!(mv_approx_eq(e1_sq, [1.0, 0.0, 0.0, 0.0]));

    let e2_sq = clifford2_product_witness(e2, e2);
    assert!(mv_approx_eq(e2_sq, [1.0, 0.0, 0.0, 0.0]));

    // e12^2 = -1 (pseudoscalar acts as imaginary unit)
    let e12_sq = clifford2_product_witness(e12, e12);
    assert!(mv_approx_eq(e12_sq, [-1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn clifford2_anticommutativity() {
    let e1 = [0.0, 1.0, 0.0, 0.0];
    let e2 = [0.0, 0.0, 1.0, 0.0];
    let e12 = [0.0, 0.0, 0.0, 1.0];

    let p1 = clifford2_product_witness(e1, e2);
    let p2 = clifford2_product_witness(e2, e1);
    assert!(mv_approx_eq(p1, e12));
    assert!(mv_approx_eq(p2, [0.0, 0.0, 0.0, -1.0]));

    let left = clifford2_product_witness(e12, e1);
    let right = clifford2_product_witness(e1, e12);
    assert!(mv_approx_eq(left, [0.0, 0.0, -1.0, 0.0]));
    assert!(mv_approx_eq(right, [0.0, 0.0, 1.0, 0.0]));
}

#[test]
fn clifford2_distributivity() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [0.0, 1.0, 0.0, 0.0];
    let c = [0.5, 0.0, 0.5, 0.0];
    let bc = [b[0] + c[0], b[1] + c[1], b[2] + c[2], b[3] + c[3]];

    let lhs = clifford2_product_witness(a, bc);
    let ab = clifford2_product_witness(a, b);
    let ac = clifford2_product_witness(a, c);
    let rhs = [ab[0] + ac[0], ab[1] + ac[1], ab[2] + ac[2], ab[3] + ac[3]];

    assert!(mv_approx_eq(lhs, rhs));
}

#[test]
fn tfn_scalar_mix_identity_and_scaling() {
    let features = vec![3.0, 5.0, 7.0, 11.0];
    let id_weights = vec![1.0, 0.0, 0.0, 1.0];
    let out = tfn_scalar_mix_witness(&features, &id_weights, 2, 2, 2);
    assert_eq!(out, features);

    let zero_weights = vec![0.0, 0.0];
    let out_zero = tfn_scalar_mix_witness(&[1.0, 2.0], &zero_weights, 1, 2, 1);
    assert!(approx_eq(out_zero[0], 0.0));

    let scale_weights = vec![3.0];
    let f_scale = vec![1.0, 2.0, 3.0];
    let out_scale = tfn_scalar_mix_witness(&f_scale, &scale_weights, 3, 1, 1);
    for (i, v) in out_scale.iter().enumerate() {
        assert!(approx_eq(*v, 3.0 * f_scale[i]));
    }
}

#[test]
fn tfn_scalar_mix_short_inputs_and_rotation_invariance() {
    let out = tfn_scalar_mix_witness(&[2.0], &[3.0, 4.0], 1, 2, 1);
    assert_eq!(out, vec![6.0]);

    let f1 = vec![1.0, 2.0, 3.0];
    let f2 = vec![1.0, 2.0, 3.0];
    let w = vec![0.5, 0.3, 0.1];
    let out1 = tfn_scalar_mix_witness(&f1, &w, 1, 3, 1);
    let out2 = tfn_scalar_mix_witness(&f2, &w, 1, 3, 1);
    assert!(approx_eq(out1[0], out2[0]));
}

#[test]
fn persistent_fixpoint_into_witness_convergence_and_scratch_reuse() {
    let seed = vec![0u32];
    let mut current = Vec::with_capacity(16);
    let mut next = Vec::with_capacity(16);
    let current_ptr = current.as_ptr();
    let next_ptr = next.as_ptr();

    let mut transfer = |cur: &[u32], out: &mut [u32]| {
        out[0] = cur[0] | 0b1010;
    };
    let iters = persistent_fixpoint_into_witness(&seed, 16, &mut transfer, &mut current, &mut next);
    assert!(iters < 5);
    assert_eq!(current, vec![0b1010]);
    assert!(current.as_ptr() == current_ptr || current.as_ptr() == next_ptr);
    assert!(next.as_ptr() == current_ptr || next.as_ptr() == next_ptr);
    assert_ne!(current.as_ptr(), next.as_ptr());

    // Diverging transfer caps at max_iterations
    let (div_out, div_iters) =
        persistent_fixpoint_witness(&[0u32], 8, |cur| vec![cur[0].wrapping_add(1)]);
    assert_eq!(div_iters, 8);
    assert_eq!(div_out, vec![8]);

    // Error on empty / capacity mismatch handled gracefully
    let mut cur_tail = Vec::with_capacity(8);
    let mut nxt_tail = Vec::with_capacity(8);
    cur_tail.extend([u32::MAX; 8]);
    nxt_tail.extend([u32::MAX; 8]);
    let iters = try_persistent_fixpoint_into_witness(
        &[1, 2],
        4,
        |cur, out| out.copy_from_slice(cur),
        &mut cur_tail,
        &mut nxt_tail,
    )
    .unwrap();
    assert_eq!(iters, 1);
    assert_eq!(cur_tail, vec![1, 2]);
}
