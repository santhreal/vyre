//! Tier 3 - Parity: drives the ACTUAL iterative Sinkhorn matrix-scaling IR (`math::sinkhorn_iterate`)
//! through `reference_eval` and asserts BIT-EXACT equality against the shipped fixed-point oracle
//! `sinkhorn_iterate::cpu_ref`. The op had NO `reference_eval` test.
//!
//! Unlike amg (f64 oracle -> tolerance), sinkhorn's `cpu_ref` returns FIXED-POINT u32 `(u, v, iters)`,
//! so parity is EXACT: the IR and the oracle run the identical 16.16 fixed-point iteration
//! (Kv = K·v ; u = a./Kv ; Ktu = Kᵀ·u ; v = b./Ktu) inside the same `persistent_fixpoint` convergence
//! loop (ping-pong u_curr/u_next, per-word `changed` flag, `max_iterations` cap). Because both use the
//! same fixed-point arithmetic they reach the `changed == 0` fixpoint at the SAME iteration, so the
//! whole data-dependent loop is deterministic and must agree bit-for-bit. `persistent_fixpoint` copies
//! `next -> current` every step, so the final `u` is always in `u_curr` (binding 0). The updates are
//! STATIC-index (i/j loops, not data-derived scatter), the class `reference_eval` executes faithfully.
//!
//! A wrong gemm index, a swapped Kv/Ktu, a mis-ordered ping-pong, a broken `a./Kv` fixed-point divide,
//! or an off-by-one on the convergence check makes the exact comparison fail.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use vyre_reference::value::Value;

use vyre_primitives::math::sinkhorn_iterate::{cpu_ref, sinkhorn_iterate};

/// 16.16 fixed-point encode.
fn enc(v: f64) -> u32 {
    (v * 65536.0).round() as i32 as u32
}
fn enc_vec(v: &[f64]) -> Vec<u32> {
    v.iter().copied().map(enc).collect()
}

fn words(v: &Value) -> Vec<u32> {
    v.to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Returns (u_final, v_final) raw fixed-point words from the IR.
#[allow(clippy::too_many_arguments)]
fn run_ir(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_init: &[u32],
    v_init: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
) -> (Vec<u32>, Vec<u32>) {
    let program = sinkhorn_iterate(
        "u_curr",
        "u_next",
        "changed",
        "k",
        "k_t",
        "a",
        "b",
        "v",
        "kv",
        "ktu",
        m,
        n,
        max_iterations,
    );
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let (mm, nn) = (m as usize, n as usize);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(u_init),          // u_curr (0, RW) <- final u
            pack(&vec![0u32; mm]), // u_next (1, RW)
            pack(&[0u32]),         // changed (2, RW)
            pack(k),               // k (3, RO)
            pack(k_t),             // k_t (4, RO)
            pack(a),               // a (5, RO)
            pack(b),               // b (6, RO)
            pack(v_init),          // v (7, RW) <- final v
            pack(&vec![0u32; mm]), // kv (8, RW)
            pack(&vec![0u32; nn]), // ktu (9, RW)
        ],
    )
    .expect("sinkhorn_iterate reference evaluation must succeed");
    // RW buffers in binding order: u_curr(0), u_next(1), changed(2), v(7), kv(8), ktu(9).
    let u = words(&outputs[0]);
    let v = words(&outputs[3]);
    (u, v)
}

#[test]
fn sinkhorn_iterate_ir_matches_fixed_point_oracle() {
    let m = 2u32;
    let n = 2u32;
    let max_iterations = 8u32;

    // Symmetric positive kernel K (so Kᵀ = K), positive marginals, unit init. Small magnitudes keep
    // every 16.16 product/quotient in range and Kv/Ktu strictly positive (no divide-by-zero).
    let k = enc_vec(&[1.0, 0.5, 0.5, 1.0]);
    let k_t = enc_vec(&[1.0, 0.5, 0.5, 1.0]);
    let a = enc_vec(&[1.0, 1.0]);
    let b = enc_vec(&[1.0, 1.0]);
    let u_init = enc_vec(&[1.0, 1.0]);
    let v_init = enc_vec(&[1.0, 1.0]);

    let (u_ir, v_ir) = run_ir(&k, &k_t, &a, &b, &u_init, &v_init, m, n, max_iterations);
    let (u_ref, v_ref, iters) = cpu_ref(&k, &k_t, &a, &b, &u_init, &v_init, m, n, max_iterations);

    // Non-vacuous: the iteration must actually move off the unit init and do real work.
    assert!(
        iters >= 1,
        "oracle must run at least one iteration, got {iters}"
    );
    assert_ne!(u_ref, u_init, "solution must differ from the unit init");

    assert_eq!(
        u_ir, u_ref,
        "u diverged: IR={u_ir:?} oracle={u_ref:?} (iters={iters})"
    );
    assert_eq!(
        v_ir, v_ref,
        "v diverged: IR={v_ir:?} oracle={v_ref:?} (iters={iters})"
    );
}

#[test]
fn sinkhorn_iterate_ir_matches_oracle_asymmetric() {
    // Asymmetric marginals + a non-symmetric K (k_t is the true transpose) exercise the Kᵀ path.
    let m = 2u32;
    let n = 2u32;
    let k = enc_vec(&[1.0, 0.25, 0.75, 0.5]); // row-major 2x2
    let k_t = enc_vec(&[1.0, 0.75, 0.25, 0.5]); // transpose
    let a = enc_vec(&[2.0, 1.0]);
    let b = enc_vec(&[1.0, 2.0]);
    let u_init = enc_vec(&[1.0, 1.0]);
    let v_init = enc_vec(&[1.0, 1.0]);

    let (u_ir, v_ir) = run_ir(&k, &k_t, &a, &b, &u_init, &v_init, m, n, 12);
    let (u_ref, v_ref, iters) = cpu_ref(&k, &k_t, &a, &b, &u_init, &v_init, m, n, 12);

    assert!(iters >= 1);
    assert_eq!(u_ir, u_ref, "asymmetric u diverged (iters={iters})");
    assert_eq!(v_ir, v_ref, "asymmetric v diverged (iters={iters})");
}
