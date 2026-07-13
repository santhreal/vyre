//! End-to-end parity for `math::natural_gradient_autotuner::precondition_autotune_gradient_fixed_via`.
//!
//! The dispatched kernel is `math::natural_gradient_block_apply`: the natural-gradient precondition
//! matvec `grad_nat[i] = Σ_j M_inv_sqrt[i*n+j] · grad[j]` in 16.16 fixed point. It had NO IR-execution
//! coverage: `rg -l natural_gradient vyre-primitives/tests/` = zero files, and its only self-substrate
//! consumer (`precondition_autotune_gradient_fixed_via`) was exercised solely by a
//! `NaturalGradientDispatcher` mock that IGNORES the `_program` IR and hand-computes the matvec, so
//! the real kernel never ran (the mock-dispatcher-coherence gap; see the SWEEP-self-substrate row in
//! BACKLOG.md). Its only host oracle (`natural_gradient_block_apply_cpu`) is f64, giving no exact
//! reference for the u32 fixed-point dispatch path.
//!
//! This runs the real fixed-point `natural_gradient_block_apply` Program through the shared
//! `ReferenceEvalDispatcher` and asserts it EXACTLY (no tolerance) reproduces a u32 16.16 matvec
//! oracle. The oracle mirrors the IR bit-for-bit: `fixed_mul_16_16(a, b) =
//! ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32` (matching the corrected SIGNED
//! `fixed_mul_16_16_expr` = bits 16..47 of the SIGNED 64-bit product), accumulated with wrapping u32
//! addition (`Expr::add` = GPU u32 add semantics). Because both sides use identical integer
//! arithmetic, any divergence is a real IR/dispatch defect, not a rounding artifact.
#![forbid(unsafe_code)]

use vyre_self_substrate::math::natural_gradient_autotuner::precondition_autotune_gradient_fixed_via;

mod common;
use common::fixed_mul as fixed_mul_16_16;
use common::ReferenceEvalDispatcher;

/// Exact u32 16.16 matvec oracle mirroring `natural_gradient_block_apply`: per output row `i`,
/// `acc = 0; for j: acc = acc.wrapping_add(fixed_mul_16_16(M[i*n+j], grad[j]))`.
fn natural_gradient_fixed(m_inv_sqrt: &[u32], grad: &[u32], n: usize) -> Vec<u32> {
    (0..n)
        .map(|i| {
            let mut acc = 0u32;
            for j in 0..n {
                acc = acc.wrapping_add(fixed_mul_16_16(m_inv_sqrt[i * n + j], grad[j]));
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
fn precondition_autotune_gradient_fixed_via_matches_exact_fixed_point_matvec() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x9E37_79B9u32;
    let mut moved_cases = 0u32;
    for case in 0..400u32 {
        let n = 1 + xorshift(&mut state) % 8; // 1..=8 gradient dimensions
        let cells = (n * n) as usize;
        // 16.16 values up to ~16.0 (20-bit magnitude): products stay within u64; n-term sums
        // occasionally wrap u32, which BOTH the IR and the oracle do identically, so exact equality
        // still holds.
        let m_inv_sqrt: Vec<u32> = (0..cells)
            .map(|_| xorshift(&mut state) & 0x000F_FFFF)
            .collect();
        let grad: Vec<u32> = (0..n).map(|_| xorshift(&mut state) & 0x000F_FFFF).collect();

        let via = precondition_autotune_gradient_fixed_via(&dispatcher, &m_inv_sqrt, &grad, n)
            .expect("precondition_autotune_gradient_fixed_via must dispatch the matvec kernel");
        let oracle = natural_gradient_fixed(&m_inv_sqrt, &grad, n as usize);
        if oracle.iter().any(|&w| w != 0) {
            moved_cases += 1;
        }
        assert_eq!(
            via, oracle,
            "case {case} (n={n}): natural-gradient _via {via:?} != exact fixed-point oracle \
             {oracle:?} (m_inv_sqrt={m_inv_sqrt:?}, grad={grad:?})"
        );
    }
    assert!(
        moved_cases > 380,
        "only {moved_cases}/400 preconditions were non-zero, the matvec is not being exercised"
    );
}

/// A signed 16.16 value in roughly `[-8.0, 8.0)`: a 19-bit magnitude, optionally negated (top bit
/// set on the negative half (the operand class the old UNSIGNED high-word multiply corrupted)).
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
fn precondition_autotune_gradient_fixed_via_matches_signed_precondition_with_negative_gradients() {
    // A natural gradient `M_inv_sqrt · ∇L` is inherently SIGNED: any component of the loss gradient
    // `grad` can be negative, and an inverse-curvature preconditioner carries negative off-diagonal
    // coupling. The base sweep masks every value to `& 0x000F_FFFF` (top bit clear → all
    // non-negative), so it never sent a single negative operand through the fixed-point matvec. This
    // sweep feeds SIGNED M and grad and asserts the dispatched kernel bit-exactly matches the SIGNED
    // oracle, locking the signed `fixed_mul_16_16_expr` fix at the natural-gradient consumer (pre-fix
    // a negative `M[i,j]·grad[j]` term diverged: the unsigned high word treated a negative operand as
    // ~2^32 and produced garbage).
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0xC2B2_AE35u32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved_cases = 0u32;
    for case in 0..400u32 {
        let n = 1 + xorshift(&mut state) % 8; // 1..=8 gradient dimensions
        let cells = (n * n) as usize;
        let m_inv_sqrt: Vec<u32> = (0..cells).map(|_| signed_fixed(&mut state)).collect();
        let grad: Vec<u32> = (0..n).map(|_| signed_fixed(&mut state)).collect();

        neg_inputs += m_inv_sqrt.iter().filter(|&&v| (v as i32) < 0).count() as u32;
        neg_inputs += grad.iter().filter(|&&v| (v as i32) < 0).count() as u32;

        let via = precondition_autotune_gradient_fixed_via(&dispatcher, &m_inv_sqrt, &grad, n)
            .expect("precondition_autotune_gradient_fixed_via must dispatch the matvec kernel");
        let oracle = natural_gradient_fixed(&m_inv_sqrt, &grad, n as usize);
        assert_eq!(
            via, oracle,
            "case {case} (n={n}): SIGNED natural-gradient _via {via:?} != exact signed fixed-point \
             oracle {oracle:?} (m_inv_sqrt={m_inv_sqrt:?}, grad={grad:?})"
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
        "signed preconditions must produce negative natural-gradient entries, got {neg_outputs}"
    );
    assert!(
        moved_cases > 380,
        "only {moved_cases}/400 preconditions were non-zero, the matvec is not being exercised"
    );
}

#[test]
fn precondition_autotune_gradient_fixed_via_hand_checked_negative_precondition() {
    // M_inv_sqrt = [[2.0, -0.5], [0.0, 1.0]], grad = [-3.0, 4.0]:
    //   out[0] = (2.0)(-3.0) + (-0.5)(4.0) = -6.0 - 2.0 = -8.0
    //   out[1] = (0.0)(-3.0) + ( 1.0)(4.0) =  0.0 + 4.0 =  4.0
    let dispatcher = ReferenceEvalDispatcher;
    let m_inv_sqrt = vec![to_fixed(2.0), to_fixed(-0.5), to_fixed(0.0), to_fixed(1.0)];
    let grad = vec![to_fixed(-3.0), to_fixed(4.0)];
    let via = precondition_autotune_gradient_fixed_via(&dispatcher, &m_inv_sqrt, &grad, 2)
        .expect("precondition_autotune_gradient_fixed_via must dispatch");
    let oracle = natural_gradient_fixed(&m_inv_sqrt, &grad, 2);
    assert_eq!(
        oracle,
        vec![to_fixed(-8.0), to_fixed(4.0)],
        "sanity: signed fixed-point matvec = [-8.0, 4.0]"
    );
    assert_eq!(
        via, oracle,
        "the dispatched precondition kernel must preserve sign: [-8.0, 4.0]"
    );
}

#[test]
fn precondition_autotune_gradient_fixed_via_computes_a_known_precondition() {
    // 2x2 M_inv_sqrt = [[2.0, 0.0],[1.0, 3.0]] in 16.16; grad = [3.0, 4.0].
    // out[0] = 2.0*3.0 + 0.0*4.0 = 6.0; out[1] = 1.0*3.0 + 3.0*4.0 = 15.0.
    let dispatcher = ReferenceEvalDispatcher;
    let one = 1u32 << 16;
    let m_inv_sqrt = vec![2 * one, 0, one, 3 * one];
    let grad = vec![3 * one, 4 * one];
    let via = precondition_autotune_gradient_fixed_via(&dispatcher, &m_inv_sqrt, &grad, 2)
        .expect("precondition_autotune_gradient_fixed_via must dispatch");
    let oracle = natural_gradient_fixed(&m_inv_sqrt, &grad, 2);
    assert_eq!(
        oracle,
        vec![6 * one, 15 * one],
        "sanity: exact fixed-point matvec = [6.0, 15.0]"
    );
    assert_eq!(
        via, oracle,
        "the dispatched precondition kernel must equal the exact fixed-point matvec"
    );
}
