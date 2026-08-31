//! End-to-end parity for `analysis::dataflow_fixpoint::semiring_gemm_via_*` through the shared
//! faithful [`vyre_driver_reference::ReferenceSemanticExecutor`], across all three semirings the consumer exposes
//! (boolean-OR reachability, min-plus shortest-path, lineage/provenance).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! `semiring_gemm`'s IR (built on the shared `u32_matmul_program` per-output-cell kernel) is run by
//! NO `vyre-primitives/tests/*` file and the consumer's only coverage is its own in-file dispatcher,
//! so this is the FIRST-EVER execution of the semiring-GEMM kernel through a dispatch boundary that
//! models the real backend, for each semiring's distinct combine/accumulate lowering.
//!
//! `semiring_gemm` binds `a` and `b` as read-only inputs and `c` as a
//! backend-allocated output. The consumer passes exactly two inputs and decodes
//! the sole writable buffer at `outputs[0]`. The kernel is per-output-cell, so
//! Every semiring operation is exact integer or bitwise arithmetic, so the oracle is bit-exact.
//!
//! MinPlus note: the IR's finite-operand combine is `Expr::add` (u32 wrapping) with a MAX-guard, while
//! the CPU oracle uses `saturating_add`; the two agree exactly as long as no finite `a+b` overflows
//! u32. The generated systems bound finite entries so every `a+b` stays well under u32::MAX, and inject
//! `u32::MAX` (∞ / no-edge) to exercise the guarded branch and the `min` accumulate against ∞.

mod bounded_compile_policy;

use vyre_libs::analysis::dataflow_fixpoint::{
    semiring_gemm_via_bool_or, semiring_gemm_via_lineage, semiring_gemm_via_min_plus,
};
use vyre_libs::math::semiring_gemm::semiring_gemm;
use vyre_libs::math::semiring_gemm::Semiring;
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_reference::composition_witness::semiring_gemm_witness;
use vyre_test_support::fixed_point::xorshift32 as xorshift;

const INF: u32 = u32::MAX;

fn reference_semiring_gemm(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Vec<u32> {
    semiring_gemm_witness(a, b, m as usize, n as usize, k as usize, semiring)
}

#[test]
fn bool_or_gemm_via_matches_cpu_reachability_matmul() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0xB0_01_A0_01u32;
    let mut nontrivial = 0u32;
    for case in 0..300u32 {
        let m = 1 + (case % 5);
        let k = 1 + ((case / 5) % 5);
        let n = 1 + ((case / 25) % 5);
        // boolean 0/1 entries → classic reachability matmul; ~50% density exercises real OR-of-ANDs.
        let a: Vec<u32> = (0..m * k).map(|_| xorshift(&mut state) & 1).collect();
        let b: Vec<u32> = (0..k * n).map(|_| xorshift(&mut state) & 1).collect();

        let got = semiring_gemm_via_bool_or(
            &dispatcher,
            &bounded_compile_policy::policy(),
            &a,
            &b,
            m,
            n,
            k,
        )
        .expect("semiring_gemm_via_bool_or must dispatch the reachability matmul");
        let want = reference_semiring_gemm(&a, &b, m, n, k, Semiring::BoolOr);
        assert_eq!(
            got, want,
            "case {case}: bool-or GEMM must match the CPU reachability matmul; m={m} k={k} n={n} a={a:?} b={b:?}"
        );
        // A cell that is 1 only because some intermediate k both-connects exercises the OR accumulate.
        if k > 1 && got.iter().any(|&c| c == 1) && got.iter().any(|&c| c == 0) {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 100,
        "expected >100 mixed 0/1 reachability results, got {nontrivial}"
    );
}

#[test]
fn min_plus_gemm_via_matches_cpu_shortest_path_matmul() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0x5A_FE_11_01u32;
    let mut hit_inf_combine = 0u32;
    let mut real_relaxation = 0u32;
    for case in 0..300u32 {
        let m = 1 + (case % 5);
        let k = 1 + ((case / 5) % 5);
        let n = 1 + ((case / 25) % 5);
        // Finite weights in [0, 100_000) so any single a+b <= ~200_000 never overflows u32; ~1/6 of
        // entries are ∞ (u32::MAX = no edge) to drive the MAX-guarded combine and min-against-∞.
        let gen = |state: &mut u32| {
            let r = xorshift(state);
            if r % 6 == 0 {
                INF
            } else {
                r % 100_000
            }
        };
        let a: Vec<u32> = (0..m * k).map(|_| gen(&mut state)).collect();
        let b: Vec<u32> = (0..k * n).map(|_| gen(&mut state)).collect();

        let got = semiring_gemm_via_min_plus(
            &dispatcher,
            &bounded_compile_policy::policy(),
            &a,
            &b,
            m,
            n,
            k,
        )
        .expect("semiring_gemm_via_min_plus must dispatch the shortest-path matmul");
        let want = reference_semiring_gemm(&a, &b, m, n, k, Semiring::MinPlus);
        assert_eq!(
            got, want,
            "case {case}: min-plus GEMM must match the CPU shortest-path matmul; m={m} k={k} n={n} a={a:?} b={b:?}"
        );
        if a.contains(&INF) || b.contains(&INF) {
            hit_inf_combine += 1;
        }
        // A finite path that is strictly shorter than ∞ (a real relaxation) exercises the min fold.
        if got.iter().any(|&c| c != INF) {
            real_relaxation += 1;
        }
    }
    assert!(
        hit_inf_combine > 100 && real_relaxation > 100,
        "min-plus sweep must exercise both ∞ edges and real relaxations: inf={hit_inf_combine} finite={real_relaxation}"
    );
}

#[test]
fn lineage_gemm_via_matches_cpu_provenance_matmul() {
    let dispatcher = ReferenceSemanticExecutor;
    let mut state = 0x11_5E_A0_01u32;
    let mut zero_guard_hits = 0u32;
    for case in 0..300u32 {
        let m = 1 + (case % 5);
        let k = 1 + ((case / 5) % 5);
        let n = 1 + ((case / 25) % 5);
        // Arbitrary provenance bitmasks with ~1/4 zeros → drives the `either_zero → 0` combine guard
        // as well as the bit-OR combine/accumulate on nonzero pairs.
        let gen = |state: &mut u32| {
            let r = xorshift(state);
            if r % 4 == 0 {
                0
            } else {
                r & 0xFFFF
            }
        };
        let a: Vec<u32> = (0..m * k).map(|_| gen(&mut state)).collect();
        let b: Vec<u32> = (0..k * n).map(|_| gen(&mut state)).collect();

        let got = semiring_gemm_via_lineage(
            &dispatcher,
            &bounded_compile_policy::policy(),
            &a,
            &b,
            m,
            n,
            k,
        )
        .expect("semiring_gemm_via_lineage must dispatch the provenance matmul");
        let want = reference_semiring_gemm(&a, &b, m, n, k, Semiring::Lineage);
        assert_eq!(
            got, want,
            "case {case}: lineage GEMM must match the CPU provenance matmul; m={m} k={k} n={n} a={a:?} b={b:?}"
        );
        if a.contains(&0) || b.contains(&0) {
            zero_guard_hits += 1;
        }
    }
    assert!(
        zero_guard_hits > 100,
        "lineage sweep must exercise the either-zero combine guard, got {zero_guard_hits}"
    );
}

/// WHY: max-plus and max-times share max accumulation but must retain distinct combines.
#[test]
fn max_semirings_execute_their_declared_algebra() {
    for (semiring, expected) in [(Semiring::MaxPlus, 9), (Semiring::MaxTimes, 20)] {
        let outputs = vyre_reference::reference_eval(
            &semiring_gemm("a", "b", "c", 1, 1, 2, semiring),
            &[
                Value::from(pack_u32_slice(&[2, 5])),
                Value::from(pack_u32_slice(&[3, 4])),
            ],
        )
        .expect("max semiring GEMM must evaluate");
        assert_eq!(
            decode_u32_le_bytes_all(&outputs[0].to_bytes()),
            vec![expected],
            "{semiring:?}"
        );
    }
}

#[test]
fn semiring_gemm_via_matches_hand_checked_cases() {
    let d = ReferenceSemanticExecutor;

    // Boolean reachability: A = [[1,0],[0,1]] (identity), B = [[1,1],[0,1]] → A·B = B.
    let got = semiring_gemm_via_bool_or(
        &d,
        &bounded_compile_policy::policy(),
        &[1, 0, 0, 1],
        &[1, 1, 0, 1],
        2,
        2,
        2,
    )
    .unwrap();
    assert_eq!(got, vec![1, 1, 0, 1], "identity · B = B under bool-or");

    // Boolean OR-of-ANDs: A = [[1,1]] (1x2), B = [[0],[1]] (2x1) → c[0,0] = (1&0)|(1&1) = 1.
    let got = semiring_gemm_via_bool_or(
        &d,
        &bounded_compile_policy::policy(),
        &[1, 1],
        &[0, 1],
        1,
        1,
        2,
    )
    .unwrap();
    assert_eq!(got, vec![1], "OR of (1&0),(1&1) = 1");

    // Min-plus shortest path: A = [[1,4]] (1x2), B = [[2],[1]] (2x1) → min(1+2, 4+1) = min(3,5) = 3.
    let got = semiring_gemm_via_min_plus(
        &d,
        &bounded_compile_policy::policy(),
        &[1, 4],
        &[2, 1],
        1,
        1,
        2,
    )
    .unwrap();
    assert_eq!(got, vec![3], "min(1+2, 4+1) = 3");

    // Min-plus with a no-edge (∞): A = [[∞,4]], B = [[2],[1]] → min(∞, 4+1) = 5 (∞+2 stays ∞).
    let got = semiring_gemm_via_min_plus(
        &d,
        &bounded_compile_policy::policy(),
        &[INF, 4],
        &[2, 1],
        1,
        1,
        2,
    )
    .unwrap();
    assert_eq!(got, vec![5], "∞ edge is skipped; finite path 4+1=5 wins");

    // Lineage: A = [[0b01, 0]], B = [[0b10],[0b11]] → term0 has a 0 factor → 0; term1 = 0&anything...
    // term0 = combine(0b01, 0b10) = 0b11 (neither zero); term1 = combine(0, 0b11) = 0 (b factor... a=0) → 0.
    // accumulate = 0b11 | 0 = 0b11.
    let got = semiring_gemm_via_lineage(
        &d,
        &bounded_compile_policy::policy(),
        &[0b01, 0],
        &[0b10, 0b11],
        1,
        1,
        2,
    )
    .unwrap();
    assert_eq!(
        got,
        vec![0b11],
        "lineage ORs the nonzero-pair provenance, zero factor contributes 0"
    );
}
