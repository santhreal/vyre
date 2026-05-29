//! GPU borrow checker on CUDA: the NLL loan-liveness reachability that the
//! front-end-agnostic engine (`vyre_libs::borrowck::analyze`) computes on the
//! CPU, evaluated instead on the CUDA device via the iterated-closure graph
//! primitive, and parity-gated against the CPU engine across the full
//! control-flow-correctness suite.
//!
//! The engine's two monotone bitset dataflows are equivalent to per-loan
//! reachability:
//!   * a loan's issue reaches a point  <=>  the point is in the forward closure
//!     of {issue} over the CFG;
//!   * a use of the loan is reachable from a point  <=>  the point is in the
//!     forward closure of the loan's use points over the REVERSED CFG.
//! A loan is live at a point iff the point is in both closures, and two loans
//! of one place conflict when one is live at the other's issue and at least one
//! is mutable. Computing both closures on the GPU and pairing on the host must
//! reproduce the CPU engine's verdict exactly. The speedup over the CPU engine
//! comes at scale by batching every loan's closures into one device dispatch;
//! this test fixes correctness first.

#![cfg(test)]

mod common;

use common::with_cuda_optimizer_dispatcher;
use vyre_libs::borrowck::{analyze, BorrowFacts, Conflict, ConflictKind, LoanKind};
use vyre_self_substrate::csr_forward_or_changed::forward_closure_via_change_flag_gpu;
use vyre_self_substrate::optimizer::dispatcher::OptimizerDispatcher;

const ALLOW_ALL: u32 = 0xFFFF_FFFF;

fn bitset_words(n: u32) -> usize {
    (n as usize).div_ceil(32)
}

fn set_bit(words: &mut [u32], bit: u32) {
    words[(bit / 32) as usize] |= 1u32 << (bit % 32);
}

fn test_bit(words: &[u32], bit: u32) -> bool {
    (words[(bit / 32) as usize] >> (bit % 32)) & 1 == 1
}

/// Build a CSR adjacency from CFG edges. `reverse` swaps each edge so the
/// closure walks predecessors instead of successors.
fn build_csr(n: u32, edges: &[(u32, u32)], reverse: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = n as usize;
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        let (from, to) = if reverse { (b, a) } else { (a, b) };
        if (from as usize) < n && (to as usize) < n {
            adj[from as usize].push(to);
        }
    }
    let mut offsets = Vec::with_capacity(n + 1);
    let mut targets = Vec::new();
    offsets.push(0u32);
    for successors in &adj {
        targets.extend_from_slice(successors);
        offsets.push(targets.len() as u32);
    }
    let masks = vec![1u32; targets.len()];
    (offsets, targets, masks)
}

/// The CUDA borrow checker: same conflict construction as the CPU engine, but
/// live ranges computed via GPU iterated closures.
fn cuda_conflicts(dispatcher: &dyn OptimizerDispatcher, facts: &BorrowFacts) -> Vec<Conflict> {
    let n = facts.point_count;
    let loans = facts.loan_count();
    if loans < 2 || n == 0 {
        return Vec::new();
    }
    let words = bitset_words(n);
    // A monotone closure over n points reaches its fixpoint within n steps; the
    // change-flag kernel also stops early once stable.
    let max_iters = n.max(1);

    let (fwd_off, fwd_tgt, fwd_msk) = build_csr(n, &facts.cfg_edges, false);
    let (rev_off, rev_tgt, rev_msk) = build_csr(n, &facts.cfg_edges, true);

    // live[a] = forward-reach({issue_a}) & reverse-reach(uses_a).
    let mut live: Vec<Vec<u32>> = Vec::with_capacity(loans);
    for a in 0..loans {
        let mut issue_seed = vec![0u32; words];
        set_bit(&mut issue_seed, facts.loan_issued_at[a]);
        let forward = forward_closure_via_change_flag_gpu(
            dispatcher, n, &fwd_off, &fwd_tgt, &fwd_msk, &issue_seed, ALLOW_ALL, max_iters,
        )
        .expect("forward closure dispatch must succeed on the CUDA device");

        let mut use_seed = vec![0u32; words];
        for &(loan, point) in &facts.loan_used_at {
            if loan as usize == a {
                set_bit(&mut use_seed, point);
            }
        }
        let backward = forward_closure_via_change_flag_gpu(
            dispatcher, n, &rev_off, &rev_tgt, &rev_msk, &use_seed, ALLOW_ALL, max_iters,
        )
        .expect("backward closure dispatch must succeed on the CUDA device");

        let live_a: Vec<u32> = forward
            .iter()
            .zip(backward.iter())
            .map(|(f, b)| f & b)
            .collect();
        live.push(live_a);
    }

    let mut conflicts = Vec::new();
    for a in 0..loans {
        for b in (a + 1)..loans {
            if facts.loan_place[a] != facts.loan_place[b] {
                continue;
            }
            let a_mut = facts.loan_kind[a] == LoanKind::Mut;
            let b_mut = facts.loan_kind[b] == LoanKind::Mut;
            if !(a_mut || b_mut) {
                continue;
            }
            let issue_a = facts.loan_issued_at[a];
            let issue_b = facts.loan_issued_at[b];
            let a_live_at_b = test_bit(&live[a], issue_b);
            let b_live_at_a = test_bit(&live[b], issue_a);
            if a_live_at_b || b_live_at_a {
                let (first, second) = if issue_a <= issue_b { (a, b) } else { (b, a) };
                conflicts.push(Conflict {
                    first: first as u32,
                    second: second as u32,
                    offset: facts.loan_offset[second],
                    kind: if a_mut && b_mut {
                        ConflictKind::TwoMutable
                    } else {
                        ConflictKind::MutableAndShared
                    },
                });
            }
        }
    }
    conflicts
}

fn facts(
    point_count: u32,
    cfg: &[(u32, u32)],
    loans: &[(u32, LoanKind, u32, u32)],
    uses: &[(u32, u32)],
) -> BorrowFacts {
    BorrowFacts {
        point_count,
        cfg_edges: cfg.to_vec(),
        loan_place: loans.iter().map(|l| l.0).collect(),
        loan_kind: loans.iter().map(|l| l.1).collect(),
        loan_issued_at: loans.iter().map(|l| l.2).collect(),
        loan_offset: loans.iter().map(|l| l.3).collect(),
        loan_used_at: uses.to_vec(),
    }
}

/// The full control-flow-correctness suite, mirroring the CPU engine tests.
/// Each case asserts the CUDA verdict equals the CPU engine verdict exactly.
fn corpus() -> Vec<(&'static str, BorrowFacts)> {
    vec![
        (
            "straight_line_two_mutable_conflict",
            facts(
                3,
                &[(0, 1), (1, 2)],
                &[(0, LoanKind::Mut, 0, 10), (0, LoanKind::Mut, 1, 20)],
                &[(0, 2), (1, 2)],
            ),
        ),
        (
            "straight_line_mutable_and_shared_conflict",
            facts(
                3,
                &[(0, 1), (1, 2)],
                &[(0, LoanKind::Shared, 0, 10), (0, LoanKind::Mut, 1, 20)],
                &[(0, 2), (1, 2)],
            ),
        ),
        (
            "two_shared_do_not_conflict",
            facts(
                3,
                &[(0, 1), (1, 2)],
                &[(0, LoanKind::Shared, 0, 10), (0, LoanKind::Shared, 1, 20)],
                &[(0, 2), (1, 2)],
            ),
        ),
        (
            "unused_first_mutable_is_dead",
            facts(
                3,
                &[(0, 1), (1, 2)],
                &[(0, LoanKind::Mut, 0, 10), (0, LoanKind::Mut, 1, 20)],
                &[(1, 2)],
            ),
        ),
        (
            "sequential_non_overlapping",
            facts(
                4,
                &[(0, 1), (1, 2), (2, 3)],
                &[(0, LoanKind::Mut, 0, 10), (0, LoanKind::Mut, 2, 20)],
                &[(0, 1), (1, 3)],
            ),
        ),
        (
            "borrows_live_across_branch_conflict",
            facts(
                6,
                &[(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)],
                &[(0, LoanKind::Mut, 0, 10), (0, LoanKind::Mut, 1, 20)],
                &[(0, 3), (1, 4)],
            ),
        ),
        (
            "borrows_in_exclusive_branches_no_conflict",
            facts(
                6,
                &[(0, 1), (1, 2), (2, 5), (0, 3), (3, 4), (4, 5)],
                &[(0, LoanKind::Mut, 1, 10), (0, LoanKind::Mut, 3, 20)],
                &[(0, 2), (1, 4)],
            ),
        ),
    ]
}

#[test]
fn cuda_borrow_checker_matches_cpu_engine_across_cfg_suite() {
    let cases = corpus();
    with_cuda_optimizer_dispatcher("cuda borrowck reachability", |dispatcher| {
        for (label, f) in &cases {
            let cpu = analyze(f);
            let gpu = cuda_conflicts(dispatcher, f);
            assert_eq!(
                gpu, cpu,
                "{label}: CUDA borrow-check verdict diverged from the CPU engine"
            );
        }
    });
}

#[test]
fn cuda_borrow_checker_scales_to_a_long_chain() {
    // 2000-point straight chain; two &mut of one place issued early, both used
    // at the end -> live across the whole chain -> exactly one conflict, the
    // same verdict the CPU engine reaches, computed via device closures.
    let n = 2000u32;
    let cfg: Vec<(u32, u32)> = (0..n - 1).map(|i| (i, i + 1)).collect();
    let f = facts(
        n,
        &cfg,
        &[(0, LoanKind::Mut, 0, 10), (0, LoanKind::Mut, 1, 20)],
        &[(0, n - 1), (1, n - 1)],
    );
    let cpu = analyze(&f);
    let gpu = with_cuda_optimizer_dispatcher("cuda borrowck chain", |dispatcher| {
        cuda_conflicts(dispatcher, &f)
    });
    assert_eq!(gpu, cpu, "CUDA long-chain verdict diverged from the CPU engine");
    assert_eq!(gpu.len(), 1, "expected exactly one conflict, got {gpu:?}");
    assert_eq!(gpu[0].kind, ConflictKind::TwoMutable);
}
