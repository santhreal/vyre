//! GPU borrow checker: the NLL loan-liveness verdict computed on a device via
//! batched reachability, producing the identical [`Conflict`] set as the CPU
//! [`analyze`](super::analyze) engine.
//!
//! The CPU engine's two monotone bitset dataflows are equivalent to per-loan
//! reachability: a loan's issue reaches a point iff the point lies in the
//! forward closure of `{issue}` over the CFG; a use is reachable from a point
//! iff the point lies in the forward closure of the loan's uses over the
//! reversed CFG; the loan is live where both hold. Two loans of one place
//! conflict when one is live at the other's issue and at least one is mutable.
//!
//! [`analyze_batched`] computes every loan's forward closure in one device
//! dispatch and every loan's backward closure in a second: two dispatches per
//! function regardless of loan count, so launch overhead is amortized across
//! the whole loan set (the path that makes a crate-scale borrow check
//! GPU-fast). It is backend-agnostic: the caller supplies an
//! [`OptimizerDispatcher`], which on the fleet is the CUDA backend.

use vyre_self_substrate::optimizer::dispatcher::{DispatchError, OptimizerDispatcher};
use vyre_self_substrate::persistent_bfs::{
    bfs_expand_resident_graph_batch_with_scratch_into, upload_resident_bfs_graph,
    PersistentBfsResidentScratch,
};

use super::{BorrowFacts, Conflict, ConflictKind, LoanKind};

/// All edge kinds traversed (the CFG is unlabeled for borrow liveness).
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

/// Test `bit` in query `q`'s slice of a flattened `query_count * words` bitset.
fn slice_test_bit(flat: &[u32], words: usize, q: usize, bit: u32) -> bool {
    test_bit(&flat[q * words..(q + 1) * words], bit)
}

/// Build a CSR adjacency from CFG edges. `reverse` swaps each edge so a closure
/// walks predecessors instead of successors.
fn build_csr(n: u32, edges: &[(u32, u32)], reverse: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = n as usize;
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        let (from, to) = if reverse { (b, a) } else { (a, b) };
        if (from as usize) < n && (to as usize) < n {
            adjacency[from as usize].push(to);
        }
    }
    let mut offsets = Vec::with_capacity(n + 1);
    let mut targets = Vec::new();
    offsets.push(0u32);
    for successors in &adjacency {
        targets.extend_from_slice(successors);
        offsets.push(targets.len() as u32);
    }
    let masks = vec![1u32; targets.len()];
    (offsets, targets, masks)
}

/// Compute every conflicting-borrow violation (rustc E0499 / E0502) for one
/// function on a GPU device, batched across all loans.
///
/// Returns exactly the conflicts the CPU [`analyze`](super::analyze) engine
/// returns, in the same order: a per-`Conflict` parity holds by construction
/// (same live-range definition, same pairing).
///
/// # Errors
///
/// Returns [`DispatchError`] if a device upload or batch dispatch fails.
pub fn analyze_batched(
    dispatcher: &dyn OptimizerDispatcher,
    facts: &BorrowFacts,
) -> Result<Vec<Conflict>, DispatchError> {
    let n = facts.point_count;
    let loans = facts.loan_count();
    if loans < 2 || n == 0 {
        return Ok(Vec::new());
    }
    let words = bitset_words(n);
    // A monotone closure over n points reaches its fixpoint within n steps; the
    // change-flag kernel also stops early once stable.
    let max_iters = n.max(1);

    let (fwd_off, fwd_tgt, fwd_msk) = build_csr(n, &facts.cfg_edges, false);
    let (rev_off, rev_tgt, rev_msk) = build_csr(n, &facts.cfg_edges, true);
    let fwd_graph = upload_resident_bfs_graph(dispatcher, n, &fwd_off, &fwd_tgt, &fwd_msk)?;
    let rev_graph = upload_resident_bfs_graph(dispatcher, n, &rev_off, &rev_tgt, &rev_msk)?;

    // Per-loan seeds, flattened `loans * words`: issue point forward, uses backward.
    let mut issue_seeds = vec![0u32; loans * words];
    let mut use_seeds = vec![0u32; loans * words];
    for a in 0..loans {
        set_bit(&mut issue_seeds[a * words..(a + 1) * words], facts.loan_issued_at[a]);
    }
    for &(loan, point) in &facts.loan_used_at {
        let a = loan as usize;
        if a < loans {
            set_bit(&mut use_seeds[a * words..(a + 1) * words], point);
        }
    }

    let mut scratch = PersistentBfsResidentScratch::default();
    let mut forward = Vec::new();
    let mut forward_changed = Vec::new();
    bfs_expand_resident_graph_batch_with_scratch_into(
        dispatcher,
        &fwd_graph,
        &issue_seeds,
        loans,
        ALLOW_ALL,
        max_iters,
        &mut scratch,
        &mut forward,
        &mut forward_changed,
    )?;
    let mut backward = Vec::new();
    let mut backward_changed = Vec::new();
    bfs_expand_resident_graph_batch_with_scratch_into(
        dispatcher,
        &rev_graph,
        &use_seeds,
        loans,
        ALLOW_ALL,
        max_iters,
        &mut scratch,
        &mut backward,
        &mut backward_changed,
    )?;

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
            // `a` is live at `issue_b` iff issue_a reaches issue_b and a use of
            // `a` is reachable from there: forward[a] & backward[a] at issue_b.
            let a_live_at_b = slice_test_bit(&forward, words, a, issue_b)
                && slice_test_bit(&backward, words, a, issue_b);
            let b_live_at_a = slice_test_bit(&forward, words, b, issue_a)
                && slice_test_bit(&backward, words, b, issue_a);
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
    Ok(conflicts)
}
