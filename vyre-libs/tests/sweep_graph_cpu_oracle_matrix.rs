//! Sweep oracle matrix for graph CPU references.
//!
//! One parameterized sweep over generated hostile CSR shapes, covering both
//! layers of the graph stack against independent bitset oracles: the
//! `vyre-primitives` CPU references that own graph semantics, and the
//! `vyre-pass-engine` reference wrappers built on them. The shape stream,
//! the bitset helpers and the successor/closure oracles exist once here and
//! every family draws from them, so no two families can disagree about what a
//! given seed means. CPU reference paths only - no mock dispatchers.

#![forbid(unsafe_code)]

use vyre_driver_reference::ReferenceEvalDispatcher;
use vyre_libs::graph::csr_backward_or_changed;
use vyre_libs::graph::csr_closure_inputs::CsrClosureInputs;
use vyre_libs::graph::csr_forward_or_changed;
use vyre_libs::graph::dispatch::csr_bidirectional::reference_bidirectional_step;
use vyre_libs::graph::dispatch::csr_forward_or_changed::reference_forward_step_with_change_flag;
use vyre_libs::graph::dispatch::exploded::{
    build_ifds_csr_via, reference_build_ifds_csr, reference_canonicalize_csr_within_rows,
};
use vyre_libs::graph::dispatch::persistent_bfs::bfs_expand;
use vyre_libs::graph::exploded::build_cpu_reference;
use vyre_libs::graph::motif::{self, MotifEdge};
use vyre_libs::graph::path_reconstruct;
use vyre_libs::graph::persistent_bfs;

/// Shapes per substrate-wrapper family. The wrappers delegate to the primitive
/// references swept below, so they need breadth, not depth.
const CASES_PER_FAMILY: u64 = 512;
/// Shapes per primitive CSR family. These own the semantics every other layer
/// inherits, so they get the widest sweep.
const PRIMITIVE_CSR_CASES: u64 = 4096;
/// Shapes per primitive batch family (path reconstruction, motif matching).
const PRIMITIVE_BATCH_CASES: u64 = 2048;

#[path = "../../tests/support/csr_sweep/mod.rs"]
mod csr_sweep;

use csr_sweep::Rng;

fn bitset_words(node_count: u32) -> usize {
    node_count.div_ceil(32) as usize
}

/// One generated CSR shape: case index, then the fields of a
/// `csr_sweep::CsrSweepCase`.
type CsrCase = (u64, u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32);

/// Deterministic CSR shape stream for one sweep family.
///
/// `seed` keeps a family's shapes distinct from every other family's; `stride`
/// decorrelates successive cases within it. Both are part of the sweep's
/// coverage: changing either moves the family onto different graphs.
fn csr_cases(group: &str, cases: u64, seed: u64, stride: u64) -> impl Iterator<Item = CsrCase> {
    let shape = csr_sweep::group(group);
    (0..cases).map(move |case| {
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            csr_sweep::generate(shape, seed ^ case.wrapping_mul(stride)).into_parts();
        (
            case, node_count, offsets, targets, masks, frontier, allow_mask,
        )
    })
}

fn bit_is_set(words: &[u32], node: u32) -> bool {
    let word = (node / 32) as usize;
    let bit = 1u32 << (node % 32);
    words.get(word).is_some_and(|value| value & bit != 0)
}

fn oracle_forward_or_changed(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let words = bitset_words(node_count);
    let mut out = frontier.to_vec();
    out.resize(words, 0);
    let mut changed = 0;
    for src in 0..node_count {
        if !bit_is_set(&out, src) {
            continue;
        }
        let start = offsets[src as usize] as usize;
        let end = offsets[src as usize + 1] as usize;
        for edge in start..end {
            if masks[edge] & allow_mask == 0 {
                continue;
            }
            let dst = targets[edge];
            let word = (dst / 32) as usize;
            let bit = 1u32 << (dst % 32);
            let before = out[word];
            out[word] |= bit;
            if out[word] != before {
                changed = 1;
            }
        }
    }
    (out, changed)
}

fn oracle_bidirectional_step(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let words = bitset_words(node_count);
    let mut out = vec![0u32; words];
    for src in 0..node_count {
        let src_word = (src / 32) as usize;
        let src_bit = 1u32 << (src % 32);
        let src_in_frontier = bit_is_set(frontier, src);
        let edge_start = offsets[src as usize] as usize;
        let edge_end = offsets[src as usize + 1] as usize;
        let mut backward_hit = false;
        for edge in edge_start..edge_end {
            if masks[edge] & allow_mask == 0 {
                continue;
            }
            let dst = targets[edge];
            if src_in_frontier && dst < node_count {
                out[(dst / 32) as usize] |= 1u32 << (dst % 32);
            }
            if bit_is_set(frontier, dst) {
                backward_hit = true;
            }
        }
        if backward_hit {
            out[src_word] |= src_bit;
        }
    }
    out
}

/// Independent model of the reverse-or-changed FIXED POINT: the set of nodes that can
/// reach an initial-frontier node along kind-passing edges. Built as an explicit reverse
/// adjacency list + an iterative worklist BFS, a wholly different structure from the
/// production `cpu_ref_closure` (which iterates a per-source bitset pass to convergence),
/// so agreement is a real cross-check, not a restatement. Seed bits (including padding
/// bits above `node_count`) are monotonically retained to match the in-place accumulator.
fn oracle_backward_closure(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let n = node_count as usize;
    let mut reverse: Vec<Vec<u32>> = vec![Vec::new(); n];
    for src in 0..node_count {
        let start = offsets[src as usize] as usize;
        let end = offsets[src as usize + 1] as usize;
        for edge in start..end {
            if masks[edge] & allow_mask == 0 {
                continue;
            }
            let dst = targets[edge];
            if dst < node_count {
                // src → dst forward ⇒ dst can be reached-from src ⇒ reverse edge dst → src.
                reverse[dst as usize].push(src);
            }
        }
    }
    let mut visited = vec![false; n];
    let mut stack = Vec::new();
    for node in 0..node_count {
        if bit_is_set(frontier, node) {
            visited[node as usize] = true;
            stack.push(node);
        }
    }
    while let Some(node) = stack.pop() {
        for &pred in &reverse[node as usize] {
            if !visited[pred as usize] {
                visited[pred as usize] = true;
                stack.push(pred);
            }
        }
    }
    let words = bitset_words(node_count);
    let mut out = frontier.to_vec();
    out.resize(words, 0);
    for node in 0..node_count {
        if visited[node as usize] {
            out[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    out
}

fn canonical_ifds_csr(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra: &[(u32, u32, u32)],
    inter: &[(u32, u32, u32, u32)],
    gen: &[(u32, u32, u32)],
    kill: &[(u32, u32, u32)],
) -> (Vec<u32>, Vec<u32>) {
    let (row_ptr, col_idx) = build_cpu_reference(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra,
        inter,
        gen,
        kill,
    );
    reference_canonicalize_csr_within_rows(&row_ptr, &col_idx)
}

type GeneratedIfdsRules = (
    u32,
    u32,
    u32,
    Vec<(u32, u32, u32)>,
    Vec<(u32, u32, u32, u32)>,
    Vec<(u32, u32, u32)>,
    Vec<(u32, u32, u32)>,
);

fn generated_ifds_rules(seed: u64) -> GeneratedIfdsRules {
    let mut rng = Rng::new(seed);
    let num_procs = 1 + rng.range(4);
    let blocks_per_proc = 1 + rng.range(8);
    let facts_per_proc = 1 + rng.range(8);
    let mut intra_edges = Vec::new();
    let mut inter_edges = Vec::new();
    let mut flow_gen = Vec::new();
    let mut flow_kill = Vec::new();

    for p in 0..num_procs {
        for b in 0..blocks_per_proc {
            if blocks_per_proc > 1 && rng.next_u32() & 1 == 0 {
                intra_edges.push((p, b, (b + 1) % blocks_per_proc));
            }
            let fact = rng.range(facts_per_proc);
            if rng.next_u32() % 3 == 0 {
                flow_gen.push((p, b, fact));
            }
            if rng.next_u32() % 5 == 0 && fact != 0 {
                flow_kill.push((p, b, fact));
            }
        }
    }
    if num_procs > 1 {
        for p in 0..num_procs - 1 {
            if rng.next_u32() & 1 == 0 {
                inter_edges.push((p, 0, p + 1, 0));
            }
        }
    }

    (
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra_edges,
        inter_edges,
        flow_gen,
        flow_kill,
    )
}

/// Deterministic IFDS rule-set stream for one sweep family.
fn ifds_cases(
    cases: u64,
    seed: u64,
    stride: u64,
) -> impl Iterator<Item = (u64, GeneratedIfdsRules)> {
    (0..cases).map(move |index| {
        (
            index,
            generated_ifds_rules(seed ^ index.wrapping_mul(stride)),
        )
    })
}

fn generated_parent(seed: u64) -> (Vec<u32>, Vec<u32>, u32) {
    let mut rng = Rng::new(seed);
    let len = 1 + rng.range(128);
    let mut parent = Vec::with_capacity(len as usize);
    for node in 0..len {
        let p = if node == 0 { 0 } else { rng.range(node + 1) };
        parent.push(p);
    }
    let target_count = 1 + rng.range(16);
    let mut targets = Vec::with_capacity(target_count as usize);
    for _ in 0..target_count {
        let target = if rng.next_u32() & 15 == 0 {
            len + rng.range(8)
        } else {
            rng.range(len)
        };
        targets.push(target);
    }
    let max_depth = 1 + rng.range(64);
    (parent, targets, max_depth)
}

// ---------------------------------------------------------------------------
// Primitive families: the CPU references that own graph semantics.
// ---------------------------------------------------------------------------

#[test]
fn generated_csr_and_persistent_bfs_oracles_cover_4096_shapes() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_cases(
        "padded_tail_masked_kinds",
        PRIMITIVE_CSR_CASES,
        0xC5A5_1D00_D00D_0001,
        0x9E37_79B9,
    ) {
        let expected_step = oracle_forward_or_changed(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual_step = csr_forward_or_changed::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(actual_step, expected_step, "case={case} forward_or_changed");

        let max_iters = (case as u32 % 9) + 1;
        let expected_bfs = csr_sweep::oracle_persistent_closure(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let actual_bfs = persistent_bfs::cpu_ref(
            CsrClosureInputs::new(
                node_count, &offsets, &targets, &masks, allow_mask, max_iters,
            ),
            &frontier,
        );
        assert_eq!(actual_bfs, expected_bfs, "case={case} persistent_bfs");
    }
}

#[test]
fn generated_csr_backward_or_changed_oracles_cover_4096_shapes() {
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_cases(
        "multi_source_restricted_kinds",
        PRIMITIVE_CSR_CASES,
        0x8ACC_1234_D00D_0007,
        0x9E37_79B9,
    ) {
        let max_iters = node_count.saturating_add(2);

        // 1. The production reverse-or-changed fixed point == the independent reverse-BFS
        //    closure. This is the op's real contract: a single node-parallel pass reads the
        //    live accumulator and is order-dependent for multi-hop chains, but the CONVERGED
        //    set is unique regardless of pass order.
        let (closure, _changed) = csr_backward_or_changed::cpu_ref_closure(
            CsrClosureInputs::new(
                node_count, &offsets, &targets, &masks, allow_mask, max_iters,
            ),
            &frontier,
        );
        let expected = oracle_backward_closure(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(closure, expected, "case={case} backward closure");

        // 2. Idempotent at the fixed point: one more snapshot pass sets no new bit.
        let (again, second_changed) = csr_backward_or_changed::cpu_ref(
            node_count, &offsets, &targets, &masks, &closure, allow_mask,
        );
        assert_eq!(again, closure, "case={case} backward idempotent");
        assert_eq!(
            second_changed, 0,
            "case={case} backward fixpoint changed flag"
        );

        // 3. Monotone: every initial-frontier node survives to the closure.
        for node in 0..node_count {
            if bit_is_set(&frontier, node) {
                assert!(
                    bit_is_set(&closure, node),
                    "case={case} backward closure dropped seed node {node}"
                );
            }
        }
    }
}

#[test]
fn generated_path_reconstruction_oracles_cover_2048_batches() {
    for case in 0..PRIMITIVE_BATCH_CASES {
        let (parent, targets, max_depth) =
            generated_parent(0x9A7E_5EED_0123_0000 ^ case.wrapping_mul(0xD1B5_4A32));
        let mut batched_paths = Vec::new();
        let mut batched_lens = Vec::new();
        path_reconstruct::cpu_ref_batched(
            &parent,
            &targets,
            max_depth,
            &mut batched_paths,
            &mut batched_lens,
        );

        assert_eq!(batched_lens.len(), targets.len(), "case={case} lens len");
        assert_eq!(
            batched_paths.len(),
            targets.len() * max_depth as usize,
            "case={case} path matrix len"
        );

        let mut scratch = Vec::new();
        for (index, &target) in targets.iter().enumerate() {
            let len = path_reconstruct::cpu_ref(&parent, target, max_depth, &mut scratch);
            assert_eq!(batched_lens[index], len, "case={case} target_index={index}");
            let start = index * max_depth as usize;
            let end = start + max_depth as usize;
            assert_eq!(
                &batched_paths[start..end],
                scratch.as_slice(),
                "case={case} target_index={index} segment"
            );
        }
    }
}

#[test]
fn generated_motif_oracles_cover_2048_patterns() {
    for (case, node_count, offsets, targets, masks, _, _) in csr_cases(
        "topology_only_all_kinds",
        PRIMITIVE_BATCH_CASES,
        0xF00D_BA5E_4455_0000,
        0xA24B_AED4,
    ) {
        let mut rng = Rng::new(0xBADC_0FFE_EE11_0000 ^ case);
        let motif_len = rng.range(5) as usize;
        let mut motif_edges = Vec::with_capacity(motif_len);
        for _ in 0..motif_len {
            motif_edges.push(MotifEdge {
                from: rng.range(node_count),
                kind_mask: 1u32 << rng.range(5),
                to: rng.range(node_count),
            });
        }

        let witness = motif::cpu_ref(node_count, &offsets, &targets, &masks, &motif_edges);
        let counted = motif::cpu_ref_participation_count(
            node_count,
            &offsets,
            &targets,
            &masks,
            &motif_edges,
        );
        let summed = witness.iter().copied().sum::<u32>();
        assert_eq!(counted, summed, "case={case} motif participation");
        assert_eq!(
            witness.len(),
            node_count as usize,
            "case={case} witness len"
        );
    }
}

// ---------------------------------------------------------------------------
// Substrate families: the wrappers built on those references.
// ---------------------------------------------------------------------------

#[test]
fn sweep_csr_forward_or_changed_matches_independent_oracle_matrix() {
    let mut assertions = 0usize;
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_cases(
        "single_source_all_kinds",
        CASES_PER_FAMILY,
        0xF0C5_0001,
        0x9E37_79B9,
    ) {
        let expected = oracle_forward_or_changed(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = reference_forward_step_with_change_flag(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_forward_or_changed case {case} node_count={node_count} must match independent oracle."
        );
        assert_eq!(actual.0.len(), bitset_words(node_count));
        assertions += 2;
    }
    assert_eq!(assertions, CASES_PER_FAMILY as usize * 2);
}

#[test]
fn sweep_csr_bidirectional_step_matches_independent_oracle_matrix() {
    let mut assertions = 0usize;
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_cases(
        "padded_tail_masked_kinds",
        CASES_PER_FAMILY,
        0xB1D1_0002,
        0xD1B5_4A32,
    ) {
        let expected = oracle_bidirectional_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual = reference_bidirectional_step(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(
            actual, expected,
            "Fix: csr_bidirectional case {case} node_count={node_count} must match independent oracle."
        );
        assert_ne!(
            actual.len(),
            0,
            "Fix: csr_bidirectional case {case} must return bitset words."
        );
        assertions += 2;
    }
    assert_eq!(assertions, CASES_PER_FAMILY as usize * 2);
}

#[test]
fn sweep_persistent_bfs_matches_independent_oracle_matrix() {
    let mut assertions = 0usize;
    for (case, node_count, offsets, targets, masks, frontier, allow_mask) in csr_cases(
        "multi_source_restricted_kinds",
        CASES_PER_FAMILY,
        0xBFC0_0003,
        0xA24B_AED4,
    ) {
        let max_iters = (case as u32 % 9) + 1;
        let expected = csr_sweep::oracle_persistent_closure(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let actual = bfs_expand(
            CsrClosureInputs::new(
                node_count, &offsets, &targets, &masks, allow_mask, max_iters,
            ),
            &frontier,
        );
        assert_eq!(
            actual, expected,
            "Fix: persistent_bfs case {case} node_count={node_count} max_iters={max_iters} must match independent oracle."
        );
        assert_eq!(actual.0.len(), bitset_words(node_count));
        assertions += 2;
    }
    assert_eq!(assertions, CASES_PER_FAMILY as usize * 2);
}

#[test]
fn sweep_exploded_ifds_substrate_matches_primitive_oracle_matrix() {
    let mut assertions = 0usize;
    for (case, (num_procs, blocks_per_proc, facts_per_proc, intra, inter, gen, kill)) in
        ifds_cases(CASES_PER_FAMILY, 0x1F05_0004, 0x85EB_CA6B)
    {
        let expected = canonical_ifds_csr(
            num_procs,
            blocks_per_proc,
            facts_per_proc,
            &intra,
            &inter,
            &gen,
            &kill,
        );
        let (row_ptr, col_idx) = reference_build_ifds_csr(
            num_procs,
            blocks_per_proc,
            facts_per_proc,
            &intra,
            &inter,
            &gen,
            &kill,
        );
        let actual = reference_canonicalize_csr_within_rows(&row_ptr, &col_idx);
        assert_eq!(
            actual, expected,
            "Fix: exploded IFDS substrate reference case {case} procs={num_procs} blocks={blocks_per_proc} facts={facts_per_proc} must match primitive CPU oracle."
        );
        assertions += 2;
    }
    assert_eq!(assertions, CASES_PER_FAMILY as usize * 2);
}

#[test]
fn sweep_exploded_ifds_via_matches_cpu_oracle_matrix() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut assertions = 0usize;
    for (case, (num_procs, blocks_per_proc, facts_per_proc, intra, inter, gen, kill)) in
        ifds_cases(CASES_PER_FAMILY, 0x1F05_0005, 0xC2B2_AE35)
    {
        let expected = canonical_ifds_csr(
            num_procs,
            blocks_per_proc,
            facts_per_proc,
            &intra,
            &inter,
            &gen,
            &kill,
        );
        let actual = build_ifds_csr_via(
            &dispatcher,
            num_procs,
            blocks_per_proc,
            facts_per_proc,
            &intra,
            &inter,
            &gen,
            &kill,
        )
        .unwrap_or_else(|error| {
            panic!("Fix: exploded IFDS via CPU oracle case {case} must dispatch: {error:?}")
        });
        assert_eq!(
            actual, expected,
            "Fix: exploded IFDS via CPU oracle case {case} procs={num_procs} blocks={blocks_per_proc} facts={facts_per_proc} must match reference CSR."
        );
        assertions += 2;
    }
    assert_eq!(assertions, CASES_PER_FAMILY as usize * 2);
}
