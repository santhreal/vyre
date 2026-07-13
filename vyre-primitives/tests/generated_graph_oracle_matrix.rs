//! Generated adversarial oracle matrix for graph primitives.
//!
//! These tests pin the primitive crate as the graph authority by comparing
//! production CPU oracles against independent deterministic models across
//! thousands of CSR, path-reconstruction, and motif shapes.

use vyre_primitives::graph::csr_backward_or_changed;
use vyre_primitives::graph::csr_forward_or_changed;
use vyre_primitives::graph::motif::{self, MotifEdge};
use vyre_primitives::graph::path_reconstruct;
use vyre_primitives::graph::persistent_bfs;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.0 = x;
        (x >> 16) as u32
    }

    fn range(&mut self, upper: u32) -> u32 {
        if upper == 0 {
            0
        } else {
            self.next_u32() % upper
        }
    }
}

fn bitset_words(node_count: u32) -> usize {
    node_count.div_ceil(32) as usize
}

fn generated_csr(seed: u64) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = Rng::new(seed);
    let node_count = 1 + rng.range(96);
    let words = bitset_words(node_count);
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0);
    for _ in 0..node_count {
        let degree = rng.range(5);
        for _ in 0..degree {
            targets.push(rng.range(node_count));
            let bit = 1u32 << rng.range(5);
            let noise = if rng.next_u32() & 7 == 0 {
                1u32 << rng.range(5)
            } else {
                0
            };
            masks.push(bit | noise);
        }
        offsets.push(targets.len() as u32);
    }
    let mut frontier = vec![0u32; words];
    for node in 0..node_count {
        if rng.next_u32() & 3 == 0 {
            frontier[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    if rng.next_u32() & 1 == 0 {
        let word = (node_count - 1) / 32;
        let used = node_count % 32;
        if used != 0 {
            frontier[word as usize] |= !((1u32 << used) - 1);
        }
    }
    let allow_mask = match rng.range(6) {
        0 => 0,
        1 => 1,
        2 => 0b10,
        3 => 0b101,
        _ => 0xFFFF_FFFF,
    };
    (node_count, offsets, targets, masks, frontier, allow_mask)
}

fn bit_is_set(words: &[u32], node: u32) -> bool {
    let word = (node / 32) as usize;
    let bit = 1u32 << (node % 32);
    words.get(word).is_some_and(|value| value & bit != 0)
}

fn expected_forward_or_changed(
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

fn snapshot_successors(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> Vec<u32> {
    let mut out = vec![0u32; bitset_words(node_count)];
    for src in 0..node_count {
        if !bit_is_set(frontier, src) {
            continue;
        }
        let start = offsets[src as usize] as usize;
        let end = offsets[src as usize + 1] as usize;
        for edge in start..end {
            if masks[edge] & allow_mask == 0 {
                continue;
            }
            let dst = targets[edge];
            out[(dst / 32) as usize] |= 1u32 << (dst % 32);
        }
    }
    out
}

fn expected_persistent_bfs(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> (Vec<u32>, u32) {
    let words = bitset_words(node_count);
    let mut out = frontier.to_vec();
    out.resize(words, 0);
    let mut changed = 0;
    for _ in 0..max_iters {
        let step = snapshot_successors(node_count, offsets, targets, masks, &out, allow_mask);
        let mut step_changed = false;
        for word in 0..words {
            let before = out[word];
            out[word] |= step[word];
            if out[word] != before {
                step_changed = true;
            }
        }
        if step_changed {
            changed = 1;
        } else {
            break;
        }
    }
    (out, changed)
}

#[test]
fn generated_csr_and_persistent_bfs_oracles_cover_4096_shapes() {
    for case in 0..4096u64 {
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr(0xC5A5_1D00_D00D_0001 ^ case.wrapping_mul(0x9E37_79B9));

        let expected_step = expected_forward_or_changed(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        let actual_step = csr_forward_or_changed::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask,
        );
        assert_eq!(actual_step, expected_step, "case={case} forward_or_changed");

        let max_iters = (case as u32 % 9) + 1;
        let expected_bfs = expected_persistent_bfs(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let actual_bfs = persistent_bfs::cpu_ref(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        assert_eq!(actual_bfs, expected_bfs, "case={case} persistent_bfs");
    }
}

/// Independent model of the reverse-or-changed FIXED POINT: the set of nodes that can
/// reach an initial-frontier node along kind-passing edges. Built as an explicit reverse
/// adjacency list + an iterative worklist BFS, a wholly different structure from the
/// production `cpu_ref_closure` (which iterates a per-source bitset pass to convergence),
/// so agreement is a real cross-check, not a restatement. Seed bits (including padding
/// bits above `node_count`) are monotonically retained to match the in-place accumulator.
fn expected_backward_closure(
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

#[test]
fn generated_csr_backward_or_changed_oracles_cover_4096_shapes() {
    for case in 0..4096u64 {
        let (node_count, offsets, targets, masks, frontier, allow_mask) =
            generated_csr(0x8ACC_1234_D00D_0007 ^ case.wrapping_mul(0x9E37_79B9));
        let max_iters = node_count.saturating_add(2);

        // 1. The production reverse-or-changed fixed point == the independent reverse-BFS
        //    closure. This is the op's real contract: a single node-parallel pass reads the
        //    live accumulator and is order-dependent for multi-hop chains, but the CONVERGED
        //    set is unique regardless of pass order.
        let (closure, _changed) = csr_backward_or_changed::cpu_ref_closure(
            node_count, &offsets, &targets, &masks, &frontier, allow_mask, max_iters,
        );
        let expected = expected_backward_closure(
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

#[test]
fn generated_path_reconstruction_oracles_cover_2048_batches() {
    for case in 0..2048u64 {
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
    for case in 0..2048u64 {
        let (node_count, offsets, targets, masks, _, _) =
            generated_csr(0xF00D_BA5E_4455_0000 ^ case.wrapping_mul(0xA24B_AED4));
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
