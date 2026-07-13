//! GPU-IR vs CPU-ref parity for `graph::vast_tree_walk` preorder + postorder
//! traversal of a VAST first-child / next-sibling tree.
//!
//! A single serial invocation walks the tree from the root (node 0), emitting
//! node indices in the requested order into `out`. The walk is pure pointer
//! chasing over the node struct (offset 1 = parent, 2 = first_child, 3 =
//! next_sibling; stride `NODE_STRIDE_U32`), with SENTINEL / out-of-range guards.
//! Every shipped test builds fixed trees and checks Program shape or an INLINE
//! (test-private) oracle; the actual traversal IR was never driven through a
//! faithful executor. This test carries an INDEPENDENT re-implementation of both
//! orders (a second walk, not a copy of the IR node-by-node) and drives the real
//! Program through `reference_eval` over random proper trees. A first-child /
//! next-sibling swap, a broken parent-climb for the next sibling, or a dropped
//! descend-to-leftmost-leaf (postorder) all diverge here.
//!
//! Grid: workgroup is [1,1,1] and the walk is serial (no InvocationId use), so an
//! over-fired grid (reference_eval infers `out_cap` lanes from the output buffer)
//! is idempotent - every lane performs the identical deterministic walk and
//! writes identical output words.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_foundation::vast::{NODE_STRIDE_U32, SENTINEL};
use vyre_primitives::graph::vast_tree_walk::{ast_walk_postorder, ast_walk_preorder};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

const STRIDE: usize = NODE_STRIDE_U32;

struct Tree {
    node_count: u32,
    /// Flattened node table: `node i` occupies `[i*STRIDE .. (i+1)*STRIDE]`,
    /// with offset 1 = parent, 2 = first_child, 3 = next_sibling.
    words: Vec<u32>,
}

fn valid(idx: u32, node_count: u32) -> bool {
    idx != SENTINEL && idx < node_count
}

impl Tree {
    fn parent(&self, n: u32) -> u32 {
        self.words[n as usize * STRIDE + 1]
    }
    fn first_child(&self, n: u32) -> u32 {
        self.words[n as usize * STRIDE + 2]
    }
    fn next_sibling(&self, n: u32) -> u32 {
        self.words[n as usize * STRIDE + 3]
    }

    /// Independent preorder walk mirroring the primitive's bounded-loop contract
    /// (climb via parent to find the next sibling when a subtree is exhausted).
    fn preorder(&self, out_cap: u32) -> Vec<u32> {
        let nc = self.node_count;
        let mut out = Vec::new();
        let mut n = 0u32;
        for _ in 0..nc {
            if out.len() as u32 >= out_cap || n >= nc {
                break;
            }
            out.push(n);
            let fc = self.first_child(n);
            if valid(fc, nc) {
                n = fc;
            } else {
                let mut next = SENTINEL;
                let mut walk = n;
                for _ in 0..nc {
                    if next == SENTINEL && valid(walk, nc) {
                        let sib = self.next_sibling(walk);
                        if valid(sib, nc) {
                            next = sib;
                        } else {
                            walk = self.parent(walk);
                        }
                    }
                }
                if next == SENTINEL {
                    break;
                }
                n = next;
            }
        }
        out
    }

    /// Independent postorder walk: descend to the leftmost leaf, emit, then step
    /// to the next sibling (descending again) or up to the parent.
    fn postorder(&self, out_cap: u32) -> Vec<u32> {
        let nc = self.node_count;
        if nc == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut n = 0u32;
        let descend = |n: &mut u32| {
            for _ in 0..nc {
                if valid(*n, nc) {
                    let fc = self.first_child(*n);
                    if valid(fc, nc) {
                        *n = fc;
                    }
                }
            }
        };
        descend(&mut n);
        for _ in 0..nc {
            if out.len() as u32 >= out_cap || n >= nc {
                break;
            }
            out.push(n);
            if n == 0 {
                break;
            }
            let sib = self.next_sibling(n);
            if valid(sib, nc) {
                n = sib;
                descend(&mut n);
            } else {
                let parent = self.parent(n);
                if !valid(parent, nc) {
                    break;
                }
                n = parent;
            }
        }
        out
    }
}

/// Build a random PROPER tree (node 0 = root, every other node has a parent
/// strictly less than itself), so the full traversal reaches every node and
/// `out_cap == node_count` leaves no unwritten tail slots.
fn generated_tree(seed: u64) -> Tree {
    let mut rng = seed;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 32) as u32
    };
    // Kept small: the serial [1,1,1] walk over-fires to out_cap == node_count
    // lanes under reference_eval and the preorder parent-climb is O(node_count)
    // per step, so per-case cost is ~O(node_count^3) through the interpreter.
    let node_count = 1 + next() % 12; // 1..=12 nodes
    let mut words = vec![0u32; node_count as usize * STRIDE];
    // Children lists per parent, in insertion order (defines sibling chains).
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); node_count as usize];
    for i in 1..node_count {
        let parent = next() % i; // strictly smaller -> acyclic, rooted at 0
        children[parent as usize].push(i);
    }
    for n in 0..node_count {
        let base = n as usize * STRIDE;
        words[base + 1] = SENTINEL; // parent (overwritten below for non-root)
        words[base + 2] = children[n as usize].first().copied().unwrap_or(SENTINEL);
        words[base + 3] = SENTINEL; // next_sibling (wired below)
    }
    // Wire parent pointers and sibling chains.
    words[1] = SENTINEL; // root parent
    for p in 0..node_count {
        let kids = &children[p as usize];
        for (k, &c) in kids.iter().enumerate() {
            words[c as usize * STRIDE + 1] = p; // parent
            let sib = kids.get(k + 1).copied().unwrap_or(SENTINEL);
            words[c as usize * STRIDE + 3] = sib; // next_sibling
        }
    }
    Tree { node_count, words }
}

fn gpu_walk(program: &vyre_foundation::ir::Program, nodes: &[u32], out_cap: u32) -> Vec<u32> {
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack(nodes)),
            Value::from(pack(&vec![0u32; out_cap as usize])),
        ],
    )
    .expect("vast_tree_walk reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(600))]

    #[test]
    fn preorder_ir_matches_reference(seed in any::<u64>()) {
        let tree = generated_tree(seed);
        let out_cap = tree.node_count;
        let program = ast_walk_preorder("nodes", "out", tree.node_count, out_cap);
        let expected = tree.preorder(out_cap);
        prop_assert_eq!(expected.len() as u32, tree.node_count, "proper tree: preorder visits all nodes");
        let got = gpu_walk(&program, &tree.words, out_cap);
        prop_assert_eq!(got, expected, "preorder IR diverged (node_count={})", tree.node_count);
    }

    #[test]
    fn postorder_ir_matches_reference(seed in any::<u64>()) {
        let tree = generated_tree(seed);
        let out_cap = tree.node_count;
        let program = ast_walk_postorder("nodes", "out", tree.node_count, out_cap);
        let expected = tree.postorder(out_cap);
        prop_assert_eq!(expected.len() as u32, tree.node_count, "proper tree: postorder visits all nodes");
        let got = gpu_walk(&program, &tree.words, out_cap);
        prop_assert_eq!(got, expected, "postorder IR diverged (node_count={})", tree.node_count);
    }
}

/// Deterministic anchor: a fixed 6-node tree with known pre/postorder, so a
/// regression that happened to match a buggy random-oracle still fails here.
#[test]
fn walks_match_reference_on_fixed_tree() {
    // Tree:      0
    //          / | \
    //         1  2  5
    //        / \
    //       3   4
    // children(0)=[1,2,5], children(1)=[3,4].
    let node_count = 6u32;
    let mut words = vec![0u32; node_count as usize * STRIDE];
    let set = |w: &mut [u32], n: usize, parent: u32, fc: u32, ns: u32| {
        w[n * STRIDE + 1] = parent;
        w[n * STRIDE + 2] = fc;
        w[n * STRIDE + 3] = ns;
    };
    set(&mut words, 0, SENTINEL, 1, SENTINEL);
    set(&mut words, 1, 0, 3, 2);
    set(&mut words, 2, 0, SENTINEL, 5);
    set(&mut words, 3, 1, SENTINEL, 4);
    set(&mut words, 4, 1, SENTINEL, SENTINEL);
    set(&mut words, 5, 0, SENTINEL, SENTINEL);
    let tree = Tree { node_count, words };

    let pre = tree.preorder(node_count);
    assert_eq!(pre, vec![0, 1, 3, 4, 2, 5], "reference preorder");
    let post = tree.postorder(node_count);
    assert_eq!(post, vec![3, 4, 1, 2, 5, 0], "reference postorder");

    let pre_prog = ast_walk_preorder("nodes", "out", node_count, node_count);
    assert_eq!(
        gpu_walk(&pre_prog, &tree.words, node_count),
        pre,
        "preorder IR"
    );
    let post_prog = ast_walk_postorder("nodes", "out", node_count, node_count);
    assert_eq!(
        gpu_walk(&post_prog, &tree.words, node_count),
        post,
        "postorder IR"
    );
}
