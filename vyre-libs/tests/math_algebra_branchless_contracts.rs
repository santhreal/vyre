//! Structural contracts for algebraic math kernels.
//!
//! These tests protect performance-critical IR shape. Boolean semiring matrix
//! multiplication is used as a GraphBLAS-style substrate for reachability and
//! parser closure, so its inner loop must remain branchless on SIMT backends.

#![cfg(feature = "math-algebra")]

use vyre::ir::Node;

fn bool_mm_loop_body(nodes: &[Node]) -> Option<&[Node]> {
    for node in nodes {
        match node {
            Node::Loop { var, body, .. } if var.as_str() == "bool_mm_k" => {
                return Some(body);
            }
            Node::Loop { body, .. } | Node::Block(body) => {
                if let Some(found) = bool_mm_loop_body(body) {
                    return Some(found);
                }
            }
            Node::Region { body, .. } => {
                if let Some(found) = bool_mm_loop_body(body) {
                    return Some(found);
                }
            }
            Node::If {
                then, otherwise, ..
            } => {
                if let Some(found) = bool_mm_loop_body(then) {
                    return Some(found);
                }
                if let Some(found) = bool_mm_loop_body(otherwise) {
                    return Some(found);
                }
            }
            _ => {}
        }
    }
    None
}

fn contains_branch(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::If { .. } => true,
        Node::Loop { body, .. } | Node::Block(body) => contains_branch(body),
        Node::Region { body, .. } => contains_branch(body),
        _ => false,
    })
}

/// The looping kernel accumulates with select/bitor, never a per-k branch.
///
/// The shape matters. `bool_semiring_matmul` fully unrolls whenever
/// `out_count <= 64 && inner <= 64`, and the old 2 x 3 x 2 arguments produced
/// 4 output cells over an inner extent of 3, so this test looked for a
/// `bool_mm_k` loop in a program that has no loops at all and failed on the
/// `expect` before it ever reached the branchless assertion. 16 x 8 x 8 is 128
/// output cells, which clears the threshold and builds the real loop.
#[test]
fn bool_semiring_inner_loop_is_branchless() {
    let program = vyre_libs::math::algebra::bool_semiring_matmul("a", "b", "out", 16, 8, 8);
    let body = bool_mm_loop_body(program.entry()).expect("bool_mm_k loop must exist");
    assert!(
        !contains_branch(body),
        "bool-semiring matmul must accumulate with select/bitor instead of divergent per-k branches"
    );
}

/// Small shapes unroll instead of looping, and stay branchless doing it.
///
/// The unrolled path is the other half of the same performance contract: it
/// emits one straight-line `bitor`/`select` expression per output cell behind
/// a single lane guard. Pinning it stops a future rewrite from reintroducing
/// per-k control flow on the shapes that skip the loop, which is exactly the
/// blind spot that let the test above sit red.
#[test]
fn bool_semiring_small_shapes_unroll_without_a_loop() {
    let program = vyre_libs::math::algebra::bool_semiring_matmul("a", "b", "out", 2, 3, 2);
    assert!(
        bool_mm_loop_body(program.entry()).is_none(),
        "2 x 3 x 2 is below the 64-cell unroll threshold, so it must emit no bool_mm_k loop"
    );

    // The unrolled stores sit behind one `invocation == 0` guard. Everything
    // inside that guard must be straight-line: no nested branch, no loop.
    let [Node::Region { body, .. }] = program.entry() else {
        panic!("bool_semiring_matmul emits one top-level region");
    };
    let [Node::If { then, .. }] = body.as_slice() else {
        panic!("the unrolled kernel is a single lane guard, got {body:?}");
    };
    assert!(
        !contains_branch(then),
        "the unrolled body must be straight-line select/bitor, not nested control flow"
    );
    assert!(
        !then.is_empty(),
        "the unrolled body must actually store the 4 output cells"
    );
}
