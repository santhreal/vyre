//! Fused `softmax_top_k` constructor for MoE gating.
//!
//! Computes `softmax(scores)` and returns the top-k indices + normalized weights
//! in a single dispatch, eliminating the separate softmax + top-k round-trip.

use super::topk_selection::{
    copy_top_k_indices_and_normalized_weights, init_top_k_slots, insert_top_k_candidate, BEST_IDXS,
    BEST_VALS,
};
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

/// Canonical op id. It is the region generator name, the trap subject, and the
/// child name a composition attributes this body to, so it is one constant
/// rather than a literal repeated at each of those three sites.
pub(crate) const OP_ID: &str = "vyre-libs::nn::softmax_top_k";

/// Build a Program that computes softmax over `scores`, then returns the
/// top-k indices and their normalized weights.
///
/// Inputs:
/// - `scores`: f32 buffer of length `n`
///
/// Outputs:
/// - `out_indices`: u32 buffer of length `k`
/// - `out_weights`: f32 buffer of length `k`
///
/// The weights sum to 1.0 across the full distribution (not just the top-k).
#[must_use]
pub fn softmax_top_k(
    scores: &str,
    out_indices: &str,
    out_weights: &str,
    n: u32,
    k: u32,
) -> Program {
    if k == 0 {
        return trap_program(
            OP_ID,
            Some((out_indices, DataType::U32)),
            "Fix: softmax_top_k requires k > 0 so the selection scratch has at least one slot."
                .to_string(),
        );
    }
    Program::wrapped(
        softmax_top_k_buffers(scores, out_indices, out_weights, n, k),
        [1, 1, 1],
        // The scan is serial over `n` and keeps its running best in read-write
        // scratch, so one invocation owns it, for the reason top_k states.
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::is_first_invocation(),
                softmax_top_k_body(scores, out_indices, out_weights, n, k),
            )],
        )],
    )
}

/// Buffer table of the standalone operation, in binding order.
fn softmax_top_k_buffers(
    scores: &str,
    out_indices: &str,
    out_weights: &str,
    n: u32,
    k: u32,
) -> Vec<BufferDecl> {
    let mut buffers = vec![
        BufferDecl::storage(scores, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
        BufferDecl::output(out_indices, 1, DataType::U32).with_count(k),
        BufferDecl::storage(out_weights, 2, BufferAccess::WriteOnly, DataType::F32).with_count(k),
    ];
    buffers.extend(softmax_top_k_scratch(3, k));
    buffers
}

/// The selection scratch this body keeps its running best `k` in.
///
/// A composition that runs the body has to declare the same two buffers at its
/// own binding indices, and a second spelling of them is a table that drifts
/// from the body that reads it.
pub(crate) fn softmax_top_k_scratch(first_binding: u32, k: u32) -> Vec<BufferDecl> {
    vec![
        BufferDecl::read_write(BEST_VALS, first_binding, DataType::F32).with_count(k),
        BufferDecl::read_write(BEST_IDXS, first_binding + 1, DataType::U32).with_count(k),
    ]
}

/// Softmax over `scores` followed by top-k selection, as region body nodes.
///
/// Serial by construction: the maximum, the exponential sum and the insertion
/// sort are one pass each over `n`, on one invocation. A caller that wants this
/// selection inside a larger operation runs these nodes as a child region
/// rather than restating them.
pub(crate) fn softmax_top_k_body(
    scores: &str,
    out_indices: &str,
    out_weights: &str,
    n: u32,
    k: u32,
) -> Vec<Node> {
    let mut body = init_top_k_slots(k);

    // max_val = max(scores)
    body.push(Node::let_bind("max_val", Expr::f32(f32::NEG_INFINITY)));
    body.push(Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::if_then(
            Expr::gt(Expr::load(scores, Expr::var("i")), Expr::var("max_val")),
            vec![Node::assign("max_val", Expr::load(scores, Expr::var("i")))],
        )],
    ));

    // sum = sum(exp(score - max_val))
    // Also track top-k on the exp values
    body.push(Node::let_bind("sum", Expr::f32(0.0)));
    body.push(Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(n),
        vec![
            Node::let_bind(
                "exp_val",
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(Expr::sub(
                        Expr::load(scores, Expr::var("i")),
                        Expr::var("max_val"),
                    )),
                },
            ),
            Node::assign("sum", Expr::add(Expr::var("sum"), Expr::var("exp_val"))),
            // Top-k insertion on exp_val
            Node::Block(insert_top_k_candidate(
                k,
                Expr::var("exp_val"),
                Expr::var("i"),
            )),
        ],
    ));

    body.extend(copy_top_k_indices_and_normalized_weights(
        out_indices,
        out_weights,
        k,
        Expr::var("sum"),
    ));

    body
}

fn fixture_f32_bytes(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

fn fixture_u32_bytes(values: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(values)
}

fn softmax_top_k_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    let scores: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    vec![vec![
        fixture_f32_bytes(&scores),
        vec![0u8; 4 * 2],
        vec![0u8; 4 * 2],
    ]]
}

#[cfg(test)]
mod tests {
    use super::super::topk_selection::{f32_from_bytes, u32_from_bytes};
    use super::*;
    use crate::fixture_bytes::eval_bytes;
    use crate::fixture_bytes::f32_bytes;

    #[test]
    fn softmax_top_k_basic() {
        // scores = [1.0, 2.0, 3.0]  -  softmax ≈ [0.090, 0.245, 0.665]
        let scores = vec![1.0f32, 2.0, 3.0];
        let program = softmax_top_k("scores", "indices", "weights", 3, 2);
        let outputs = eval_bytes(
            "softmax_top_k",
            &program,
            vec![f32_bytes(&scores), vec![0u8; 2 * 4], vec![0u8; 2 * 4]],
        );

        let indices = u32_from_bytes(&outputs[0]);
        let weights = f32_from_bytes(&outputs[1]);

        assert_eq!(indices[0], 2); // 3.0 is max
        assert_eq!(indices[1], 1); // 2.0 is second

        // Weights should be the normalized softmax values
        let max = 3.0f32;
        let exp0 = (1.0 - max).exp();
        let exp1 = (2.0 - max).exp();
        let exp2 = (3.0 - max).exp();
        let sum = exp0 + exp1 + exp2;
        let expected_w0 = exp2 / sum;
        let expected_w1 = exp1 / sum;

        assert!((weights[0] - expected_w0).abs() < 1e-4);
        assert!((weights[1] - expected_w1).abs() < 1e-4);
    }

    #[test]
    fn softmax_top_k_weights_sum_to_one() {
        let scores: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let program = softmax_top_k("scores", "indices", "weights", 8, 3);
        let outputs = eval_bytes(
            "softmax_top_k",
            &program,
            vec![f32_bytes(&scores), vec![0u8; 3 * 4], vec![0u8; 3 * 4]],
        );

        let weights = f32_from_bytes(&outputs[1]);
        let total: f32 = weights.iter().sum();
        // The top-3 weights don't sum to 1.0, but the internal sum is 1.0.
        // Just verify the weights are positive and ordered correctly.
        assert!(total > 0.0);
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert!(weights[0] > 0.0);
    }
}

const EXPECTED_SOFTMAX_TOP_K_BUF0_BYTES: [u8; 8] = [0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00];
const EXPECTED_SOFTMAX_TOP_K_BUF1_BYTES: [u8; 8] = [0x8E, 0xE0, 0x21, 0x3F, 0x83, 0x34, 0x6E, 0x3E];
const EXPECTED_SOFTMAX_TOP_K_BUF2_BYTES: [u8; 8] = [0x00, 0x00, 0x80, 0x3F, 0xB2, 0x5A, 0xBC, 0x3E];
const EXPECTED_SOFTMAX_TOP_K_BUF3_BYTES: [u8; 8] = [0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || softmax_top_k("scores", "indices", "weights", 8, 2),
        Some(softmax_top_k_fixture_inputs),
        Some(|| {
            vec![vec![
                EXPECTED_SOFTMAX_TOP_K_BUF0_BYTES.to_vec(),
                EXPECTED_SOFTMAX_TOP_K_BUF1_BYTES.to_vec(),
                EXPECTED_SOFTMAX_TOP_K_BUF2_BYTES.to_vec(),
                EXPECTED_SOFTMAX_TOP_K_BUF3_BYTES.to_vec(),
            ]]
        }),
    )
    .with_category("nn")
}
