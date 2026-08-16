//! MoE Gating: softmax(scores) + top-k selection.
//!
//! Category-A composition over the tiled reduce skeleton `nn::softmax` uses and
//! the repeated-argmax top-k body `nn::quest_select_top_k` publishes. The gate
//! is a softmax denominator with a duplicate-suppressed top-k on the end, so it
//! reduces the score vector across the lanes of one workgroup exactly as
//! softmax does rather than walking it in lane zero.

use crate::builder::cooperative::chunks;
use crate::builder::tiled_reduce::{tiled_reduce_program, ReducePhase, TiledReduceProgram};
use crate::builder::{strided_accumulate_child, strided_writeback_child};
use crate::nn::quest_paging_passes::{quest_select_top_k_body, QUEST_SELECT_TOP_K_OP_ID};
use crate::reduce::workgroup_tree::{max_f32_child, sum_f32_child, WorkgroupReductionScope};
use vyre_foundation::composition::wrap_child_region;
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp, PORTABLE_WORKGROUP_INVOCATIONS,
};

const OP_ID: &str = "vyre-libs::nn::moe_gate";
const SCORES_SCRATCH: &str = "__moe_gate_scores_scratch";
const STATS_SCRATCH: &str = "__moe_gate_stats";
const LANE_SCRATCH: &str = "__moe_gate_lane_scratch";
const SOFTMAX_STATS_OP_ID: &str = "vyre-libs::nn::moe_gate::softmax_stats";
const WEIGHT_WRITE_OP_ID: &str = "vyre-libs::nn::moe_gate::weight_write";

/// Lanes the gate reduces over.
///
/// `nn::softmax` tiles the same score vector at this width and gating is that
/// softmax with a top-k on the end, so the two arms state one geometry: a plan
/// that fuses them keeps the workgroup shape both were built for.
const GATE_TILE: u32 = PORTABLE_WORKGROUP_INVOCATIONS;

/// Build a Program that computes MoE gating.
/// `input_scores`: `num_experts`, `output_indices`: `k`, `output_weights`: `k`.
#[must_use]
pub fn moe_gate(
    input_scores: &str,
    output_indices: &str,
    output_weights: &str,
    num_experts: u32,
    k: u32,
) -> Program {
    let expert_chunks = chunks(num_experts, GATE_TILE);
    // The top-k pass suppresses a chosen expert by overwriting its score, so it
    // reads a workgroup copy rather than the caller's read-only input.
    let mut phases = vec![ReducePhase {
        accumulate: strided_writeback_child(
            OP_ID,
            GATE_TILE,
            expert_chunks,
            num_experts,
            SCORES_SCRATCH,
            Vec::new(),
            |idx| Expr::load(input_scores, idx),
        ),
        reductions: Vec::new(),
        publish: Vec::new(),
    }];
    phases.extend(softmax_stats_phases(OP_ID, input_scores, num_experts));
    // Selection pass `j` reads the expert pass `j - 1` suppressed, so the scan
    // is sequential in the selection index and stays in one lane.
    phases.push(ReducePhase {
        accumulate: Node::if_then(
            Expr::and(
                Expr::is_first_workgroup(),
                Expr::eq(Expr::var("local"), Expr::u32(0)),
            ),
            vec![wrap_child_region(
                QUEST_SELECT_TOP_K_OP_ID,
                Ident::from(OP_ID),
                quest_select_top_k_body(SCORES_SCRATCH, output_indices, num_experts, k, f32::MIN),
            )],
        ),
        reductions: Vec::new(),
        publish: Vec::new(),
    });

    // V022: a Program may declare at most one ::output buffer.
    // `output_weights` is the scalar gating result the reference
    // interpreter compares against; `output_indices` is a read-write
    // storage buffer the caller consumes alongside.
    tiled_reduce_program(TiledReduceProgram {
        generator: OP_ID,
        buffers: vec![
            BufferDecl::storage(input_scores, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(num_experts),
            BufferDecl::storage(output_indices, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(k),
            BufferDecl::output(output_weights, 2, DataType::F32).with_count(k),
            BufferDecl::workgroup(SCORES_SCRATCH, num_experts, DataType::F32),
            BufferDecl::workgroup(STATS_SCRATCH, 2, DataType::F32),
            BufferDecl::workgroup(LANE_SCRATCH, GATE_TILE, DataType::F32),
        ],
        workgroup: [GATE_TILE, 1, 1],
        phases,
        writeback: Some(weight_write_child(
            OP_ID,
            input_scores,
            output_indices,
            output_weights,
            k,
        )),
    })
}

/// The two reduce passes that leave `STATS_SCRATCH` holding the stable-softmax
/// statistics `[max(scores), sum(exp(scores - max))]`.
///
/// Both passes reduce through the same lane scratch, so the maximum has to be
/// published before the sum pass overwrites the slot it was reduced in.
fn softmax_stats_phases(
    parent: &'static str,
    input_scores: &str,
    num_experts: u32,
) -> Vec<ReducePhase> {
    let expert_chunks = chunks(num_experts, GATE_TILE);
    vec![
        ReducePhase {
            accumulate: strided_accumulate_child(
                parent,
                GATE_TILE,
                expert_chunks,
                num_experts,
                "lane_max",
                Expr::f32(f32::MIN),
                LANE_SCRATCH,
                |idx, acc| Expr::max(acc, Expr::load(input_scores, idx)),
            ),
            reductions: vec![max_f32_child(
                parent,
                GATE_TILE,
                LANE_SCRATCH,
                WorkgroupReductionScope::FirstWorkgroup,
            )],
            publish: vec![Node::store(
                STATS_SCRATCH,
                Expr::u32(0),
                Expr::load(LANE_SCRATCH, Expr::u32(0)),
            )],
        },
        ReducePhase {
            accumulate: strided_accumulate_child(
                parent,
                GATE_TILE,
                expert_chunks,
                num_experts,
                "lane_sum_exp",
                Expr::f32(0.0),
                LANE_SCRATCH,
                |idx, acc| {
                    Expr::add(
                        acc,
                        exp_expr(Expr::sub(
                            Expr::load(input_scores, idx),
                            Expr::load(STATS_SCRATCH, Expr::u32(0)),
                        )),
                    )
                },
            ),
            reductions: vec![sum_f32_child(
                parent,
                GATE_TILE,
                LANE_SCRATCH,
                WorkgroupReductionScope::FirstWorkgroup,
            )],
            publish: vec![Node::store(
                STATS_SCRATCH,
                Expr::u32(1),
                Expr::load(LANE_SCRATCH, Expr::u32(0)),
            )],
        },
    ]
}

/// The writeback that turns `k` selected expert indices into gating weights.
fn weight_write_child(
    parent: &'static str,
    input_scores: &str,
    output_indices: &str,
    output_weights: &str,
    k: u32,
) -> Node {
    strided_writeback_child(
        parent,
        GATE_TILE,
        chunks(k, GATE_TILE),
        k,
        output_weights,
        vec![
            Node::let_bind("weight_max_score", Expr::load(STATS_SCRATCH, Expr::u32(0))),
            Node::let_bind("weight_sum_exp", Expr::load(STATS_SCRATCH, Expr::u32(1))),
        ],
        |j| {
            Expr::div(
                exp_expr(Expr::sub(
                    Expr::load(input_scores, Expr::load(output_indices, j)),
                    Expr::var("weight_max_score"),
                )),
                Expr::var("weight_sum_exp"),
            )
        },
    )
}

fn exp_expr(operand: Expr) -> Expr {
    Expr::UnOp {
        op: UnOp::Exp,
        operand: Box::new(operand),
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || moe_gate("scores", "indices", "weights", 8, 2),
        // Buffer order: scores (read-only f32 × 8), indices
        // (read-write u32 × 2), weights (output f32 × 2).
        Some(|| {
            let scores: [f32; 8] = [0.5, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            let scores_bytes = vyre_primitives::wire::pack_f32_slice(&scores);
            vec![vec![scores_bytes, vec![0u8; 4 * 2], vec![0u8; 4 * 2]]]
        }),
        Some(|| {
            let scores: [f32; 8] = [0.5, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = scores
                .iter()
                .map(|score| libm::expf(*score - max_score))
                .sum::<f32>();
            let indices: [u32; 2] = [5, 3];
            let idx_bytes = vyre_primitives::wire::pack_u32_slice(&indices);
            let expected_weights = [
                libm::expf(scores[5] - max_score) / sum_exp,
                libm::expf(scores[3] - max_score) / sum_exp,
            ];
            let weights = vyre_primitives::wire::pack_f32_slice(&expected_weights);
            vec![vec![idx_bytes, weights]]
        }),
    )
    .with_category("nn")
}

fn f32_fixture(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

fn u32_fixture(values: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(values)
}

fn softmax_stats_program() -> Program {
    tiled_reduce_program(TiledReduceProgram {
        generator: SOFTMAX_STATS_OP_ID,
        buffers: vec![
            BufferDecl::storage("scores", 0, BufferAccess::ReadOnly, DataType::F32).with_count(8),
            BufferDecl::storage(STATS_SCRATCH, 1, BufferAccess::ReadWrite, DataType::F32)
                .with_count(2),
            BufferDecl::workgroup(LANE_SCRATCH, GATE_TILE, DataType::F32),
        ],
        workgroup: [GATE_TILE, 1, 1],
        phases: softmax_stats_phases(SOFTMAX_STATS_OP_ID, "scores", 8),
        writeback: None,
    })
}

fn weight_write_program() -> Program {
    tiled_reduce_program(TiledReduceProgram {
        generator: WEIGHT_WRITE_OP_ID,
        buffers: vec![
            BufferDecl::storage("scores", 0, BufferAccess::ReadOnly, DataType::F32).with_count(8),
            BufferDecl::storage("indices", 1, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::storage(STATS_SCRATCH, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(2),
            BufferDecl::output("weights", 3, DataType::F32).with_count(2),
        ],
        workgroup: [GATE_TILE, 1, 1],
        phases: Vec::new(),
        writeback: Some(weight_write_child(
            WEIGHT_WRITE_OP_ID,
            "scores",
            "indices",
            "weights",
            2,
        )),
    })
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SOFTMAX_STATS_OP_ID,
        softmax_stats_program,
        Some(|| {
            let scores = [0.5_f32, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            vec![vec![f32_fixture(&scores), f32_fixture(&[0.0; 2])]]
        }),
        Some(|| {
            let scores = [0.5_f32, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = scores
                .iter()
                .map(|score| libm::expf(*score - max_score))
                .sum::<f32>();
            vec![vec![f32_fixture(&[max_score, sum_exp])]]
        }),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        WEIGHT_WRITE_OP_ID,
        weight_write_program,
        Some(|| {
            let scores = [0.5_f32, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = scores
                .iter()
                .map(|score| libm::expf(*score - max_score))
                .sum::<f32>();
            vec![vec![
                f32_fixture(&scores),
                u32_fixture(&[5, 3]),
                f32_fixture(&[max_score, sum_exp]),
                f32_fixture(&[0.0; 2]),
            ]]
        }),
        Some(|| {
            let scores = [0.5_f32, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp = scores
                .iter()
                .map(|score| libm::expf(*score - max_score))
                .sum::<f32>();
            vec![vec![f32_fixture(&[
                libm::expf(scores[5] - max_score) / sum_exp,
                libm::expf(scores[3] - max_score) / sum_exp,
            ])]]
        }),
    )
    .with_category("nn")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::f32_bytes;
    use vyre_reference::value::Value;

    fn u32_words(bytes: &[u8]) -> Vec<u32> {
        vyre_primitives::wire::decode_u32_le_bytes_all(bytes)
    }

    fn f32_words(bytes: &[u8]) -> Vec<f32> {
        vyre_primitives::wire::decode_f32_le_bytes_all(bytes)
    }

    #[test]
    fn moe_gate_outputs_unique_top_k_softmax_weights() {
        let scores: [f32; 8] = [0.5, 1.0, 0.1, 2.0, 0.3, 3.0, 0.2, 0.4];
        let program = moe_gate("scores", "indices", "weights", 8, 2);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&scores)),
                Value::from(vec![0u8; 8]),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: moe_gate must execute in the reference interpreter.");

        assert_eq!(u32_words(&outputs[0].to_bytes()), vec![5, 3]);
        let weights = f32_words(&outputs[1].to_bytes());
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp = scores
            .iter()
            .map(|score| libm::expf(*score - max_score))
            .sum::<f32>();
        let expected = [
            libm::expf(scores[5] - max_score) / sum_exp,
            libm::expf(scores[3] - max_score) / sum_exp,
        ];
        for (actual, expected) in weights.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn moe_gate_covers_an_expert_count_past_one_lane_pass() {
        // 300 experts over `GATE_TILE` lanes needs two chunks, so the last
        // chunk overshoots the space and the walk's bounds guard is what keeps
        // the reduction exact. A ceiling that dropped the tail would miss the
        // maximum at 299 and a guard that let the tail run would fold garbage
        // into the denominator.
        let mut scores: Vec<f32> = (0..300).map(|i| 0.001 * i as f32).collect();
        scores[150] = 5.0;
        scores[299] = 9.0;
        let program = moe_gate("scores", "indices", "weights", 300, 2);
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(&scores)),
                Value::from(vec![0u8; 8]),
                Value::from(vec![0u8; 8]),
            ],
        )
        .expect("Fix: moe_gate must execute in the reference interpreter.");

        assert_eq!(u32_words(&outputs[0].to_bytes()), vec![299, 150]);
        let sum_exp = scores
            .iter()
            .map(|score| libm::expf(*score - 9.0))
            .sum::<f32>();
        let expected = [1.0 / sum_exp, libm::expf(5.0 - 9.0) / sum_exp];
        let weights = f32_words(&outputs[1].to_bytes());
        for (actual, expected) in weights.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "got {actual}, want {expected}"
            );
        }
    }
}
