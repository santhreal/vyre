//! FlashAttention-2 tiling  -  sequence-tiled online-softmax attention.
//!
//! Two builders are provided:
//!
//! - [`flash_attention_2`]  -  tiled variant. Each invocation handles one
//!   query row. The KV sequence is processed in tiles of `tile_size`;
//!   scores for a whole tile are computed first, then the online-softmax
//!   state `(m, l, o_acc)` is updated once per tile.
//!
//! - [`flash_attention_2_reference`]  -  scalar per-row online-softmax
//!   without tiling. Used as a parity oracle.
//!
//! Category-A composition.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

use super::planner::plan_flash_attention_tiled;
use crate::region::wrap_anonymous;
use vyre_primitives::nn::attention_stability::{
    bounded_exp_arg, bounded_score, flush_tiny, positive_denominator,
};

const OP_ID: &str = "vyre-libs::nn::flash_attention_2";
const REFERENCE_OP_ID: &str = "vyre-libs::nn::flash_attention_2_reference";

/// Build a Program that computes FlashAttention-2 with explicit
/// sequence tiling.
///
/// Each invocation handles one query row.  The KV sequence is iterated
/// in tiles of `tile_size`.  For every tile all scores are computed
/// first, then the row-level online-softmax accumulator `(m, l, o_acc)`
/// is updated in one batched step.
///
/// # Parameters
///
/// * `tile_size`  -  number of keys processed per tile (e.g. 64 or 128).
///
/// # Errors
///
/// Returns a trap program when `seq_len == 0`, `head_dim == 0` or
/// `tile_size == 0`.
#[must_use]
pub fn flash_attention_2(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    seq_len: u32,
    head_dim: u32,
    tile_size: u32,
) -> Program {
    if seq_len == 0 || head_dim == 0 || tile_size == 0 {
        return crate::builder::invalid_output_program(
            OP_ID,
            out,
            DataType::F32,
            "Fix: flash_attention_2 seq_len, head_dim, and tile_size must all be > 0".to_string(),
        );
    }
    let plan = match plan_flash_attention_tiled(seq_len, head_dim, tile_size) {
        Ok(plan) => plan,
        Err(error) => {
            return crate::builder::invalid_output_program(OP_ID, out, DataType::F32, error);
        }
    };
    let scale_expr = Expr::f32(1.0f32 / (head_dim as f32).sqrt());
    let q_idx = |local: Expr, d: Expr| Expr::add(Expr::mul(local, Expr::u32(head_dim)), d);
    let score_idx = |local: Expr, j: Expr| Expr::add(Expr::mul(local, Expr::u32(tile_size)), j);
    let o_idx = |local: Expr, d: Expr| Expr::add(Expr::mul(local, Expr::u32(head_dim)), d);
    let compute_tile_scores = vec![Node::loop_for(
        "tile_j",
        Expr::u32(0),
        Expr::var("tile_len"),
        vec![
            Node::let_bind("dot_val", Expr::f32(0.0)),
            Node::loop_for(
                "score_d",
                Expr::u32(0),
                Expr::u32(head_dim),
                vec![Node::assign(
                    "dot_val",
                    Expr::add(
                        Expr::var("dot_val"),
                        Expr::mul(
                            Expr::load(
                                "q_scratch",
                                q_idx(Expr::var("local"), Expr::var("score_d")),
                            ),
                            Expr::load(
                                k,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("tile_start"), Expr::var("tile_j")),
                                        Expr::u32(head_dim),
                                    ),
                                    Expr::var("score_d"),
                                ),
                            ),
                        ),
                    ),
                )],
            ),
            Node::let_bind("raw_score", Expr::mul(Expr::var("dot_val"), scale_expr)),
            Node::let_bind("score", bounded_score(Expr::var("raw_score"))),
            Node::store(
                "score_tile",
                score_idx(Expr::var("local"), Expr::var("tile_j")),
                Expr::var("score"),
            ),
        ],
    )];
    let update_o_acc = vec![Node::loop_for(
        "out_d",
        Expr::u32(0),
        Expr::u32(head_dim),
        vec![
            Node::let_bind("weighted_v", Expr::f32(0.0)),
            Node::loop_for(
                "v_j",
                Expr::u32(0),
                Expr::var("tile_len"),
                vec![Node::assign(
                    "weighted_v",
                    Expr::add(
                        Expr::var("weighted_v"),
                        Expr::mul(
                            Expr::UnOp {
                                op: UnOp::Exp,
                                operand: Box::new(bounded_exp_arg(Expr::sub(
                                    Expr::load(
                                        "score_tile",
                                        score_idx(Expr::var("local"), Expr::var("v_j")),
                                    ),
                                    Expr::var("m_new"),
                                ))),
                            },
                            Expr::load(
                                v,
                                Expr::add(
                                    Expr::mul(
                                        Expr::add(Expr::var("tile_start"), Expr::var("v_j")),
                                        Expr::u32(head_dim),
                                    ),
                                    Expr::var("out_d"),
                                ),
                            ),
                        ),
                    ),
                )],
            ),
            Node::store(
                "o_acc",
                o_idx(Expr::var("local"), Expr::var("out_d")),
                Expr::add(
                    Expr::mul(
                        Expr::var("rescale"),
                        Expr::load("o_acc", o_idx(Expr::var("local"), Expr::var("out_d"))),
                    ),
                    Expr::var("weighted_v"),
                ),
            ),
        ],
    )];
    let body = super::tiled_online_softmax::tiled_online_softmax_body(
        super::tiled_online_softmax::TiledOnlineSoftmaxSpec {
            q,
            out,
            item_var: "row",
            item_count: seq_len,
            seq_len,
            head_dim,
            tile_size,
            tile_count: plan.tile_count,
        },
        compute_tile_scores,
        update_o_acc,
    );
    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(plan.logical_elements),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(plan.logical_elements),
            BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(plan.logical_elements),
            BufferDecl::workgroup("q_scratch", plan.q_scratch_elements, DataType::F32),
            BufferDecl::workgroup("score_tile", plan.score_scratch_elements, DataType::F32),
            BufferDecl::workgroup("o_acc", plan.o_acc_scratch_elements, DataType::F32),
            BufferDecl::output(out, 3, DataType::F32).with_count(plan.logical_elements),
        ],
        [plan.workgroup_lanes, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    )
}

/// Simple scalar online-softmax attention reference (no tiling).
///
/// One invocation per query row, workgroup size `[1, 1, 1]`.  This is
/// the standard flash-attention recurrence processed key-by-key.
#[must_use]
pub fn flash_attention_2_reference(
    q: &str,
    k: &str,
    v: &str,
    out: &str,
    seq_len: u32,
    head_dim: u32,
) -> Program {
    if seq_len == 0 || head_dim == 0 {
        return crate::builder::invalid_output_program(
            REFERENCE_OP_ID,
            out,
            DataType::F32,
            "Fix: flash_attention_2_reference seq_len and head_dim must be > 0".to_string(),
        );
    }

    let elements = match seq_len.checked_mul(head_dim) {
        Some(e) => e,
        None => {
            return crate::builder::invalid_output_program(
                REFERENCE_OP_ID,
                out,
                DataType::F32,
                "Fix: flash_attention_2_reference seq_len*head_dim overflows u32".to_string(),
            );
        }
    };

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let scale_expr = Expr::f32(scale);

    // Per-key online-softmax update.
    let key_body = vec![
        // score = scale * dot(Q[row,:], K[j,:])
        Node::let_bind("dot_val", Expr::f32(0.0)),
        Node::loop_for(
            "kd",
            Expr::u32(0),
            Expr::u32(head_dim),
            vec![Node::assign(
                "dot_val",
                Expr::add(
                    Expr::var("dot_val"),
                    Expr::mul(
                        Expr::load(
                            q,
                            Expr::add(
                                Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                                Expr::var("kd"),
                            ),
                        ),
                        Expr::load(
                            k,
                            Expr::add(
                                Expr::mul(Expr::var("j"), Expr::u32(head_dim)),
                                Expr::var("kd"),
                            ),
                        ),
                    ),
                ),
            )],
        ),
        Node::let_bind(
            "raw_score",
            Expr::mul(Expr::var("dot_val"), scale_expr.clone()),
        ),
        Node::let_bind("score", bounded_score(Expr::var("raw_score"))),
        // m_new = max(m, score)
        Node::let_bind(
            "m_new",
            Expr::select(
                Expr::gt(Expr::var("score"), Expr::var("m")),
                Expr::var("score"),
                Expr::var("m"),
            ),
        ),
        // rescale = exp(m - m_new)
        Node::let_bind(
            "rescale",
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(bounded_exp_arg(Expr::sub(
                    Expr::var("m"),
                    Expr::var("m_new"),
                ))),
            },
        ),
        // prob = exp(score - m_new)
        Node::let_bind(
            "prob",
            Expr::UnOp {
                op: UnOp::Exp,
                operand: Box::new(bounded_exp_arg(Expr::sub(
                    Expr::var("score"),
                    Expr::var("m_new"),
                ))),
            },
        ),
        // l = rescale * l + prob
        Node::let_bind(
            "l_new",
            Expr::add(
                Expr::mul(Expr::var("rescale"), Expr::var("l")),
                Expr::var("prob"),
            ),
        ),
        // o[d] = rescale * o[d] + prob * V[j, d]
        Node::loop_for(
            "od",
            Expr::u32(0),
            Expr::u32(head_dim),
            vec![Node::store(
                "o_ref",
                Expr::add(
                    Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                    Expr::var("od"),
                ),
                Expr::add(
                    Expr::mul(
                        Expr::var("rescale"),
                        Expr::load(
                            "o_ref",
                            Expr::add(
                                Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                                Expr::var("od"),
                            ),
                        ),
                    ),
                    Expr::mul(
                        Expr::var("prob"),
                        Expr::load(
                            v,
                            Expr::add(
                                Expr::mul(Expr::var("j"), Expr::u32(head_dim)),
                                Expr::var("od"),
                            ),
                        ),
                    ),
                ),
            )],
        ),
        Node::assign("m", Expr::var("m_new")),
        Node::assign("l", Expr::var("l_new")),
    ];

    let per_row = vec![
        Node::let_bind("m", Expr::f32(f32::MIN)),
        Node::let_bind("l", Expr::f32(0.0)),
        // Zero o_ref for this row
        Node::loop_for(
            "init_d",
            Expr::u32(0),
            Expr::u32(head_dim),
            vec![Node::store(
                "o_ref",
                Expr::add(
                    Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                    Expr::var("init_d"),
                ),
                Expr::f32(0.0),
            )],
        ),
        Node::loop_for("j", Expr::u32(0), Expr::u32(seq_len), key_body),
        // Write final output
        Node::let_bind("denom", positive_denominator(Expr::var("l"))),
        Node::loop_for(
            "write_d",
            Expr::u32(0),
            Expr::u32(head_dim),
            vec![Node::store(
                out,
                Expr::add(
                    Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                    Expr::var("write_d"),
                ),
                flush_tiny(Expr::div(
                    Expr::load(
                        "o_ref",
                        Expr::add(
                            Expr::mul(Expr::var("row"), Expr::u32(head_dim)),
                            Expr::var("write_d"),
                        ),
                    ),
                    Expr::var("denom"),
                )),
            )],
        ),
    ];

    let body = vec![
        Node::let_bind("row", Expr::InvocationId { axis: 0 }),
        Node::if_then(Expr::lt(Expr::var("row"), Expr::u32(seq_len)), per_row),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::storage(v, 2, BufferAccess::ReadOnly, DataType::F32).with_count(elements),
            BufferDecl::workgroup("o_ref", elements, DataType::F32),
            BufferDecl::output(out, 3, DataType::F32).with_count(elements),
        ],
        [1, 1, 1],
        vec![wrap_anonymous(REFERENCE_OP_ID, body)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_pack::decode_f32;
    use crate::test_support::byte_pack::f32_bytes;
    use vyre_reference::value::Value;

    fn run_program(program: Program, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let out_bytes = program
            .buffers()
            .iter()
            .find(|b| b.name() == "out")
            .map(|b| b.count() as usize * core::mem::size_of::<f32>())
            .expect("Fix: output buffer present");
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(f32_bytes(q)),
                Value::from(f32_bytes(k)),
                Value::from(f32_bytes(v)),
                Value::from(vec![0u8; out_bytes]),
            ],
        )
        .expect("Fix: reference eval must succeed");
        decode_f32(&outputs[0].to_bytes())
    }

    /// Tiled FlashAttention-2 agrees with the scalar reference on a
    /// non-trivial random fixture.
    #[test]
    fn flash_attention_2_matches_reference() {
        let seq_len = 8_u32;
        let head_dim = 16_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let q: Vec<f32> = (0..elements)
            .map(|i| ((i as f32) * 0.13).sin() - 0.5)
            .collect();
        let k: Vec<f32> = (0..elements)
            .map(|i| ((i as f32) * 0.07).cos() + 0.25)
            .collect();
        let v: Vec<f32> = (0..elements)
            .map(|i| ((i as f32) * 0.19).sin() * 2.0)
            .collect();

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            flash_attention_2_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        assert_eq!(actual.len(), expected.len(), "output length must match");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "flash_attention_2 vs reference mismatch at index {i}: {a} != {e}"
            );
        }
    }

    /// Output shape is `[seq_len, head_dim]`.
    #[test]
    fn flash_attention_2_output_shape() {
        let seq_len = 5_u32;
        let head_dim = 7_u32;
        let tile_size = 3_u32;
        let elements = (seq_len * head_dim) as usize;

        let q = vec![1.0f32; elements];
        let k = vec![0.5f32; elements];
        let v = vec![2.0f32; elements];

        let out = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        assert_eq!(out.len(), elements);
    }

    /// Edge case: `seq_len == 1` degenerates to passing V through
    /// (softmax of a length-1 vector is `[1.0]`).
    #[test]
    fn flash_attention_2_seq_len_one() {
        let seq_len = 1_u32;
        let head_dim = 4_u32;
        let tile_size = 1_u32;
        let q = vec![1.0f32, 2.0, 3.0, 4.0];
        let k = vec![0.5f32, 1.5, 2.5, 3.5];
        let v = vec![10.0f32, 20.0, 30.0, 40.0];

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            flash_attention_2_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len=1 mismatch at {i}: {a} != {e}"
            );
        }
    }

    /// Edge case: `seq_len == tile_size`.
    #[test]
    fn flash_attention_2_seq_len_eq_tile_size() {
        let seq_len = 4_u32;
        let head_dim = 8_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let q: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.1).collect();
        let k: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.05 + 0.2).collect();
        let v: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.3 - 0.1).collect();

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            flash_attention_2_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len==tile_size mismatch at {i}: {a} != {e}"
            );
        }
    }

    /// Edge case: `seq_len == tile_size + 1`.
    #[test]
    fn flash_attention_2_seq_len_eq_tile_size_plus_one() {
        let seq_len = 5_u32;
        let head_dim = 8_u32;
        let tile_size = 4_u32;
        let elements = (seq_len * head_dim) as usize;

        let q: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.11).collect();
        let k: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.06 + 0.15).collect();
        let v: Vec<f32> = (0..elements).map(|i| (i as f32) * 0.25 - 0.05).collect();

        let actual = run_program(
            flash_attention_2("q", "k", "v", "out", seq_len, head_dim, tile_size),
            &q,
            &k,
            &v,
        );
        let expected = run_program(
            flash_attention_2_reference("q", "k", "v", "out", seq_len, head_dim),
            &q,
            &k,
            &v,
        );

        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= 1.0e-3,
                "seq_len==tile_size+1 mismatch at {i}: {a} != {e}"
            );
        }
    }
}
