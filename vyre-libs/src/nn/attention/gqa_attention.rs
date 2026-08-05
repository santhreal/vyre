//! Grouped-Query Attention: n_q Q heads, n_kv KV heads (replicate K/V).
//!
//! Full 3-pass softmax (max, sum, weighted-write) with KV-head broadcasting.

use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::ir::model::expr::GeneratorRef;
use vyre_primitives::nn::attention_passes::{
    attention_max_pass_with_bases, attention_sum_pass_with_bases, attention_write_pass_with_bases,
    ATTENTION_MAX_PASS_OP_ID, ATTENTION_SUM_PASS_OP_ID, ATTENTION_WRITE_PASS_OP_ID,
};
use vyre_primitives::nn::attention_stability::positive_denominator;

use crate::region::{wrap_anonymous, wrap_child};

const OP_ID: &str = "vyre-libs::nn::gqa_attention";

/// Build GQA attention (F32). n_q_heads must be a multiple of n_kv_heads.
///
/// # Errors
/// Returns `Err` on dimension violations.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &str,
    k: &str,
    v_buf: &str,
    output: &str,
    n_q_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
) -> Result<Program, String> {
    if n_q_heads == 0 || n_kv_heads == 0 || seq_len == 0 || head_dim == 0 {
        return Err("Fix: gqa_attention requires non-zero dims".into());
    }
    if n_q_heads % n_kv_heads != 0 {
        return Err("Fix: n_q_heads must be multiple of n_kv_heads".into());
    }
    let group_size = n_q_heads / n_kv_heads;
    let per_head = seq_len.checked_mul(head_dim).ok_or_else(|| {
        "gqa_attention sequence and head dimensions overflow u32. Fix: shard the attention input"
            .to_string()
    })?;
    let q_rows = n_q_heads.checked_mul(seq_len).ok_or_else(|| {
        "gqa_attention query row count overflows u32. Fix: shard the query heads".to_string()
    })?;
    let q_total = n_q_heads.checked_mul(per_head).ok_or_else(|| {
        "gqa_attention query element count overflows u32. Fix: shard the query heads".to_string()
    })?;
    let kv_total = n_kv_heads.checked_mul(per_head).ok_or_else(|| {
        "gqa_attention key/value element count overflows u32. Fix: shard the key/value heads"
            .to_string()
    })?;
    let scale_expr = Expr::f32(1.0f32 / (head_dim as f32).sqrt());

    let row_index = Expr::var("i");
    let q_head = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(row_index.clone()),
        right: Box::new(Expr::u32(seq_len)),
    };
    let kv_head = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(q_head),
        right: Box::new(Expr::u32(group_size)),
    };
    let query_base = Expr::mul(row_index.clone(), Expr::u32(head_dim));
    let kv_base = Expr::mul(kv_head, Expr::u32(per_head));
    let parent = GeneratorRef {
        name: OP_ID.to_string(),
    };

    let body = vec![
        Node::let_bind("i", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(row_index, Expr::u32(q_rows)),
            vec![
                Node::let_bind("max_val", Expr::f32(f32::MIN)),
                wrap_child(
                    ATTENTION_MAX_PASS_OP_ID,
                    parent.clone(),
                    attention_max_pass_with_bases(
                        q,
                        k,
                        head_dim,
                        seq_len,
                        scale_expr.clone(),
                        query_base.clone(),
                        kv_base.clone(),
                    ),
                ),
                Node::let_bind("sum_val", Expr::f32(0.0)),
                wrap_child(
                    ATTENTION_SUM_PASS_OP_ID,
                    parent.clone(),
                    attention_sum_pass_with_bases(
                        q,
                        k,
                        head_dim,
                        seq_len,
                        scale_expr.clone(),
                        query_base.clone(),
                        kv_base.clone(),
                    ),
                ),
                Node::let_bind("denom", positive_denominator(Expr::var("sum_val"))),
                wrap_child(
                    ATTENTION_WRITE_PASS_OP_ID,
                    parent,
                    attention_write_pass_with_bases(
                        q,
                        k,
                        v_buf,
                        head_dim,
                        seq_len,
                        scale_expr,
                        output,
                        query_base.clone(),
                        kv_base.clone(),
                        kv_base,
                        query_base,
                    ),
                ),
            ],
        ),
    ];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(q, 0, BufferAccess::ReadOnly, DataType::F32).with_count(q_total),
            BufferDecl::storage(k, 1, BufferAccess::ReadOnly, DataType::F32).with_count(kv_total),
            BufferDecl::storage(v_buf, 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count(kv_total),
            BufferDecl::output(output, 3, DataType::F32).with_count(q_total),
        ],
        [64, 1, 1],
        vec![wrap_anonymous(OP_ID, body)],
    ))
}

inventory::submit! {
    crate::fixture_catalog::OpEntry {
        id: OP_ID,
        build: || {
            gqa_attention("q", "k", "v", "out", 2, 1, 2, 2)
                .unwrap_or_else(|error| crate::invalid_program(OP_ID, format!("Fix: gqa_attention fixture must build: {error}")))
        },
        test_inputs: Some(|| {
            let f = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                f(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
                f(&[1.0, 0.0, 0.0, 1.0]),
                f(&[10.0, 20.0, 30.0, 40.0]),
                vec![0u8; 32],
            ]]
        }),
        expected_output: Some(|| {
            vec![vec![vec![
                145, 214, 132, 65, 146, 214, 212, 65, 111, 41, 187, 65, 183, 148, 5, 66, 111,
                41, 187, 65, 183, 148, 5, 66, 145, 214, 132, 65, 146, 214, 212, 65,
            ]]]
        }),
        category: Some("nn"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_pack::decode_f32;
    use crate::test_support::byte_pack::f32_bytes;
    use vyre_reference::value::Value;

    #[test]
    fn gqa_attention_zero_sequence_length_rejected() {
        let err =
            gqa_attention("q", "k", "v", "out", 2, 1, 0, 4).expect_err("zero seq_len must error");
        assert!(err.contains("seq_len=0") || err.contains("non-zero"));
    }

    #[test]
    fn gqa_attention_single_token() {
        let n_q = 2u32;
        let n_kv = 1u32;
        let s = 1u32;
        let d = 2u32;
        let q = [1.0f32, 0.0, 0.0, 1.0];
        let k = [1.0f32, 0.0];
        let v = [10.0f32, 20.0];
        let prog = gqa_attention("q", "k", "v", "out", n_q, n_kv, s, d).expect("Fix: build");
        let outputs = vyre_reference::reference_eval(
            &prog,
            &[
                Value::from(f32_bytes(&q)),
                Value::from(f32_bytes(&k)),
                Value::from(f32_bytes(&v)),
                Value::from(vec![0u8; (n_q * s * d) as usize * 4]),
            ],
        )
        .expect("Fix: gqa_attention single token must execute");
        let out = decode_f32(&outputs[0].to_bytes());
        // With one token, softmax is [1.0], so output equals V broadcast.
        for (i, &v) in out.iter().enumerate() {
            let expected = if i % 2 == 0 { 10.0 } else { 20.0 };
            assert!(
                (v - expected).abs() <= 1.0e-4,
                "gqa_attention single token mismatch at {i}: {v} != {expected}"
            );
        }
    }

    #[test]
    fn gqa_attention_very_large_qk_values_stay_finite() {
        let n_q = 1u32;
        let n_kv = 1u32;
        let s = 2u32;
        let d = 2u32;
        let q = [1e20f32; 4];
        let k = [1e20f32; 4];
        let v = [1.0f32; 4];
        let prog = gqa_attention("q", "k", "v", "out", n_q, n_kv, s, d).expect("Fix: build");
        let outputs = vyre_reference::reference_eval(
            &prog,
            &[
                Value::from(f32_bytes(&q)),
                Value::from(f32_bytes(&k)),
                Value::from(f32_bytes(&v)),
                Value::from(vec![0u8; (n_q * s * d) as usize * 4]),
            ],
        )
        .expect("Fix: gqa_attention must not panic on large QK values");
        let out = decode_f32(&outputs[0].to_bytes());
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "gqa_attention output at {i} must be finite for large QK values, got {v}"
            );
        }
    }

    #[test]
    fn gqa_attention_nan_in_q_k_v_propagates() {
        let n_q = 1u32;
        let n_kv = 1u32;
        let s = 1u32;
        let d = 2u32;
        let q = [f32::NAN, 0.0];
        let k = [0.0f32, 0.0];
        let v = [1.0f32, 2.0];
        let prog = gqa_attention("q", "k", "v", "out", n_q, n_kv, s, d).expect("Fix: build");
        let outputs = vyre_reference::reference_eval(
            &prog,
            &[
                Value::from(f32_bytes(&q)),
                Value::from(f32_bytes(&k)),
                Value::from(f32_bytes(&v)),
                Value::from(vec![0u8; (n_q * s * d) as usize * 4]),
            ],
        )
        .expect("Fix: gqa_attention must not panic on NaN input");
        let out = decode_f32(&outputs[0].to_bytes());
        assert!(
            out.iter().any(|v| v.is_nan()),
            "gqa_attention must propagate NaN in Q/K/V instead of silently producing finite output {:?}",
            out
        );
    }
}
