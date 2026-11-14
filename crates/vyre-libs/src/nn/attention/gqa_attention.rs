//! Grouped-Query Attention: n_q Q heads, n_kv KV heads (replicate K/V).
//!
//! Full 3-pass softmax (max, sum, weighted-write) with KV-head broadcasting.

use crate::nn::attention_passes::{
    attention_max_pass_bounded, attention_sum_pass_bounded, attention_write_pass_bounded_typed,
    ATTENTION_MAX_PASS_OP_ID, ATTENTION_SUM_PASS_OP_ID, ATTENTION_WRITE_PASS_OP_ID,
};
use crate::nn::attention_stability::positive_denominator;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

const OP_ID: &str = "vyre-libs::nn::gqa_attention";

#[allow(clippy::too_many_arguments)]
fn three_pass_softmax_row_nodes(
    q: &str,
    k: &str,
    v_buf: &str,
    output: &str,
    head_dim: u32,
    key_limit: Expr,
    scale: Expr,
    query_base: Expr,
    kv_base: Expr,
    dtype: DataType,
) -> Vec<Node> {
    let parent = Ident::from(OP_ID);
    vec![
        Node::let_bind("max_val", Expr::f32(f32::MIN)),
        wrap_child_region(
            ATTENTION_MAX_PASS_OP_ID,
            parent.clone(),
            attention_max_pass_bounded(
                q,
                k,
                head_dim,
                key_limit.clone(),
                scale.clone(),
                query_base.clone(),
                kv_base.clone(),
            ),
        ),
        Node::let_bind("sum_val", Expr::f32(0.0)),
        wrap_child_region(
            ATTENTION_SUM_PASS_OP_ID,
            parent.clone(),
            attention_sum_pass_bounded(
                q,
                k,
                head_dim,
                key_limit.clone(),
                scale.clone(),
                query_base.clone(),
                kv_base.clone(),
            ),
        ),
        Node::let_bind("denom", positive_denominator(Expr::var("sum_val"))),
        wrap_child_region(
            ATTENTION_WRITE_PASS_OP_ID,
            parent,
            attention_write_pass_bounded_typed(
                q,
                k,
                v_buf,
                head_dim,
                key_limit,
                scale,
                output,
                query_base.clone(),
                kv_base.clone(),
                kv_base,
                query_base,
                dtype,
            ),
        ),
    ]
}

fn attention_buffers<'a>(
    q: &'a str,
    k: &'a str,
    v_buf: &'a str,
    output: &'a str,
    q_total: u32,
    kv_total: u32,
    dtype: DataType,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(q, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(q_total),
        BufferDecl::storage(k, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(kv_total),
        BufferDecl::storage(v_buf, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(kv_total),
        BufferDecl::output(output, 3, dtype).with_count(q_total),
    ]
}

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
        String::from(
            "gqa_attention sequence and head dimensions overflow u32. Fix: shard the attention input",
        )
    })?;
    let q_rows = n_q_heads.checked_mul(seq_len).ok_or_else(|| {
        String::from("gqa_attention query row count overflows u32. Fix: shard the query heads")
    })?;
    let q_total = n_q_heads.checked_mul(per_head).ok_or_else(|| {
        String::from("gqa_attention query element count overflows u32. Fix: shard the query heads")
    })?;
    let kv_total = n_kv_heads.checked_mul(per_head).ok_or_else(|| {
        String::from(
            "gqa_attention key/value element count overflows u32. Fix: shard the key/value heads",
        )
    })?;
    let scale_expr = Expr::f32(1.0f32 / (head_dim as f32).sqrt());

    let q_head = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(Expr::var("i")),
        right: Box::new(Expr::u32(seq_len)),
    };
    let kv_head = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(q_head),
        right: Box::new(Expr::u32(group_size)),
    };
    let query_base = Expr::mul(Expr::var("i"), Expr::u32(head_dim));
    let kv_base = Expr::mul(kv_head, Expr::u32(per_head));

    let body = vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("i"), Expr::u32(q_rows)),
            three_pass_softmax_row_nodes(
                q,
                k,
                v_buf,
                output,
                head_dim,
                Expr::u32(seq_len),
                scale_expr,
                query_base,
                kv_base,
                DataType::F32,
            ),
        ),
    ];

    Ok(Program::wrapped(
        attention_buffers(q, k, v_buf, output, q_total, kv_total, DataType::F32),
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}

/// Build batch-aware causal GQA over a prompt or cached decode window.
///
/// Query rows use `query_len`; K/V cache rows use `kv_len`. Query token `t`
/// may attend exactly through `cache_offset + t`, never later cache entries.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_causal(
    q: &str,
    k: &str,
    v_buf: &str,
    output: &str,
    batch: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    query_len: u32,
    kv_len: u32,
    head_dim: u32,
    cache_offset: u32,
) -> Result<Program, String> {
    gqa_attention_causal_typed(
        q,
        k,
        v_buf,
        output,
        batch,
        n_q_heads,
        n_kv_heads,
        query_len,
        kv_len,
        head_dim,
        cache_offset,
        DataType::F32,
    )
}

/// Build typed batch-aware causal GQA with F32 score and value accumulation.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_causal_typed(
    q: &str,
    k: &str,
    v_buf: &str,
    output: &str,
    batch: u32,
    n_q_heads: u32,
    n_kv_heads: u32,
    query_len: u32,
    kv_len: u32,
    head_dim: u32,
    cache_offset: u32,
    dtype: DataType,
) -> Result<Program, String> {
    if batch == 0
        || n_q_heads == 0
        || n_kv_heads == 0
        || query_len == 0
        || kv_len == 0
        || head_dim == 0
    {
        return Err("Fix: gqa_attention_causal requires non-zero dimensions".into());
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: gqa_attention_causal_typed supports F16, BF16, or F32 tensors; got {dtype:?}"
        ));
    }
    if n_q_heads % n_kv_heads != 0 {
        return Err("Fix: n_q_heads must be multiple of n_kv_heads".into());
    }
    if cache_offset
        .checked_add(query_len)
        .is_none_or(|end| end > kv_len)
    {
        return Err(format!(
            "Fix: causal GQA query range offset={cache_offset}, query_len={query_len} exceeds kv_len={kv_len}"
        ));
    }
    let checked = |values: &[u32], label: &str| {
        values.iter().try_fold(1_u32, |product, value| {
            product
                .checked_mul(*value)
                .ok_or_else(|| format!("Fix: causal GQA {label} element count overflows u32"))
        })
    };
    let q_total = checked(&[batch, n_q_heads, query_len, head_dim], "query")?;
    let kv_total = checked(&[batch, n_kv_heads, kv_len, head_dim], "KV")?;
    let rows = checked(&[batch, n_q_heads, query_len], "row")?;
    let q_head_span = checked(&[query_len, head_dim], "query head span")?;
    let kv_head_span = checked(&[kv_len, head_dim], "KV head span")?;
    let q_batch_span = checked(&[n_q_heads, q_head_span], "query batch span")?;
    let kv_batch_span = checked(&[n_kv_heads, kv_head_span], "KV batch span")?;
    let group = n_q_heads / n_kv_heads;
    let q_rows_per_batch = n_q_heads * query_len;
    let batch_index = Expr::div(Expr::var("row"), Expr::u32(q_rows_per_batch));
    let batch_row = Expr::sub(
        Expr::var("row"),
        Expr::mul(batch_index.clone(), Expr::u32(q_rows_per_batch)),
    );
    let query_head = Expr::div(batch_row.clone(), Expr::u32(query_len));
    let query_token = Expr::sub(
        batch_row,
        Expr::mul(query_head.clone(), Expr::u32(query_len)),
    );
    let kv_head = Expr::div(query_head.clone(), Expr::u32(group));
    let query_base = Expr::add(
        Expr::mul(batch_index.clone(), Expr::u32(q_batch_span)),
        Expr::add(
            Expr::mul(query_head, Expr::u32(q_head_span)),
            Expr::mul(query_token.clone(), Expr::u32(head_dim)),
        ),
    );
    let kv_base = Expr::add(
        Expr::mul(batch_index, Expr::u32(kv_batch_span)),
        Expr::mul(kv_head, Expr::u32(kv_head_span)),
    );
    let key_limit = Expr::add(
        Expr::add(Expr::u32(cache_offset), query_token),
        Expr::u32(1),
    );
    let scale = Expr::f32(1.0 / (head_dim as f32).sqrt());
    let body = vec![
        Node::let_bind("row", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(
            Expr::lt(Expr::var("row"), Expr::u32(rows)),
            three_pass_softmax_row_nodes(
                q,
                k,
                v_buf,
                output,
                head_dim,
                key_limit,
                scale,
                query_base,
                kv_base,
                dtype.clone(),
            ),
        ),
    ];
    Ok(Program::wrapped(
        attention_buffers(q, k, v_buf, output, q_total, kv_total, dtype),
        [64, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::nn::gqa_attention_causal",
            body,
        )],
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || {
            gqa_attention("q", "k", "v", "out", 2, 1, 2, 2)
                .unwrap_or_else(|error| trap_program(OP_ID, None, format!("Fix: gqa_attention fixture must build: {error}")))
        },
        Some(|| {
            let f = vyre_primitives::wire::pack_f32_slice;
            vec![vec![
                f(&[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
                f(&[1.0, 0.0, 0.0, 1.0]),
                f(&[10.0, 20.0, 30.0, 40.0]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                145, 214, 132, 65, 146, 214, 212, 65, 111, 41, 187, 65, 183, 148, 5, 66, 111,
                41, 187, 65, 183, 148, 5, 66, 145, 214, 132, 65, 146, 214, 212, 65,
            ]]]
        }),
    )
    .with_category("nn")
    .with_numeric(vyre_foundation::numeric::NumericContract::ieee_f32(4))
    .with_uncharacterized()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_test_support::test_parity_oracles::{decode_f32, eval_bytes, f32_bytes};

    fn eval_gqa(n_q: u32, n_kv: u32, s: u32, d: u32, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        let prog = gqa_attention("q", "k", "v", "out", n_q, n_kv, s, d).expect("Fix: build");
        let outputs = eval_bytes(
            "gqa_attention",
            &prog,
            vec![
                f32_bytes(q),
                f32_bytes(k),
                f32_bytes(v),
                vec![0u8; (n_q * s * d) as usize * 4],
            ],
        );
        decode_f32(&outputs[0])
    }

    #[test]
    fn gqa_attention_zero_sequence_length_rejected() {
        let err =
            gqa_attention("q", "k", "v", "out", 2, 1, 0, 4).expect_err("zero seq_len must error");
        assert!(err.contains("seq_len=0") || err.contains("non-zero"));
    }

    #[test]
    fn gqa_attention_single_token() {
        let out = eval_gqa(
            2,
            1,
            1,
            2,
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0],
            &[10.0, 20.0],
        );
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
        let out = eval_gqa(1, 1, 2, 2, &[1e20; 4], &[1e20; 4], &[1.0; 4]);
        for (i, &v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "gqa_attention output at {i} must be finite for large QK values, got {v}"
            );
        }
    }

    #[test]
    fn gqa_attention_nan_in_q_k_v_propagates() {
        let out = eval_gqa(1, 1, 1, 2, &[f32::NAN, 0.0], &[0.0, 0.0], &[1.0, 2.0]);
        assert!(
            out.iter().any(|v| v.is_nan()),
            "gqa_attention must propagate NaN in Q/K/V instead of silently producing finite output {out:?}",
        );
    }
}
