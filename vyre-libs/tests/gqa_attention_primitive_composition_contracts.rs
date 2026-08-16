//! Grouped-query attention primitive composition and boundary contracts.

#![cfg(feature = "nn-attention")]

mod wire_words;
use wire_words::{f32_bytes, f32_words as decode_f32};

use std::collections::HashMap;

use vyre::ir::Node;
use vyre_foundation::visit::walk_nodes;
use vyre_libs::nn::attention::gqa_attention;
use vyre_libs::math::dot_partial::OP_ID as DOT_PARTIAL_OP_ID;
use vyre_libs::nn::attention_passes::{
    ATTENTION_MAX_PASS_OP_ID, ATTENTION_SUM_PASS_OP_ID, ATTENTION_WRITE_PASS_OP_ID,
};
use vyre_reference::value::Value;

/// GQA must compose canonical attention pass bodies, not duplicate their IR under copied op ids.
#[test]
fn gqa_contains_dot_partial_children_owned_by_each_canonical_attention_pass() {
    let program = gqa_attention("q", "k", "v", "out", 4, 2, 2, 4)
        .expect("valid grouped-query dimensions must build");
    let mut pass_regions = HashMap::<String, usize>::new();
    let mut dot_parents = HashMap::<String, usize>::new();

    walk_nodes(&program, |node| {
        let Node::Region {
            generator,
            source_region,
            ..
        } = node
        else {
            return;
        };
        if [
            ATTENTION_MAX_PASS_OP_ID,
            ATTENTION_SUM_PASS_OP_ID,
            ATTENTION_WRITE_PASS_OP_ID,
        ]
        .contains(&generator.as_str())
        {
            *pass_regions
                .entry(generator.as_str().to_string())
                .or_default() += 1;
        }
        if generator.as_str() == DOT_PARTIAL_OP_ID {
            let parent = source_region
                .as_ref()
                .expect("canonical dot-partial regions must record their attention-pass parent");
            *dot_parents.entry(parent.as_str().to_string()).or_default() += 1;
        }
    });

    assert_eq!(
        pass_regions,
        HashMap::from([
            (ATTENTION_MAX_PASS_OP_ID.to_string(), 1),
            (ATTENTION_SUM_PASS_OP_ID.to_string(), 1),
            (ATTENTION_WRITE_PASS_OP_ID.to_string(), 1),
        ])
    );
    assert_eq!(
        dot_parents,
        HashMap::from([
            (ATTENTION_MAX_PASS_OP_ID.to_string(), 1),
            (ATTENTION_SUM_PASS_OP_ID.to_string(), 1),
            (ATTENTION_WRITE_PASS_OP_ID.to_string(), 1),
        ])
    );
}

/// A one-token GQA row must broadcast each KV head to exactly its assigned query-head group.
#[test]
fn gqa_single_token_broadcasts_distinct_kv_heads_to_their_query_groups() {
    let program = gqa_attention("q", "k", "v", "out", 4, 2, 1, 2)
        .expect("valid grouped-query dimensions must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(f32_bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0])),
            Value::from(f32_bytes(&[1.0, 0.0, 0.0, 1.0])),
            Value::from(f32_bytes(&[10.0, 20.0, 30.0, 40.0])),
            Value::from(vec![0u8; 8 * size_of::<f32>()]),
        ],
    )
    .expect("reference execution must evaluate canonical GQA composition");

    assert_eq!(
        decode_f32(&outputs[0].to_bytes()),
        vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
    );
}

/// Query-head groups that cannot map evenly onto KV heads must fail before IR construction.
#[test]
fn gqa_rejects_non_divisible_query_to_kv_head_groups() {
    let error = gqa_attention("q", "k", "v", "out", 3, 2, 1, 4)
        .expect_err("three query heads cannot be partitioned across two KV heads");
    assert_eq!(error, "Fix: n_q_heads must be multiple of n_kv_heads");
}

/// Element-count overflow must return a sharding error instead of wrapping buffer declarations.
#[test]
fn gqa_rejects_u32_element_count_overflow() {
    let error = gqa_attention("q", "k", "v", "out", u32::MAX, 1, 1, 2)
        .expect_err("query element count must not wrap u32");
    assert_eq!(
        error,
        "gqa_attention query element count overflows u32. Fix: shard the query heads"
    );
}
