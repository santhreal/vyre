//! Overflow contracts for QK-gain tensor shapes.

#![cfg(feature = "nn-attention")]

use vyre_libs::nn::attention::qk_gain;

fn rejected_shape_message(num_heads: u32, seq_len: u32, head_dim: u32) -> String {
    let program = qk_gain("q_in", "q_out", "gain", num_heads, seq_len, head_dim);
    vyre_reference::reference_eval(&program, &[])
        .expect_err("an overflowing QK-gain shape must produce a trap program")
        .to_string()
}

/// Per-head shape multiplication must fail closed instead of wrapping to an undersized buffer.
#[test]
fn per_head_element_count_overflow_is_rejected() {
    let message = rejected_shape_message(1, u32::MAX, 2);
    assert!(
        message.contains(
            "qk_gain per-head element count overflows u32 for seq_len=4294967295, head_dim=2"
        ),
        "unexpected overflow diagnostic: {message}"
    );
}

/// Total shape multiplication must reject overflow even when one head fits in `u32`.
#[test]
fn total_element_count_overflow_is_rejected() {
    let message = rejected_shape_message(2, u32::MAX, 1);
    assert!(
        message.contains(
            "qk_gain total element count overflows u32 for num_heads=2, seq_len=4294967295, head_dim=1"
        ),
        "unexpected overflow diagnostic: {message}"
    );
}
