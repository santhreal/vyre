//! Verifies that INT4 Cat-A wrappers retain their Tier-2.5 primitive provenance.

#![cfg(feature = "nn-linear-4bit")]

use vyre::ir::{Node, Program};
use vyre_libs::nn::quant::{
    int4_batched_matmul_f32_scaled, int4_batched_matmul_top1_f32_scaled,
    int4_batched_matvec_f32_scaled, int4_dot_f32_scaled, int4_dot_i32, int4_matvec_f32_scaled,
};

fn assert_composition_chain(program: &Program, parent_id: &str, primitive_id: &str) {
    let [Node::Region {
        generator,
        source_region: None,
        body,
    }] = program.entry()
    else {
        panic!("expected one Cat-A parent region");
    };
    assert_eq!(generator.as_ref(), parent_id);
    let [Node::Region {
        generator,
        source_region: Some(parent),
        ..
    }] = body.as_slice()
    else {
        panic!("expected one Tier-2.5 child region");
    };
    assert_eq!(generator.as_ref(), primitive_id);
    assert_eq!(parent.as_str(), parent_id);
}

/// This test prevents the two public INT4 dot wrappers from presenting primitive IR as parent-owned implementation work.
#[test]
fn dot_wrappers_record_cat_a_parent_and_primitive_child() {
    let dot = int4_dot_i32("lhs", "rhs", "out", 8);
    assert_composition_chain(
        &dot,
        "vyre-libs::quant::int4_dot_i32",
        "vyre-libs::math::quantized::i4x8_dot_i32",
    );

    let scaled = int4_dot_f32_scaled("lhs", "rhs", "lhs_scale", "rhs_scale", "out", 8);
    assert_composition_chain(
        &scaled,
        "vyre-libs::quant::int4_dot_f32_scaled",
        "vyre-libs::math::quantized::i4x8_dot_f32_scaled",
    );
}

/// This test locks every matrix-vector and matrix-matrix wrapper to its canonical Tier-2.5 implementation instead of a copied body.
#[test]
fn matrix_wrappers_record_their_exact_primitive_children() {
    let matvec = int4_matvec_f32_scaled("weights", "x", "row_scales", "out", 3, 9);
    assert_composition_chain(
        &matvec,
        "vyre-libs::quant::int4_matvec_f32_scaled",
        "vyre-libs::math::quantized::i4x8_matvec_f32_scaled",
    );

    let batched_matvec =
        int4_batched_matvec_f32_scaled("weights", "x", "row_scales", "out", 2, 3, 9);
    assert_composition_chain(
        &batched_matvec,
        "vyre-libs::quant::int4_batched_matvec_f32_scaled",
        "vyre-libs::math::quantized::i4x8_batched_matvec_f32_scaled",
    );

    let batched_matmul = int4_batched_matmul_f32_scaled(
        "weights",
        "activations",
        "row_scales",
        "batch_scales",
        "out",
        2,
        3,
        9,
    );
    assert_composition_chain(
        &batched_matmul,
        "vyre-libs::quant::int4_batched_matmul_f32_scaled",
        "vyre-libs::math::quantized::i4x8_batched_matmul_f32_scaled",
    );

    let top1 = int4_batched_matmul_top1_f32_scaled(
        "weights",
        "activations",
        "row_scales",
        "batch_scales",
        "out",
        2,
        3,
        9,
    );
    assert_composition_chain(
        &top1,
        "vyre-libs::quant::int4_batched_matmul_top1_f32_scaled",
        "vyre-libs::math::quantized::i4x8_batched_matmul_top1_f32_scaled",
    );
}

/// This boundary test proves invalid zero-lane requests keep their primitive diagnostic body beneath the Cat-A wrapper rather than bypassing provenance.
#[test]
fn zero_lane_dot_keeps_composition_chain_and_trap() {
    let program = int4_dot_i32("lhs", "rhs", "out", 0);

    assert_composition_chain(
        &program,
        "vyre-libs::quant::int4_dot_i32",
        "vyre-libs::math::quantized::i4x8_dot_i32",
    );
    assert!(program.stats().trap());
}
