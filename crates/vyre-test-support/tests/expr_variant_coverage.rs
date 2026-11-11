//! Integration test verifying complete derived coverage of all Expr variants and operand slots.

use vyre_foundation::ir::Expr;
use vyre_test_support::expr_variants::{
    assert_covers_every_expr_variant, expr_operand_slot_samples, expr_variant_samples,
};

#[test]
fn expr_variant_universe_is_completely_covered() {
    let samples = expr_variant_samples();
    assert_covers_every_expr_variant(&samples);
}

#[test]
fn expr_operand_slots_plant_marker_correctly() {
    let marker = Expr::var("MARKER");
    let slot_samples = expr_operand_slot_samples(&marker);
    assert!(!slot_samples.is_empty());
    for sample in slot_samples {
        assert!(sample.slot.is_some());
        let debug = format!("{:?}", sample.expr);
        assert!(
            debug.contains("MARKER"),
            "sample for {} did not contain the planted marker",
            sample.label()
        );
    }
}
