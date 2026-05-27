//! Shape-predicate scheduler enforcement.

use super::*;
use crate::validate::shape_predicate::check_shape_predicates;

#[test]
fn shape_predicate_enforcement_disabled_by_default_keeps_shape_breaking_rewrites() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ShapeBreakingPass {
        metadata: PassMetadata::new("shape_break_default_off", &[], &[]),
    })]);
    assert!(
        !scheduler.shape_predicate_enforcement(),
        "shape-predicate enforcement must default to OFF for compatibility"
    );

    let post = scheduler
        .run(shape_predicate_program())
        .expect("Fix: scheduler must converge");

    assert!(
        !check_shape_predicates(&post).is_empty(),
        "with the gate disabled, a shape-breaking rewrite still lands"
    );
}

#[test]
fn shape_predicate_enforcement_reverts_new_shape_violations() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ShapeBreakingPass {
        metadata: PassMetadata::new("shape_break_forbidden", &[], &[]),
    })])
    .with_shape_predicate_enforcement(true);

    let post = scheduler
        .run(shape_predicate_program())
        .expect("Fix: shape-predicate revert must converge");

    assert!(
        check_shape_predicates(&post).is_empty(),
        "shape-predicate gate must revert rewrites that break declared BufferDecl refinements"
    );
}

#[test]
fn shape_predicate_enforcement_metrics_reflect_post_revert_state() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ShapeBreakingPass {
        metadata: PassMetadata::new("shape_break_metric_check", &[], &[]),
    })])
    .with_shape_predicate_enforcement(true);

    let report = scheduler
        .run_with_metrics(shape_predicate_program())
        .expect("Fix: metrics run must converge");
    assert_eq!(report.passes.len(), 1);
    let metric = &report.passes[0];

    assert!(metric.ran, "shape-breaking pass must have run");
    assert!(
        !metric.changed,
        "reverted shape-predicate violations must not land"
    );
    assert_eq!(metric.decision, PassRunDecision::ShapePredicateReverted);
    assert_eq!(metric.shape_predicate_violations_before, 0);
    assert_eq!(metric.shape_predicate_violations_after, 0);
    assert_eq!(metric.refusal_kind, None);
}

#[test]
fn shape_predicate_enforcement_allows_shape_preserving_rewrites() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ExprOnlyPass {
        metadata: PassMetadata::new("shape_preserving_expr_rewrite", &[], &[]),
    })])
    .with_shape_predicate_enforcement(true);

    let report = scheduler
        .run_with_metrics(shape_predicate_program())
        .expect("Fix: shape-preserving rewrite must converge");
    let metric = report
        .passes
        .iter()
        .find(|metric| metric.ran)
        .expect("Fix: preserving rewrite should produce one ran metric row");

    assert!(metric.changed, "value-only rewrite should land");
    assert_eq!(metric.decision, PassRunDecision::Changed);
    assert_eq!(metric.shape_predicate_violations_before, 0);
    assert_eq!(metric.shape_predicate_violations_after, 0);
}

#[test]
fn shape_predicate_enforcement_allows_repairs_of_existing_shape_violations() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ShapeRepairingPass {
        metadata: PassMetadata::new("shape_repair_existing_violation", &[], &[]),
    })])
    .with_shape_predicate_enforcement(true);

    let pre = invalid_shape_predicate_program();
    assert_eq!(
        check_shape_predicates(&pre).len(),
        1,
        "Fix: repair test must start from a genuine shape-predicate violation"
    );

    let report = scheduler
        .run_with_metrics(pre)
        .expect("Fix: shape-repairing rewrite must converge");
    let metric = report
        .passes
        .iter()
        .find(|metric| metric.ran)
        .expect("Fix: repairing rewrite should produce one ran metric row");

    assert!(metric.changed, "repairing rewrite should land");
    assert_eq!(metric.decision, PassRunDecision::Changed);
    assert_eq!(metric.shape_predicate_violations_before, 1);
    assert_eq!(metric.shape_predicate_violations_after, 0);
    assert!(
        check_shape_predicates(&report.program).is_empty(),
        "shape-predicate gate must allow rewrites that reduce existing violations"
    );
}
