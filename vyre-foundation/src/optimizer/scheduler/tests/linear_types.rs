//! Linear-type scheduler enforcement.

use super::*;
use crate::validate::linear_type::check_linear_types;

#[test]
fn linear_type_enforcement_disabled_by_default_keeps_linear_breaking_rewrites() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(LinearBreakingPass {
        metadata: PassMetadata::new("linear_break_default_off", &[], &[]),
    })]);
    assert!(
        !scheduler.linear_type_enforcement(),
        "linear-type enforcement must default to OFF for compatibility"
    );

    let post = scheduler
        .run(linear_program())
        .expect("Fix: scheduler must converge");

    assert!(
        !check_linear_types(&post).is_empty(),
        "with the gate disabled, a linear-breaking rewrite still lands"
    );
}

#[test]
fn linear_type_enforcement_reverts_new_linear_violations() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(LinearBreakingPass {
        metadata: PassMetadata::new("linear_break_forbidden", &[], &[]),
    })])
    .with_linear_type_enforcement(true);

    let post = scheduler
        .run(linear_program())
        .expect("Fix: linear-type revert must converge");

    assert!(
        check_linear_types(&post).is_empty(),
        "linear-type gate must revert rewrites that break declared BufferAccess discipline"
    );
}

#[test]
fn linear_type_enforcement_metrics_reflect_post_revert_state() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(LinearBreakingPass {
        metadata: PassMetadata::new("linear_break_metric_check", &[], &[]),
    })])
    .with_linear_type_enforcement(true);

    let report = scheduler
        .run_with_metrics(linear_program())
        .expect("Fix: metrics run must converge");
    assert_eq!(report.passes.len(), 1);
    let metric = &report.passes[0];

    assert!(metric.ran, "linear-breaking pass must have run");
    assert!(
        !metric.changed,
        "reverted linear-type violations must not land"
    );
    assert_eq!(metric.decision, PassRunDecision::LinearTypeReverted);
    assert_eq!(metric.linear_type_violations_before, 0);
    assert_eq!(metric.linear_type_violations_after, 0);
    assert_eq!(metric.refusal_kind, None);
}

#[test]
fn linear_type_enforcement_allows_linear_preserving_rewrites() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ExprOnlyPass {
        metadata: PassMetadata::new("linear_preserving_expr_rewrite", &[], &[]),
    })])
    .with_linear_type_enforcement(true);

    let report = scheduler
        .run_with_metrics(linear_program())
        .expect("Fix: linear-preserving rewrite must converge");
    let metric = report
        .passes
        .iter()
        .find(|metric| metric.ran)
        .expect("Fix: preserving rewrite should produce one ran metric row");

    assert!(metric.changed, "value-only rewrite should land");
    assert_eq!(metric.decision, PassRunDecision::Changed);
    assert_eq!(metric.linear_type_violations_before, 0);
    assert_eq!(metric.linear_type_violations_after, 0);
}
