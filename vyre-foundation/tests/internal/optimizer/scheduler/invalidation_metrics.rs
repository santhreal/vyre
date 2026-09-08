//! Integration test crate for the containing Vyre package.

use super::*;

#[test]
fn invalidating_prior_requirement_does_not_break_current_iteration() {
    let scheduler = PassScheduler::with_passes(vec![
        ProgramPassKind::new(TestPass {
            metadata: PassMetadata::new("prepare", &[], &[]),
            changes: false,
        }),
        ProgramPassKind::new(TestPass {
            metadata: PassMetadata::new("rewrite", &[], &["prepare"]),
            changes: true,
        }),
        ProgramPassKind::new(TestPass {
            metadata: PassMetadata::new("consume", &["prepare"], &[]),
            changes: false,
        }),
    ]);
    let report = scheduler
        .run_with_metrics(trivial_program())
        .expect("Fix: invalidating a prior requirement must queue a rerun, not make later passes unschedulable");

    assert!(
        report.passes.len() >= 6,
        "first iteration must queue prepare and consume for a second dirty-flag iteration"
    );
    assert!(
        report
            .passes
            .iter()
            .any(|metric| metric.iteration == 0 && metric.pass == "rewrite" && metric.changed),
        "the rewrite pass must land a change during the first metrics iteration"
    );
    assert_eq!(report.passes[3].pass, "prepare");
    assert!(
        report.passes[3].ran,
        "invalidating `prepare` must rerun the named pass on the next metrics iteration"
    );
    assert!(
        report
            .passes
            .iter()
            .any(|metric| metric.iteration == 1 && metric.pass == "consume" && metric.ran),
        "invalidating `prepare` must rerun dependents that require it"
    );
}

#[test]
fn run_with_metrics_tracks_expression_only_rewrites() {
    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(ExprOnlyPass {
        metadata: PassMetadata::new("expr_only", &[], &["value_numbering"]),
    })]);

    let report = scheduler
        .run_with_metrics(trivial_program())
        .expect("Fix: metrics run must converge for expression-only rewrites");
    assert_eq!(report.passes.len(), 2);
    let first = &report.passes[0];
    assert_eq!(first.pass, "expr_only");
    assert!(
        first.changed,
        "expression-only rewrites keep node_count stable but still changed the program and must invalidate downstream facts"
    );
    assert_eq!(
        first.nodes_before, first.nodes_after,
        "the regression target is a same-node-count expression rewrite"
    );
    assert!(
        !report.passes[1].changed,
        "the second iteration must observe convergence after the expression rewrite landed"
    );
}

/// A pass fingerprint separates two instances configured differently.
///
/// WHY: caches key a pass result on the fingerprint, so two instances that
/// rewrite to different values and report the same fingerprint let the first
/// result be served for the second. `StoreValueRewritePass` carries its
/// configuration in the fingerprint, and this holds it to that: same
/// configuration, same fingerprint; different target value, different
/// fingerprint. The rewrite itself converges, which is what makes the
/// difference observable rather than a claim about a number.
#[test]
fn a_configured_rewrite_fingerprints_its_configuration() {
    let rewrite_to_seven = StoreValueRewritePass {
        metadata: PassMetadata::new("store_rewrite", &[], &[]),
        from: 42,
        to: 7,
    };
    let rewrite_to_eight = StoreValueRewritePass {
        metadata: PassMetadata::new("store_rewrite", &[], &[]),
        from: 42,
        to: 8,
    };
    let program = trivial_program();

    assert_eq!(
        rewrite_to_seven.fingerprint(&program),
        StoreValueRewritePass {
            metadata: PassMetadata::new("store_rewrite", &[], &[]),
            from: 42,
            to: 7,
        }
        .fingerprint(&program),
        "the same configuration must fingerprint the same, or no result is ever reused"
    );
    assert_ne!(
        rewrite_to_seven.fingerprint(&program),
        rewrite_to_eight.fingerprint(&program),
        "two target values must not share a fingerprint; a cache would serve one for the other"
    );

    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(StoreValueRewritePass {
        metadata: PassMetadata::new("store_rewrite", &[], &[]),
        from: 42,
        to: 7,
    })]);
    let report = scheduler
        .run_with_metrics(trivial_program())
        .expect("Fix: a store-value rewrite must converge");
    assert!(
        report.passes[0].changed,
        "the first run must land the rewrite the configuration names"
    );
    assert!(
        !report.passes[1].changed,
        "the second run must find nothing left to rewrite"
    );
}
