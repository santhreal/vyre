//! Every registered pass belongs to the pipeline of the level it declares.
//!
//! WHY this suite exists: the scheduler holds one flat order over every pass,
//! so nothing stated which IR level a pass acts at and nothing stopped a
//! logical rewrite from being scheduled after a schedule rewrite, reading a
//! program that already carries the physical constructs its preconditions were
//! stated without. `optimizer::level_pipeline` partitions the scheduled order
//! by the level each pass's rewrite contract declares, and this suite holds
//! that partition to the two sources it is derived from:
//!
//! - the `IrLevel` variants `vyre-spec` declares, read from source, so adding a
//!   level turns this suite red until a pipeline row exists for it;
//! - the registered pass set, so a pass that declares no contract is not
//!   quietly dropped out of every pipeline.
//!
//! It also asserts the order itself: within a level the scheduled order is
//! preserved, and across levels the scheduled order never runs a level before
//! one that precedes it.
//!
//! What this does NOT catch: whether the level a pass declares is the level it
//! actually rewrites. `rewrite_contract_closure` holds the declared level
//! against the pass phase, and the level's own verifier is what proves a
//! program leaving the pipeline is in that level's canonical form.

use std::collections::BTreeSet;

use vyre_foundation::optimizer::level_pipeline::{
    level_inversions, level_of_pass, level_pipelines,
};
use vyre_foundation::optimizer::registered_pass_registrations;
use vyre_test_support::declared_level_variants;

fn registered_pass_names() -> BTreeSet<&'static str> {
    registered_pass_registrations()
        .expect("the registered pass set must schedule")
        .iter()
        .map(|registration| registration.metadata.name)
        .collect()
}

/// Adding an IR level turns this suite red until it has a pipeline.
#[test]
fn every_declared_ir_level_has_a_pipeline() {
    let declared = declared_level_variants();
    assert!(
        declared.len() >= 5,
        "Fix: the IrLevel source enumeration found only {} variants; the scan is broken, not the \
         enum",
        declared.len()
    );

    let pipelines = level_pipelines().expect("the registered pass set must schedule");
    assert_eq!(
        pipelines.len(),
        declared.len(),
        "Fix: {} levels are declared and {} pipelines exist; every level owns one pipeline, \
         empty or not",
        declared.len(),
        pipelines.len()
    );

    let rendered: BTreeSet<&'static str> = pipelines
        .iter()
        .map(|pipeline| pipeline.level().name())
        .collect();
    assert_eq!(
        rendered.len(),
        pipelines.len(),
        "Fix: two pipelines render the same level name: {rendered:?}"
    );
}

/// The pipelines partition the registered pass set.
#[test]
fn the_pipelines_partition_the_registered_pass_set() {
    let pipelines = level_pipelines().expect("the registered pass set must schedule");
    let mut seen: Vec<&'static str> = Vec::new();
    for pipeline in &pipelines {
        seen.extend_from_slice(pipeline.passes());
    }
    let unique: BTreeSet<&'static str> = seen.iter().copied().collect();
    assert_eq!(
        unique.len(),
        seen.len(),
        "Fix: a pass appears in more than one level pipeline; a pass declares one level"
    );

    let registered = registered_pass_names();
    let missing: Vec<&&'static str> = registered
        .iter()
        .filter(|name| !unique.contains(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: these registered passes belong to no level pipeline, so nothing states the level \
         they rewrite: {missing:?}"
    );

    let unknown: Vec<&&'static str> = unique
        .iter()
        .filter(|name| !registered.contains(*name))
        .collect();
    assert!(
        unknown.is_empty(),
        "Fix: these pipeline members are not registered passes: {unknown:?}"
    );
}

/// A pipeline preserves the scheduled order of its own passes.
#[test]
fn each_pipeline_preserves_the_scheduled_order() {
    let scheduled: Vec<&'static str> = registered_pass_registrations()
        .expect("the registered pass set must schedule")
        .iter()
        .map(|registration| registration.metadata.name)
        .collect();
    let position = |name: &str| {
        scheduled
            .iter()
            .position(|scheduled| *scheduled == name)
            .unwrap_or_else(|| panic!("Fix: pipeline member {name} is not in the scheduled order"))
    };

    for pipeline in level_pipelines().expect("the registered pass set must schedule") {
        let positions: Vec<usize> = pipeline
            .passes()
            .iter()
            .map(|name| position(name))
            .collect();
        let mut sorted = positions.clone();
        sorted.sort_unstable();
        assert_eq!(
            positions,
            sorted,
            "Fix: the {} pipeline reorders its passes against the scheduled order",
            pipeline.level().name()
        );
    }
}

/// Every pipeline member declares the level of the pipeline it is in.
#[test]
fn a_pipeline_member_declares_that_pipeline_level() {
    for pipeline in level_pipelines().expect("the registered pass set must schedule") {
        for pass in pipeline.passes() {
            assert_eq!(
                level_of_pass(pass),
                Some(pipeline.level()),
                "Fix: {pass} is in the {} pipeline but declares another level",
                pipeline.level().name()
            );
        }
    }
}

/// The scheduled order never runs a level before a level that precedes it.
///
/// A logical rewrite scheduled after a schedule rewrite reads a program that
/// already carries physical constructs. Its preconditions were stated about a
/// program that did not, so the inversion is a correctness question and not an
/// ordering preference.
#[test]
fn the_scheduled_order_runs_levels_in_order() {
    let inversions = level_inversions().expect("the registered pass set must schedule");
    assert!(
        inversions.is_empty(),
        "Fix: the scheduled order runs a level before one that precedes it. Either the pass \
         declares the wrong level in its rewrite contract, or it needs a declared requirement \
         that orders it before the deeper pass: {inversions:?}"
    );
}

/// The inversion scan reports a shallower level against the deepest pass that
/// preceded it.
///
/// The live registry is in order, so every assertion about the scan itself is
/// made against a synthetic order. Without these cases the scan could report
/// nothing at all and the suite would still pass.
#[test]
fn the_inversion_scan_reads_an_arbitrary_order() {
    use vyre_foundation::optimizer::level_pipeline::inversions_in_order;
    use vyre_spec::IrLevel::{Logical, PhysicalKernel, Schedule, WholeGraph};

    assert_eq!(
        inversions_in_order(&[("a", WholeGraph), ("b", Logical), ("c", Schedule)]),
        Vec::new(),
        "Fix: a deepening order holds no inversion"
    );

    let one = inversions_in_order(&[("a", Logical), ("b", Schedule), ("c", Logical)]);
    assert_eq!(
        one.len(),
        1,
        "Fix: the scan missed a logical pass after a schedule pass: {one:?}"
    );
    assert_eq!(one[0].earlier, "b");
    assert_eq!(one[0].earlier_level, Schedule);
    assert_eq!(one[0].later, "c");
    assert_eq!(one[0].later_level, Logical);

    let both = inversions_in_order(&[
        ("a", Logical),
        ("b", PhysicalKernel),
        ("c", Logical),
        ("d", Schedule),
    ]);
    assert_eq!(
        both.len(),
        2,
        "Fix: every pass after the deepest one is reported, not only the first: {both:?}"
    );
    assert!(
        both.iter().all(|inversion| inversion.earlier == "b"),
        "Fix: an inversion is reported against the deepest pass that preceded it: {both:?}"
    );
}
