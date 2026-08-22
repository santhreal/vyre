//! Integration test crate for the containing Vyre package.

use super::*;

/// Twenty distinct pass names, enough that the scheduler's lookup tables are
/// exercised by index rather than by a single-entry map.
const STRESS_PASS_NAMES: [&str; 20] = [
    "stress_pass_00",
    "stress_pass_01",
    "stress_pass_02",
    "stress_pass_03",
    "stress_pass_04",
    "stress_pass_05",
    "stress_pass_06",
    "stress_pass_07",
    "stress_pass_08",
    "stress_pass_09",
    "stress_pass_10",
    "stress_pass_11",
    "stress_pass_12",
    "stress_pass_13",
    "stress_pass_14",
    "stress_pass_15",
    "stress_pass_16",
    "stress_pass_17",
    "stress_pass_18",
    "stress_pass_19",
];

#[test]
fn scheduler_lookup_tables_use_static_str_keys() {
    // Structural assertion: pass_index must be FxHashMap<&'static str, usize>.
    fn assert_static_str_map(_: &FxHashMap<&'static str, usize>) {}

    let scheduler = PassScheduler::try_default().expect("Fix: built-in passes must be valid");
    assert_static_str_map(&scheduler.pass_index);

    // N=20 passes: build scheduler, topo-sort runs inside with_passes, then
    // exercise the lookup loop via the metrics runner and direct query methods.
    //
    // The names are literals rather than leaked boxes. A leak is what a
    // `&'static str` key costs, and the free that paid it back reconstituted a
    // `Box` from a shared reference while the scheduler still held that key:
    // a retag the aliasing model rejects over memory a live table points at.
    let names: Vec<&'static str> = STRESS_PASS_NAMES.to_vec();
    let passes: Vec<_> = names
        .iter()
        .map(|&name| {
            ProgramPassKind::new(TestPass {
                metadata: PassMetadata::new(name, &[], &[]),
                changes: false,
            })
        })
        .collect();
    let scheduler20 = PassScheduler::with_passes(passes);
    assert_static_str_map(&scheduler20.pass_index);

    // Lookup phase: run_with_metrics iterates execution_order and checks the
    // indexed dirty flags for every pass.
    let report = scheduler20
        .run_with_metrics(trivial_program())
        .expect("Fix: stress scheduler must run");
    assert_eq!(
        report.passes.len(),
        names.len(),
        "all clean test passes should be considered exactly once before convergence"
    );

    // Direct pass_index lookups via public query API.
    for &name in &names {
        assert!(
            !scheduler20.reaches(name, name),
            "a pass must not reach itself"
        );
        assert!(
            scheduler20.pass_index.contains_key(name),
            "pass_index must contain {name}"
        );
    }
}

#[test]
fn scheduler_preserves_program_identity_when_pass_skips() {
    let program = trivial_program();
    let original_entry = Arc::clone(program.entry_arc());

    let scheduler = PassScheduler::with_passes(vec![ProgramPassKind::new(SkipPass)]);
    let result = scheduler
        .run(program)
        .expect("Fix: scheduler must converge when all passes SKIP");

    assert!(
        Arc::ptr_eq(&original_entry, result.entry_arc()),
        "scheduler must preserve entry Arc identity when a pass returns SKIP; \
         reconcile_runnable_top_level must not allocate a fresh Vec or Arc"
    );
}
