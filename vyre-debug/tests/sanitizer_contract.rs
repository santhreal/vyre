//! Tests for sanitizer correctness failures vs PMU performance evidence.

use vyre_debug::{PmuExpectation, PmuMeasurement, PmuWarning, SanitizerFailure};
use vyre_foundation::diagnostics::{DiagnosticStage, Severity};

#[test]
fn sanitizer_failures_map_to_hard_error_diagnostics() {
    let failure = SanitizerFailure::data_race(
        "read-after-write data race on buffer `shared_acc`",
        0x7fff_0000_1234,
        [32, 0, 0],
    );

    let diag = failure.diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "SAN003_DATA_RACE");
    assert_eq!(diag.stage, DiagnosticStage::Materialize);
    assert!(diag.suggested_fix.unwrap().contains("insert Barrier"));
    assert!(diag.notes.iter().any(|n| n.contains("0x00007fff00001234")));
    assert!(diag.notes.iter().any(|n| n.contains("[32, 0, 0]")));
}

#[test]
fn out_of_bounds_sanitizer_maps_to_actionable_diagnostic() {
    let failure = SanitizerFailure::out_of_bounds(
        "global memory access beyond buffer allocation",
        0x1000_dead_beef,
    );

    let diag = failure.diagnostic();
    assert_eq!(diag.code.as_str(), "SAN004_OUT_OF_BOUNDS");
    assert!(diag
        .suggested_fix
        .unwrap()
        .contains("clamp index expressions"));
}

#[test]
fn pmu_evaluates_dense_vs_sparse_workload_expectations() {
    // Dense regular expectation
    let dense_exp = PmuExpectation::dense_regular();
    assert!(!dense_exp.allow_uncoalesced_traffic);

    let dense_measurement = PmuMeasurement {
        spill_bytes: 0,
        bank_conflicts: 0,
        uncoalesced_transactions: 128,
        occupancy_pct: 75.0,
        achieved_bandwidth_gb_s: 850.0,
    };
    let warnings = dense_measurement.evaluate(&dense_exp);
    assert_eq!(warnings.len(), 1);
    assert!(matches!(
        warnings[0],
        PmuWarning::UncoalescedTrafficOnDenseWorkload { observed: 128 }
    ));

    // Sparse / gather expectation
    let sparse_exp = PmuExpectation::sparse_or_gather();
    assert!(sparse_exp.allow_uncoalesced_traffic);

    let sparse_measurement = PmuMeasurement {
        spill_bytes: 0,
        bank_conflicts: 16,
        uncoalesced_transactions: 1024, // permitted for gather/sparse
        occupancy_pct: 40.0,
        achieved_bandwidth_gb_s: 320.0,
    };
    let sparse_warnings = sparse_measurement.evaluate(&sparse_exp);
    assert!(
        sparse_warnings.is_empty(),
        "sparse workload allows uncoalesced transactions and minor bank conflicts"
    );
}
