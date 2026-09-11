//! What a sanitizer report owes a consumer, and what PMU evidence owes a
//! reviewer.
//!
//! WHY: a sanitizer report is read by tooling that jumps to the faulting
//! access, so each coordinate is a typed `context_values` entry keyed by name.
//! Formatting one into prose and leaving it in `notes` puts the address behind
//! a parser, which is what these cases rule out. A coordinate the failure does
//! not carry emits no key at all, so a consumer can tell "unknown" from zero.

use vyre_debug::{PmuExpectation, PmuMeasurement, PmuWarning, SanitizerFailure, SanitizerKind};
use vyre_foundation::diagnostics::{Diagnostic, DiagnosticStage, Severity};

fn context_value<'a>(diag: &'a Diagnostic, key: &str) -> Option<&'a str> {
    diag.context_values
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.as_str())
}

/// Every coordinate key the diagnostic builder can emit.
const COORDINATE_KEYS: &[&str] = &[
    "device_address",
    "invocation_id",
    "instruction_offset",
    "tool_raw_output",
];

#[test]
fn a_data_race_reports_its_coordinates_as_typed_context_values() {
    let failure = SanitizerFailure::data_race(
        "read-after-write data race on buffer `shared_acc`",
        0x7fff_0000_1234,
        [32, 0, 0],
    );

    let diag = failure.diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "SAN003_DATA_RACE");
    assert_eq!(diag.stage, DiagnosticStage::Materialize);
    assert!(diag
        .suggested_fix
        .as_deref()
        .is_some_and(|fix| fix.contains("insert Barrier")));
    assert_eq!(
        context_value(&diag, "device_address"),
        Some("0x00007fff00001234")
    );
    assert_eq!(context_value(&diag, "invocation_id"), Some("32,0,0"));
}

#[test]
fn an_unknown_coordinate_emits_no_key() {
    // `out_of_bounds` carries an address and nothing else. A builder that
    // defaulted the rest to zero would report invocation [0, 0, 0] as fact.
    let diag = SanitizerFailure::out_of_bounds(
        "global memory access beyond buffer allocation",
        0x1000_dead_beef,
    )
    .diagnostic();

    assert_eq!(diag.code.as_str(), "SAN004_OUT_OF_BOUNDS");
    assert!(diag
        .suggested_fix
        .as_deref()
        .is_some_and(|fix| fix.contains("clamp index expressions")));
    assert_eq!(
        context_value(&diag, "device_address"),
        Some("0x00001000deadbeef")
    );
    for key in ["invocation_id", "instruction_offset", "tool_raw_output"] {
        assert_eq!(
            context_value(&diag, key),
            None,
            "a coordinate the failure does not carry must not appear as {key}"
        );
    }
}

#[test]
fn every_coordinate_a_failure_carries_reaches_context_values_and_none_reaches_notes() {
    // Populated by field so a coordinate added to `SanitizerFailure` without a
    // `context_values` arm leaves its key missing here.
    let failure = SanitizerFailure {
        kind: SanitizerKind::IllegalInstruction,
        message: "illegal instruction on an unsupported ISA profile".to_string(),
        device_address: Some(0xdead_0000_0010),
        invocation_coords: Some([7, 3, 1]),
        instruction_offset: Some(0x2a),
        raw_tool_output: Some("tool: fault at pc 0x2a".to_string()),
    };

    let diag = failure.diagnostic();
    let observed: Vec<&str> = diag
        .context_values
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(
        observed, COORDINATE_KEYS,
        "a fully populated failure must emit every coordinate key once, in builder order"
    );
    assert_eq!(context_value(&diag, "instruction_offset"), Some("0x002a"));
    assert_eq!(
        context_value(&diag, "tool_raw_output"),
        Some("tool: fault at pc 0x2a")
    );

    assert!(
        diag.notes.is_empty(),
        "coordinates belong in context_values; notes reintroduces the prose form a consumer would \
         have to parse: {:?}",
        diag.notes
    );
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
