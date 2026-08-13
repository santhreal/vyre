//! Registered-witness proofs for the five per-row typedef phase ops.
//!
//! WHY: these five ops shipped with `None, None` fixtures, so nothing ever
//! executed them. Every case here pulls the entry out of the operation
//! registry, runs its registered program on the reference backend with its
//! registered `test_inputs`, and compares against its registered
//! `expected_output`, which the fixtures derive from the independent CPU
//! oracles in `ref_typedef`.
//!
//! The non-degeneracy and extent cases close the class the empty fixtures
//! invited. A one-row witness asked about row 0 makes every backward scan run
//! zero iterations and return the sentinel it started from, which satisfies a
//! presence gate while proving nothing; so does a declared buffer extent too
//! small to hold the witness the fixtures actually pass.
//!
//! Not caught here: whether the phases agree with the annotator that inlines
//! them. That is the annotator's own parity coverage.

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::all_entries;
use vyre_reference::value::Value;

const SCOPE_OPEN: &str = "vyre-libs::parsing::c11_typedef_scope_open_for_row";
const VISIBLE_NAME: &str = "vyre-libs::parsing::c11_typedef_visible_name_for_row";
const VISIBLE_NAME_PACKED: &str =
    "vyre-libs::parsing::c11_typedef_visible_name_for_row_packed_haystack";
const DECL_KIND: &str = "vyre-libs::parsing::c11_typedef_decl_kind_for_row";
const DECL_KIND_PACKED: &str = "vyre-libs::parsing::c11_typedef_decl_kind_for_row_packed_haystack";

const PHASE_OPS: [&str; 5] = [
    SCOPE_OPEN,
    VISIBLE_NAME,
    VISIBLE_NAME_PACKED,
    DECL_KIND,
    DECL_KIND_PACKED,
];

/// `SENTINEL` from the VAST row model: what a scope walk returns when it found
/// no enclosing `{`.
const SENTINEL: u32 = u32::MAX;

/// Declaration index of the `phase_row` scalar in each signature.
const SCOPE_OPEN_ROW_INPUT: usize = 1;
const HAYSTACK_PHASE_ROW_INPUT: usize = 2;

fn entry(id: &str) -> SemanticOperation {
    all_entries()
        .find(|entry| entry.id == id)
        .unwrap_or_else(|| panic!("Fix: {id} must be registered in the operation catalog"))
}

fn program(entry: &SemanticOperation) -> Program {
    entry
        .program()
        .unwrap_or_else(|| panic!("Fix: {} must provide a neutral builder", entry.id))
}

fn fixture_cases(entry: &SemanticOperation) -> (Vec<Vec<Vec<u8>>>, Vec<Vec<Vec<u8>>>) {
    let inputs = (entry
        .test_inputs
        .unwrap_or_else(|| panic!("Fix: {} must register witness inputs", entry.id)))(
    );
    let expected = (entry
        .expected_output
        .unwrap_or_else(|| panic!("Fix: {} must register oracle expected outputs", entry.id)))(
    );
    assert_eq!(
        inputs.len(),
        expected.len(),
        "Fix: {} registers {} input case(s) against {} expected case(s)",
        entry.id,
        inputs.len(),
        expected.len()
    );
    assert!(
        !inputs.is_empty(),
        "Fix: {} registers zero witness cases, which is zero coverage",
        entry.id
    );
    (inputs, expected)
}

/// Read the sole `u32` a phase op writes, for the case at `case_index`.
fn phase_result(case: &[Vec<u8>], id: &str) -> u32 {
    assert_eq!(
        case.len(),
        1,
        "Fix: {id} is a single-output phase but the fixture carries {} buffer(s)",
        case.len()
    );
    let bytes = &case[0];
    assert_eq!(
        bytes.len(),
        4,
        "Fix: {id} must produce exactly one u32, got {} byte(s)",
        bytes.len()
    );
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn row_argument(case: &[Vec<u8>], input_index: usize) -> u32 {
    let bytes = &case[input_index];
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn every_phase_op_matches_its_registered_oracle_expectation() {
    for id in PHASE_OPS {
        let entry = entry(id);
        let program = program(&entry);
        let (inputs, expected) = fixture_cases(&entry);

        for (case_index, (case_inputs, case_expected)) in
            inputs.iter().zip(expected.iter()).enumerate()
        {
            let values = case_inputs
                .iter()
                .cloned()
                .map(Value::from)
                .collect::<Vec<_>>();
            let actual = vyre_reference::reference_eval(&program, &values)
                .unwrap_or_else(|error| {
                    panic!("Fix: {id} case {case_index} failed on the reference backend: {error}")
                })
                .into_iter()
                .map(|value| value.to_bytes())
                .collect::<Vec<_>>();
            assert_eq!(
                actual, *case_expected,
                "{id} case {case_index}: reference backend produced {:?} but the CPU oracle expects {:?}",
                actual, case_expected
            );
        }
    }
}

/// WHY: every one of these phases starts from a sentinel and only leaves it by
/// scanning backwards. A witness that never makes a scan iterate would agree
/// with the oracle trivially, so pin that each phase left its start state and
/// that the row it was asked about is not row 0.
#[test]
fn phase_witnesses_leave_their_initial_scan_state() {
    for (id, row_input) in [
        (SCOPE_OPEN, SCOPE_OPEN_ROW_INPUT),
        (VISIBLE_NAME, HAYSTACK_PHASE_ROW_INPUT),
        (VISIBLE_NAME_PACKED, HAYSTACK_PHASE_ROW_INPUT),
        (DECL_KIND, HAYSTACK_PHASE_ROW_INPUT),
        (DECL_KIND_PACKED, HAYSTACK_PHASE_ROW_INPUT),
    ] {
        let entry = entry(id);
        let (inputs, expected) = fixture_cases(&entry);
        for (case_index, (case_inputs, case_expected)) in
            inputs.iter().zip(expected.iter()).enumerate()
        {
            let row = row_argument(case_inputs, row_input);
            assert_ne!(
                row, 0,
                "Fix: {id} case {case_index} asks about row 0, where every backward scan runs zero iterations"
            );
            let result = phase_result(case_expected, id);
            let initial = if id == SCOPE_OPEN { SENTINEL } else { 0 };
            assert_ne!(
                result, initial,
                "Fix: {id} case {case_index} returns its initial scan state, so the witness proves nothing"
            );
        }
    }
}

/// WHY: the callee buffer extents are declaration-only, but they still have to
/// describe the witness the registry hands the interpreter. Shrinking one back
/// to a single row or a single haystack word must not go unnoticed.
#[test]
fn declared_buffer_extents_hold_the_registered_witness() {
    for id in PHASE_OPS {
        let entry = entry(id);
        let program = program(&entry);
        let (inputs, _) = fixture_cases(&entry);

        let declared = program
            .buffers()
            .iter()
            .filter(|buffer| {
                !buffer.is_backend_allocated_output() && buffer.access() != BufferAccess::Workgroup
            })
            .map(|buffer| (buffer.name().to_string(), buffer.count() as usize * 4))
            .collect::<Vec<_>>();

        for (case_index, case_inputs) in inputs.iter().enumerate() {
            assert_eq!(
                case_inputs.len(),
                declared.len(),
                "Fix: {id} case {case_index} passes {} buffer(s) against {} declared input buffer(s)",
                case_inputs.len(),
                declared.len()
            );
            for (bytes, (name, declared_bytes)) in case_inputs.iter().zip(declared.iter()) {
                assert_eq!(
                    bytes.len(),
                    *declared_bytes,
                    "Fix: {id} case {case_index} buffer `{name}` carries {} byte(s) against a declared extent of {declared_bytes}",
                    bytes.len()
                );
            }
        }
    }
}

/// WHY: the packed and resident-expanded haystack variants read the same
/// source text through different loads, so their fixtures must differ in bytes
/// while agreeing on the answer.
#[test]
fn packed_and_resident_haystack_variants_agree_on_the_answer() {
    for (resident, packed) in [
        (VISIBLE_NAME, VISIBLE_NAME_PACKED),
        (DECL_KIND, DECL_KIND_PACKED),
    ] {
        let (resident_inputs, resident_expected) = fixture_cases(&entry(resident));
        let (packed_inputs, packed_expected) = fixture_cases(&entry(packed));
        assert_eq!(
            resident_expected, packed_expected,
            "Fix: {resident} and {packed} must answer identically for one source text"
        );
        assert_ne!(
            resident_inputs[0][1], packed_inputs[0][1],
            "Fix: {packed} must encode its haystack four bytes per word, not one"
        );
    }
}
