//! Registered-witness proofs for the per-row typedef phase ops.
//!
//! WHY: these ops shipped with `None, None` fixtures, so nothing ever executed
//! them. Every case here pulls the entry out of the operation registry, runs its
//! registered program on the reference backend with its registered
//! `test_inputs`, and compares against its registered `expected_output`, which
//! the fixtures derive from the independent CPU oracles in `ref_typedef`.
//!
//! The phase set is discovered from the registry by the buffer contract in
//! `vast::phase_program`, so registering a new phase op joins it to every case
//! here and a new phase without fixtures fails instead of passing silently.
//!
//! The non-degeneracy and extent cases close the class the empty fixtures
//! invited. A witness asked about row 0 makes every backward scan run zero
//! iterations and return the state it started from, which satisfies a presence
//! gate while proving nothing; so does a declared buffer extent too small to
//! hold the witness the fixtures actually pass.
//!
//! Not caught here: whether the phases agree with the annotator that inlines
//! them. That is the annotator's own parity coverage.

use vyre_foundation::ir::{BufferAccess, Program};
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::all_entries;
use vyre_reference::value::Value;

/// The node table every phase reads, and the marker that an op is a phase.
const NODES_INPUT: &str = "phase_vast_nodes";
/// The row a phase answers about.
const ROW_INPUT: &str = "phase_row";
/// Suffix of the variant that reads four source bytes per haystack word.
const PACKED_SUFFIX: &str = "_packed_haystack";

/// Every registered operation built on the per-row phase buffer contract.
fn phase_ops() -> Vec<SemanticOperation> {
    let ops = all_entries()
        .filter(|entry| {
            entry
                .program()
                .is_some_and(|program| declares_phase_contract(&program))
        })
        .collect::<Vec<_>>();
    assert!(
        ops.len() >= 5,
        "Fix: the registry exposes {} phase op(s); the typedef row phases are gone or renamed",
        ops.len()
    );
    ops
}

fn declares_phase_contract(program: &Program) -> bool {
    let names = input_buffers(program);
    names.first().is_some_and(|(name, _)| name == NODES_INPUT)
        && names.iter().any(|(name, _)| name == ROW_INPUT)
}

/// Declared input buffers in argument order, with the byte extent each declares.
fn input_buffers(program: &Program) -> Vec<(String, usize)> {
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            !buffer.is_backend_allocated_output() && buffer.access() != BufferAccess::Workgroup
        })
        .map(|buffer| (buffer.name().to_string(), buffer.count() as usize * 4))
        .collect()
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

/// Read the sole `u32` a phase op writes, for one case.
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

/// Declaration index of the row scalar in this op's argument list.
fn row_input_index(program: &Program, id: &str) -> usize {
    input_buffers(program)
        .iter()
        .position(|(name, _)| name == ROW_INPUT)
        .unwrap_or_else(|| panic!("Fix: {id} declares no `{ROW_INPUT}` argument"))
}

fn row_argument(case: &[Vec<u8>], input_index: usize) -> u32 {
    let bytes = &case[input_index];
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn run(program: &Program, case_inputs: &[Vec<u8>], id: &str, case_index: usize) -> Vec<Vec<u8>> {
    let values = case_inputs
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    vyre_reference::reference_eval(program, &values)
        .unwrap_or_else(|error| {
            panic!("Fix: {id} case {case_index} failed on the reference backend: {error}")
        })
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

#[test]
fn every_phase_op_matches_its_registered_oracle_expectation() {
    for entry in phase_ops() {
        let id = entry.id;
        let program = program(&entry);
        let (inputs, expected) = fixture_cases(&entry);

        for (case_index, (case_inputs, case_expected)) in
            inputs.iter().zip(expected.iter()).enumerate()
        {
            let actual = run(&program, case_inputs, id, case_index);
            assert_eq!(
                actual, *case_expected,
                "{id} case {case_index}: reference backend produced {actual:?} but the CPU oracle expects {case_expected:?}"
            );
        }
    }
}

/// WHY: every one of these phases starts from a fixed state and only leaves it
/// by scanning away from the row it was asked about. A witness that never makes
/// a scan iterate would agree with the oracle trivially, so pin that the row is
/// not row 0 and that asking about row 0 gives a different answer.
#[test]
fn phase_witnesses_depend_on_the_row_they_ask_about() {
    for entry in phase_ops() {
        let id = entry.id;
        let program = program(&entry);
        let row_input = row_input_index(&program, id);
        let (inputs, expected) = fixture_cases(&entry);

        for (case_index, (case_inputs, case_expected)) in
            inputs.iter().zip(expected.iter()).enumerate()
        {
            assert_ne!(
                row_argument(case_inputs, row_input),
                0,
                "Fix: {id} case {case_index} asks about row 0, where every backward scan runs zero iterations"
            );
            let mut row_zero = case_inputs.clone();
            row_zero[row_input] = 0u32.to_le_bytes().to_vec();
            let degenerate = run(&program, &row_zero, id, case_index);
            assert_ne!(
                phase_result(case_expected, id),
                phase_result(&degenerate, id),
                "Fix: {id} case {case_index} answers row 0 the same way, so the witness proves nothing"
            );
        }
    }
}

/// WHY: the callee buffer extents are declaration-only, but they still have to
/// describe the witness the registry hands the interpreter. Shrinking one back
/// to a single row or a single haystack word must not go unnoticed.
#[test]
fn declared_buffer_extents_hold_the_registered_witness() {
    for entry in phase_ops() {
        let id = entry.id;
        let program = program(&entry);
        let (inputs, _) = fixture_cases(&entry);
        let declared = input_buffers(&program);

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

/// WHY: the packed and resident-expanded haystack variants read the same source
/// text through different loads, so their fixtures must differ in bytes while
/// agreeing on the answer.
#[test]
fn packed_and_resident_haystack_variants_agree_on_the_answer() {
    let ops = phase_ops();
    let packed = ops
        .iter()
        .filter(|entry| entry.id.ends_with(PACKED_SUFFIX))
        .collect::<Vec<_>>();
    assert!(
        !packed.is_empty(),
        "Fix: no packed-haystack phase variant is registered"
    );
    for entry in packed {
        let resident_id = entry
            .id
            .strip_suffix(PACKED_SUFFIX)
            .expect("filtered on the suffix");
        let resident = ops
            .iter()
            .find(|candidate| candidate.id == resident_id)
            .unwrap_or_else(|| {
                panic!("Fix: {} has no resident-haystack sibling {resident_id}", entry.id)
            });
        let (resident_inputs, resident_expected) = fixture_cases(resident);
        let (packed_inputs, packed_expected) = fixture_cases(entry);
        assert_eq!(
            resident_expected, packed_expected,
            "Fix: {resident_id} and {} must answer identically for one source text",
            entry.id
        );
        let haystack = input_buffers(&program(entry))
            .iter()
            .position(|(name, _)| name == "phase_haystack")
            .unwrap_or_else(|| panic!("Fix: {} declares no haystack argument", entry.id));
        assert_ne!(
            resident_inputs[0][haystack], packed_inputs[0][haystack],
            "Fix: {} must encode its haystack four bytes per word, not one",
            entry.id
        );
    }
}
