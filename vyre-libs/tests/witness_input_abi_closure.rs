//! WHY: a registered witness case is the input list every backend, the
//! reference interpreter, and the conformance harness replay. Two shapes were
//! in circulation: one value per `vyre_reference::is_reference_input` buffer,
//! and one value per non-workgroup buffer, which carries a zeroed placeholder
//! for every backend-allocated output. The wgpu record-and-readback path used
//! to accept both and pick by list length, so a placeholder-shaped witness ran
//! there and was rejected by the reference interpreter and by cuda. Only wgpu
//! admitted it, so the disagreement surfaced on whichever backend a fixture
//! happened not to exercise.
//!
//! Closes: the witness input ABI across the whole operation catalog. The entry
//! set is read from `all_entries()` at run time, so a newly registered
//! operation is covered with nothing else edited, and every case of every entry
//! that declares `test_inputs` is checked. `vyre-libs/tests/universal_harness.rs`
//! makes the same assertion but only over `fixture_entries()`, which requires
//! both `test_inputs` and `expected_output`; an entry that declares witnesses
//! without an oracle still reaches the wgpu op-pairwise harness through
//! `all_entries()` and escaped that check.
//!
//! Does not catch: whether the bytes in a case are the right bytes. That is the
//! oracle's job, pinned in `vyre-libs/tests/cpu_witnesses.rs`. It also does not
//! prove a case is long enough for the buffer it feeds; `declared_min_byte_len`
//! in the reference interpreter owns that.

#![allow(deprecated)]

use vyre_libs::operation_catalog::all_entries;

/// Buffers a witness case must supply a value for, and the ones it must not.
struct WitnessShape {
    logical: Vec<String>,
    placeholder_shaped: usize,
}

fn witness_shape(program: &vyre::Program) -> WitnessShape {
    let logical = program
        .buffers()
        .iter()
        .filter(|buffer| vyre_reference::is_reference_input(buffer))
        .map(|buffer| buffer.name().to_string())
        .collect::<Vec<_>>();
    let placeholder_shaped = program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access() != vyre::ir::BufferAccess::Workgroup)
        .count();
    WitnessShape {
        logical,
        placeholder_shaped,
    }
}

#[test]
fn every_registered_witness_case_supplies_one_value_per_reference_input() {
    let mut checked_entries = 0usize;
    let mut checked_cases = 0usize;

    for entry in all_entries() {
        let (Some(build), Some(test_inputs)) = (entry.build, entry.test_inputs) else {
            continue;
        };
        let program = build();
        let shape = witness_shape(&program);
        checked_entries += 1;

        for (case_idx, case) in test_inputs().into_iter().enumerate() {
            checked_cases += 1;
            if case.len() == shape.logical.len() {
                continue;
            }
            let diagnosis = if case.len() == shape.placeholder_shaped {
                format!(
                    "that is the placeholder-shaped list of {} non-workgroup buffers, which no \
                     backend accepts",
                    shape.placeholder_shaped
                )
            } else {
                "that matches neither the reference input count nor any other buffer count"
                    .to_string()
            };
            panic!(
                "{} case {case_idx} supplies {} value(s) for {} reference input buffer(s) \
                 {:?}: {diagnosis}. Fix: declare one witness value per buffer accepted by \
                 `vyre_reference::is_reference_input`, in declaration order, and none for a \
                 backend-allocated output.",
                entry.id,
                case.len(),
                shape.logical.len(),
                shape.logical,
            );
        }
    }

    assert!(
        checked_entries > 0,
        "the operation catalog reported no entry with both a builder and witnesses, so this \
         test proved nothing. Fix: check `vyre_libs::operation_catalog::all_entries`."
    );
    assert!(
        checked_cases >= checked_entries,
        "{checked_entries} entries declared witnesses but only {checked_cases} cases were \
         checked, so an entry declared an empty case vector. Fix: every `test_inputs` fixture \
         returns at least one case."
    );
}

/// WHY: the mirror of the case above, on the side the harness compares against.
/// `reference_eval` returns one buffer per `vyre_reference::is_reference_output`
/// decl, so a fused composition returns each stage's result alongside the final
/// one. Five registrations declared an expected value only for the final
/// buffer, and the comparison then failed on the count with nothing naming
/// which stage was unaccounted for. A declaration that is short also leaves
/// every intermediate unchecked, which is the same as having no oracle for the
/// stages that produce them.
///
/// The entry set is `all_entries()` read at run time, so a newly registered
/// operation is covered with nothing else edited.
///
/// Does not catch: whether the declared bytes are the right bytes. That is the
/// parity oracle's job.
#[test]
fn every_registered_oracle_case_declares_one_value_per_returned_buffer() {
    let mut checked_entries = 0usize;
    let mut checked_cases = 0usize;
    let mut failures = Vec::new();

    for entry in all_entries() {
        let (Some(build), Some(expected_output)) = (entry.build, entry.expected_output) else {
            continue;
        };
        let program = build();
        let returned = program
            .buffers()
            .iter()
            .filter(|buffer| vyre_reference::is_reference_output(buffer))
            .map(|buffer| buffer.name())
            .collect::<Vec<_>>();
        checked_entries += 1;

        for (case_idx, case) in expected_output().into_iter().enumerate() {
            checked_cases += 1;
            if case.len() == returned.len() {
                continue;
            }
            failures.push(format!(
                "{} case {case_idx} declares {} expected value(s) for {} returned buffer(s) {:?}",
                entry.id,
                case.len(),
                returned.len(),
                returned,
            ));
        }
    }

    assert!(
        checked_entries > 0,
        "the operation catalog reported no entry with both a builder and an oracle, so this \
         test proved nothing. Fix: check `vyre_libs::operation_catalog::all_entries`."
    );
    assert!(
        checked_cases >= checked_entries,
        "{checked_entries} entries declared an oracle but only {checked_cases} cases were \
         checked, so an entry declared an empty case vector. Fix: every `expected_output` \
         fixture returns at least one case."
    );
    assert!(
        failures.is_empty(),
        "{} oracle case(s) of {checked_cases} declare the wrong number of outputs. Fix: declare \
         one expected value per buffer accepted by `vyre_reference::is_reference_output`, in \
         `Program::buffers` order, including every intermediate a fused stage leaves \
         writable:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The two shapes are distinguishable for at least one registered operation, so
/// the check above is not comparing a count against itself.
///
/// A catalog in which every program's reference-input count equals its
/// non-workgroup buffer count would make the assertion above pass for any
/// witness shape at all. That is the failure mode of a differential test whose
/// two sides collapsed into one.
#[test]
fn the_placeholder_shape_differs_from_the_logical_shape_somewhere_in_the_catalog() {
    let mut distinguishing = Vec::new();

    for entry in all_entries() {
        let Some(build) = entry.build else {
            continue;
        };
        let shape = witness_shape(&build());
        if shape.placeholder_shaped != shape.logical.len() {
            distinguishing.push(entry.id);
        }
    }

    assert!(
        !distinguishing.is_empty(),
        "no registered program declares a backend-allocated output, so the placeholder shape and \
         the logical shape are the same list everywhere and the ABI check above is vacuous. \
         Fix: this is either a catalog regression or a change to \
         `vyre_reference::is_reference_input`."
    );
}
