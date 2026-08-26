// Cross-backend parity matrix: registered backends, wire shapes, and buffer comparison.
// `#![forbid(unsafe_code)]` lives on the parent `parity_matrix.rs` crate root.

#[path = "parity_matrix__divergence.rs"]
mod parity_matrix_divergence;
#[path = "parity_matrix__entries.rs"]
mod parity_matrix_entries;
#[path = "parity_matrix__runner.rs"]
mod parity_matrix_runner;
#[path = "parity_matrix__synthetic_entries.rs"]
mod parity_matrix_synthetic_entries;

use parity_matrix_divergence::{Divergence, OpFailure, Summary};
use parity_matrix_entries::{unified_entries, UnifiedEntry};
use parity_matrix_runner::{backend_runners, BackendKind, BackendRunner};
use parity_matrix_synthetic_entries::*;

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use blake3::Hash;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::witness_plan::WitnessInputPlan;
use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};
use vyre_spec::expr_variants;

#[test]
fn parity_reference_runner_uses_planned_zeroed_read_write_inputs() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "scratch",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    );
    let plan = WitnessInputPlan::for_program(&program)
        .expect("Fix: static read-write zero-fill planning must succeed.");
    let runner = BackendRunner {
        id: "reference",
        kind: BackendKind::ReferenceBackend,
    };
    let inputs = vec![1u32.to_le_bytes().to_vec()];
    let mut values = Vec::new();
    let mut borrowed_inputs = Vec::new();

    let outputs = runner
        .execute_with_plan(
            &program,
            &inputs,
            &mut values,
            Some(&plan),
            &mut borrowed_inputs,
        )
        .expect("Fix: reference parity runner must receive planned zeroed read-write inputs.");

    assert_eq!(
        outputs,
        vec![1u32.to_le_bytes().to_vec()],
        "Fix: reference and backend parity paths must use the same planned input buffer expansion."
    );
}

// Asserts `runners.len() >= 2`, which means at least one dispatch-capable
// backend in addition to vyre-reference must be linked. If the crate is built
// without the `gpu` feature, this test must fail loudly instead of compiling
// out the parity gate.
/// Measure one operation on every runner and record what it found.
///
/// `Err` names a defect that stops this operation: a missing fixture, a program
/// the validator rejects, a reference backend that refused the dispatch. Every
/// defect that leaves the remaining witnesses measurable is pushed onto the
/// summary instead, so one broken backend does not hide the operations behind
/// it.
fn measure_entry(
    entry: &UnifiedEntry,
    runners: &[BackendRunner],
    summary: &mut Summary,
) -> Result<(), OpFailure> {
    let test_inputs = entry.test_inputs.ok_or_else(|| {
        OpFailure::harness(
            entry.id,
            "fixtures",
            "missing test_inputs; every registered op must provide fixture inputs".to_string(),
        )
    })?;
    let expected_output = entry.expected_output.ok_or_else(|| {
        OpFailure::harness(
            entry.id,
            "fixtures",
            "missing expected_output; every registered op must provide a fixture oracle"
                .to_string(),
        )
    })?;

    let program = entry.program();
    validate_program(entry.id, &program)
        .map_err(|detail| OpFailure::harness(entry.id, "validation", detail))?;
    check_region_chain(&program)
        .map_err(|detail| OpFailure::harness(entry.id, "region chain", detail))?;

    let input_cases = test_inputs();
    let expected_cases = expected_output();
    if input_cases.is_empty() {
        return Err(OpFailure::harness(
            entry.id,
            "fixtures",
            "registered empty test_inputs; empty witnesses are zero execution coverage".to_string(),
        ));
    }
    if expected_cases.is_empty() {
        return Err(OpFailure::harness(
            entry.id,
            "fixtures",
            "registered empty expected_output; empty oracles are zero execution coverage"
                .to_string(),
        ));
    }
    if input_cases.len() != expected_cases.len() {
        return Err(OpFailure::harness(
            entry.id,
            "fixtures",
            format!(
                "test_inputs / expected_output case count mismatch ({} vs {})",
                input_cases.len(),
                expected_cases.len()
            ),
        ));
    }

    summary.ops_covered += 1;
    let input_plan = WitnessInputPlan::for_program(&program)
        .map_err(|error| OpFailure::harness(entry.id, "input plan", error.to_string()))?;
    let mut reference_values = Vec::with_capacity(program.buffers().len());
    let mut outputs = Vec::<(&'static str, Vec<Vec<u8>>)>::with_capacity(runners.len());
    let mut borrowed_inputs = Vec::with_capacity(input_plan.source_count());
    for (case_index, (inputs, expected)) in
        input_cases.iter().zip(expected_cases.iter()).enumerate()
    {
        let input_hash = hash_buffers(inputs);
        let program_hash_before = hash_program(&program)
            .map_err(|detail| OpFailure::harness(entry.id, "program hash", detail))?;
        outputs.clear();
        borrowed_inputs.clear();

        let reference_output = runners[0]
            .execute_with_plan(
                &program,
                inputs,
                &mut reference_values,
                Some(&input_plan),
                &mut borrowed_inputs,
            )
            .map_err(|error| {
                OpFailure::backend(
                    entry.id,
                    runners[0].id,
                    "dispatch",
                    format!("case {case_index}: {error}"),
                )
            })?;
        let reference_hash = hash_buffers(&reference_output);
        if hash_program(&program)
            .map_err(|detail| OpFailure::harness(entry.id, "program hash", detail))?
            != program_hash_before
        {
            return Err(OpFailure::backend(
                entry.id,
                runners[0].id,
                "program stability",
                format!(
                    "case {case_index} mutated the Program during dispatch; the region chain must remain stable post-run"
                ),
            ));
        }
        compare_outputs(
            entry.id,
            "reference",
            "expected_output",
            input_hash,
            &reference_output,
            expected,
            &program,
            &mut summary.divergences,
        );
        outputs.push(("reference", reference_output));

        for runner in runners.iter().skip(1) {
            match catch_unwind(AssertUnwindSafe(|| {
                runner.execute_with_plan(
                    &program,
                    inputs,
                    &mut reference_values,
                    Some(&input_plan),
                    &mut borrowed_inputs,
                )
            })) {
                Ok(Ok(output)) => match hash_program(&program) {
                    Ok(after) if after == program_hash_before => {
                        outputs.push((runner.id, output));
                    }
                    Ok(_) => summary.failures.push(OpFailure::backend(
                        entry.id,
                        runner.id,
                        "program stability",
                        format!(
                            "case {case_index} mutated the Program during dispatch; the region chain must remain stable post-run"
                        ),
                    )),
                    Err(detail) => {
                        summary
                            .failures
                            .push(OpFailure::harness(entry.id, "program hash", detail));
                    }
                },
                Ok(Err(error)) => summary.failures.push(OpFailure::backend(
                    entry.id,
                    runner.id,
                    "dispatch",
                    format!("case {case_index}: {error}"),
                )),
                Err(payload) => {
                    summary.divergences.push(Divergence {
                        op_id: entry.id,
                        backend_a: runner.id,
                        backend_b: "reference",
                        input_hash,
                        output_a_hash: hash_buffers(&[]),
                        output_b_hash: reference_hash,
                        detail: format!("dispatch panic: {}", panic_message(payload)),
                    });
                }
            }
        }

        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                let (backend_a, output_a) = &outputs[i];
                let (backend_b, output_b) = &outputs[j];
                compare_outputs(
                    entry.id,
                    backend_a,
                    backend_b,
                    input_hash,
                    output_a,
                    output_b,
                    &program,
                    &mut summary.divergences,
                );
            }
        }
    }

    Ok(())
}

#[test]
fn parity_matrix_across_all_registered_ops() {
    // Validation resolves every `Expr::Call` through the immutable canonical
    // operation registry, so an unknown operation is rejected with V016 before
    // it reaches a backend.
    let mut summary = Summary::default();
    let runners = backend_runners(&mut summary);
    let entries = unified_entries();
    let expr_rows = expr_variant_rows(&entries);
    let filter = env::var("VYRE_PARITY_FILTER").ok();

    assert!(
        runners.len() >= 2,
        "Fix: parity_matrix requires at least one linked dispatch-capable backend in addition to vyre-reference. Link a concrete driver crate for this gate."
    );
    assert!(
        !entries.is_empty(),
        "Fix: parity matrix linked zero canonical operation registrations. Ensure vyre-libs and vyre-primitives are linked into this test binary."
    );
    let missing_expr_variants = expr_variants()
        .iter()
        .copied()
        .filter(|variant| !expr_rows.contains_key(variant))
        .collect::<Vec<_>>();
    assert!(
        missing_expr_variants.is_empty(),
        "Fix: parity matrix must cover every Expr variant from vyre-spec; missing rows for {}",
        missing_expr_variants.join(", ")
    );

    for entry in &entries {
        if filter.as_deref().is_some_and(|needle| {
            needle
                .strip_prefix('=')
                .map_or_else(|| !entry.id.contains(needle), |exact| entry.id != exact)
        }) {
            continue;
        }
        summary.ops_total += 1;
        if let Err(failure) = measure_entry(entry, &runners, &mut summary) {
            summary.failures.push(failure);
        }
    }

    eprintln!(
        "PARITY-SUMMARY ops_total={} ops_covered={} backends_linked={} backends_runnable={} divergences={} unmeasured={}",
        summary.ops_total,
        summary.ops_covered,
        summary.backends_linked,
        summary.backends_runnable,
        summary.divergences.len(),
        summary.failures.len()
    );
    for variant in expr_variants() {
        if let Some(op_ids) = expr_rows.get(variant) {
            eprintln!(
                "PARITY-EXPR-COVERAGE variant={} rows={}",
                variant,
                op_ids.join(",")
            );
        }
    }

    assert!(
        summary.failures.is_empty() && summary.divergences.is_empty(),
        "{}",
        format_summary_failures(&summary)
    );
    assert_eq!(
        summary.ops_covered, summary.ops_total,
        "parity matrix under-coverage: ops_covered={} ops_total={}. Fix: every registered op must run at least one witness case.",
        summary.ops_covered, summary.ops_total
    );
}

/// The coverage bundle survives the neutral wire and still dispatches on every
/// runnable backend when it is rebuilt from that wire image.
///
/// The bundle carries `Expr::Opaque`, whose payload is an out-of-tree extension
/// node. Encoding, decoding and then dispatching the decoded program proves the
/// extension reaches each backend through the wire rather than through the
/// in-process value this test constructed.
#[test]
fn the_synthetic_opaque_extension_round_trips_through_the_wire() {
    let mut summary = Summary::default();
    let runners = backend_runners(&mut summary);
    assert!(
        runners.len() >= 2,
        "Fix: this contract requires at least one linked dispatch-capable backend in addition to vyre-reference. Link a concrete driver crate for this gate."
    );
    let entry = synthetic_entries()
        .into_iter()
        .find(|entry| entry.id == SYNTHETIC_BUNDLE_OP_ID)
        .expect(
            "Fix: synthetic_entries() must register the coverage bundle under SYNTHETIC_BUNDLE_OP_ID.",
        );
    let program = entry.program();
    let wire = program
        .to_wire()
        .expect("Fix: the coverage bundle must encode to the neutral wire.");
    let decoded = Program::from_wire(&wire)
        .expect("Fix: the coverage bundle must decode from its own wire image.");
    assert_eq!(
        decoded
            .to_wire()
            .expect("Fix: the decoded coverage bundle must re-encode to the neutral wire."),
        wire,
        "Fix: the coverage bundle wire image is not byte-stable across one decode and re-encode round trip."
    );
    let decoded_variants = expr_variants_in_program(&decoded);
    assert!(
        decoded_variants.contains("Opaque"),
        "Fix: the decoded coverage bundle carries {:?} and no Opaque node; the opaque extension does not survive the wire.",
        decoded_variants
    );
    assert_eq!(
        decoded_variants,
        expr_variants_in_program(&program),
        "Fix: the wire round trip changed which Expr variants the coverage bundle contains."
    );

    let input_cases = (entry
        .test_inputs
        .expect("Fix: the coverage bundle must register test_inputs."))();
    let expected_cases = (entry
        .expected_output
        .expect("Fix: the coverage bundle must register expected_output."))(
    );
    let input_plan = WitnessInputPlan::for_program(&decoded)
        .expect("Fix: the decoded coverage bundle must yield a witness input plan.");
    let mut values = Vec::with_capacity(decoded.buffers().len());
    let mut borrowed_inputs = Vec::with_capacity(input_plan.source_count());
    for runner in &runners {
        for (case_index, (inputs, expected)) in
            input_cases.iter().zip(expected_cases.iter()).enumerate()
        {
            borrowed_inputs.clear();
            let output = runner
                .execute_with_plan(
                    &decoded,
                    inputs,
                    &mut values,
                    Some(&input_plan),
                    &mut borrowed_inputs,
                )
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: {} refused the decoded coverage bundle on case {case_index}: {error}",
                        runner.id
                    )
                });
            assert_eq!(
                hash_buffers(&output),
                hash_buffers(expected),
                "Fix: {} produced a different result for the decoded coverage bundle on case {case_index} than the fixture oracle.",
                runner.id
            );
        }
    }
}
