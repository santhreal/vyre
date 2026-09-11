//! WHY: `vyre_reference::step_budget::MAX_REFERENCE_STEPS` is a termination
//! contract, and a ceiling chosen by preference either refuses a legitimate
//! program or admits a runaway. This measures the corpus instead: every
//! registered library operation fixture case is evaluated through the counting
//! entry point, and the heaviest run must stay at or below the recorded
//! measurement the ceiling is derived from.
//!
//! Closes: the ceiling's justification. The heaviest legitimate run is
//! re-measured on every execution, so a fixture that grows past the recorded
//! number turns this red and the constant moves to the value this test printed
//! rather than to a value someone liked. It also proves every registered
//! fixture still evaluates under the shipped ceiling, which is the regression a
//! work bound can introduce, and that the ceiling is still the product of the
//! measurement and the headroom rather than a hand-edited number.
//!
//! Does not catch: a run outside the registered operation fixtures. The
//! benchmark oracle runs in `vyre-bench` and the parity lanes in
//! `vyre-driver-wgpu` and `vyre-conform` evaluate the same interpreter under
//! the same ceiling, so a heavier legitimate run there fails in that crate's
//! suite naming the program and the ceiling, and the recorded measurement moves
//! to it. A program a downstream consumer builds heavier than any of them is
//! refused by name, which is the contract rather than a defect; such a caller
//! states its own bound through `reference_eval_with_step_ceiling`.

#![allow(deprecated)]

use vyre::ir::Program;
use vyre_foundation::operation::SemanticOperation;
use vyre_libs::operation_catalog::fixture_entries;
use vyre_reference::step_budget::{
    MAX_REFERENCE_STEPS, MEASURED_HEAVIEST_CORPUS_STEPS, STEP_CEILING_HEADROOM,
};
use vyre_reference::value::Value;

fn program(entry: &SemanticOperation) -> Program {
    entry
        .program()
        .expect("Fix: registered library operation must provide a neutral builder")
}

/// The heaviest fixture run in the corpus, and the operation that produced it.
fn heaviest_registered_fixture() -> (&'static str, u64) {
    let mut heaviest = ("none", 0u64);
    let mut evaluated = 0usize;
    for entry in fixture_entries() {
        let cases = (entry
            .test_inputs
            .expect("Fix: fixture_entries yields only entries that ship test_inputs"))(
        );
        for (index, case) in cases.iter().enumerate() {
            let inputs = case.iter().cloned().map(Value::from).collect::<Vec<_>>();
            let (_outputs, steps) =
                vyre_reference::ReferenceRequest::standard(&program(&entry), &inputs).outputs_and_steps()
                    .unwrap_or_else(|error| {
                        panic!(
                            "Fix: registered fixture {} case {index} must evaluate under the shipped step ceiling: {error}",
                            entry.id
                        )
                    });
            evaluated += 1;
            if steps > heaviest.1 {
                heaviest = (entry.id, steps);
            }
        }
    }
    assert!(
        evaluated > 0,
        "Fix: this measurement is vacuous with no fixture cases; check the feature set the corpus needs"
    );
    println!(
        "[step-ceiling] evaluated {evaluated} fixture cases; heaviest {} at {} steps; recorded {MEASURED_HEAVIEST_CORPUS_STEPS}; ceiling {MAX_REFERENCE_STEPS}",
        heaviest.0, heaviest.1
    );
    heaviest
}

/// The corpus is walked once: every claim here is about the same measurement,
/// and evaluating every registered fixture twice buys nothing.
#[test]
fn the_reference_step_ceiling_is_derived_from_the_corpus() {
    assert_eq!(
        MAX_REFERENCE_STEPS,
        MEASURED_HEAVIEST_CORPUS_STEPS * STEP_CEILING_HEADROOM,
        "Fix: the step ceiling is the recorded measurement times the headroom. Move MEASURED_HEAVIEST_CORPUS_STEPS or STEP_CEILING_HEADROOM, never MAX_REFERENCE_STEPS"
    );

    let (id, steps) = heaviest_registered_fixture();
    assert!(
        steps <= MEASURED_HEAVIEST_CORPUS_STEPS,
        "Fix: registered fixture {id} charges {steps} steps, above the recorded heaviest legitimate run {MEASURED_HEAVIEST_CORPUS_STEPS}. Record {steps} in MEASURED_HEAVIEST_CORPUS_STEPS, which raises the derived ceiling with it, or reduce the fixture's work"
    );
    assert!(
        steps < MAX_REFERENCE_STEPS,
        "Fix: registered fixture {id} charges {steps} steps, which the shipped ceiling {MAX_REFERENCE_STEPS} refuses"
    );
}
