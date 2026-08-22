//! Whole-registry parity guard for the library surface: no registered
//! composition may access a buffer out of bounds, change its answer when the
//! dispatch is over-fired, or depend on the order lanes are stepped.
//!
//! WHY: `vyre-primitives` has carried these nets since the gather-class audit,
//! and the library surface is where the compositions live. It had none, so a
//! composition that indexed past a buffer end was caught only if somebody wrote
//! a test for that one builder. One did not exist for the grid-stride tree
//! reduction, whose tail loads were guarded by `Expr::select`: that evaluates
//! both arms, so every lane past the element count read past the buffer end.
//! The reference interpreter absorbs the read as a zero, which is precisely the
//! masking a backend that does not bounds-check will not do.
//!
//! The four nets themselves live in `vyre_test_support::registry_nets`, which
//! both registered surfaces call, so neither can drift into judging its
//! population by a different rule. What is crate-specific is here: the
//! population comes from the library catalog at run time, so a registration
//! added tomorrow is judged tomorrow.

#![forbid(unsafe_code)]

use vyre_reference::value::Value;
use vyre_test_support::registry_nets::{RegistrySweep, SweepCase};

/// Every fixture case the library catalog publishes.
///
/// The catalog is refused when empty: an empty walk passes every net without
/// proving anything.
fn sweep() -> RegistrySweep {
    let entries: Vec<_> = vyre_libs::operation_catalog::all_entries().collect();
    assert!(
        !entries.is_empty(),
        "Fix: the library catalog is empty, so this run judges no registration at all"
    );

    let mut cases = Vec::new();
    for entry in entries {
        let Some(inputs_fn) = entry.test_inputs else {
            continue;
        };
        let program = entry
            .program()
            .expect("Fix: registered library operation must provide a neutral builder");
        for (index, case) in inputs_fn().into_iter().enumerate() {
            let inputs: Vec<Value> = case.into_iter().map(Value::from).collect();
            cases.push(SweepCase::new(
                format!("{} (fixture case {index})", entry.id),
                program.clone(),
                inputs,
            ));
        }
    }
    RegistrySweep::new("the library catalog", cases)
}

#[test]
fn every_registered_composition_is_oob_clean_on_its_fixtures() {
    sweep().assert_oob_clean();
}

#[test]
fn every_registered_composition_is_oob_clean_under_grid_overfire() {
    sweep().assert_oob_clean_under_overfire();
}

#[test]
fn every_registered_composition_output_is_invariant_under_grid_overfire() {
    sweep().assert_output_invariant_under_overfire();
}

#[test]
fn every_registered_composition_is_race_free_under_lane_reversal() {
    sweep().assert_race_free_under_lane_reversal();
}
