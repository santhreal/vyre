//! Whole-registry parity guard: no registered Tier-2.5 primitive may access a
//! buffer out of bounds while running its own valid fixture inputs, change its
//! answer when the dispatch over-fires, or depend on the order lanes are
//! stepped.
//!
//! The reference interpreter silently absorbs an out-of-bounds access (see
//! `vyre-reference/src/oob.rs`: zero-fill loads, dropped stores) so its output
//! stays deterministic, but that masking hides the gather-class parity hazard:
//! an IR program with an ungated data-derived index works on the reference,
//! while a backend that bounds-checks nothing reads garbage or corrupts memory.
//!
//! The nets and the catalog walk live in `vyre_test_support::registry_nets`,
//! which the library surface calls with its own catalog. They were written
//! twice before that, once per crate, and the copies drifted: one refused a
//! case it could not evaluate and the other skipped it, so the same class of
//! defect failed on one surface and went unjudged on the other.
//!
//! What is crate-specific is the catalog named here, plus IR validity, which is
//! asserted over every registered primitive rather than only the fixtured ones.
#![cfg(feature = "inventory-registry")]

mod gate_fixtures;

use vyre_reference::value::Value;
use vyre_test_support::registry_nets::RegistrySweep;

fn sweep() -> RegistrySweep {
    RegistrySweep::from_catalog(
        "the primitive catalog",
        vyre_primitives::operation_catalog::all_entries(),
    )
}

#[test]
fn every_registered_primitive_is_oob_clean_on_its_fixtures() {
    sweep().assert_oob_clean();
}

#[test]
fn every_registered_primitive_is_oob_clean_under_grid_overfire() {
    sweep().assert_oob_clean_under_overfire();
}

#[test]
fn every_registered_primitive_output_is_invariant_under_grid_overfire() {
    sweep().assert_output_invariant_under_overfire();
}

#[test]
fn every_registered_primitive_is_race_free_under_lane_reversal() {
    sweep().assert_race_free_under_lane_reversal();
}

/// Every registered primitive emits IR that passes validation, fixtured or not.
///
/// The four nets above reach only the entries that carry a fixture, so an
/// unfixtured registration with an IR defect (a duplicate-binding shadow, which
/// the no-shadowing validator and the CUDA backend both reject) would land
/// undetected. Validation runs before input binding, so an IR-invalid program
/// reports `failed IR validation` whatever inputs are supplied, while a valid
/// program on empty inputs reports a benign missing input, ignored here. This
/// closes the gap the `union_find` shadow defect exposed.
#[test]
fn every_registered_primitive_program_is_ir_valid() {
    let mut invalid = Vec::new();
    let mut total = 0usize;
    for entry in vyre_primitives::operation_catalog::all_entries() {
        total += 1;
        let program = entry
            .program()
            .expect("Fix: registered primitive must provide a neutral builder");
        let values: Vec<Value> = match entry.test_inputs {
            Some(inputs_fn) => inputs_fn()
                .into_iter()
                .next()
                .unwrap_or_default()
                .into_iter()
                .map(Value::from)
                .collect(),
            None => Vec::new(),
        };
        if let Err(err) = vyre_reference::reference_eval(&program, &values) {
            if format!("{err}").contains("failed IR validation") {
                invalid.push(format!("{}: {err}", entry.id));
            }
        }
    }
    assert!(
        total > 0,
        "Fix: no registered primitives seen; select the domain features (--features inventory-registry,all-lego)."
    );
    assert!(
        invalid.is_empty(),
        "Fix: {} registered primitive(s) emit IR that FAILS validation, which the no-shadowing validator and the \
         CUDA backend both reject, so the op runs neither on the reference nor on a device. A registered op emits \
         valid IR whether or not it carries a fixture. Invalid:\n{}",
        invalid.len(),
        invalid.join("\n")
    );
}
