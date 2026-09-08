//! Every registered security composition declares one caller-visible output.
//!
//! WHY: a security bitset composition computes its answer through intermediate
//! buffers (arithmetic lanes, reachability sets, domination sets) and returns a
//! scalar verdict. An intermediate left flagged as an output is projected into
//! artifact completion, so a caller receives device-internal pipeline state and
//! a registration's expected-output fixture has to describe bytes that carry no
//! meaning. Nine such fixtures existed and each named an intermediate.
//!
//! The variant space is the inventory registry read at run time, so a security
//! operation added to `vyre-libs/src/security/catalog.rs` enters this closure
//! without an edit here. It does not catch an operation that declares the wrong
//! single buffer as its terminal output; the parity oracle covers that.

use vyre_foundation::operation::OperationRegistration;

/// Category tag every security registration in the catalog carries.
const SECURITY_CATEGORY: &str = "security";

#[test]
fn every_registered_security_operation_declares_one_output() {
    let mut checked = 0_usize;
    let mut failures = Vec::new();

    for registration in inventory::iter::<OperationRegistration> {
        if registration.category != Some(SECURITY_CATEGORY) {
            continue;
        }
        let Some(build) = registration.build else {
            continue;
        };
        checked += 1;
        let program = build();
        let outputs: Vec<&str> = program
            .buffers()
            .iter()
            .filter(|buffer| buffer.is_output())
            .map(|buffer| buffer.name())
            .collect();
        if outputs.len() != 1 {
            failures.push(format!(
                "{}: declares {} outputs {:?}, expected the single terminal verdict",
                registration.id,
                outputs.len(),
                outputs
            ));
        }
    }

    assert!(
        checked > 0,
        "no registration carried the {SECURITY_CATEGORY} category with a program builder, so this closure scanned nothing"
    );
    assert!(
        failures.is_empty(),
        "{} security operation(s) of {checked} declare an output set that is not one terminal buffer:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// An intermediate buffer stays declared, so demoting it from the output set is
/// a change to the completion boundary and not a change to the computation.
#[test]
fn a_security_operation_still_declares_its_intermediate_buffers() {
    let program = vyre_libs::security::integer_overflow_arith(
        4,
        "arith",
        "reach",
        "dom",
        "intermediate",
        "out",
    );
    let declared: Vec<&str> = program
        .buffers()
        .iter()
        .map(|buffer| buffer.name())
        .collect();
    for expected in ["arith", "reach", "dom", "intermediate", "out"] {
        assert!(
            declared.contains(&expected),
            "integer_overflow_arith stopped declaring {expected}; declared {declared:?}"
        );
    }
    let outputs: Vec<&str> = program
        .buffers()
        .iter()
        .filter(|buffer| buffer.is_output())
        .map(|buffer| buffer.name())
        .collect();
    assert_eq!(
        outputs,
        ["out"],
        "only the terminal verdict crosses the completion boundary"
    );
}
