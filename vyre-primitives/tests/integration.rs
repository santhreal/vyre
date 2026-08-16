//! Integration smoke: the intrinsic registry is linked and one op builds.
//!
//! Requires `hardware` (see `Cargo.toml` `[[test]]`). Every composition domain
//! moved to `vyre-libs`, so the only registrations this crate submits are
//! Category C hardware intrinsics, and their ids are what the conformance
//! harness discovers here.
#![forbid(unsafe_code)]

use vyre_foundation::ir::Program;
use vyre_foundation::validate::{validate_with_options, BackendCapabilities, ValidationOptions};

#[test]
fn inventory_registry_exposes_only_primitive_ids() {
    let ids: Vec<_> = vyre_primitives::operation_catalog::all_entries()
        .map(|entry| entry.id)
        .collect();
    assert!(
        !ids.is_empty(),
        "Fix: vyre-primitives with `hardware` must register at least one op; an empty catalog means the inventory link dropped."
    );
    let foreign: Vec<&&str> = ids
        .iter()
        .filter(|id| !id.starts_with("vyre-primitives::"))
        .collect();
    assert!(
        foreign.is_empty(),
        "Fix: every id this crate registers must be namespaced `vyre-primitives::`, found {foreign:?}"
    );
}

/// Capabilities of a backend that has every arm a Category C op can need.
///
/// An intrinsic is admitted to this crate only because it needs a dedicated
/// emitter arm, so the validator that judges one has to be told the arms exist.
fn every_capability() -> BackendCapabilities {
    BackendCapabilities {
        supports_subgroup_ops: true,
        supports_indirect_dispatch: true,
        supports_specialization_constants: true,
        supports_distributed_collectives: true,
        ..BackendCapabilities::default()
    }
}

fn built_programs() -> Vec<(&'static str, Program)> {
    let programs: Vec<(&'static str, Program)> = vyre_primitives::operation_catalog::all_entries()
        .filter_map(|entry| entry.build.map(|build| (entry.id, build())))
        .collect();
    assert!(
        !programs.is_empty(),
        "Fix: no registered intrinsic carries a program builder, so this file proves nothing about what the crate registers"
    );
    programs
}

/// WHY: this used to build one hand-picked op, `subgroup_add`, and validate it
/// with no backend. That passed for the wrong reason: the op named after a
/// subgroup reduction was the one intrinsic that used no subgroup expression,
/// summing thirty-two memory neighbours in a loop instead, so the single case
/// the smoke test covered was the single case that needed no capability. The
/// roster is read from the registry at run time, so a new intrinsic is covered
/// the moment it registers rather than when somebody remembers this file.
#[test]
fn every_registered_intrinsic_builds_a_program_a_capable_backend_accepts() {
    for (id, program) in built_programs() {
        let report = validate_with_options(
            &program,
            ValidationOptions::default().with_backend_capabilities(every_capability()),
        );
        assert!(
            report.errors.is_empty(),
            "Fix: registered intrinsic `{id}` builds a Program a fully capable backend rejects: {:?}",
            report.errors
        );
    }
}

/// WHY: a Category C op exists because it needs a backend arm, so validation
/// against a backend without that arm has to refuse it rather than pass it
/// through to a lowering that cannot emit it. At least one intrinsic must be
/// refused by the capability-free validator, or V041 and its siblings are
/// rules nothing in this crate exercises.
#[test]
fn a_capability_free_validator_refuses_an_intrinsic_that_needs_an_arm() {
    let refused: Vec<&str> = built_programs()
        .iter()
        .filter(|(_, program)| {
            !validate_with_options(program, ValidationOptions::universal())
                .errors
                .is_empty()
        })
        .map(|(id, _)| *id)
        .collect();
    assert!(
        !refused.is_empty(),
        "Fix: every registered intrinsic passes validation with no backend capabilities, which means none of them uses a construct a backend has to declare support for. An op that any backend can already lower is a composition and belongs in vyre-libs"
    );
}
