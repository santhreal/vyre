//! Integration smoke: the intrinsic registry is linked and one op builds.
//!
//! Requires `hardware` (see `Cargo.toml` `[[test]]`). Every composition domain
//! moved to `vyre-libs`, so the only registrations this crate submits are
//! Category C hardware intrinsics, and their ids are what the conformance
//! harness discovers here.
#![forbid(unsafe_code)]

use vyre_foundation::ir::Program;
use vyre_primitives::hardware::subgroup_add::subgroup_add;

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

#[test]
fn a_registered_intrinsic_builds_a_valid_program() {
    let program: Program = subgroup_add("in", "out", 4);
    program
        .validate()
        .unwrap_or_else(|error| panic!("Fix: subgroup_add must build a valid Program: {error}"));
}
