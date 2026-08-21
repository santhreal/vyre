//! A whole-grid barrier is refused only when no route can run it.
//!
//! # Why this suite exists
//!
//! Capability validation gained a `grid_sync` bit so a program needing a
//! grid-scope barrier is refused while a target is being chosen, rather than
//! discovered by an emitter after one has been picked. Mapped from
//! `VyreBackend::supports_grid_sync` alone, that bit would have refused every
//! backend in this workspace except CUDA: none of them lower a cooperative
//! launch, and all of them run such a program anyway by splitting it into
//! sequential host dispatches, which the registry wrapper offers by default.
//! The refusal would have landed on the wgpu security flows whose split is the
//! reason the split exists.
//!
//! So the bit answers whether *any* route exists, and these cases pin all four
//! corners of that disjunction.
//!
//! # What this suite does NOT claim
//!
//! It does not claim the chosen route is the fast one, nor that the split point
//! is legal. A fence with no correct cut is refused by the planner, which
//! `grid_sync_nested_fence_survives_split.rs` covers.

use vyre_driver::validation::{validate_program_contract, ProgramValidationCaps};
use vyre_foundation::ir::{MemoryOrdering, Node, Program};
use vyre_foundation::validate::ValidationOptions;

fn fenced_program() -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            Node::Return,
        ],
    )
}

fn caps(supports_grid_sync: bool, allows_host_grid_sync_split: bool) -> ProgramValidationCaps {
    ProgramValidationCaps {
        backend_id: "grid-sync-route-test",
        supports_subgroup_ops: false,
        supports_f16: false,
        supports_bf16: false,
        supports_indirect_dispatch: false,
        supports_distributed_collectives: false,
        supports_trap_propagation: true,
        supports_grid_sync,
        allows_host_grid_sync_split,
        max_workgroup_size: [256, 256, 64],
    }
}

fn validate(caps: ProgramValidationCaps) -> Result<(), vyre_driver::BackendError> {
    validate_program_contract(
        &fenced_program(),
        ValidationOptions::default(),
        vyre_driver::default_supported_ops(),
        caps,
    )
}

/// Without this the suite is vacuous: every case below would pass against a
/// program that never required a grid barrier at all.
#[test]
fn the_fixture_program_actually_requires_a_grid_barrier() {
    let required = vyre_foundation::program_caps::scan(&fenced_program());
    assert!(
        required.grid_sync,
        "Fix: the fixture must make `scan` set `grid_sync`, or these cases prove nothing."
    );
}

#[test]
fn a_backend_that_lowers_the_barrier_natively_is_admitted() {
    assert_eq!(
        validate(caps(true, false)),
        Ok(()),
        "Fix: a native cooperative launch runs the barrier; refusing it strands CUDA."
    );
}

#[test]
fn a_backend_that_only_allows_the_host_split_is_admitted() {
    assert_eq!(
        validate(caps(false, true)),
        Ok(()),
        "Fix: the host split runs the barrier as sequential dispatches. Refusing here would reject every wgpu security flow the split was written for."
    );
}

#[test]
fn a_backend_offering_both_routes_is_admitted() {
    assert_eq!(validate(caps(true, true)), Ok(()));
}

#[test]
fn a_backend_with_neither_route_is_refused_by_name() {
    let error = validate(caps(false, false))
        .expect_err("Fix: a barrier with no route to run it must be refused before a target is chosen.");
    let message = error.to_string();
    assert!(
        message.contains("grid_sync"),
        "Fix: the refusal must name the missing capability; got: {message}"
    );
}
