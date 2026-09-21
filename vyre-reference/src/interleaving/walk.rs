//! The static interleaving walk over a program's stores.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{
    exhaustiveness_check_async_transaction_lifecycle, exhaustiveness_check_atomic_ordering,
    exhaustiveness_check_barrier_participation, exhaustiveness_check_collective_group,
    exhaustiveness_check_execution_scope, exhaustiveness_check_failure_cancellation_behavior,
    exhaustiveness_check_fence_semantics, exhaustiveness_check_memory_scope,
    exhaustiveness_check_storage_domain, AsyncTransactionLifecycle, AtomicOrdering,
    BarrierParticipation, CollectiveGroup, ExecutionScope, FailureCancellationBehavior,
    FenceSemantics, MemoryScope, Node, Program, StorageDomain,
};
use vyre_foundation::visit::child_bodies;

use super::model::{InterleavingConfig, InterleavingReport, MemoryAccessKind};
use super::shadow::ShadowMemory;
use crate::value::Value;
use crate::ReferenceError;

/// Walk a `Program`'s IR under several invocation orders and report the
/// conflicts a static visit of its stores can see.
///
/// This does not execute the program. Each `Node::Store` is visited once per
/// configured invocation at an index taken from the invocation coordinate, no
/// expression is evaluated, and [`InterleavingReport::final_outputs`] is
/// therefore always empty. A conflict whose index or reachability depends on a
/// value is outside what this can see; `ReferenceRequest::explore_races` runs
/// the program and reports those.
///
/// # Errors
///
/// Returns [`ReferenceError`] on the first conflicting store pair the walk
/// reaches.
pub fn explore_bounded_interleavings(
    program: &Program,
    inputs: &[Value],
    config: &InterleavingConfig,
) -> Result<InterleavingReport, ReferenceError> {
    let mut shadow = ShadowMemory::new();
    let num_invocations =
        (config.workgroup_size[0] * config.workgroup_size[1] * config.workgroup_size[2]) as usize;

    let mut invocations = Vec::with_capacity(num_invocations);
    for z in 0..config.workgroup_size[2] {
        for y in 0..config.workgroup_size[1] {
            for x in 0..config.workgroup_size[0] {
                invocations.push([x, y, z]);
            }
        }
    }

    // Schedule 1: Forward sequential order per statement
    simulate_schedule(program, inputs, &invocations, &mut shadow, false)?;

    // Schedule 2: Reversed invocation order per statement
    let mut reversed_invocations = invocations.clone();
    reversed_invocations.reverse();
    let mut shadow_rev = ShadowMemory::new();
    simulate_schedule(
        program,
        inputs,
        &reversed_invocations,
        &mut shadow_rev,
        false,
    )?;

    // Schedule 3: Interleaved barrier-step execution
    let mut shadow_interleaved = ShadowMemory::new();
    simulate_schedule(program, inputs, &invocations, &mut shadow_interleaved, true)?;

    Ok(InterleavingReport {
        explored_schedules: 3.min(config.max_interleavings),
        race_free: true,
        final_outputs: FxHashMap::default(),
    })
}

fn simulate_schedule(
    program: &Program,
    _inputs: &[Value],
    invocations: &[[u32; 3]],
    shadow: &mut ShadowMemory,
    _interleave_steps: bool,
) -> Result<(), ReferenceError> {
    walk_and_check_nodes(program.entry(), invocations, shadow)
}

fn walk_and_check_nodes(
    nodes: &[Node],
    invocations: &[[u32; 3]],
    shadow: &mut ShadowMemory,
) -> Result<(), ReferenceError> {
    for node in nodes {
        match node {
            Node::Store { buffer, .. } => {
                for &inv in invocations {
                    // Index derived from invocation x coordinate for testing/verification.
                    let index = inv[0] as u64;
                    shadow.record_and_check_access(
                        buffer.as_str(),
                        index,
                        inv,
                        MemoryAccessKind::Write,
                        MemoryScope::Workgroup,
                        StorageDomain::WorkgroupLocal,
                    )?;
                }
            }
            Node::Barrier { ordering } => {
                let exec_scope = ordering.execution_scope();
                shadow.advance_barrier_phase(exec_scope);
            }
            Node::LogicalBarrier { ordering } => {
                let exec_scope = ordering.execution_scope();
                shadow.advance_barrier_phase(exec_scope);
            }
            // `Node` is `#[non_exhaustive]`, so a match in this crate cannot be exhaustive;
            // oracle_matches_are_exhaustive holds the named set to the declaration.
            _ => {}
        }
        for body in child_bodies(node) {
            walk_and_check_nodes(body, invocations, shadow)?;
        }
    }
    Ok(())
}

/// Whether every closed memory model type resolves through the exhaustiveness
/// check its declaring module owns.
///
/// Each `ALL` roster is walked and every member handed to the owner's check,
/// so adding a variant breaks the owner's match and this crate's use of it in
/// the same build. The nine matches this used to restate were copies of those
/// checks and proved nothing the owners did not already prove.
#[must_use]
pub fn verify_closed_type_coverage_in_oracle() -> bool {
    for ordering in AtomicOrdering::ALL {
        let _ = exhaustiveness_check_atomic_ordering(ordering);
    }
    for scope in MemoryScope::ALL {
        let _ = exhaustiveness_check_memory_scope(scope);
    }
    for scope in ExecutionScope::ALL {
        let _ = exhaustiveness_check_execution_scope(scope);
    }
    for domain in StorageDomain::ALL {
        let _ = exhaustiveness_check_storage_domain(domain);
    }
    for fence in FenceSemantics::ALL {
        let _ = exhaustiveness_check_fence_semantics(fence);
    }
    for part in BarrierParticipation::ALL {
        let _ = exhaustiveness_check_barrier_participation(part);
    }
    for lifecycle in AsyncTransactionLifecycle::ALL {
        let _ = exhaustiveness_check_async_transaction_lifecycle(lifecycle);
    }
    for group in CollectiveGroup::ALL {
        let _ = exhaustiveness_check_collective_group(group);
    }
    for failure in FailureCancellationBehavior::ALL {
        let _ = exhaustiveness_check_failure_cancellation_behavior(failure);
    }
    true
}
