//! Barrier and uniform-control-flow checks for the HashMap interpreter.
//!
//! The executor calls these helpers between round-robin steps to preserve the
//! reference interpreter's workgroup-wide barrier semantics.

use super::invocation::HashmapInvocation;
use crate::ReferenceError;
use smallvec::SmallVec;
use vyre_foundation::ir::BufferDecl;

#[cfg(feature = "subgroup-ops")]
pub(crate) use crate::execution::node_tree::node_reads_peer_lanes;
pub(crate) use crate::execution::node_tree::{contains_rendezvous, node_id, RendezvousKind};

pub(crate) fn release_barrier_if_ready(invocations: &mut [HashmapInvocation<'_>]) -> bool {
    let active = invocations.iter().filter(|inv| !inv.done()).count();
    let waiting = live_waiting_count(invocations);
    if active > 0 && active == waiting {
        for inv in invocations {
            inv.waiting_at_barrier = false;
        }
        true
    } else {
        false
    }
}

pub(crate) fn live_waiting_count(invocations: &[HashmapInvocation<'_>]) -> usize {
    invocations
        .iter()
        .filter(|inv| !inv.done() && inv.waiting_at_barrier)
        .count()
}

/// Release every lane holding for its collective peers, once each live lane of
/// the workgroup has arrived at a rendezvous.
///
/// A lane parked at a `Barrier` counts as arrived, so a program that holds one
/// lane at a barrier while another holds at a collective still converges: the
/// collective runs first, and its lanes then reach the barrier themselves. A
/// lane held at a grid fence is not counted, because the dispatch driver, not
/// this workgroup, releases it.
pub(crate) fn release_collective_rendezvous(invocations: &mut [HashmapInvocation<'_>]) -> bool {
    let live = invocations
        .iter()
        .filter(|inv| !inv.done() && !inv.waiting_at_grid_fence)
        .count();
    let arrived = invocations
        .iter()
        .filter(|inv| {
            !inv.done()
                && !inv.waiting_at_grid_fence
                && (inv.waiting_for_collective_peers || inv.waiting_at_barrier)
        })
        .count();
    let holding = invocations
        .iter()
        .any(|inv| !inv.done() && inv.waiting_for_collective_peers);
    if !holding || live == 0 || live != arrived {
        return false;
    }
    for invocation in invocations.iter_mut() {
        if invocation.waiting_for_collective_peers {
            invocation.waiting_for_collective_peers = false;
            invocation.collective_peers_arrived = true;
        }
    }
    true
}

pub(crate) fn live_collective_waiting_count(invocations: &[HashmapInvocation<'_>]) -> usize {
    invocations
        .iter()
        .filter(|inv| !inv.done() && inv.waiting_for_collective_peers)
        .count()
}

/// Reject a branch whose condition differs across a rendezvous's own scope
/// while its body can reach that rendezvous.
///
/// Every lane that entered such a branch waits for peers that the diverged
/// lanes never send: the rendezvous releases once the diverged lane retires,
/// so accepting it would have the oracle issue a result computed over a subset
/// of the lanes the construct is defined over, which no target guarantees.
///
/// The scope is the construct's own, not the workgroup's in both cases. A
/// barrier synchronizes the workgroup, so its condition must agree across
/// every lane of the workgroup. A subgroup collective reads only its own
/// subgroup, so its condition must agree across that subgroup and may differ
/// between subgroups. Holding a collective to workgroup uniformity refuses the
/// standard shape where each subgroup owns one output and the tail workgroup
/// masks the subgroups that have none.
pub(crate) fn verify_uniform_control_flow(
    invocations: &[HashmapInvocation<'_>],
) -> Result<(), ReferenceError> {
    let mut observed = SmallVec::<[(usize, usize, bool); 8]>::new();
    for invocation in invocations.iter().filter(|inv| !inv.done()) {
        for (id, value, rendezvous) in &invocation.uniform_checks {
            let scope = rendezvous_scope(*rendezvous, invocation.linear_local_index);
            if let Some((_, _, previous)) = observed
                .iter()
                .find(|(seen_scope, seen_id, _)| *seen_scope == scope && seen_id == id)
            {
                if previous != value {
                    return Err(ReferenceError::new(format!(
                        "program violates uniform-control-flow rule: {} appears inside an If whose condition differs across the {}. Fix: make the condition uniform or move {} outside the branch.",
                        rendezvous.describe(),
                        rendezvous.scope_name(),
                        rendezvous.describe(),
                    )));
                }
            } else {
                observed.push((scope, *id, *value));
            }
        }
    }
    Ok(())
}

/// Which set of lanes a rendezvous of this kind agrees over, for the lane at
/// `linear_local_index`.
///
/// A barrier's scope is the whole workgroup, so every lane reports the same
/// one. A subgroup collective's scope is the subgroup the lane sits in.
fn rendezvous_scope(
    rendezvous: RendezvousKind,
    #[cfg_attr(not(feature = "subgroup-ops"), allow(unused_variables))] linear_local_index: u32,
) -> usize {
    match rendezvous {
        RendezvousKind::Barrier => 0,
        #[cfg(feature = "subgroup-ops")]
        RendezvousKind::SubgroupCollective => {
            linear_local_index as usize / crate::execution::hashmap::subgroup::subgroup_width()
        }
    }
}

pub(crate) fn element_count(decl: &BufferDecl, byte_len: usize) -> Result<u32, ReferenceError> {
    if let Some(bits) = decl.element().bit_width() {
        let total_bits = byte_len.checked_mul(8).ok_or_else(|| {
            ReferenceError::new(format!(
                "buffer `{}` has {} bytes and overflows host bit counting. Fix: shrink declaration footprint or split work.",
                decl.name(),
                byte_len,
            ))
        })?;
        let elements = total_bits / bits;
        return u32::try_from(elements).map_err(|_| {
            ReferenceError::new(format!(
                "buffer `{}` has {} bytes for {}-bit elements and overflows u32 elements. Fix: shrink declaration footprint or split work.",
                decl.name(),
                byte_len,
                bits,
            ))
        });
    }
    let Some(stride) = decl.element().size_bytes() else {
        return Err(ReferenceError::new(format!(
            "buffer `{}` has unsized element type {}. Fix: provide a fixed-width buffer element type before invoking the reference interpreter.",
            decl.name(),
            decl.element()
        )));
    };
    if stride == 0 {
        return u32 :: try_from (byte_len) . map_err (| _ | { ReferenceError::new(format ! ("buffer `{}` has {} bytes and cannot be indexed within u32 address space. Fix: shrink or split the invocation." , decl . name () , byte_len ,)) }) ;
    }
    let elements = byte_len / stride;
    u32 :: try_from (elements) . map_err (| _ | { ReferenceError::new(format ! ("buffer `{}` has {} bytes for stride {} and overflows u32 elements. Fix: shrink declaration footprint or split work." , decl . name () , byte_len , stride ,)) })
}

// Inline: covers the crate-private `element_count`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;

    #[test]
    fn element_count_uses_bit_width_for_packed_i4_buffers() {
        let decl = BufferDecl::read("packed", 0, DataType::I4).with_count(8);

        assert_eq!(
            element_count(&decl, 4).expect("Fix: packed I4 count must be computable."),
            8,
            "Fix: four bytes of I4 storage contain eight logical elements."
        );
        assert_eq!(
            element_count(&decl, 3).expect("Fix: packed I4 count must be computable."),
            6,
            "Fix: partial packed I4 buffers must report logical element count from bits, not bytes."
        );
    }
}
