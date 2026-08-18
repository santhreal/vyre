//! Round-robin node stepping and expression-adjacent execution helpers.
mod node_step;
pub(crate) use node_step::{eval_call, step_loop_frame, step_nodes_frame};

#[cfg(feature = "subgroup-ops")]
use super::invocation::HashmapInvocationSnapshot;
use super::{invocation::HashmapInvocation, memory::HashmapMemory};
use crate::ReferenceError;
use crate::{value::Value, workgroup::Frame};

pub(crate) fn step_round_robin(
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] uses_subgroup_ops: bool,
) -> Result<bool, ReferenceError> {
    let mut made_progress = false;
    #[cfg(feature = "subgroup-ops")]
    let snapshots = if uses_subgroup_ops {
        capture_invocation_snapshots(invocations)
    } else {
        Vec::new()
    };
    for index in 0..invocations.len() {
        if invocations[index].done() || invocations[index].waiting_at_barrier {
            continue;
        }
        step(
            index,
            memory,
            invocations,
            #[cfg(feature = "subgroup-ops")]
            &snapshots,
        )?;
        made_progress = true;
    }
    Ok(made_progress)
}

fn step(
    index: usize,
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<(), ReferenceError> {
    let invocation = &mut invocations[index];
    if invocation.done() || invocation.waiting_at_barrier {
        return Ok(());
    }
    loop {
        let Some(frame) = invocation.frames.pop() else {
            return Ok(());
        };
        match frame {
            Frame::Nodes {
                nodes,
                index,
                scoped,
            } => {
                if step_nodes_frame(
                    invocation,
                    memory,
                    nodes,
                    index,
                    scoped,
                    #[cfg(feature = "subgroup-ops")]
                    snapshots,
                )? {
                    return Ok(());
                }
            }
            Frame::Loop {
                var,
                next,
                to,
                body,
            } => {
                step_loop_frame(invocation, var, next, to, body)?;
                return Ok(());
            }
        }
    }
}

pub(crate) fn axis_value(values: [u32; 3], axis: u8) -> Result<Value, ReferenceError> {
    (axis < 3)
        .then(|| Value::U32(values[axis as usize]))
        .ok_or_else(|| ReferenceError::new(format!("invocation/workgroup ID axis {axis} out of range. Fix: use 0, 1, or 2.")))
}

pub(crate) fn eval_to_index(
    expr: &vyre_foundation::ir::Expr,
    label: &str,
    invocation: &mut HashmapInvocation<'_>,
    memory: &mut HashmapMemory,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<u32, ReferenceError> {
    super::eval_expr(
        expr,
        invocation,
        memory,
        #[cfg(feature = "subgroup-ops")]
        snapshots,
    )?
    .try_as_u32()
    .ok_or_else(|| {
        ReferenceError::new(format!(
            "{label} cannot be represented as u32. Fix: use a non-negative scalar index within u32."
        ))
    })
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_expr_snapshot(
    expr: &vyre_foundation::ir::Expr,
    snapshot: &HashmapInvocationSnapshot,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &HashmapMemory,
) -> Result<Value, ReferenceError> {
    let empty_entry: &[vyre_foundation::ir::Node] = &[];
    let mut invocation =
        HashmapInvocation::new(snapshot.ids, snapshot.linear_local_index, empty_entry);
    invocation.locals.locals = snapshot.locals.locals.clone();
    let mut snapshot_memory = HashmapMemory {
        storage: memory.storage.clone(),
        workgroup: memory.workgroup.clone(),
    };
    super::eval_expr(expr, &mut invocation, &mut snapshot_memory, snapshots)
}

/// Capture every lane's locals for cross-lane collective evaluation, INDEXED BY LANE.
///
/// `subgroup_slice` carves a lane window out of this vector POSITIONALLY, and
/// `eval_subgroup_shuffle` addresses a lane inside that window by its
/// `linear_local_index % subgroup_width`, so position in this vector IS lane identity.
/// `invocations` is in STEP order, which a non-`Forward` `LaneOrder` permutes, so
/// capturing in iteration order made every subgroup collective read the wrong lanes
/// under a permuted schedule: a ballot returned the mask of a different subgroup and a
/// shuffle sourced a different lane, which is a change in RESULT, not in scheduling.
/// Sorting by lane index restores the invariant the readers assume for every schedule.
#[cfg(feature = "subgroup-ops")]
fn capture_invocation_snapshots(
    invocations: &[HashmapInvocation<'_>],
) -> Vec<HashmapInvocationSnapshot> {
    let mut snapshots: Vec<HashmapInvocationSnapshot> = invocations
        .iter()
        .map(|invocation| HashmapInvocationSnapshot {
            ids: invocation.ids,
            linear_local_index: invocation.linear_local_index,
            locals: invocation.locals.snapshot(),
        })
        .collect();
    snapshots.sort_unstable_by_key(|snapshot| snapshot.linear_local_index);
    snapshots
}

#[cfg(all(test, feature = "subgroup-ops"))]
mod tests {
    use super::capture_invocation_snapshots;
    use crate::execution::hashmap::invocation::HashmapInvocation;
    use crate::value::Value;
    use crate::workgroup::InvocationIds;
    use std::sync::Arc;
    use vyre_foundation::ir::Node;

    #[test]
    fn subgroup_snapshots_share_persistent_local_maps() {
        let entry: &[Node] = &[];
        let mut invocation = HashmapInvocation::new(InvocationIds::ZERO, 0, entry);
        for index in 0..256 {
            invocation
                .locals
                .bind(
                    &format!("lane_value_{index}"),
                    Value::Bytes(Arc::from(vec![index as u8; 4096])),
                )
                .expect("Fix: generated locals must bind once");
        }
        invocation.locals.push_scope();
        invocation
            .locals
            .bind("scoped", Value::U32(7))
            .expect("Fix: scoped local must bind once");

        let invocations = [invocation];
        let snapshots = capture_invocation_snapshots(&invocations);

        assert!(
            snapshots[0].locals.locals.ptr_eq(&invocations[0].locals.locals),
            "Fix: subgroup snapshots must clone the persistent locals root instead of rebuilding or deep-cloning values"
        );
        assert_eq!(
            snapshots[0].locals.local("scoped"),
            Some(Value::U32(7)),
            "Fix: subgroup snapshots must retain active locals without copying scope stacks"
        );
    }
}
