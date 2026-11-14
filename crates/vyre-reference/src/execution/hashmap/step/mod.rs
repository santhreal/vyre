//! Round-robin node stepping and expression-adjacent execution helpers.
mod node_step;
pub(crate) use crate::execution::axis_value;
pub(crate) use node_step::{eval_call, step_loop_frame, step_nodes_frame};

#[cfg(feature = "subgroup-ops")]
use super::invocation::HashmapInvocationSnapshot;
use super::{
    invocation::{HashmapInvocation, InvocationView},
    memory::HashmapMemory,
};
#[cfg(feature = "subgroup-ops")]
use crate::value::Value;
use crate::workgroup::Frame;
use crate::ReferenceError;

pub(crate) fn step_round_robin(
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] uses_subgroup_ops: bool,
) -> Result<bool, ReferenceError> {
    #[cfg(feature = "subgroup-ops")]
    if uses_subgroup_ops
        && invocations
            .iter()
            .any(|inv| !inv.done() && inv.collective_peers_arrived)
    {
        let mut snapshots = std::mem::take(&mut memory.snapshot_scratch);
        capture_invocation_snapshots_into(invocations, &mut snapshots);
        let res = step_active_lanes(memory, invocations, true, &snapshots);
        memory.snapshot_scratch = snapshots;
        return res;
    }
    step_active_lanes(
        memory,
        invocations,
        #[cfg(feature = "subgroup-ops")]
        uses_subgroup_ops,
        #[cfg(feature = "subgroup-ops")]
        &[],
    )
}

fn step_active_lanes(
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] uses_subgroup_ops: bool,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<bool, ReferenceError> {
    let mut made_progress = false;
    for index in 0..invocations.len() {
        if invocations[index].done()
            || invocations[index].waiting_at_barrier
            || invocations[index].waiting_at_grid_fence
            || invocations[index].waiting_for_collective_peers
        {
            continue;
        }
        step(
            index,
            memory,
            invocations,
            #[cfg(feature = "subgroup-ops")]
            uses_subgroup_ops,
            #[cfg(feature = "subgroup-ops")]
            snapshots,
        )?;
        made_progress = true;
    }
    Ok(made_progress)
}

fn step(
    index: usize,
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] uses_subgroup_ops: bool,
    #[cfg(feature = "subgroup-ops")] snapshots: &[HashmapInvocationSnapshot],
) -> Result<(), ReferenceError> {
    let invocation = &mut invocations[index];
    if invocation.done()
        || invocation.waiting_at_barrier
        || invocation.waiting_at_grid_fence
        || invocation.waiting_for_collective_peers
    {
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
                    uses_subgroup_ops,
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

pub(crate) fn eval_to_index(
    expr: &vyre_foundation::ir::Expr,
    label: &str,
    invocation: InvocationView<'_>,
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

/// Refuse a collective argument that writes memory across more than one lane.
///
/// A collective gathers every lane in the subgroup window, and every lane in
/// that window reaches the collective, so a `width`-lane window evaluates the
/// argument `width * width` times. Buffer bytes live behind a shared handle,
/// so an atomic in the argument commits every one of those times where the
/// device commits once per lane: a 32-lane
/// `subgroupAdd(atomicAdd(counter, 1))` left `counter` at 1024 and handed each
/// lane a different reduction, where the device leaves it at 32 and every lane
/// reads the same sum. Evaluating each lane's argument exactly once instead
/// would still leave the per-lane values decided by the order the atomics
/// commit, which is a property of the schedule rather than of the program, so
/// there is no expected output to issue and the oracle refuses instead of
/// inventing one.
///
/// A window of one lane has neither multiplicity nor an order to depend on.
/// The argument is evaluated once, commits once, and the device commits once,
/// so the reduction over that single value is exact and is issued.
///
/// `lanes` is the length of the gathered window, so the refusal is decided
/// once per collective rather than once per lane.
#[cfg(feature = "subgroup-ops")]
pub(crate) fn refuse_effectful_collective_argument(
    expr: &vyre_foundation::ir::Expr,
    lanes: usize,
) -> Result<(), ReferenceError> {
    if lanes < 2 {
        return Ok(());
    }
    // Children come from foundation's single `expr_children` owner, so a new
    // operand-carrying variant is searched without a second traversal here,
    // and the walk is an explicit worklist rather than recursion. A new
    // memory-writing variant is still refused: `eval_expr` has no arm for it
    // and its catch-all errors, so the fail-closed direction does not depend
    // on this predicate naming it.
    if !vyre_foundation::visit::any_subexpr(expr, &mut |sub| {
        matches!(sub, vyre_foundation::ir::Expr::Atomic { .. })
    }) {
        return Ok(());
    }
    Err(ReferenceError::incomplete_dispatch_semantics(
        "a subgroup collective argument writes memory through an atomic and the collective \
         gathers more than one lane. Every lane reaches the collective, so the argument would \
         commit once per lane per lane rather than once per lane, and which commits land is a \
         property of the gather and of the order the atomics commit rather than of the program. \
         Fix: bind the atomic with a Let before the collective and pass the bound variable as \
         the argument.",
    ))
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_expr_snapshot(
    expr: &vyre_foundation::ir::Expr,
    snapshot: &HashmapInvocationSnapshot,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &mut HashmapMemory,
) -> Result<Value, ReferenceError> {
    super::eval_expr(expr, snapshot.view(), memory, snapshots)
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
fn capture_invocation_snapshots_into(
    invocations: &[HashmapInvocation<'_>],
    snapshots: &mut Vec<HashmapInvocationSnapshot>,
) {
    if snapshots.len() > invocations.len() {
        snapshots.truncate(invocations.len());
    }
    for (idx, invocation) in invocations.iter().enumerate() {
        if let Some(slot) = snapshots.get_mut(idx) {
            slot.ids = invocation.ids;
            slot.linear_local_index = invocation.linear_local_index;
            slot.locals.clone_from_locals(&invocation.locals);
        } else {
            let mut locals = super::invocation::HashmapLocalSnapshot::empty();
            locals.clone_from_locals(&invocation.locals);
            snapshots.push(HashmapInvocationSnapshot {
                ids: invocation.ids,
                linear_local_index: invocation.linear_local_index,
                locals,
            });
        }
    }
    snapshots.sort_unstable_by_key(|snapshot| snapshot.linear_local_index);
}

#[cfg(all(test, feature = "subgroup-ops"))]
fn capture_invocation_snapshots(
    invocations: &[HashmapInvocation<'_>],
) -> Vec<HashmapInvocationSnapshot> {
    let mut snapshots = Vec::with_capacity(invocations.len());
    capture_invocation_snapshots_into(invocations, &mut snapshots);
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
    /// A snapshot is taken per lane every time a collective is reached, so it must
    /// stay proportional to the number of live locals and never touch their payloads.
    /// The map is a flat hash map, so this is what a reader can rely on: every value
    /// in the snapshot is the same allocation the invocation still holds, and a
    /// local bound in an inner scope is present. A snapshot that rebuilt values, or
    /// that captured only the outermost scope, fails here.
    #[test]
    fn subgroup_snapshots_copy_no_local_payload() {
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

        for index in 0..256 {
            let name = format!("lane_value_{index}");
            let live = invocations[0].locals.local(&name);
            let captured = snapshots[0].locals.local(&name);
            let (Some(Value::Bytes(live)), Some(Value::Bytes(captured))) = (live, captured) else {
                panic!("Fix: subgroup snapshots must retain every bound local `{name}` as bytes");
            };
            assert!(
                Arc::ptr_eq(&live, &captured),
                "Fix: subgroup snapshots must share local payloads instead of copying them"
            );
        }
        assert_eq!(
            snapshots[0].locals.local("scoped"),
            Some(Value::U32(7)),
            "Fix: subgroup snapshots must retain active locals without copying scope stacks"
        );
    }
}
