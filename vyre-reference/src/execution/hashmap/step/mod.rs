//! Round-robin node stepping and expression-adjacent execution helpers.
mod node_step;
pub(crate) use crate::execution::axis_value;
pub(crate) use node_step::{eval_call, step_loop_frame, step_nodes_frame};

#[cfg(feature = "subgroup-ops")]
use super::invocation::HashmapInvocationSnapshot;
use super::{invocation::HashmapInvocation, memory::HashmapMemory};
#[cfg(feature = "subgroup-ops")]
use crate::value::Value;
use crate::workgroup::Frame;
use crate::ReferenceError;

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

/// Refuse a collective argument that writes memory.
///
/// A lane that reaches a collective evaluates the argument for every lane in
/// its subgroup, and every lane in the subgroup reaches the collective, so the
/// argument is evaluated `width * width` times. Buffer bytes live behind a
/// shared handle, so an atomic in the argument commits every one of those
/// times where the device commits once per lane: a 32-lane
/// `subgroupAdd(atomicAdd(counter, 1))` left `counter` at 1024 and handed each
/// lane a different reduction, where the device leaves it at 32 and every lane
/// reads the same sum. How many commits land is a property of how this
/// interpreter gathers lanes rather than of the program, so there is no
/// expected output to issue and the oracle refuses instead of inventing one.
#[cfg(feature = "subgroup-ops")]
fn refuse_effectful_collective_argument(
    expr: &vyre_foundation::ir::Expr,
) -> Result<(), ReferenceError> {
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
        "a subgroup collective argument writes memory through an atomic. The collective gathers \
         every lane in the subgroup and every lane reaches it, so the argument would commit once \
         per lane per lane rather than once per lane, and which commits land is a property of \
         the gather rather than of the program. Fix: bind the atomic with a Let before the \
         collective and pass the bound variable as the argument.",
    ))
}

#[cfg(feature = "subgroup-ops")]
pub(crate) fn eval_expr_snapshot(
    expr: &vyre_foundation::ir::Expr,
    snapshot: &HashmapInvocationSnapshot,
    snapshots: &[HashmapInvocationSnapshot],
    memory: &mut HashmapMemory,
) -> Result<Value, ReferenceError> {
    refuse_effectful_collective_argument(expr)?;
    let empty_entry: &[vyre_foundation::ir::Node] = &[];
    let mut invocation =
        HashmapInvocation::new(snapshot.ids, snapshot.linear_local_index, empty_entry);
    invocation.locals.locals = snapshot.locals.locals.clone();
    // The argument cannot write, so evaluating it against the live memory is
    // byte-identical to evaluating it against a copy. The copy this used to
    // take allocated both buffer maps once per lane per collective and
    // isolated nothing: buffer bytes sit behind a shared handle, so every
    // write reached the original anyway.
    super::eval_expr(expr, &mut invocation, memory, snapshots)
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
