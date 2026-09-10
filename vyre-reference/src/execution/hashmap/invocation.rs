//! Invocation state, local scopes, and workgroup scheduling.
//!
//! This module owns the mutable per-lane interpreter state. It delegates node
//! stepping to `step` and synchronization checks to `sync`; it does not evaluate
//! expressions or resolve buffers directly.

use super::{
    memory::HashmapMemory,
    step::step_round_robin,
    sync::{
        live_collective_waiting_count, live_waiting_count, release_barrier_if_ready,
        release_collective_rendezvous, verify_uniform_control_flow,
    },
};
use crate::execution::async_transfer::{AsyncTransfer, PendingAsyncTransfers};
use crate::ReferenceError;
use crate::{
    value::Value,
    workgroup::{Frame, InvocationIds},
};
use rustc_hash::FxHashMap;
use std::sync::Arc;
use vyre_foundation::ir::{Node, Program, Tile};

/// Local variable environment for one invocation.
///
/// A flat hash map, not a persistent one. The persistent map this used to hold
/// made a subgroup snapshot one reference-count bump, and charged every other
/// operation for it: a `Let` inside a loop body allocated trie nodes on every
/// iteration, and `pop_scope` allocated again to unbind. The interpreter is the
/// parity oracle for every registered operation, so that allocation was on the
/// hot path of the whole conformance corpus while the snapshot it bought is
/// taken only by a program that uses subgroup collectives.
///
/// A snapshot is now a map clone: one `Arc` copy per live local and no value
/// payload, because every large [`Value`] is already reference-counted. Every
/// lane is captured once per round while a collective is reachable, so the
/// cost a subgroup program pays is live locals rather than a constant, which
/// is a few entries for the programs that use collectives at all.
///
/// Immutability travels with the value rather than in a second map. Reading it
/// separately meant two hashes of the same name on every assignment.
pub(crate) struct HashmapLocals {
    pub(crate) locals: FxHashMap<Arc<str>, Local>,
    pub(crate) scopes: Vec<Vec<Arc<str>>>,
}

/// One bound local: its value, and whether an assignment may replace it.
#[derive(Clone)]
pub(crate) struct Local {
    pub(crate) value: Value,
    pub(crate) immutable: bool,
}

impl HashmapLocals {
    pub(crate) fn new() -> Self {
        Self {
            locals: FxHashMap::default(),
            scopes: vec![Vec::new()],
        }
    }
    pub(crate) fn local(&self, name: &str) -> Option<Value> {
        self.locals.get(name).map(|local| local.value.clone())
    }

    #[cfg(feature = "subgroup-ops")]
    pub(crate) fn snapshot(&self) -> HashmapLocalSnapshot {
        HashmapLocalSnapshot {
            locals: self.locals.clone(),
        }
    }
    pub(crate) fn bind(&mut self, name: &str, value: Value) -> Result<Arc<str>, ReferenceError> {
        self.insert_binding(name, value, false)
    }
    /// Bind the induction variable of a loop, which no assignment may replace.
    pub(crate) fn bind_loop_var(&mut self, name: &str, value: Value) -> Result<(), ReferenceError> {
        self.insert_binding(name, value, true).map(|_| ())
    }
    fn insert_binding(
        &mut self,
        name: &str,
        value: Value,
        immutable: bool,
    ) -> Result<Arc<str>, ReferenceError> {
        if self.locals.contains_key(name) {
            return Err(ReferenceError::new(format!(
                "duplicate local binding `{name}`. Fix: choose a unique local name; shadowing is not allowed."
            )));
        }
        let name: Arc<str> = Arc::from(name);
        self.locals
            .insert(Arc::clone(&name), Local { value, immutable });
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(Arc::clone(&name));
        }
        Ok(name)
    }
    pub(crate) fn assign(&mut self, name: &str, value: Value) -> Result<(), ReferenceError> {
        let Some(local) = self.locals.get_mut(name) else {
            return Err(ReferenceError::new(format!(
                "assignment to undeclared variable `{name}`. Fix: add a Let before assigning it."
            )));
        };
        if local.immutable {
            return Err(ReferenceError::new(format!(
                "assignment to loop variable `{name}`. Fix: loop variables are immutable."
            )));
        }
        local.value = value;
        Ok(())
    }
    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }
    pub(crate) fn pop_scope(&mut self) {
        if let Some(names) = self.scopes.pop() {
            for name in names {
                self.locals.remove(&name);
            }
        }
    }
    pub(crate) fn remove(&mut self, name: &str) -> Option<Value> {
        self.locals.remove(name).map(|local| local.value)
    }
}

#[cfg(feature = "subgroup-ops")]
/// Local values of one lane, as they stood when a collective was reached.
#[derive(Clone)]
pub(crate) struct HashmapLocalSnapshot {
    pub(crate) locals: FxHashMap<Arc<str>, Local>,
}

#[cfg(feature = "subgroup-ops")]
impl HashmapLocalSnapshot {
    #[cfg(test)]
    pub(crate) fn local(&self, name: &str) -> Option<Value> {
        self.locals.get(name).map(|local| local.value.clone())
    }
}

pub(crate) struct HashmapInvocation<'a> {
    pub(crate) ids: InvocationIds,
    #[cfg_attr(
        not(feature = "subgroup-ops"),
        expect(
            dead_code,
            reason = "the lane index is read only by the subgroup reductions, so the field has no reader without that feature"
        )
    )]
    pub(crate) linear_local_index: u32,
    pub(crate) locals: HashmapLocals,
    pub(crate) returned: bool,
    pub(crate) waiting_at_barrier: bool,
    pub(crate) uniform_checks: Vec<(usize, bool)>,
    /// Set when this lane reached a `MemoryOrdering::GridSync` fence.
    ///
    /// A workgroup barrier is released by the lanes of one workgroup, so
    /// [`run_invocations`] owns it. A grid fence is released only once every
    /// workgroup in the dispatch has arrived, which no single workgroup can
    /// observe, so the lane stops here and the dispatch driver resumes it.
    pub(crate) waiting_at_grid_fence: bool,
    /// Set while this lane holds at a statement whose operands read its peers.
    ///
    /// A subgroup collective is defined over the lanes of one subgroup at one
    /// program point. Lanes step one node per round, and a branch whose
    /// condition is not lane-uniform gives one lane more nodes to run, so the
    /// lanes drift apart and stay apart across a loop back-edge. Reading a
    /// peer's locals at that point reads them from a different statement:
    /// either a local the peer has already unbound, which refused a name the
    /// program does bind, or the previous iteration's value, which is a wrong
    /// answer with no diagnostic at all. Hardware reconverges a subgroup
    /// before a converged collective, so the lane holds here until every live
    /// lane of the workgroup has arrived, and the statement then runs against
    /// peers that stand at the same point.
    pub(crate) waiting_for_collective_peers: bool,
    /// Set once this lane has been released from its rendezvous and may run
    /// the collective statement it holds at.
    pub(crate) collective_peers_arrived: bool,
    pub(crate) frames: Vec<Frame<'a>>,
    pub(crate) pending_async: PendingAsyncTransfers,
    pub(crate) op_cache: crate::execution::call::OpCache,
    /// The declared [`Tile`] of every tile this lane has bound, by name.
    ///
    /// `Node::TileMatmul` and `Node::TileReduce` name their operands and carry
    /// no shape of their own, so the only statement of a tile's extents and
    /// element type is the `TileDecl` or `TileLoad` that bound it. Without
    /// that record those two nodes have an element count and nothing else, and
    /// an element count does not determine a 2-D shape.
    pub(crate) tile_shapes: FxHashMap<Arc<str>, Arc<Tile>>,
}

impl<'a> HashmapInvocation<'a> {
    pub(crate) fn new(ids: InvocationIds, linear_local_index: u32, entry: &'a [Node]) -> Self {
        Self {
            ids,
            linear_local_index,
            locals: HashmapLocals::new(),
            returned: false,
            waiting_at_barrier: false,
            uniform_checks: Vec::new(),
            waiting_at_grid_fence: false,
            waiting_for_collective_peers: false,
            collective_peers_arrived: false,
            pending_async: PendingAsyncTransfers::new(),
            op_cache: FxHashMap::default(),
            tile_shapes: FxHashMap::default(),
            frames: vec![Frame::Nodes {
                nodes: entry,
                index: 0,
                scoped: false,
            }],
        }
    }
    pub(crate) fn done(&self) -> bool {
        self.returned || self.frames.is_empty()
    }

    /// Push one frame, refusing a lane whose nesting has reached the armed
    /// recursion ceiling.
    ///
    /// Every frame a lane enters goes through here, so the depth contract is
    /// stated once instead of at each nesting statement.
    ///
    /// # Errors
    /// Refuses with `BudgetExhaustion` when the frame ceiling is crossed.
    #[inline]
    pub(crate) fn push_frame(&mut self, frame: Frame<'a>) -> Result<(), ReferenceError> {
        self.frames.push(frame);
        crate::execution::step_budget::check_recursion_depth(self.frames.len())
    }

    #[inline]
    pub(crate) fn is_leader(&self) -> bool {
        self.linear_local_index == 0
    }

    #[inline]
    pub(crate) fn begin_async(
        &mut self,
        tag: &str,
        transfer: AsyncTransfer,
    ) -> Result<(), ReferenceError> {
        self.pending_async.begin(tag, transfer)
    }

    #[inline]
    pub(crate) fn finish_async(&mut self, tag: &str) -> Result<AsyncTransfer, ReferenceError> {
        self.pending_async.finish(tag)
    }
}

#[cfg(feature = "subgroup-ops")]
#[derive(Clone)]
pub(crate) struct HashmapInvocationSnapshot {
    pub(crate) ids: InvocationIds,
    pub(crate) linear_local_index: u32,
    pub(crate) locals: HashmapLocalSnapshot,
}

pub(crate) fn create_invocations<'a>(
    program: &Program,
    workgroup: [u32; 3],
    entry: &'a [Node],
) -> Result<Vec<HashmapInvocation<'a>>, ReferenceError> {
    let [sx, sy, sz] = program.workgroup_size();
    let total = sx.checked_mul(sy).and_then(|c| c.checked_mul(sz)).ok_or_else(|| {
        ReferenceError::new("workgroup invocation count overflows u32. Fix: reduce workgroup dimensions before reference execution.")
    })?;
    let cap = usize::try_from(total).map_err(|_| {
        ReferenceError::new("workgroup invocation count exceeds host usize. Fix: reduce workgroup dimensions before reference execution.")
    })?;
    let mut invocations = Vec::with_capacity(cap);
    for z in 0..sz {
        let gz = workgroup[2].checked_mul(sz).and_then(|b| b.checked_add(z)).ok_or_else(|| {
            ReferenceError::new("workgroup * dispatch dimensions overflow u32 global id. Fix: reduce workgroup id or workgroup size so each global_invocation_id component fits in u32.")
        })?;
        for y in 0..sy {
            let gy = workgroup[1].checked_mul(sy).and_then(|b| b.checked_add(y)).ok_or_else(|| {
                ReferenceError::new("workgroup * dispatch dimensions overflow u32 global id. Fix: reduce workgroup id or workgroup size so each global_invocation_id component fits in u32.")
            })?;
            for x in 0..sx {
                let gx = workgroup[0].checked_mul(sx).and_then(|b| b.checked_add(x)).ok_or_else(|| {
                    ReferenceError::new("workgroup * dispatch dimensions overflow u32 global id. Fix: reduce workgroup id or workgroup size so each global_invocation_id component fits in u32.")
                })?;
                let idx = invocations.len() as u32;
                invocations.push(HashmapInvocation::new(
                    InvocationIds {
                        global: [gx, gy, gz],
                        workgroup,
                        local: [x, y, z],
                    },
                    idx,
                    entry,
                ));
            }
        }
    }
    Ok(invocations)
}

/// Step this workgroup's lanes until each one is finished or suspended on a
/// whole-grid fence.
///
/// Returns `true` when at least one live lane is holding at a grid fence. The
/// dispatch driver owns that release, because it is the only caller that can
/// see whether the rest of the grid has arrived.
pub(crate) fn run_invocations(
    memory: &mut HashmapMemory,
    invocations: &mut [HashmapInvocation<'_>],
    #[cfg(feature = "subgroup-ops")] uses_subgroup_ops: bool,
) -> Result<bool, ReferenceError> {
    while invocations
        .iter()
        .any(|inv| !inv.done() && !inv.waiting_at_grid_fence)
    {
        // Charged per round as well as per statement, so a barrier-release
        // cycle that advances no statement is still bounded.
        crate::execution::step_budget::charge()?;
        let made_progress = step_round_robin(
            memory,
            invocations,
            #[cfg(feature = "subgroup-ops")]
            uses_subgroup_ops,
        )?;
        verify_uniform_control_flow(invocations)?;
        if release_collective_rendezvous(invocations) {
            continue;
        }
        if release_barrier_if_ready(invocations) {
            continue;
        }
        if !made_progress && live_collective_waiting_count(invocations) > 0 {
            return Err(ReferenceError::new("program violates uniform-control-flow rule: not every live invocation reached the same subgroup collective. Fix: evaluate the collective in control flow every lane of the subgroup enters."));
        }
        if !made_progress && live_waiting_count(invocations) > 0 {
            return Err(ReferenceError::new("program violates uniform-control-flow rule: not every live invocation reached the same barrier. Fix: move Barrier to uniform control flow."));
        }
    }
    let fenced = invocations
        .iter()
        .any(|inv| !inv.done() && inv.waiting_at_grid_fence);
    // A lane suspended mid-program may legitimately still hold an async
    // transfer it will wait on after the fence, so the drain contract is
    // checked once the workgroup has actually run out.
    if !fenced {
        for invocation in invocations.iter() {
            invocation.pending_async.assert_drained(invocation.ids)?;
        }
    }
    Ok(fenced)
}
