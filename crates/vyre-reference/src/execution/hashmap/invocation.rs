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
        release_collective_rendezvous, verify_uniform_control_flow, RendezvousKind,
    },
};
use crate::execution::async_transfer::{AsyncTransfer, PendingAsyncTransfers};
use crate::ReferenceError;
use crate::{
    value::Value,
    workgroup::{Frame, InvocationIds},
};
use rustc_hash::{FxHashMap, FxHashSet};
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
/// payload, because every [`Value`] payload is reference-counted. Every lane
/// is captured once per round while a collective is reachable, so the cost a
/// subgroup program pays is live locals rather than a constant, which is a few
/// entries for the programs that use collectives at all.
///
/// Immutability travels with the value rather than in a second map. Reading it
/// separately meant two hashes of the same name on every assignment.
pub(crate) struct HashmapLocals {
    pub(crate) locals: FxHashMap<Arc<str>, Local>,
    pub(crate) scopes: ScopeArena,
}

/// Names bound by every open scope of one lane, in one streaming region.
///
/// # What it reclaims
///
/// The slots of a scope, at the moment that scope closes. `open` records where
/// the current scope starts in `names`; closing truncates back to that mark,
/// and the next scope to open writes over the same slots. A lane that enters
/// and leaves a million loop bodies therefore reaches the high-water mark of
/// one body and stays there, rather than taking and returning a vector per
/// entry.
///
/// The name storage itself, through `held`. A `Let` inside a loop body binds
/// the same name on every iteration, and building an `Arc<str>` for it each
/// time charged the allocator once per iteration per binding. A name a closing
/// scope retires stays in `held`, so the next scope that binds it takes a
/// reference count instead of an allocation. That makes name allocation a
/// function of the distinct names a lane binds, which the program fixes,
/// rather than of the number of times it binds them, which the trip counts do.
///
/// # What it does not reclaim
///
/// Capacity. `names` and `open` keep the largest allocation they have needed
/// for the life of the lane, and hand it back only when the lane is dropped.
///
/// A name the lane will not bind again. `held` grows to the distinct names the
/// lane has bound and stays there until the lane is dropped, so a program that
/// binds a fresh name per iteration pays for each one. Nothing in the IR
/// generates names that way: a binding's name is written in the program text.
///
/// The values. Unbinding a name removes it from the local map, and the payload
/// behind it lives until the last value sharing it is dropped.
pub(crate) struct ScopeArena {
    /// Every name bound by every open scope, oldest scope first.
    names: Vec<Arc<str>>,
    /// Index into `names` at which each open scope's names begin.
    open: Vec<usize>,
    /// Every name this lane has bound, kept for the next scope that binds it.
    held: FxHashSet<Arc<str>>,
}

impl ScopeArena {
    /// An arena with one open scope and nothing bound in it.
    fn new() -> Self {
        Self {
            names: Vec::new(),
            open: vec![0],
            held: FxHashSet::default(),
        }
    }

    /// The arena's `Arc<str>` for `name`, allocating one only the first time
    /// this lane binds it.
    fn name(&mut self, name: &str) -> Arc<str> {
        if let Some(held) = self.held.get(name) {
            return Arc::clone(held);
        }
        let owned: Arc<str> = Arc::from(name);
        self.held.insert(Arc::clone(&owned));
        owned
    }

    /// Record `name` against the innermost open scope.
    fn bind(&mut self, name: Arc<str>) {
        self.names.push(name);
    }

    /// Open a scope whose names start after every name bound so far.
    fn push(&mut self) {
        self.open.push(self.names.len());
    }

    /// Close the innermost scope, yielding the names it bound.
    ///
    /// Yields nothing when no scope is open, which is the same outcome as
    /// closing an empty one.
    fn pop(&mut self) -> std::vec::Drain<'_, Arc<str>> {
        let start = self.open.pop().unwrap_or(0);
        self.names.drain(start..)
    }

    /// Slots the arena holds, whether or not an open scope is using them.
    #[cfg(test)]
    fn slot_capacity(&self) -> usize {
        self.names.capacity()
    }

    /// Slots the open scopes are using.
    #[cfg(test)]
    fn slots_in_use(&self) -> usize {
        self.names.len()
    }
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
            scopes: ScopeArena::new(),
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
        let name = self.scopes.name(name);
        self.locals
            .insert(Arc::clone(&name), Local { value, immutable });
        self.scopes.bind(Arc::clone(&name));
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
        self.scopes.push();
    }
    pub(crate) fn pop_scope(&mut self) {
        for name in self.scopes.pop() {
            self.locals.remove(&name);
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
    /// Position of this lane within its workgroup, in the linear order the
    /// dispatch assigns.
    ///
    /// `is_leader` reads it whatever the feature set, and the subgroup
    /// collectives read it as lane identity.
    pub(crate) linear_local_index: u32,
    pub(crate) locals: HashmapLocals,
    pub(crate) returned: bool,
    pub(crate) waiting_at_barrier: bool,
    /// Branch conditions this lane evaluated whose body can reach a
    /// rendezvous, keyed by `If` node identity, with the rendezvous kind for
    /// the diagnostic.
    pub(crate) uniform_checks: Vec<(usize, bool, RendezvousKind)>,
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

/// Invocations one workgroup of `program` runs.
///
/// The single owner of that product. Dispatch geometry divides the grid by it
/// and [`create_invocations`] allocates by it, and a product those two disagree
/// on dispatches a grid the program does not describe, so both read it here.
///
/// # Errors
/// Refuses when the extents multiply past what a `u32` counts. Validation
/// refuses a zero extent but caps no product, so this is the only rule such a
/// program meets.
pub(crate) fn invocations_per_workgroup(program: &Program) -> Result<u32, ReferenceError> {
    let [sx, sy, sz] = program.workgroup_size();
    sx.checked_mul(sy)
        .and_then(|count| count.checked_mul(sz))
        .ok_or_else(|| {
            ReferenceError::overflow(format!(
                "workgroup size {sx}x{sy}x{sz} states more invocations per workgroup than a u32 \
                 counts. Fix: state workgroup extents whose product is at most {}.",
                u32::MAX
            ))
        })
}

pub(crate) fn create_invocations<'a>(
    program: &Program,
    workgroup: [u32; 3],
    entry: &'a [Node],
) -> Result<Vec<HashmapInvocation<'a>>, ReferenceError> {
    let total = invocations_per_workgroup(program)?;
    let [sx, sy, sz] = program.workgroup_size();
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
            crate::interleaving::note_barrier_release();
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

// Inline: `ScopeArena` and `HashmapLocals` are crate-private, and neither the
// slot reuse nor the name sharing is observable through a public entry point.
#[cfg(test)]
mod tests {
    use super::{HashmapLocals, ScopeArena};
    use crate::value::Value;
    use std::sync::Arc;

    /// WHY: the arena exists so a lane that enters a scope a million times
    /// does not charge the allocator a million times. Two properties carry
    /// that: a closing scope's slots are reused by the next scope rather than
    /// released, and a name a closing scope retires is handed back rather than
    /// rebuilt. An arena that grew its slot vector per scope, or that built a
    /// fresh `Arc<str>` per binding, would still pass every behavioural test
    /// in the suite and cost one allocation per scope entry.
    ///
    /// It does not catch the allocation count of a whole evaluation, which
    /// `entering_a_scope_reuses_the_region_the_previous_scope_retired` in the
    /// integration suite measures against the allocator.
    #[test]
    fn a_closing_scope_leaves_its_slots_and_its_names_for_the_next_scope() {
        let mut locals = HashmapLocals::new();
        let mut first: Option<Arc<str>> = None;

        for round in 0..8u32 {
            locals.push_scope();
            let bound = locals
                .bind("step", Value::U32(round))
                .expect("Fix: a name unbound by the previous scope binds again");
            match &first {
                None => first = Some(bound),
                Some(first) => assert!(
                    Arc::ptr_eq(first, &bound),
                    "Fix: a name the previous scope retired is handed back, not rebuilt"
                ),
            }
            assert_eq!(
                locals.scopes.slots_in_use(),
                1,
                "Fix: one scope binding one name uses one slot, so the slots a closed scope \
                 retired are the slots the next one writes"
            );
            locals.pop_scope();
            assert!(
                locals.local("step").is_none(),
                "Fix: closing a scope unbinds the names it bound"
            );
        }

        assert!(
            locals.scopes.slot_capacity() <= 4,
            "Fix: eight open-and-close cycles must not grow the arena, got {} slots",
            locals.scopes.slot_capacity()
        );
    }

    /// WHY: the arena hands back a retired name by identity, so a lookup that
    /// matched the wrong name would bind one local under another's key. The
    /// case above proves the same name is shared; this proves a different name
    /// is not.
    #[test]
    fn two_distinct_names_never_share_one_allocation() {
        let mut arena = ScopeArena::new();
        let left = arena.name("alpha");
        let right = arena.name("beta");
        assert!(
            !Arc::ptr_eq(&left, &right),
            "Fix: two distinct names are two distinct allocations"
        );
        assert!(
            Arc::ptr_eq(&left, &arena.name("alpha")),
            "Fix: the same name is one allocation however often it is bound"
        );
        assert_eq!(
            &*right, "beta",
            "Fix: the arena hands back the name asked for"
        );
    }
}
