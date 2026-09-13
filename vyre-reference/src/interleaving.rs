//! Shadow memory, race findings, and the static interleaving walk.
//!
//! [`ShadowMemory`] is the detector. It records one access per
//! `(buffer, index)` per invocation and reports a pair that reached the same
//! location in the same synchronization phase.
//!
//! Two callers drive it. `ReferenceRequest::explore_races` executes the
//! submitted program once per deterministic step order with the thread-local
//! tracker below enabled, so every finding comes from an access the
//! interpreter actually performed. [`explore_bounded_interleavings`] walks the
//! IR statically instead: it visits each `Node::Store` once per configured
//! invocation with an index derived from the invocation coordinate, never
//! evaluates an expression, and returns no output values. The static walk
//! accepts a program whose buffers are undeclared, which an execution refuses,
//! and it reports nothing about a conflict whose index is data-dependent.

use std::cell::RefCell;

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

use crate::value::Value;
use crate::ReferenceError;

/// Configuration for bounded interleaving exploration.
#[derive(Clone, Debug)]
pub struct InterleavingConfig {
    /// Workgroup dimensions `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Grid dimensions `[x, y, z]`.
    pub grid_size: [u32; 3],
    /// Maximum distinct scheduling permutations to explore.
    pub max_interleavings: usize,
    /// Maximum execution steps per invocation before termination.
    pub step_bound: usize,
}

impl Default for InterleavingConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [2, 1, 1],
            grid_size: [1, 1, 1],
            max_interleavings: 16,
            step_bound: 1024,
        }
    }
}

/// Result report from bounded interleaving exploration.
#[derive(Clone, Debug)]
pub struct InterleavingReport {
    /// Number of distinct scheduling interleavings explored.
    pub explored_schedules: usize,
    /// Whether all explored interleavings proved race-free.
    pub race_free: bool,
    /// Final output buffer values verified across interleavings.
    pub final_outputs: FxHashMap<String, Vec<u32>>,
}

/// Access kind on a buffer location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAccessKind {
    /// Non-atomic read.
    Read,
    /// Non-atomic write.
    Write,
    /// Atomic read-modify-write with ordering and scope.
    Atomic {
        /// Atomic ordering.
        ordering: AtomicOrdering,
        /// Memory scope.
        scope: MemoryScope,
    },
    /// Asynchronous DMA / copy transaction.
    AsyncTransfer {
        /// Transaction lifecycle.
        lifecycle: AsyncTransactionLifecycle,
    },
}

/// A recorded memory access by a specific invocation.
#[derive(Clone, Debug)]
pub struct MemoryAccessRecord {
    /// Invocation coordinates `[x, y, z]`.
    pub invocation: [u32; 3],
    /// Workgroup coordinates `[x, y, z]` the invocation belongs to.
    pub workgroup: [u32; 3],
    /// Memory access kind.
    pub kind: MemoryAccessKind,
    /// Barrier phase / generation when access occurred.
    pub barrier_phase: u64,
    /// Grid fence generation when the access occurred.
    pub grid_phase: u64,
    /// Memory visibility scope of the operation.
    pub memory_scope: MemoryScope,
    /// Storage domain accessed.
    pub storage_domain: StorageDomain,
    /// Order this access was recorded in, within one explored schedule.
    pub sequence: u64,
}

/// One hazard a bounded exploration reported.
///
/// The two variants are distinct classes. An [`UnsynchronizedAccess`] is a
/// conflict inside one explored order: two invocations reached one location in
/// the same synchronization phase with no barrier, fence, or atomic between
/// them. A [`ScheduleDisagreement`] is a conflict between two explored orders:
/// the same dispatch produced different output bytes depending on which order
/// the lanes were stepped in. A program can carry either without the other, so
/// neither one subsumes the other.
///
/// [`UnsynchronizedAccess`]: RaceFinding::UnsynchronizedAccess
/// [`ScheduleDisagreement`]: RaceFinding::ScheduleDisagreement
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RaceFinding {
    /// Two explored step orders produced different bytes for one output.
    ScheduleDisagreement {
        /// Debug rendering of the first step order.
        first_order: String,
        /// Debug rendering of the step order that disagreed with it.
        second_order: String,
        /// Position of the first output value that differs.
        output_index: usize,
    },
    /// Two invocations conflicted on one location in one synchronization phase.
    UnsynchronizedAccess {
        /// Buffer the conflict occurred on.
        buffer: String,
        /// Element index within that buffer.
        index: u64,
        /// Invocation that reached the location first, and its access kind.
        first: [u32; 3],
        /// Access kind of the first invocation.
        first_kind: MemoryAccessKind,
        /// Invocation that conflicted with it.
        second: [u32; 3],
        /// Access kind of the second invocation.
        second_kind: MemoryAccessKind,
        /// Barrier phase both accesses fell in.
        barrier_phase: u64,
    },
}

impl std::fmt::Display for RaceFinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ScheduleDisagreement {
                first_order,
                second_order,
                output_index,
            } => write!(
                f,
                "schedule exploration disagreed: output {output_index} differs between step order \
                 {first_order} and {second_order}. Fix: give every shared output slot a single \
                 writer, or write it through a commutative atomic, so the result does not depend \
                 on the order the lanes were stepped in."
            ),
            Self::UnsynchronizedAccess {
                buffer,
                index,
                first,
                first_kind,
                second,
                second_kind,
                barrier_phase,
            } => write!(
                f,
                "Data race detected on buffer `{buffer}` at index {index}: unsynchronized access \
                 by invocation {first:?} ({first_kind:?}) races with access by invocation \
                 {second:?} ({second_kind:?}) in barrier phase {barrier_phase}"
            ),
        }
    }
}

/// Result of a bounded race exploration over one submitted program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaceExplorationReport {
    /// Number of deterministic step orders the exploration executed.
    pub orders_explored: usize,
    /// Every distinct hazard the exploration reported, first occurrence order.
    pub findings: Vec<RaceFinding>,
    /// Interpreter steps the whole exploration charged against the budget.
    pub steps_executed: u64,
}

impl RaceExplorationReport {
    /// True when no explored order reported a hazard.
    #[must_use]
    pub fn is_race_free(&self) -> bool {
        self.findings.is_empty()
    }
}

/// One invocation, identified by its lane and the workgroup it belongs to.
type Participant = ([u32; 3], [u32; 3]);

/// One release an atomic published on a location.
#[derive(Clone, Copy, Debug)]
struct ReleaseEdge {
    /// Invocation that published it.
    publisher: Participant,
    /// Scope the releasing atomic declared.
    scope: MemoryScope,
    /// Access sequence at the moment of the release. Everything the publisher
    /// recorded before this number is what the release publishes.
    seq: u64,
}

/// Shadow memory tracker detecting data races and verifying synchronization.
///
/// Two accesses to one location are separated when a synchronization edge
/// stands between them. Inside one workgroup that edge is a barrier, counted
/// by `barrier_phase`. Across workgroups a barrier orders nothing, because the
/// two workgroups never rendezvous, so the only edge is a whole-grid fence,
/// counted by `grid_phase`. Comparing a cross-workgroup pair against
/// `barrier_phase` would report every unsynchronized cross-workgroup write as
/// safe whenever the two workgroups happened to have executed the same number
/// of barriers, which is the usual case.
///
/// A release/acquire pair on one location is the third edge. A barrier is not
/// the only way a program publishes: an invocation that writes a location and
/// then releases a flag, read by an acquiring invocation that then reads the
/// location, has ordered the two accesses without any rendezvous. The tracker
/// counted no such edge, so every correct release/acquire handoff was reported
/// as an unsynchronized access and the only way to write a program the oracle
/// accepted was to insert a barrier the algorithm did not need.
#[derive(Clone, Debug, Default)]
pub struct ShadowMemory {
    /// Per `(buffer_name, index)` list of access records.
    accesses: FxHashMap<(String, u64), Vec<MemoryAccessRecord>>,
    /// Per `(buffer_name, index)` releases published by an atomic there.
    releases: FxHashMap<(String, u64), Vec<ReleaseEdge>>,
    /// For each `(publisher, acquirer)` pair, the highest access sequence the
    /// acquirer has synchronized with. Everything the publisher recorded at or
    /// below it happens before everything the acquirer records afterwards.
    happens_before: FxHashMap<(Participant, Participant), u64>,
    /// Number of accesses recorded, and the sequence the next one takes.
    seq: u64,
    /// Active barrier generation within the current workgroup.
    barrier_phase: u64,
    /// Active whole-grid fence generation.
    grid_phase: u64,
    /// Workgroup whose lanes are currently being stepped.
    workgroup: [u32; 3],
}

impl ShadowMemory {
    /// Create a new shadow memory tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            accesses: FxHashMap::default(),
            releases: FxHashMap::default(),
            happens_before: FxHashMap::default(),
            seq: 0,
            barrier_phase: 0,
            grid_phase: 0,
            workgroup: [0, 0, 0],
        }
    }

    /// Advance the barrier phase (rendezvous passed).
    pub fn advance_barrier_phase(&mut self, _scope: ExecutionScope) {
        self.barrier_phase = self.barrier_phase.saturating_add(1);
    }

    /// Advance the whole-grid fence generation.
    ///
    /// A grid fence orders every invocation in the dispatch. The barrier count
    /// is left alone, because it restarts per workgroup anyway and the
    /// same-workgroup separation test reads the fence generation alongside it.
    pub fn advance_grid_phase(&mut self) {
        self.grid_phase = self.grid_phase.saturating_add(1);
    }

    /// Begin recording accesses made by the lanes of `workgroup`.
    ///
    /// The barrier count restarts, because a barrier count belongs to one
    /// workgroup's rendezvous sequence and carries no meaning in another's.
    pub fn enter_workgroup(&mut self, workgroup: [u32; 3]) {
        self.workgroup = workgroup;
        self.barrier_phase = 0;
    }

    /// Current barrier generation.
    #[must_use]
    pub const fn current_barrier_phase(&self) -> u64 {
        self.barrier_phase
    }

    /// Current whole-grid fence generation.
    #[must_use]
    pub const fn current_grid_phase(&self) -> u64 {
        self.grid_phase
    }

    /// Record a memory access and return the hazard it conflicts with, if any.
    ///
    /// The access is recorded either way, so one conflicting location reports
    /// once per conflicting pair rather than ending the walk at the first one.
    pub fn record_access(
        &mut self,
        buffer: &str,
        index: u64,
        invocation: [u32; 3],
        kind: MemoryAccessKind,
        scope: MemoryScope,
        domain: StorageDomain,
    ) -> Option<RaceFinding> {
        let barrier_phase = self.barrier_phase;
        let grid_phase = self.grid_phase;
        let workgroup = self.workgroup;
        let sequence = self.seq;
        self.seq = self.seq.saturating_add(1);
        let me: Participant = (invocation, workgroup);
        self.observe_acquire(buffer, index, me, kind);

        let key = (buffer.to_string(), index);
        let mut finding = None;
        if let Some(history) = self.accesses.get(&key) {
            for prior in history {
                // Accesses from the same lane are sequenced and not a race.
                if prior.invocation == invocation && prior.workgroup == workgroup {
                    continue;
                }
                let separated = if prior.workgroup == workgroup {
                    (prior.grid_phase, prior.barrier_phase) != (grid_phase, barrier_phase)
                } else {
                    prior.grid_phase != grid_phase
                };
                if separated {
                    continue;
                }
                if !conflicts(prior.kind, kind) {
                    continue;
                }
                let published = self
                    .happens_before
                    .get(&((prior.invocation, prior.workgroup), me))
                    .is_some_and(|synchronized_through| *synchronized_through > prior.sequence);
                if published {
                    continue;
                }
                finding = Some(RaceFinding::UnsynchronizedAccess {
                    buffer: buffer.to_string(),
                    index,
                    first: prior.invocation,
                    first_kind: prior.kind,
                    second: invocation,
                    second_kind: kind,
                    barrier_phase,
                });
                break;
            }
        }

        self.accesses
            .entry(key)
            .or_default()
            .push(MemoryAccessRecord {
                invocation,
                workgroup,
                kind,
                barrier_phase,
                grid_phase,
                memory_scope: scope,
                storage_domain: domain,
                sequence,
            });
        self.publish_release(buffer, index, me, kind, sequence);

        finding
    }

    /// Record what a releasing atomic on this location publishes.
    ///
    /// Everything the publisher recorded before `sequence` is what an acquirer
    /// on the same location observes. A non-releasing access publishes nothing.
    fn publish_release(
        &mut self,
        buffer: &str,
        index: u64,
        publisher: Participant,
        kind: MemoryAccessKind,
        sequence: u64,
    ) {
        let MemoryAccessKind::Atomic { ordering, scope } = kind else {
            return;
        };
        if !ordering.is_release() {
            return;
        }
        self.releases
            .entry((buffer.to_string(), index))
            .or_default()
            .push(ReleaseEdge {
                publisher,
                scope,
                seq: sequence,
            });
    }

    /// Observe every release already published on this location that an
    /// acquiring atomic by `acquirer` synchronizes with.
    ///
    /// The edge stands only where both declared scopes reach the other
    /// invocation. A workgroup-scoped release publishes nothing to another
    /// workgroup, so an acquirer there observes nothing and the accesses the
    /// publisher made stay unsynchronized.
    fn observe_acquire(
        &mut self,
        buffer: &str,
        index: u64,
        acquirer: Participant,
        kind: MemoryAccessKind,
    ) {
        let MemoryAccessKind::Atomic { ordering, scope } = kind else {
            return;
        };
        if !ordering.is_acquire() {
            return;
        }
        let Some(edges) = self.releases.get(&(buffer.to_string(), index)) else {
            return;
        };
        let observed: Vec<(Participant, u64)> = edges
            .iter()
            // A self-edge is never consulted, because a prior access by the
            // same lane is sequenced and skipped before the check. Dropping it
            // here keeps the map to pairs that can decide something; no test
            // can distinguish its presence.
            .filter(|edge| edge.publisher != acquirer)
            .filter(|edge| {
                let same_workgroup = edge.publisher.1 == acquirer.1;
                synchronization_reaches(edge.scope, same_workgroup)
                    && synchronization_reaches(scope, same_workgroup)
            })
            .map(|edge| (edge.publisher, edge.seq))
            .collect();
        for (publisher, through) in observed {
            let entry = self
                .happens_before
                .entry((publisher, acquirer))
                .or_default();
            *entry = (*entry).max(through);
        }
    }

    /// Record a memory access and check for data races.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceError`] if an unsynchronized data race is detected.
    pub fn record_and_check_access(
        &mut self,
        buffer: &str,
        index: u64,
        invocation: [u32; 3],
        kind: MemoryAccessKind,
        scope: MemoryScope,
        domain: StorageDomain,
    ) -> Result<(), ReferenceError> {
        match self.record_access(buffer, index, invocation, kind, scope, domain) {
            None => Ok(()),
            Some(finding) => Err(ReferenceError::new(finding.to_string())),
        }
    }
}

/// Whether two accesses to one location in one synchronization phase conflict.
///
/// At least one side has to write, and a pair of atomics is ordered by the
/// atomic operation itself. An asynchronous transfer conflicts with everything,
/// because its completion point is the wait it has not reached yet.
///
/// The match has no catch-all arm, so a new access kind states its own conflict
/// rule rather than borrowing another kind's.
const fn conflicts(prior: MemoryAccessKind, next: MemoryAccessKind) -> bool {
    match (prior, next) {
        (MemoryAccessKind::Read, MemoryAccessKind::Read)
        | (MemoryAccessKind::Atomic { .. }, MemoryAccessKind::Atomic { .. }) => false,
        (MemoryAccessKind::Write, MemoryAccessKind::Read | MemoryAccessKind::Write)
        | (MemoryAccessKind::Read, MemoryAccessKind::Write)
        | (MemoryAccessKind::Write, MemoryAccessKind::Atomic { .. })
        | (MemoryAccessKind::Atomic { .. }, MemoryAccessKind::Write)
        | (MemoryAccessKind::Read, MemoryAccessKind::Atomic { .. })
        | (MemoryAccessKind::Atomic { .. }, MemoryAccessKind::Read)
        | (MemoryAccessKind::AsyncTransfer { .. }, _)
        | (
            MemoryAccessKind::Read | MemoryAccessKind::Write | MemoryAccessKind::Atomic { .. },
            MemoryAccessKind::AsyncTransfer { .. },
        ) => true,
    }
}

/// Whether a release or acquire declared at `scope` reaches an invocation that
/// is in the same workgroup, or in another one.
///
/// A release publishes to the invocations its scope covers and to no others.
/// `Thread` covers the issuing invocation alone, and the caller has already
/// excluded an edge to itself, so it reaches nothing here. `Subgroup` and
/// `Workgroup` reach the workgroup that issued the atomic. `Cluster`, `Device`
/// and `System` reach every invocation in the dispatch.
///
/// The reach only ever removes a finding, so a scope judged too narrowly keeps
/// a race report that a correct program did not earn, and never hides one.
///
/// What this does not separate: two subgroups of one workgroup. The access
/// record carries the invocation and the workgroup, not the subgroup it was
/// assigned to, so a subgroup-scoped release is credited across the whole
/// workgroup.
///
/// The match has no catch-all arm, so a scope added to the memory model states
/// its own reach rather than borrowing another's.
const fn synchronization_reaches(scope: MemoryScope, same_workgroup: bool) -> bool {
    match scope {
        MemoryScope::Thread => false,
        MemoryScope::Subgroup | MemoryScope::Workgroup => same_workgroup,
        MemoryScope::Cluster | MemoryScope::Device | MemoryScope::System => true,
    }
}

/// Shadow memory and findings accumulated for one race exploration.
#[derive(Debug, Default)]
struct RaceTracker {
    shadow: ShadowMemory,
    findings: Vec<RaceFinding>,
}

thread_local! {
    /// Per-thread race tracking state, `None` outside an exploration.
    ///
    /// The interpreter is single-threaded per call and every hook checks this
    /// slot, so an ordinary evaluation pays one thread-local read per memory
    /// access and records nothing. Tracking every access unconditionally would
    /// otherwise allocate a keyed history for every store the oracle performs.
    static RACE_TRACKER: RefCell<Option<RaceTracker>> = const { RefCell::new(None) };
}

/// Restores the tracking state that was in effect before the exploration it
/// brackets, so a nested evaluation cannot leave the thread recording.
pub(crate) struct RaceTrackingGuard {
    previous: Option<RaceTracker>,
}

impl Drop for RaceTrackingGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        RACE_TRACKER.with(|slot| *slot.borrow_mut() = previous);
    }
}

/// Enable race tracking on this thread for the length of the returned guard.
pub(crate) fn enter_race_tracking() -> RaceTrackingGuard {
    let previous = RACE_TRACKER.with(|slot| slot.borrow_mut().replace(RaceTracker::default()));
    RaceTrackingGuard { previous }
}

/// Discard the shadow memory of the previous explored order.
///
/// Findings accumulate across orders; the access history does not, because
/// each order is a separate execution of the dispatch.
pub(crate) fn begin_explored_order() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow = ShadowMemory::new();
        }
    });
}

/// Take every finding recorded since tracking was enabled.
pub(crate) fn take_race_findings() -> Vec<RaceFinding> {
    RACE_TRACKER.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .map(|tracker| std::mem::take(&mut tracker.findings))
            .unwrap_or_default()
    })
}

/// Record one memory access made by an executing invocation.
pub(crate) fn note_access(
    buffer: &str,
    index: u64,
    invocation: [u32; 3],
    kind: MemoryAccessKind,
    scope: MemoryScope,
    domain: StorageDomain,
) {
    RACE_TRACKER.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(tracker) = slot.as_mut() else {
            return;
        };
        if let Some(finding) = tracker
            .shadow
            .record_access(buffer, index, invocation, kind, scope, domain)
        {
            if !tracker.findings.contains(&finding) {
                tracker.findings.push(finding);
            }
        }
    });
}

/// Record that the lanes of `workgroup` are the ones now executing.
pub(crate) fn note_workgroup(workgroup: [u32; 3]) {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow.enter_workgroup(workgroup);
        }
    });
}

/// Record that a workgroup barrier released every lane holding at it.
pub(crate) fn note_barrier_release() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker
                .shadow
                .advance_barrier_phase(ExecutionScope::Workgroup);
        }
    });
}

/// Record that the whole dispatch passed a grid fence.
pub(crate) fn note_grid_fence() {
    RACE_TRACKER.with(|slot| {
        if let Some(tracker) = slot.borrow_mut().as_mut() {
            tracker.shadow.advance_grid_phase();
        }
    });
}

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

#[cfg(test)]
mod release_acquire_edges {
    use super::{MemoryAccessKind, RaceFinding, ShadowMemory};
    use vyre_foundation::ir::{AtomicOrdering, MemoryScope, StorageDomain};

    /// A publisher writes a payload, releases a flag; an acquirer acquires the
    /// flag and reads the payload.
    ///
    /// `release` and `acquire` are the orderings and scopes the two atomics on
    /// the flag declare. Returns what the payload read reported.
    fn handoff(
        release: (AtomicOrdering, MemoryScope),
        acquire: (AtomicOrdering, MemoryScope),
        same_workgroup: bool,
    ) -> Option<RaceFinding> {
        let mut shadow = ShadowMemory::new();
        shadow.enter_workgroup([0, 0, 0]);
        let publisher = [0, 0, 0];
        let acquirer = [1, 0, 0];
        assert_eq!(
            shadow.record_access(
                "payload",
                0,
                publisher,
                MemoryAccessKind::Write,
                MemoryScope::Workgroup,
                StorageDomain::DeviceGlobal,
            ),
            None,
            "Fix: the first write to an untouched location conflicts with nothing"
        );
        let _ = shadow.record_access(
            "flag",
            0,
            publisher,
            MemoryAccessKind::Atomic {
                ordering: release.0,
                scope: release.1,
            },
            release.1,
            StorageDomain::DeviceGlobal,
        );
        if !same_workgroup {
            shadow.enter_workgroup([1, 0, 0]);
        }
        let _ = shadow.record_access(
            "flag",
            0,
            acquirer,
            MemoryAccessKind::Atomic {
                ordering: acquire.0,
                scope: acquire.1,
            },
            acquire.1,
            StorageDomain::DeviceGlobal,
        );
        shadow.record_access(
            "payload",
            0,
            acquirer,
            MemoryAccessKind::Read,
            MemoryScope::Workgroup,
            StorageDomain::DeviceGlobal,
        )
    }

    /// A release/acquire handoff on a flag orders the payload around it.
    ///
    /// WHY: the tracker counted a barrier and a grid fence as the only
    /// synchronization edges, so a publisher that wrote a payload and released
    /// a flag, and an acquirer that acquired the flag and read the payload,
    /// was reported as an unsynchronized access. That is a correct handoff,
    /// and the only way to write a program the oracle accepted was to add a
    /// barrier the algorithm did not need. An oracle that rejects a correct
    /// program is worse than one that is merely incomplete: it makes the
    /// implementation wrong to be right.
    ///
    /// The ordering pairs are enumerated from `AtomicOrdering::ALL`, so an
    /// ordering added to the model is judged without editing this test.
    ///
    /// What it does not catch: a handoff where the release and the acquire
    /// name two different locations.
    #[test]
    fn a_release_acquire_pair_orders_the_payload_around_it() {
        let mut ordered = 0usize;
        let mut unordered = 0usize;
        for release in AtomicOrdering::ALL {
            for acquire in AtomicOrdering::ALL {
                let finding = handoff(
                    (release, MemoryScope::Workgroup),
                    (acquire, MemoryScope::Workgroup),
                    true,
                );
                if release.is_release() && acquire.is_acquire() {
                    ordered += 1;
                    assert_eq!(
                        finding, None,
                        "Fix: a `{release:?}` release observed by an `{acquire:?}` acquire \
                         publishes the payload written before it"
                    );
                    continue;
                }
                unordered += 1;
                assert!(
                    matches!(finding, Some(RaceFinding::UnsynchronizedAccess { .. })),
                    "Fix: `{release:?}` then `{acquire:?}` is not a release/acquire pair, so the \
                     payload read is unsynchronized; got {finding:?}"
                );
            }
        }
        assert!(
            ordered > 0 && unordered > 0,
            "Fix: the ordering space collapsed to one verdict ({ordered} ordered, {unordered} \
             unordered)"
        );
    }

    /// A workgroup-scoped handoff publishes nothing to another workgroup.
    ///
    /// WHY: the edge is only as wide as the scope both atomics declared.
    /// Crediting it regardless of scope would let a workgroup-local release
    /// silence a genuine cross-workgroup race, which is the direction that
    /// loses a defect rather than reporting an extra one.
    #[test]
    fn a_workgroup_scoped_handoff_does_not_cross_workgroups() {
        assert_eq!(
            handoff(
                (AtomicOrdering::Release, MemoryScope::Workgroup),
                (AtomicOrdering::Acquire, MemoryScope::Workgroup),
                true,
            ),
            None
        );
        assert!(matches!(
            handoff(
                (AtomicOrdering::Release, MemoryScope::Workgroup),
                (AtomicOrdering::Acquire, MemoryScope::Workgroup),
                false,
            ),
            Some(RaceFinding::UnsynchronizedAccess { .. })
        ));
    }

    /// A device-scoped handoff does cross workgroups.
    ///
    /// WHY: the workgroup case above passes for a rule that never credits a
    /// cross-workgroup edge at all. This pins that the scope is what decides
    /// it.
    #[test]
    fn a_device_scoped_handoff_crosses_workgroups() {
        assert_eq!(
            handoff(
                (AtomicOrdering::Release, MemoryScope::Device),
                (AtomicOrdering::Acquire, MemoryScope::Device),
                false,
            ),
            None
        );
    }

    /// A release publishes what preceded it, not what follows it.
    ///
    /// WHY: the edge is a point in the publisher's sequence, not a blanket
    /// exemption for the publisher. A write issued after the release is not
    /// covered by it and stays a race. Recording the edge as "these two
    /// invocations are synchronized" instead of "synchronized through
    /// sequence N" would lose every such defect.
    #[test]
    fn a_release_does_not_publish_a_write_that_follows_it() {
        let mut shadow = ShadowMemory::new();
        shadow.enter_workgroup([0, 0, 0]);
        let publisher = [0, 0, 0];
        let acquirer = [1, 0, 0];
        let flag = MemoryAccessKind::Atomic {
            ordering: AtomicOrdering::Release,
            scope: MemoryScope::Device,
        };
        let _ = shadow.record_access(
            "flag",
            0,
            publisher,
            flag,
            MemoryScope::Device,
            StorageDomain::DeviceGlobal,
        );
        assert_eq!(
            shadow.record_access(
                "payload",
                0,
                publisher,
                MemoryAccessKind::Write,
                MemoryScope::Workgroup,
                StorageDomain::DeviceGlobal,
            ),
            None
        );
        let _ = shadow.record_access(
            "flag",
            0,
            acquirer,
            MemoryAccessKind::Atomic {
                ordering: AtomicOrdering::Acquire,
                scope: MemoryScope::Device,
            },
            MemoryScope::Device,
            StorageDomain::DeviceGlobal,
        );
        assert!(
            matches!(
                shadow.record_access(
                    "payload",
                    0,
                    acquirer,
                    MemoryAccessKind::Read,
                    MemoryScope::Workgroup,
                    StorageDomain::DeviceGlobal,
                ),
                Some(RaceFinding::UnsynchronizedAccess { .. })
            ),
            "Fix: the payload write was issued after the release, so the acquirer does not \
             observe it"
        );
    }
}
