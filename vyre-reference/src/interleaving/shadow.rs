//! Shadow memory: the detector that decides whether two accesses conflict.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{ExecutionScope, MemoryScope, StorageDomain};

use crate::ReferenceError;

use super::model::{MemoryAccessKind, MemoryAccessRecord, RaceFinding};

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
