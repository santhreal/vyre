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

mod model;
mod shadow;
mod tracking;
mod walk;

pub use model::{
    InterleavingConfig, InterleavingReport, MemoryAccessKind, MemoryAccessRecord,
    RaceExplorationReport, RaceFinding,
};
pub use shadow::ShadowMemory;
pub use walk::{explore_bounded_interleavings, verify_closed_type_coverage_in_oracle};

pub(crate) use tracking::{
    begin_explored_order, enter_race_tracking, note_access, note_barrier_release, note_grid_fence,
    note_workgroup, take_race_findings,
};

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
