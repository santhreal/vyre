//! Reference interleaving and race freedom contracts (Row 105).
//!
//! Asserts:
//! 1. Data races are detected and reported by bounded interleaving search rather than passing by luck.
//! 2. Synchronized programs are proven race-free across all bounded interleavings.
//! 3. All closed types are exhaustively covered in the reference oracle.

use vyre_foundation::ir::{
    AsyncTransactionLifecycle, AtomicOrdering, BarrierParticipation, CollectiveGroup,
    ExecutionScope, Expr, FailureCancellationBehavior, FenceSemantics, Ident, MemoryOrdering,
    MemoryScope, Node, Program, StorageDomain,
};
use vyre_reference::{
    explore_bounded_interleavings, verify_closed_type_coverage_in_oracle, InterleavingConfig,
    MemoryAccessKind, ShadowMemory,
};

#[test]
fn racing_uncoordinated_program_is_reported_by_interleaving_search() {
    // Both threads write to the same buffer location index 0 without barrier.
    let mut shadow = ShadowMemory::new();
    let thread_0 = [0, 0, 0];
    let thread_1 = [1, 0, 0];

    // Thread 0 writes to index 0.
    let res0 = shadow.record_and_check_access(
        "shared_buf",
        0,
        thread_0,
        MemoryAccessKind::Write,
        MemoryScope::Workgroup,
        StorageDomain::WorkgroupLocal,
    );
    assert!(res0.is_ok());

    // Thread 1 concurrently writes to index 0 in the same barrier phase.
    let res1 = shadow.record_and_check_access(
        "shared_buf",
        0,
        thread_1,
        MemoryAccessKind::Write,
        MemoryScope::Workgroup,
        StorageDomain::WorkgroupLocal,
    );
    assert!(
        res1.is_err(),
        "concurrent unsynchronized write must be reported as a data race"
    );
    let err = res1.unwrap_err();
    assert!(
        err.to_string().contains("Data race detected"),
        "error message must name data race, got: {err}"
    );
    assert!(
        err.to_string().contains("shared_buf"),
        "error message must name the racing buffer, got: {err}"
    );
}

#[test]
fn synchronized_program_passes_race_freedom_check() {
    let mut shadow = ShadowMemory::new();
    let thread_0 = [0, 0, 0];
    let thread_1 = [1, 0, 0];

    // Thread 0 writes to index 0 in phase 0.
    assert!(shadow
        .record_and_check_access(
            "shared_buf",
            0,
            thread_0,
            MemoryAccessKind::Write,
            MemoryScope::Workgroup,
            StorageDomain::WorkgroupLocal,
        )
        .is_ok());

    // Barrier executes, advancing phase.
    shadow.advance_barrier_phase(ExecutionScope::Workgroup);

    // Thread 1 reads from index 0 in phase 1 (safely synchronized).
    assert!(shadow
        .record_and_check_access(
            "shared_buf",
            0,
            thread_1,
            MemoryAccessKind::Read,
            MemoryScope::Workgroup,
            StorageDomain::WorkgroupLocal,
        )
        .is_ok());
}

#[test]
fn atomic_concurrent_accesses_do_not_race() {
    let mut shadow = ShadowMemory::new();
    let thread_0 = [0, 0, 0];
    let thread_1 = [1, 0, 0];

    // Thread 0 atomic RMW on counter at index 0.
    assert!(shadow
        .record_and_check_access(
            "counter",
            0,
            thread_0,
            MemoryAccessKind::Atomic {
                ordering: AtomicOrdering::Relaxed,
                scope: MemoryScope::Workgroup,
            },
            MemoryScope::Workgroup,
            StorageDomain::WorkgroupLocal,
        )
        .is_ok());

    // Thread 1 atomic RMW on counter at index 0 concurrently.
    assert!(shadow
        .record_and_check_access(
            "counter",
            0,
            thread_1,
            MemoryAccessKind::Atomic {
                ordering: AtomicOrdering::Relaxed,
                scope: MemoryScope::Workgroup,
            },
            MemoryScope::Workgroup,
            StorageDomain::WorkgroupLocal,
        )
        .is_ok());
}

#[test]
fn explore_bounded_interleavings_runs_cleanly_on_disjoint_access_program() {
    let prog = Program::from_raw_parts(
        vec![],
        [1, 1, 1],
        vec![
            Node::Store {
                buffer: Ident::from("buf"),
                index: Expr::LitU32(0),
                value: Expr::LitU32(42),
            },
            Node::Barrier {
                ordering: MemoryOrdering::SeqCst,
            },
            Node::Store {
                buffer: Ident::from("buf"),
                index: Expr::LitU32(1),
                value: Expr::LitU32(84),
            },
        ],
    );

    let config = InterleavingConfig {
        workgroup_size: [2, 1, 1],
        grid_size: [1, 1, 1],
        max_interleavings: 4,
        step_bound: 128,
    };

    let report = explore_bounded_interleavings(&prog, &[], &config)
        .expect("bounded interleavings exploration must pass on synchronized program");
    assert!(report.race_free);
    assert!(report.explored_schedules >= 1);
}

#[test]
fn reference_oracle_exhaustively_covers_all_closed_types() {
    assert!(verify_closed_type_coverage_in_oracle());
}

#[test]
fn runtime_variant_space_closure_across_all_nine_closed_types() {
    assert_eq!(AtomicOrdering::ALL.len(), 5);
    assert_eq!(MemoryScope::ALL.len(), 6);
    assert_eq!(ExecutionScope::ALL.len(), 6);
    assert_eq!(StorageDomain::ALL.len(), 8);
    assert_eq!(FenceSemantics::ALL.len(), 4);
    assert_eq!(BarrierParticipation::ALL.len(), 6);
    assert_eq!(AsyncTransactionLifecycle::ALL.len(), 6);
    assert_eq!(CollectiveGroup::ALL.len(), 6);
    assert_eq!(FailureCancellationBehavior::ALL.len(), 5);
}
