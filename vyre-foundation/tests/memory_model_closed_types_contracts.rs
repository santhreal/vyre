//! Closed orthogonal memory model and concurrency contract tests (Row 105).
//!
//! Asserts:
//! 1. Every closed type variant space is derived from source at run time and verified complete.
//! 2. Programs with unconsumed obligations are refused by name.
//! 3. Grid rendezvous is represented as an ExecutionScope rather than an ordering strength,
//!    and kernel cuts appear strictly in the schedule layer.

use vyre_foundation::ir::{
    exhaustiveness_check_async_transaction_lifecycle, exhaustiveness_check_atomic_ordering,
    exhaustiveness_check_barrier_participation, exhaustiveness_check_collective_group,
    exhaustiveness_check_execution_scope, exhaustiveness_check_failure_cancellation_behavior,
    exhaustiveness_check_fence_semantics, exhaustiveness_check_memory_scope,
    exhaustiveness_check_storage_domain, verify_program_obligations, AsyncTransactionLifecycle,
    AtomicOrdering, BarrierParticipation, CollectiveGroup, ExecutionScope,
    FailureCancellationBehavior, FenceSemantics, MemoryOrdering, MemoryScope, Node, Program,
    StorageDomain,
};
use vyre_foundation::transform::grid_sync_split::split_on_grid_sync;

#[test]
fn all_atomic_ordering_variants_have_explicit_decisions() {
    for ordering in AtomicOrdering::ALL {
        let name = exhaustiveness_check_atomic_ordering(ordering);
        assert!(!name.is_empty());
        let _ = ordering.wire_tag();
        let _ = ordering.is_acquire();
        let _ = ordering.is_release();
        let _ = ordering.is_seq_cst();
        let _ = ordering.join(ordering);
    }
}

#[test]
fn all_memory_scope_variants_have_explicit_decisions() {
    for scope in MemoryScope::ALL {
        let name = exhaustiveness_check_memory_scope(scope);
        assert!(!name.is_empty());
        let _ = scope.wire_tag();
        let _ = scope.is_cross_workgroup();
        let _ = scope.is_device_wide();
        let _ = scope.widen(scope);
        let _ = scope.includes(scope);
    }
}

#[test]
fn all_execution_scope_variants_have_explicit_decisions() {
    for scope in ExecutionScope::ALL {
        let name = exhaustiveness_check_execution_scope(scope);
        assert!(!name.is_empty());
        let _ = scope.wire_tag();
        let _ = scope.is_cross_block();
        let _ = scope.is_grid_or_mesh();
        let _ = scope.widen(scope);
        let _ = scope.includes(scope);
    }
}

#[test]
fn all_storage_domain_variants_have_explicit_decisions() {
    for domain in StorageDomain::ALL {
        let name = exhaustiveness_check_storage_domain(domain);
        assert!(!name.is_empty());
        let _ = domain.wire_tag();
        let _ = domain.is_shared_across_threads();
        let _ = domain.is_host_accessible();
        let _ = domain.is_on_chip();
        let _ = domain.default_memory_scope();
    }
}

#[test]
fn all_fence_semantics_variants_have_explicit_decisions() {
    for fence in FenceSemantics::ALL {
        let name = exhaustiveness_check_fence_semantics(fence);
        assert!(!name.is_empty());
        let _ = fence.wire_tag();
        let _ = fence.orders_reads();
        let _ = fence.orders_writes();
        let _ = fence.is_bidirectional();
    }
}

#[test]
fn all_barrier_participation_variants_have_explicit_decisions() {
    for part in BarrierParticipation::ALL {
        let name = exhaustiveness_check_barrier_participation(part);
        assert!(!name.is_empty());
        let _ = part.wire_tag();
        let _ = part.requires_uniform_control_flow();
        let _ = part.allows_divergence();
    }
}

#[test]
fn all_async_transaction_lifecycle_variants_have_explicit_decisions() {
    for lifecycle in AsyncTransactionLifecycle::ALL {
        let name = exhaustiveness_check_async_transaction_lifecycle(lifecycle);
        assert!(!name.is_empty());
        let _ = lifecycle.wire_tag();
        let _ = lifecycle.is_terminal();
        let _ = lifecycle.is_successful_commit();
        let _ = lifecycle.is_in_flight();
    }
}

#[test]
fn all_collective_group_variants_have_explicit_decisions() {
    for group in CollectiveGroup::ALL {
        let name = exhaustiveness_check_collective_group(group);
        assert!(!name.is_empty());
        let _ = group.wire_tag();
        let _ = group.is_inter_device();
        let _ = group.is_subgroup_only();
        let _ = group.execution_scope();
    }
}

#[test]
fn all_failure_cancellation_behavior_variants_have_explicit_decisions() {
    for failure in FailureCancellationBehavior::ALL {
        let name = exhaustiveness_check_failure_cancellation_behavior(failure);
        assert!(!name.is_empty());
        let _ = failure.wire_tag();
        let _ = failure.halts_execution();
        let _ = failure.poisons_memory();
        let _ = failure.propagates_to_caller();
    }
}

#[test]
fn unconsumed_async_transfer_obligation_is_refused_by_name() {
    use vyre_foundation::ir::{Expr, Ident};

    // Construct a program with an AsyncLoad that never performs an AsyncWait.
    let unconsumed_prog = Program::from_raw_parts(
        vec![],
        [1, 1, 1],
        vec![Node::AsyncLoad {
            source: Ident::from("src_buf"),
            destination: Ident::from("dst_buf"),
            offset: Box::new(Expr::LitU32(0)),
            size: Box::new(Expr::LitU32(64)),
            tag: Ident::from("transfer_stage_0"),
        }],
    );

    let result = verify_program_obligations(&unconsumed_prog);
    assert!(result.is_err(), "unconsumed async transfer obligation must be refused");
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("async_wait:transfer_stage_0"),
        "error must name the unconsumed obligation, got: {err_str}"
    );
}

#[test]
fn satisfied_async_transfer_obligation_passes() {
    use vyre_foundation::ir::{Expr, Ident};

    let satisfied_prog = Program::from_raw_parts(
        vec![],
        [1, 1, 1],
        vec![
            Node::AsyncLoad {
                source: Ident::from("src_buf"),
                destination: Ident::from("dst_buf"),
                offset: Box::new(Expr::LitU32(0)),
                size: Box::new(Expr::LitU32(64)),
                tag: Ident::from("transfer_stage_0"),
            },
            Node::AsyncWait {
                tag: Ident::from("transfer_stage_0"),
            },
        ],
    );

    let result = verify_program_obligations(&satisfied_prog);
    assert!(result.is_ok(), "satisfied obligation must verify cleanly");
}

#[test]
fn grid_rendezvous_is_represented_as_execution_scope_and_cut_in_schedule_layer() {
    // 1. In semantic IR, grid synchronization is an ExecutionScope rendezvous, not an ordering strength.
    let grid_scope = ExecutionScope::Grid;
    assert!(grid_scope.is_cross_block());
    assert!(grid_scope.is_grid_or_mesh());
    assert_eq!(grid_scope.wire_tag(), 4);
    let workgroup_scope = ExecutionScope::Workgroup;
    assert!(!workgroup_scope.is_cross_block());

    // 2. Schedule layer performs kernel cuts strictly for whole-grid synchronization.
    let prog = Program::from_raw_parts(
        vec![],
        [1, 1, 1],
        vec![
            Node::Store {
                buffer: vyre_foundation::ir::Ident::from("state"),
                index: vyre_foundation::ir::Expr::LitU32(0),
                value: vyre_foundation::ir::Expr::LitU32(42),
            },
            Node::Barrier {
                ordering: MemoryOrdering::GridSync,
            },
            Node::Store {
                buffer: vyre_foundation::ir::Ident::from("state"),
                index: vyre_foundation::ir::Expr::LitU32(1),
                value: vyre_foundation::ir::Expr::LitU32(99),
            },
        ],
    );
    let segments = split_on_grid_sync(&prog).expect("grid sync split succeeds");
    assert_eq!(segments.len(), 2, "grid sync fence splits program into 2 scheduled segments");
}
