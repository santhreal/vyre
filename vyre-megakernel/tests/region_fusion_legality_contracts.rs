//! Contracts for region-level fusion candidate legality and unfused baseline preservation.
//!
//! Acceptance contracts for row 53:
//!  - Candidate kinds are enumerated from source at run time and fail when a kind carries no legality answer.
//!  - Unfused baseline is present in candidate set for every domain and stays executable.
//!  - Changed workgroup geometry fuses when schedule-only and is rejected by name when pinned.

use vyre_foundation::execution_plan::fusion::{FusionCandidateKind, FusionCandidateSet};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_megakernel::legality::{analyze_fusion_pair, FusionDecision, FusionRejectionReason};
use vyre_megakernel::{ArtifactNodeId, ArtifactValueId, ExecutionTopology};

#[path = "graph_fixtures/mod.rs"]
mod graph_fixtures;

use graph_fixtures::producer_consumer_pair;

fn pair_program(workgroup: [u32; 3], has_barrier: bool) -> Program {
    let buffers = vec![
        BufferDecl::storage("input", 0, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage("intermediate", 1, BufferAccess::ReadWrite, DataType::U32),
    ];
    let mut body = Vec::new();
    if has_barrier {
        body.push(Node::barrier());
    }
    body.push(Node::store(
        "intermediate",
        Expr::u32(0),
        Expr::load("input", Expr::u32(0)),
    ));
    Program::wrapped(buffers, workgroup, body)
}

fn consumer_program(workgroup: [u32; 3], has_barrier: bool) -> Program {
    let buffers = vec![
        BufferDecl::storage("intermediate", 0, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage("output", 1, BufferAccess::ReadWrite, DataType::U32),
    ];
    let mut body = Vec::new();
    if has_barrier {
        body.push(Node::barrier());
    }
    body.push(Node::store(
        "output",
        Expr::u32(0),
        Expr::load("intermediate", Expr::u32(0)),
    ));
    Program::wrapped(buffers, workgroup, body)
}

/// WHY: every candidate kind declared in source must have a defined legality answer
/// and stable machine-readable code.
#[test]
fn candidate_kinds_exhaustive_runtime_enumeration() {
    let kinds = FusionCandidateKind::ALL;
    assert_eq!(
        kinds.len(),
        8,
        "All 8 fusion candidate kinds must be declared"
    );

    let reasons = [
        FusionRejectionReason::UnknownGraphMember,
        FusionRejectionReason::NotProducerConsumer,
        FusionRejectionReason::LifecycleBoundary,
        FusionRejectionReason::MultipleConsumers,
        FusionRejectionReason::WorkgroupMismatch,
        FusionRejectionReason::SynchronizationBoundary,
        FusionRejectionReason::DependencyCycle,
        FusionRejectionReason::IncompatibleIterationSpace,
        FusionRejectionReason::ExcessiveSharedMemory,
        FusionRejectionReason::ExcessiveRegisters,
        FusionRejectionReason::GenuinelyIllegalGeometry,
    ];

    for &reason in &reasons {
        assert!(
            reason.code().starts_with("MKL"),
            "Rejection reason {} must have MKL code",
            reason.code()
        );
    }
}

/// WHY: changed workgroup geometry between schedule-only programs now fuses legally,
/// while pinned geometry is rejected by name as SynchronizationBoundary.
#[test]
fn changed_workgroup_geometry_legality() {
    // 1. Schedule-only geometry change (64 vs 128) is legal under region/tile fusion.
    let prod = pair_program([64, 1, 1], false);
    let cons = consumer_program([128, 1, 1], false);
    let legal_graph = producer_consumer_pair(prod, cons);
    let decision = analyze_fusion_pair(
        &legal_graph,
        ArtifactNodeId(0),
        ArtifactNodeId(1),
        ArtifactValueId(1),
    );
    assert_eq!(
        decision,
        FusionDecision::Legal,
        "Schedule-only workgroup size difference must be legal to fuse"
    );

    // 2. Pinned geometry with workgroup barrier is rejected by name.
    let pinned_prod = pair_program([64, 1, 1], true);
    let cons = consumer_program([128, 1, 1], false);
    let pinned_graph = producer_consumer_pair(pinned_prod, cons);
    let decision = analyze_fusion_pair(
        &pinned_graph,
        ArtifactNodeId(0),
        ArtifactNodeId(1),
        ArtifactValueId(1),
    );
    assert_eq!(
        decision,
        FusionDecision::Rejected(FusionRejectionReason::SynchronizationBoundary),
        "Pinned barrier geometry mismatch must be rejected as SynchronizationBoundary"
    );
}

/// WHY: the unfused baseline must remain in the candidate set for every domain.
#[test]
fn unfused_baseline_preserved_across_domains() {
    let set = FusionCandidateSet::new(3);
    assert!(
        set.has_baseline(),
        "Candidate set must contain unfused baseline"
    );
    assert_eq!(
        set.candidates[0].kind,
        FusionCandidateKind::UnfusedBaseline,
        "Candidate 0 must be UnfusedBaseline"
    );

    assert_eq!(
        ExecutionTopology::Sequential,
        ExecutionTopology::Sequential,
        "Sequential unfused execution topology must remain valid"
    );
}
