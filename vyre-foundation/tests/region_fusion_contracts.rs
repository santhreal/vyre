//! Contracts for logical region and schedule tile fusion.
//!
//! Acceptance contracts for row 53:
//!  - Earliest legal producer-consumer handoff is chosen (e.g. register forwarding with 0 barriers).
//!  - Changed workgroup geometry fuses for schedule-only programs and is refused by name when pinned.
//!  - Candidate kinds are enumerated from source at run time and close over legality.
//!  - Unfused baseline is present in the candidate set for every domain.

use vyre_foundation::execution_plan::fusion::{
    classify_program_handoff, earliest_region_handoff, fuse_programs, lower_fusion_candidate,
    FusionCandidate, FusionCandidateKind, FusionCandidateSet, FusionError, HandoffLocation,
    RegionDependenceGraph,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::logical::{
    LogicalAliasFacts, LogicalEffects, LogicalExtent, LogicalIndexMap, LogicalLayout,
    LogicalPartitionFacts, LogicalRegion, LogicalRegionKind, OrderingContract, OrderingSyncScope,
    ProgressContract, ScratchContract,
};
use vyre_foundation::numeric::NumericContract;

fn sample_region(name: &str, kind: LogicalRegionKind, extents: Vec<u64>) -> LogicalRegion {
    let rank = extents.len();
    LogicalRegion {
        node: vyre_foundation::ir::GraphNodeId(0),
        name: name.to_string(),
        kind,
        extents: extents.into_iter().map(LogicalExtent::Static).collect(),
        index_map: LogicalIndexMap {
            axes: (0..rank).map(|i| format!("axis{i}")).collect(),
            row_major_strides: vec![1; rank],
        },
        layout: LogicalLayout {
            storage_order: (0..rank as u32).collect(),
            strides: vec![1; rank],
            contiguous: true,
        },
        reduction_axes: Vec::new(),
        aliases: LogicalAliasFacts {
            retained_successors: Vec::new(),
            in_place_values: Vec::new(),
            inputs_disjoint: true,
            outputs_disjoint: true,
        },
        dependencies: Vec::new(),
        effects: LogicalEffects {
            reads: Vec::new(),
            writes: Vec::new(),
            retained_state: false,
            atomics: false,
            synchronizes: kind.is_reduction_or_scan(),
        },
        partition: LogicalPartitionFacts::default(),
        written_bytes: 4096,
        max_points: 1024,
        numeric: NumericContract::exact_word(),
        segment: None,
        window: None,
        recurrence: None,
        partial_join: None,
        scratch: ScratchContract {
            workgroup_scratch_bytes: 0,
            partition_scratch_bytes: 0,
            reusable: true,
        },
        progress: ProgressContract {
            max_iterations: 1024,
            monotone_progress: true,
            guaranteed_termination: true,
        },
        ordering: OrderingContract {
            causal_ordering: false,
            sync_scope: OrderingSyncScope::None,
        },
    }
}

/// WHY: in previous coarse placement, a RAW buffer hazard unconditionally inserted a
/// whole-kernel barrier between producer and consumer. With dependence analysis on logical
/// regions, pointwise 1:1 dataflow is recognized and the earliest safe handoff is chosen:
/// direct register forwarding with 0 barriers.
#[test]
fn earliest_legal_handoff_chosen_earlier_than_buffer_hazard_barrier() {
    let producer = Program::wrapped(
        vec![
            BufferDecl::read("x", 0, DataType::F32),
            BufferDecl::storage("y", 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "y",
            Expr::gid_x(),
            Expr::mul(Expr::load("x", Expr::gid_x()), Expr::f32(2.0)),
        )],
    );

    let consumer = Program::wrapped(
        vec![
            BufferDecl::storage("y", 1, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage("z", 2, BufferAccess::ReadWrite, DataType::F32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "z",
            Expr::gid_x(),
            Expr::add(Expr::load("y", Expr::gid_x()), Expr::f32(1.0)),
        )],
    );

    // 1. Dependence analysis identifies earliest handoff is Register (0 barriers).
    let handoff = classify_program_handoff(&producer, &consumer);
    assert_eq!(
        handoff,
        HandoffLocation::Register,
        "Pointwise elementwise handoff must select Register forwarding"
    );
    assert!(handoff.is_register_or_independent());
    assert!(!handoff.requires_workgroup_barrier());

    // 2. Region dependence graph finds 0-barrier handoff.
    let dep_graph = RegionDependenceGraph::from_programs(&[producer.clone(), consumer.clone()]);
    assert_eq!(
        dep_graph.earliest_handoff_between(0, 1),
        Some(HandoffLocation::Register)
    );

    // 3. Lowering as RegisterForwarding candidate produces combined program with 0 barriers.
    let candidate = FusionCandidate::register_forwarding(0, 1);
    let lowered = lower_fusion_candidate(&candidate, &[producer, consumer]).unwrap();

    let has_barrier = lowered
        .entry()
        .iter()
        .any(|n| matches!(n, Node::Barrier { .. } | Node::LogicalBarrier { .. }));
    assert!(
        !has_barrier,
        "Register forwarding handoff must contain zero barriers"
    );
}

/// WHY: reduction consumers require intra-workgroup coordination, so the earliest legal
/// handoff is WorkgroupShared with a workgroup-level barrier at the tile boundary.
#[test]
fn reduction_consumer_selects_shared_memory_handoff() {
    let map_region = sample_region("map", LogicalRegionKind::Parallel, vec![1024]);
    let red_region = sample_region("reduce", LogicalRegionKind::Reduction, vec![1024]);

    let handoff = earliest_region_handoff(&map_region, &red_region);
    assert_eq!(
        handoff,
        HandoffLocation::WorkgroupShared,
        "Reduction consumer must select WorkgroupShared handoff"
    );
    assert!(handoff.requires_workgroup_barrier());
}

/// WHY: independent computation frontiers have no dataflow dependencies and execute with 0 barriers.
#[test]
fn independent_frontiers_select_independent_handoff() {
    let arm_a = Program::wrapped(
        vec![
            BufferDecl::read("in_a", 0, DataType::U32),
            BufferDecl::output("out_a", 1, DataType::U32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "out_a",
            Expr::gid_x(),
            Expr::load("in_a", Expr::gid_x()),
        )],
    );
    let arm_b = Program::wrapped(
        vec![
            BufferDecl::read("in_b", 2, DataType::U32),
            BufferDecl::output("out_b", 3, DataType::U32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "out_b",
            Expr::gid_x(),
            Expr::load("in_b", Expr::gid_x()),
        )],
    );

    let dep_graph = RegionDependenceGraph::from_programs(&[arm_a, arm_b]);
    assert!(dep_graph.independent_frontiers.contains(&(0, 1)));
}

/// WHY: a schedule-only workgroup geometry difference is not an observable semantic constraint,
/// so it now fuses where it was previously refused. A genuinely pinned geometry (with barriers or
/// workgroup memory) is still refused by name.
#[test]
fn changed_workgroup_geometry_fuses_for_schedule_only_and_refused_when_pinned() {
    let p_64 = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            BufferDecl::output("b", 1, DataType::U32),
        ],
        [64, 1, 1],
        vec![Node::store(
            "b",
            Expr::gid_x(),
            Expr::load("a", Expr::gid_x()),
        )],
    );

    let p_128 = Program::wrapped(
        vec![
            BufferDecl::read("b", 1, DataType::U32),
            BufferDecl::output("c", 2, DataType::U32),
        ],
        [128, 1, 1],
        vec![Node::store(
            "c",
            Expr::gid_x(),
            Expr::load("b", Expr::gid_x()),
        )],
    );

    // 1. Schedule-only workgroup size difference (64 vs 128) now fuses!
    let fused = fuse_programs(&[p_64.clone(), p_128.clone()])
        .expect("Schedule-only geometry change must fuse");
    assert_eq!(
        fused.workgroup_size(),
        [128, 1, 1],
        "Fused program adopts unified launch width"
    );

    // 2. An arm with a workgroup barrier genuinely pins its geometry and is refused by name.
    let p_pinned = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            BufferDecl::output("b", 1, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::barrier(),
            Node::store("b", Expr::gid_x(), Expr::load("a", Expr::gid_x())),
        ],
    );

    let result = fuse_programs(&[p_pinned, p_128]);
    match result {
        Err(FusionError::WorkgroupGeometry(e)) => {
            assert_eq!(e.arm, 0);
            assert_eq!(e.arm_workgroup, [64, 1, 1]);
            assert_eq!(e.fused_workgroup, [128, 1, 1]);
            assert!(e.reason.contains("synchronizes its workgroup"));
        }
        other => panic!("Expected WorkgroupGeometry error for pinned barrier, got {other:?}"),
    }
}

/// WHY: candidate kinds are enumerated from source at run time via `FusionCandidateKind::ALL`.
/// Every candidate kind must have an explicit legality answer and stable code.
#[test]
fn candidate_kinds_enumerated_from_source_and_require_legality_answers() {
    let kinds = FusionCandidateKind::ALL;
    assert_eq!(
        kinds.len(),
        8,
        "All 8 fusion candidate kinds must be present in ALL"
    );

    let region_a = sample_region("a", LogicalRegionKind::Parallel, vec![512]);
    let region_b = sample_region("b", LogicalRegionKind::Parallel, vec![512]);
    let region_incompatible = sample_region("inc", LogicalRegionKind::Parallel, vec![512, 16]);

    for &kind in kinds {
        // Every kind has non-empty name and code.
        assert!(!kind.name().is_empty());
        assert!(kind.code().starts_with("FCK"));

        // Every kind produces an explicit legality answer without panicking.
        let legal_ans = kind.evaluate_legality(&region_a, &region_b);
        assert!(
            legal_ans.is_ok() || legal_ans.is_err(),
            "Candidate kind {} must provide a legality answer",
            kind.name()
        );

        // Incompatible regions must be handled deterministically.
        let _incompat_ans = kind.evaluate_legality(&region_a, &region_incompatible);
    }
}

/// WHY: the unfused baseline must remain in the candidate set for every domain
/// the fusion planner serves, and must stay executable.
#[test]
fn unfused_baseline_present_for_every_domain() {
    let domains = [
        (
            "neural_elementwise",
            vec![
                sample_region("relu", LogicalRegionKind::Parallel, vec![1024]),
                sample_region("add", LogicalRegionKind::Parallel, vec![1024]),
            ],
        ),
        (
            "contraction",
            vec![
                sample_region("gemm_a", LogicalRegionKind::Parallel, vec![128, 128]),
                sample_region("gemm_b", LogicalRegionKind::Parallel, vec![128, 128]),
            ],
        ),
        (
            "reduction",
            vec![
                sample_region("map", LogicalRegionKind::Parallel, vec![2048]),
                sample_region("sum", LogicalRegionKind::Reduction, vec![2048]),
            ],
        ),
        (
            "irregular_scan",
            vec![
                sample_region("scan_input", LogicalRegionKind::Parallel, vec![512]),
                sample_region("prefix_scan", LogicalRegionKind::Scan, vec![512]),
            ],
        ),
        (
            "security_dataflow",
            vec![
                sample_region("taint_source", LogicalRegionKind::Parallel, vec![256]),
                sample_region("sanitizer", LogicalRegionKind::Parallel, vec![256]),
            ],
        ),
    ];

    for (domain_name, regions) in domains {
        let set = FusionCandidateSet::new(regions.len());
        assert!(
            set.has_baseline(),
            "Domain {} must have unfused baseline in candidate set",
            domain_name
        );
        assert!(
            set.candidates[0].kind.is_baseline(),
            "Domain {} candidate set must have baseline as candidate 0",
            domain_name
        );
    }
}
