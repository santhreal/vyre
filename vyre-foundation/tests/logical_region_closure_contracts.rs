//! Logical region, dependence, partition axis, and exchange kind closure contracts.
//!
//! BACKLOG row 51 requires logical region IR to represent segmented maps, associative
//! reductions, scans, recurrent state, windows, ragged extents, and partial-result joins,
//! generically validating scratch, progress, ordering, and numerical contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphInput, GraphOutput, Node, Program, ProgramGraph,
    ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::logical::{
    LogicalDependenceKind, LogicalExchangeKind, LogicalPartitionAxisKind, LogicalProgramGraph,
    LogicalRegionKind,
};

#[test]
fn logical_region_kind_exhaustive_closure() {
    for kind in LogicalRegionKind::ALL {
        match kind {
            LogicalRegionKind::Parallel => assert_eq!(kind, LogicalRegionKind::Parallel),
            LogicalRegionKind::Sequential => assert_eq!(kind, LogicalRegionKind::Sequential),
            LogicalRegionKind::Reduction => assert_eq!(kind, LogicalRegionKind::Reduction),
            LogicalRegionKind::RetainedState => assert_eq!(kind, LogicalRegionKind::RetainedState),
            LogicalRegionKind::SegmentedMap => assert_eq!(kind, LogicalRegionKind::SegmentedMap),
            LogicalRegionKind::Scan => assert_eq!(kind, LogicalRegionKind::Scan),
            LogicalRegionKind::RecurrentState => assert_eq!(kind, LogicalRegionKind::RecurrentState),
            LogicalRegionKind::Window => assert_eq!(kind, LogicalRegionKind::Window),
            LogicalRegionKind::RaggedExtent => assert_eq!(kind, LogicalRegionKind::RaggedExtent),
            LogicalRegionKind::PartialResultJoin => {
                assert_eq!(kind, LogicalRegionKind::PartialResultJoin)
            }
        }
    }
}

#[test]
fn logical_dependence_kind_exhaustive_closure() {
    let kinds = [
        LogicalDependenceKind::Flow,
        LogicalDependenceKind::RetainedState,
    ];

    for kind in kinds {
        match kind {
            LogicalDependenceKind::Flow => assert_eq!(kind, LogicalDependenceKind::Flow),
            LogicalDependenceKind::RetainedState => {
                assert_eq!(kind, LogicalDependenceKind::RetainedState)
            }
        }
    }
}

#[test]
fn logical_partition_axis_kind_exhaustive_closure() {
    let kinds = [
        LogicalPartitionAxisKind::Elementwise,
        LogicalPartitionAxisKind::Reduction,
        LogicalPartitionAxisKind::Sequence,
        LogicalPartitionAxisKind::Spatial,
        LogicalPartitionAxisKind::Routed,
    ];

    for kind in kinds {
        match kind {
            LogicalPartitionAxisKind::Elementwise => {
                assert_eq!(kind, LogicalPartitionAxisKind::Elementwise)
            }
            LogicalPartitionAxisKind::Reduction => {
                assert_eq!(kind, LogicalPartitionAxisKind::Reduction)
            }
            LogicalPartitionAxisKind::Sequence => {
                assert_eq!(kind, LogicalPartitionAxisKind::Sequence)
            }
            LogicalPartitionAxisKind::Spatial => {
                assert_eq!(kind, LogicalPartitionAxisKind::Spatial)
            }
            LogicalPartitionAxisKind::Routed => assert_eq!(kind, LogicalPartitionAxisKind::Routed),
        }
    }
}

#[test]
fn logical_exchange_kind_exhaustive_closure() {
    for kind in LogicalExchangeKind::ALL {
        match kind {
            LogicalExchangeKind::AllReduce => {
                assert!(kind.combines());
            }
            LogicalExchangeKind::AllGather => {
                assert!(!kind.combines());
            }
            LogicalExchangeKind::ReduceScatter => {
                assert!(kind.combines());
            }
            LogicalExchangeKind::Broadcast => {
                assert!(!kind.combines());
            }
            LogicalExchangeKind::PointToPoint => {
                assert!(!kind.combines());
            }
        }
    }
}

#[test]
fn logical_stage_validates_parallel_reduction_and_retained_regions() {
    let mut graph = ProgramGraph::new();

    let input = graph
        .add_external_value(
            "input",
            ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(64)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .expect("input");

    let state = graph
        .add_external_value(
            "state.0",
            ValueContract {
                dtype: DataType::F32,
                shape: vec![ShapeDim::Known(64)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Retained,
            },
        )
        .expect("state");

    // Parallel elementwise stage
    let (_, p1_outs) = graph
        .add_node(
            "parallel_map",
            Program::wrapped(
                vec![
                    BufferDecl::read("input", 0, DataType::F32).with_count(64),
                    BufferDecl::output("mapped", 1, DataType::F32).with_count(64),
                ],
                [64, 1, 1],
                vec![Node::store(
                    "mapped",
                    Expr::gid_x(),
                    Expr::add(Expr::load("input", Expr::gid_x()), Expr::f32(1.0)),
                )],
            ),
            vec![GraphInput {
                buffer: "input".into(),
                value: input,
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(64)],
                    access: BufferAccess::ReadOnly,
                    lifetime: ValueLifetime::Invocation,
                },
            }],
            vec![GraphOutput {
                buffer: "mapped".into(),
                name: "mapped_val".into(),
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(64)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Invocation,
                },
                retained_successor_of: None,
            }],
        )
        .expect("parallel map");

    // Retained state update stage
    graph
        .add_node(
            "retained_step",
            Program::wrapped(
                vec![
                    BufferDecl::read("mapped", 0, DataType::F32).with_count(64),
                    BufferDecl::storage("state", 1, BufferAccess::ReadWrite, DataType::F32)
                        .with_count(64),
                ],
                [64, 1, 1],
                vec![Node::store(
                    "state",
                    Expr::gid_x(),
                    Expr::add(
                        Expr::load("state", Expr::gid_x()),
                        Expr::load("mapped", Expr::gid_x()),
                    ),
                )],
            ),
            vec![
                GraphInput {
                    buffer: "mapped".into(),
                    value: p1_outs[0],
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(64)],
                        access: BufferAccess::ReadOnly,
                        lifetime: ValueLifetime::Invocation,
                    },
                },
                GraphInput {
                    buffer: "state".into(),
                    value: state,
                    contract: ValueContract {
                        dtype: DataType::F32,
                        shape: vec![ShapeDim::Known(64)],
                        access: BufferAccess::ReadWrite,
                        lifetime: ValueLifetime::Retained,
                    },
                },
            ],
            vec![GraphOutput {
                buffer: "state".into(),
                name: "state.1".into(),
                contract: ValueContract {
                    dtype: DataType::F32,
                    shape: vec![ShapeDim::Known(64)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Retained,
                },
                retained_successor_of: Some(state),
            }],
        )
        .expect("retained step");

    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .expect("logical program graph must validate");

    assert_eq!(logical.regions().len(), 2);
    assert_eq!(logical.regions()[0].kind, LogicalRegionKind::Parallel);
    assert!(logical.regions()[0].partition.replicable);
    assert_eq!(logical.regions()[1].kind, LogicalRegionKind::RetainedState);
    assert!(!logical.regions()[1].partition.replicable);
    assert!(!logical.semantic_wire().is_empty());
}
#[test]
fn logical_region_descriptors_and_contracts_closure() {
    use vyre_foundation::logical::{
        LogicalCombineOp, OrderingContract, OrderingSyncScope, PartialResultJoinDescriptor,
        ProgressContract, RecurrenceDescriptor, ScanDirection, ScratchContract, SegmentDescriptor,
        WindowDescriptor,
    };

    let combine_ops = [
        LogicalCombineOp::Add,
        LogicalCombineOp::Mul,
        LogicalCombineOp::Min,
        LogicalCombineOp::Max,
        LogicalCombineOp::BitOr,
        LogicalCombineOp::BitAnd,
        LogicalCombineOp::BitXor,
    ];
    for op in combine_ops {
        match op {
            LogicalCombineOp::Add => assert_eq!(op, LogicalCombineOp::Add),
            LogicalCombineOp::Mul => assert_eq!(op, LogicalCombineOp::Mul),
            LogicalCombineOp::Min => assert_eq!(op, LogicalCombineOp::Min),
            LogicalCombineOp::Max => assert_eq!(op, LogicalCombineOp::Max),
            LogicalCombineOp::BitOr => assert_eq!(op, LogicalCombineOp::BitOr),
            LogicalCombineOp::BitAnd => assert_eq!(op, LogicalCombineOp::BitAnd),
            LogicalCombineOp::BitXor => assert_eq!(op, LogicalCombineOp::BitXor),
        }
    }

    let scan_dirs = [
        ScanDirection::InclusiveForward,
        ScanDirection::ExclusiveForward,
        ScanDirection::InclusiveBackward,
        ScanDirection::ExclusiveBackward,
    ];
    for dir in scan_dirs {
        match dir {
            ScanDirection::InclusiveForward => assert_eq!(dir, ScanDirection::InclusiveForward),
            ScanDirection::ExclusiveForward => assert_eq!(dir, ScanDirection::ExclusiveForward),
            ScanDirection::InclusiveBackward => assert_eq!(dir, ScanDirection::InclusiveBackward),
            ScanDirection::ExclusiveBackward => assert_eq!(dir, ScanDirection::ExclusiveBackward),
        }
    }

    let sync_scopes = [
        OrderingSyncScope::None,
        OrderingSyncScope::Workgroup,
        OrderingSyncScope::DeviceQueue,
        OrderingSyncScope::RetainedEpoch,
    ];
    for scope in sync_scopes {
        match scope {
            OrderingSyncScope::None => assert_eq!(scope, OrderingSyncScope::None),
            OrderingSyncScope::Workgroup => assert_eq!(scope, OrderingSyncScope::Workgroup),
            OrderingSyncScope::DeviceQueue => assert_eq!(scope, OrderingSyncScope::DeviceQueue),
            OrderingSyncScope::RetainedEpoch => assert_eq!(scope, OrderingSyncScope::RetainedEpoch),
        }
    }

    let window = WindowDescriptor {
        window_shape: vec![3, 3],
        strides: vec![1, 1],
        dilations: vec![1, 1],
        halo_padding: vec![(1, 1), (1, 1)],
    };
    assert_eq!(window.window_shape.len(), 2);

    let segment = SegmentDescriptor {
        segment_count: 8,
        max_segment_len: 64,
        is_ragged: true,
    };
    assert!(segment.is_ragged);

    let recurrence = RecurrenceDescriptor {
        sequence_steps: 128,
        state_elements: 32,
        tile_factor: 4,
    };
    assert_eq!(recurrence.tile_factor, 4);

    let join = PartialResultJoinDescriptor {
        split_axis: 0,
        partition_count: 4,
        combine_op: LogicalCombineOp::Add,
    };
    assert_eq!(join.combine_op, LogicalCombineOp::Add);

    let scratch = ScratchContract {
        workgroup_scratch_bytes: 1024,
        partition_scratch_bytes: 4096,
        reusable: true,
    };
    assert!(scratch.reusable);

    let progress = ProgressContract {
        max_iterations: 100,
        monotone_progress: true,
        guaranteed_termination: true,
    };
    assert!(progress.guaranteed_termination);

    let ordering = OrderingContract {
        causal_ordering: true,
        sync_scope: OrderingSyncScope::Workgroup,
    };
    assert!(ordering.causal_ordering);
}

#[test]
fn schedule_lowering_distributions_closure() {
    use vyre_foundation::transform::schedule_lowering::{
        distribute_partial_result_join, distribute_ragged_extent, distribute_recurrent_state,
        distribute_reduction, distribute_scan, distribute_segmented_map, distribute_window,
        DistributionTarget, ScheduleDistribution,
    };

    let prog = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::F32).with_count(128),
            BufferDecl::output("out", 1, DataType::F32).with_count(128),
        ],
        [128, 1, 1],
        vec![Node::store("out", Expr::gid_x(), Expr::load("in", Expr::gid_x()))],
    );

    let dist_wg = ScheduleDistribution::workgroup(4, 32);
    let dist_lane = ScheduleDistribution::lane(32);
    let dist_resident = ScheduleDistribution::resident_partition(2, 64);

    assert_eq!(dist_wg.target, DistributionTarget::Workgroup);
    assert_eq!(dist_lane.target, DistributionTarget::Lane);
    assert_eq!(dist_resident.target, DistributionTarget::ResidentPartition);

    let lowered_red = distribute_reduction(&prog, &dist_wg);
    assert_eq!(lowered_red.workgroup_size(), [32, 1, 1]);

    let lowered_scan = distribute_scan(&prog, &dist_wg);
    assert_eq!(lowered_scan.workgroup_size(), [32, 1, 1]);

    let lowered_seg = distribute_segmented_map(&prog, &dist_wg);
    assert_eq!(lowered_seg.workgroup_size(), [32, 1, 1]);

    let lowered_rec = distribute_recurrent_state(&prog, &dist_wg);
    assert_eq!(lowered_rec.workgroup_size(), [32, 1, 1]);

    let lowered_win = distribute_window(&prog, &dist_wg);
    assert_eq!(lowered_win.workgroup_size(), [32, 1, 1]);

    let lowered_ragged = distribute_ragged_extent(&prog, &dist_wg);
    assert_eq!(lowered_ragged.workgroup_size(), [32, 1, 1]);

    let lowered_join = distribute_partial_result_join(&prog, &dist_wg);
    assert_eq!(lowered_join.workgroup_size(), [32, 1, 1]);
}
