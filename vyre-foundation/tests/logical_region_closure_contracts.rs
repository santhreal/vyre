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
    let kinds = [
        LogicalRegionKind::Parallel,
        LogicalRegionKind::Sequential,
        LogicalRegionKind::Reduction,
        LogicalRegionKind::RetainedState,
    ];

    for kind in kinds {
        match kind {
            LogicalRegionKind::Parallel => assert_eq!(kind, LogicalRegionKind::Parallel),
            LogicalRegionKind::Sequential => assert_eq!(kind, LogicalRegionKind::Sequential),
            LogicalRegionKind::Reduction => assert_eq!(kind, LogicalRegionKind::Reduction),
            LogicalRegionKind::RetainedState => assert_eq!(kind, LogicalRegionKind::RetainedState),
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
