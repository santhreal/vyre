//! Graphs, facts, and device fact sets the candidate-search contracts share.
//!
//! The grammar contracts and the evaluation-ladder contracts compile the same
//! graphs under the same facts and differ only in what they assert, so the
//! inputs are defined once. A second copy would have to be kept identical for
//! assertions that compare derivations across the two suites to keep meaning
//! what they say.
//!
//! Included with `#[path]` the same way as `tests/support/artifact_fixtures.rs`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, DataType, GraphInput, GraphOutput, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget, ValidatedCompileRequest,
};
use vyre_test_support::pass_programs::{add_program, copy_program};

/// A two-dimensional `u32` value, so a phase carries more than one axis.
pub(crate) fn matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::U32,
        shape: vec![
            ShapeDim::Symbol("rows".into()),
            ShapeDim::Symbol("cols".into()),
        ],
        access: BufferAccess::ReadWrite,
        lifetime,
    }
}

/// The same value read by a consumer that only loads it.
pub(crate) fn read_matrix(lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        access: BufferAccess::ReadOnly,
        ..matrix(lifetime)
    }
}

/// One stage, so no production has anything to fuse, overlap, or cut.
pub(crate) fn single_stage_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let input = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    graph
        .add_node(
            "n0",
            copy_program("in_a", "out"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: input,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("only stage");
    graph
}

/// A chain, an independent arm, and a fan-in.
///
/// `n0 -> n1` is a single-consumer chain the grammar may fuse, `n2` is an
/// independent arm, and `n3` reads both `n1` and `n2`, so one phase carries two
/// predecessors and the joining productions have an operand.
pub(crate) fn joined_graph() -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let in_a = graph
        .add_external_value("in_a", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let in_b = graph
        .add_external_value("in_b", matrix(ValueLifetime::Invocation))
        .expect("external input");
    let (_, mid_a) = graph
        .add_node(
            "n0",
            copy_program("in_a", "mid_a"),
            vec![GraphInput {
                buffer: "in_a".into(),
                value: in_a,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_a".into(),
                name: "mid_a".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("first stage");
    let (_, mid_b) = graph
        .add_node(
            "n1",
            copy_program("mid_a", "mid_b"),
            vec![GraphInput {
                buffer: "mid_a".into(),
                value: mid_a[0],
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_b".into(),
                name: "mid_b".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("second stage");
    let (_, mid_c) = graph
        .add_node(
            "n2",
            copy_program("in_b", "mid_c"),
            vec![GraphInput {
                buffer: "in_b".into(),
                value: in_b,
                contract: matrix(ValueLifetime::Invocation),
            }],
            vec![GraphOutput {
                buffer: "mid_c".into(),
                name: "mid_c".into(),
                contract: matrix(ValueLifetime::Invocation),
                retained_successor_of: None,
            }],
        )
        .expect("independent arm");
    graph
        .add_node(
            "n3",
            add_program("mid_b", "mid_c", "out"),
            vec![
                GraphInput {
                    buffer: "mid_b".into(),
                    value: mid_b[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
                GraphInput {
                    buffer: "mid_c".into(),
                    value: mid_c[0],
                    contract: read_matrix(ValueLifetime::Invocation),
                },
            ],
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: matrix(ValueLifetime::Output),
                retained_successor_of: None,
            }],
        )
        .expect("fan-in stage");
    graph
}

pub(crate) fn facts() -> ExternalFacts {
    ExternalFacts::new(
        Digest([0x5a; 32]),
        BTreeMap::from([("rows".into(), 64), ("cols".into(), 64)]),
    )
}

/// A device that grants every capability the grammar can ask an operand for.
pub(crate) fn rich_device() -> DeviceFacts {
    DeviceFacts::new(
        BackendCapabilities {
            supports_subgroup_ops: true,
            has_warp_shuffle: true,
            ..BackendCapabilities::default()
        },
        256,
    )
    .with_occupancy(128, 64 * 1024)
    .with_compute_units(8)
    .with_concurrent_queues(4)
    .with_spatial_partitioning(true)
    .with_cooperative_launch(true)
    .with_subgroup_size(32)
    .with_launch_costs(4224, 1000)
    .with_bandwidth_facts(3788, 3788)
}

/// A device that reports nothing beyond a launch limit.
pub(crate) fn bare_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
}

/// A device where issuing one launch dominates every other cost.
pub(crate) fn launch_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256).with_launch_costs(200_000, 1_000)
}

/// A device where launches are nearly free and resident state is scarce.
pub(crate) fn occupancy_bound_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_launch_costs(1, 1)
        .with_occupancy(1, 1)
        .with_bandwidth_facts(1, 1)
}

/// A device that partitions but guarantees no forward progress.
pub(crate) fn no_progress_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 256)
        .with_compute_units(8)
        .with_spatial_partitioning(true)
        .with_launch_costs(4224, 1000)
}

pub(crate) fn budget() -> SearchBudget {
    SearchBudget::new(512, 200_000, 4, 0, 1_000_000_000)
}

/// Artifact byte ceiling every fixture request states, well above what the
/// fixture graphs compile to.
pub(crate) const ARTIFACT_BYTES: u64 = 4_000_000;

/// Minimize the latency of one submission within the fixture byte ceiling.
pub(crate) fn latency_objective() -> CompileObjective {
    CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES)
}

/// The fixture graph and facts as an unvalidated request.
///
/// Every suite here compiles the same graph under the same external facts and
/// differs only in device, budget, and objective. A suite that also states
/// recorded measurement evidence adds it before validating.
pub(crate) fn fixture_request(
    device: DeviceFacts,
    budget: SearchBudget,
    objective: CompileObjective,
) -> CompileRequest {
    CompileRequest::new(joined_graph(), facts(), device, budget, objective)
}

/// The fixture request validated for `device`, `budget` and `objective`.
pub(crate) fn validated(
    device: DeviceFacts,
    budget: SearchBudget,
    objective: CompileObjective,
) -> ValidatedCompileRequest {
    fixture_request(device, budget, objective)
        .validate()
        .expect("Fix: state fixture facts the compiler accepts")
}

pub(crate) fn compiled(device: DeviceFacts, budget: SearchBudget) -> Artifact {
    compile(&validated(device, budget, latency_objective())).expect("compilation must succeed")
}

/// The field path a refusal names, when it names one.
///
/// A diagnostic that reports the right code against the wrong field sends a
/// caller to edit something that was already correct, so every suite asserts the
/// path and none restates how to read it.
pub(crate) fn refused_field(error: &vyre_megakernel::CompileError) -> Option<&str> {
    error
        .diagnostic
        .location
        .as_ref()
        .and_then(|location| location.path.as_deref())
}
