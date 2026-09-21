//! WHY: a whole-grid fence is a launch boundary, not an instruction.
//! `taint_pollution` is the shape that states it: program fusion inserts
//! `MemoryOrdering::GridSync` between the divergent writer arm and the arm that
//! reads what it wrote, so the fence lives INSIDE one node's body. A single-node
//! graph has no fusion pair to reject, so `legality::analyze_fusion_pair` never
//! sees it, and the fence reaches an emitter that has no instruction for it.
//!
//! The class this closes is a fence surviving into a backend as an instruction.
//! Two routes remove it, and both are asserted on the whole route rather than on
//! a single function. The emitter turns the fence into a launch boundary: the
//! fenced descriptor emits one compute entry point per dispatch segment, named in
//! submission order. The planner cut removes it earlier: request validation
//! splits the program into more than one node, each fence-free, each emitting and
//! validating as its own WGSL module.
//!
//! What it does not catch: whether the dispatches are ordered correctly at run
//! time. That is the retained-succession contract, covered by the megakernel
//! dependency and fusion-legality suites.

#![cfg(feature = "security")]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::transform::grid_sync_split::contains_grid_sync;
use vyre_foundation::validate::BackendCapabilities;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_libs::security::taint_pollution;
use vyre_megakernel::{
    CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget, ValidatedCompileRequest,
};

/// 33 nodes puts the bitset one node past a word boundary, which is the shape the
/// family guard already pins as the interesting one.
fn taint_pollution_program() -> Program {
    taint_pollution(
        ProgramGraphShape::new(33, 32),
        "source",
        "sink",
        "reach",
        "hits",
        "out_scalar",
    )
}

fn validated(program: Program) -> ValidatedCompileRequest {
    let graph = ProgramGraph::from_program("taint_pollution", program)
        .expect("a security flow program must form a single-node graph");
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        // A device with shared memory for workgroup scratch but no cooperative
        // launch is the point: the planner cut removes the fence before device
        // admission, so admission must accept the program.
        DeviceFacts::new(
            BackendCapabilities {
                has_shared_memory: true,
                ..BackendCapabilities::default()
            },
            256,
        )
        .with_occupancy(0, 4096),
        SearchBudget::new(128, 1_000_000, 8, 4, 1_000_000_000),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1 << 24),
    )
    .validate()
    .expect("a whole-grid fence must be cut, not rejected")
}

/// The segment entry points and the WGSL module `program`'s first schedule phase
/// lowers to.
fn lower_and_emit(program: &Program) -> Result<(Vec<String>, naga::Module), String> {
    let graph = ProgramGraph::from_program("taint_pollution_emit", program.clone())
        .map_err(|error| format!("{error:?}"))?;
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .map_err(|error| format!("{error:?}"))?;
    let schedule = vyre_megakernel::baseline_schedule(&logical);
    let phase = schedule
        .phases
        .first()
        .ok_or_else(|| "selected schedule has no phase".to_string())?
        .id;
    let lowered = vyre_lower::lower_scheduled(program, &schedule, phase)
        .map_err(|error| format!("{error:?}"))?;
    let segments = vyre_emit_naga::grid_segment_entry_points(lowered.descriptor())
        .map_err(|error| format!("{error}"))?;
    let module = vyre_emit_naga::emit(lowered.descriptor()).map_err(|error| format!("{error}"))?;
    Ok((segments, module))
}

fn emit_wgsl(program: &Program) -> Result<naga::Module, String> {
    lower_and_emit(program).map(|(_, module)| module)
}

/// The premise. If fusion stops inserting the fence this whole file is vacuous,
/// so the fence is asserted rather than assumed.
#[test]
fn taint_pollution_carries_a_whole_grid_fence() {
    assert!(
        contains_grid_sync(&taint_pollution_program()),
        "taint_pollution fuses a divergent writer arm with a reader arm, which requires a whole-grid fence"
    );
}

/// WGSL has no whole-grid barrier and wgpu has no cooperative launch, so the
/// fence must never lower to an instruction inside one entry point. It lowers to a
/// launch boundary: the fenced descriptor emits one compute entry point per
/// dispatch segment, and a dispatch layer submits them in that order. If this ever
/// collapses to a single entry point, the fence has been downgraded to a workgroup
/// barrier and the kernel runs with no cross-workgroup synchronization at all.
#[test]
fn the_unsplit_program_emits_one_entry_point_per_dispatch_segment() {
    let (segments, module) = lower_and_emit(&taint_pollution_program())
        .expect("a fenced descriptor must emit as ordered dispatch segments");
    assert!(
        segments.len() > 1,
        "a whole-grid fence must become a launch boundary, got {} segment",
        segments.len()
    );
    let emitted: Vec<&str> = module
        .entry_points
        .iter()
        .map(|entry| entry.name.as_str())
        .collect();
    assert_eq!(
        emitted, segments,
        "the emitted entry points must be the dispatch segments a caller submits, in order"
    );
}

/// The planner cut removes the fence before device admission, so every node it
/// yields is a fence-free WGSL module the validator accepts.
#[test]
fn the_planner_cut_yields_more_than_one_emittable_module() {
    let request = validated(taint_pollution_program());
    let graph = request.graph();
    assert!(
        graph.nodes().len() > 1,
        "a fenced program must be cut into sequential nodes, got {}",
        graph.nodes().len()
    );

    let validator = || {
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
    };
    for node in graph.nodes() {
        assert!(
            !contains_grid_sync(&node.program),
            "segment `{}` still carries the fence the cut removed",
            node.name
        );
        let module = emit_wgsl(&node.program)
            .unwrap_or_else(|error| panic!("segment `{}` must emit WGSL: {error}", node.name));
        if let Err(error) = validator().validate(&module) {
            panic!("segment `{}` emitted invalid WGSL: {error:?}", node.name);
        }
    }
}

/// A cut that dropped the retained succession would let a later pass fuse the
/// segments back together and reintroduce the fence. Every segment after the
/// first must therefore consume a value one of its predecessors produced.
#[test]
fn each_segment_after_the_first_consumes_a_produced_value() {
    let request = validated(taint_pollution_program());
    let graph = request.graph();
    for node in graph.nodes().iter().skip(1) {
        assert!(
            node.inputs.iter().any(|input| {
                graph
                    .values()
                    .get(input.value.0 as usize)
                    .and_then(|value| value.producer)
                    .is_some_and(|producer| producer.0 < node.id.0)
            }),
            "segment `{}` has no dependency on an earlier segment, so nothing orders the two dispatches",
            node.name
        );
    }
}
