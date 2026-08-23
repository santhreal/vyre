//! WHY: a whole-grid fence is a launch boundary, not an instruction, and until
//! the planner cut existed the only place that fact was enforced was the WGSL
//! emitter, which refused. `taint_pollution` is the shape that hits it: program
//! fusion inserts `MemoryOrdering::GridSync` between the divergent writer arm and
//! the arm that reads what it wrote, so the fence lives INSIDE one node's body.
//! A single-node graph has no fusion pair to reject, so
//! `legality::analyze_fusion_pair` never sees it, and every wgpu compile of this
//! op failed at emit.
//!
//! The class this closes is a fence surviving into any backend that has no
//! instruction for it. The assertions are on the whole route, not on the split
//! function: the fence is present in the built program, the emitter still refuses
//! the whole program, request validation cuts it into more than one node, and
//! every node that results emits and validates as its own WGSL module.
//!
//! What it does not catch: whether the two dispatches are ordered correctly at
//! run time. That is the retained-succession contract, covered by the megakernel
//! dependency and fusion-legality suites.

#![cfg(feature = "security")]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::schedule::SelectedSchedule;
use vyre_foundation::transform::grid_sync_split::contains_grid_sync;
use vyre_foundation::validate::BackendCapabilities;
use vyre_libs::graph::program_graph::ProgramGraphShape;
use vyre_libs::security::taint_pollution;
use vyre_megakernel::{
    CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget, ValidatedCompileRequest,
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
        1 << 24,
    )
    .validate()
    .expect("a whole-grid fence must be cut, not rejected")
}

fn emit_wgsl(program: &Program) -> Result<naga::Module, String> {
    let graph = ProgramGraph::from_program("taint_pollution_emit", program.clone())
        .map_err(|error| format!("{error:?}"))?;
    let logical = LogicalProgramGraph::validate(&graph, &BTreeMap::new())
        .map_err(|error| format!("{error:?}"))?;
    let schedule = SelectedSchedule::from_logical(&logical);
    let phase = schedule
        .phases
        .first()
        .ok_or_else(|| "selected schedule has no phase".to_string())?
        .id;
    let lowered = vyre_lower::lower_scheduled(program, &schedule, phase)
        .map_err(|error| format!("{error:?}"))?;
    vyre_emit_naga::emit(lowered.descriptor()).map_err(|error| format!("{error}"))
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

/// The pre-cut behavior, kept as a live assertion. WGSL has no whole-grid barrier
/// and wgpu has no cooperative launch, so lowering the unsplit program must stay a
/// refusal. If this ever starts succeeding, the emitter has silently downgraded
/// the fence to a workgroup barrier and the kernel runs unsynchronized.
#[test]
fn the_unsplit_program_is_still_refused_by_the_wgsl_emitter() {
    let error = emit_wgsl(&taint_pollution_program())
        .expect_err("a whole-grid fence must never lower to a WGSL barrier");
    assert!(
        error.contains("grid synchronization"),
        "the refusal must name whole-grid synchronization as the reason: {error}"
    );
}

/// The acceptance: the planner cut turns the refusal into two dispatches, and
/// each one is a WGSL module the validator accepts.
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
