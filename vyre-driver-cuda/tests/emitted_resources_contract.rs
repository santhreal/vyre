//! What the CUDA driver reports one loaded entry point allocated.
//!
//! WHY: registers and local-memory spill are assigned by `ptxas`, not by the
//! neutral artifact, so compile-time ranking used the analytic estimate and
//! never saw either figure. `ArtifactInstance::emitted_resources` is the seam
//! that answers for the loaded module, and an answer of zero registers would
//! silently mean "unreported" and leave the estimate in force for every plan.
//! This asserts the CUDA arm reports a real allocation for a real entry point,
//! one record per payload entry, so the compiler ranks on the figure the device
//! will run.

#![cfg(feature = "device-tests")]

use std::collections::BTreeMap;

mod harness;
use harness::add_one_program;
use vyre_driver_cuda::cuda_factory;
use vyre_foundation::ir::ProgramGraph;
use vyre_megakernel::{
    compile, CompileObjective, CompileRequest, Digest, ExternalFacts, ObjectiveMetric, SearchBudget,
};

/// Registers a `main` entry point cannot allocate fewer than: the driver
/// assigns at least the ones a store and an add need.
const MINIMUM_REGISTERS: u32 = 1;

#[test]
fn the_cuda_driver_reports_the_registers_it_allocated_for_every_entry_point() {
    let registration = vyre_driver::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
        .expect("the CUDA backend registration must be linked");
    let compiler = registration
        .target_compiler()
        .expect("the CUDA backend must register a target compiler");
    let materializer = registration
        .materializer()
        .expect("the CUDA backend must register an artifact materializer");
    let backend =
        cuda_factory().expect("this test requires the live CUDA device the GPU lanes own");
    let facts = backend.device_profile().compile_facts();
    let graph = ProgramGraph::from_program("cuda.emitted-resources", add_one_program())
        .expect("the fixture program must adapt to one canonical graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        facts,
        SearchBudget::new(128, 128, 0, 0, 128),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 60_000),
    )
    .validate()
    .expect("the fixture compile request must validate");
    let artifact = compile(&request).expect("the fixture graph must compile");
    let payload = compiler
        .compile(&artifact)
        .expect("the CUDA target compiler must build the fixture artifact");
    let instance = materializer
        .materialize(&artifact, &payload)
        .expect("the CUDA device must materialize the fixture payload");

    let reported = instance
        .emitted_resources()
        .expect("a loaded CUDA module must answer for what it allocated");

    assert!(
        !reported.is_empty(),
        "the fixture artifact has one entry point, so an empty report would \
         assert nothing"
    );
    assert_eq!(
        reported.len(),
        payload.entries().len(),
        "the compiler pairs records with payload entries by position, so the \
         counts must match: {reported:?}"
    );
    for (entry, resources) in payload.entries().iter().zip(&reported) {
        assert!(
            resources.registers_per_invocation >= MINIMUM_REGISTERS,
            "entry `{}` reported {} registers, which the compiler reads as \
             unreported and would rank on the estimate instead",
            entry.name,
            resources.registers_per_invocation
        );
        assert!(
            resources.registers_per_invocation
                <= facts
                    .hardware_registers_per_invocation()
                    .max(MINIMUM_REGISTERS),
            "entry `{}` cannot allocate more registers than the device has",
            entry.name
        );
    }
}
