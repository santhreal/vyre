//! Compiler artifact diagnostics preserve authoritative identities and planning evidence.

use std::collections::BTreeMap;

use vyre::compiler::{
    compile, ArtifactEnvelope, CompileRequest, Digest, ExternalFacts, SearchBudget,
};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_debug::ArtifactReport;

#[test]
fn report_round_trips_compiler_owned_identity_plan_and_abi() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "out",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(4)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
        )
        .unwrap();
    graph
        .add_node(
            "main",
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
                [4, 1, 1],
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([7; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();
    let artifact = compile(&request).unwrap();
    let envelope = ArtifactEnvelope::new(artifact.clone());
    let bytes = envelope.to_bytes().unwrap();

    let report = ArtifactReport::from_bytes(&bytes).unwrap();

    assert_eq!(report.artifact, hex(artifact.digest().as_bytes()));
    assert_eq!(
        report.source_graph,
        hex(artifact.provenance().source_graph.as_bytes())
    );
    assert_eq!(
        report.request,
        hex(artifact.provenance().request.as_bytes())
    );
    assert_eq!(report.selected_plan, artifact.selected_plan().clone());
    assert_eq!(report.abi, artifact.abi().clone());
    assert!(report.targets.is_empty());
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
