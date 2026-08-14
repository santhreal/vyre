//! Compiler artifact diagnostics preserve authoritative identities and planning evidence.

use vyre::compiler::ArtifactEnvelope;
use vyre_debug::ArtifactReport;
use vyre_foundation::ir::{BufferAccess, DataType, ValueLifetime};

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{compile_graph, contract, graph_over};

#[test]
fn report_round_trips_compiler_owned_identity_plan_and_abi() {
    let artifact = compile_graph(
        graph_over(
            "main",
            [4, 1, 1],
            &[(
                "out",
                contract(
                    DataType::U32,
                    4,
                    BufferAccess::WriteOnly,
                    ValueLifetime::Output,
                ),
            )],
        ),
        7,
    );
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
