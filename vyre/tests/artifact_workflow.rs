//! The facade routes whole-program compilation through canonical compiler artifacts.

use std::collections::BTreeMap;

use vyre::compiler::{compile, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget};
use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};

#[test]
fn facade_compiles_validated_graph_to_canonical_artifact() {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "out",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
        )
        .unwrap();
    graph
        .add_node(
            "main",
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
                [1, 1, 1],
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([9; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();

    let artifact = compile(&request).unwrap();
    let repeated = compile(&request).unwrap();

    assert_eq!(artifact.nodes().len(), 1);
    assert_eq!(artifact.abi().entries.len(), 1);
    assert_eq!(artifact.digest(), repeated.digest());
    assert_eq!(artifact.provenance().request, repeated.provenance().request);
}
