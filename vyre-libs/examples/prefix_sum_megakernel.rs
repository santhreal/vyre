//! Compile the `scan_prefix_sum` libs composition into a megakernel artifact.
//!
//! The builder returns IR. The artifact is what a runtime loads. Nothing here
//! executes the program: `vyre-megakernel` compiles a typed graph into a
//! backend-neutral artifact, and the checks below are on that artifact.
use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_libs::math::scan::scan_prefix_sum;
use vyre_megakernel::{
    compile, Artifact, CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts,
    ObjectiveMetric, SearchBudget,
};

const ELEMENTS: u32 = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let program = scan_prefix_sum("input", "output", ELEMENTS);
    let artifact = compile_artifact(&program)?;

    println!("digest: {}", hex(artifact.digest()));
    println!("nodes: {}", artifact.nodes().len());
    println!("resources: {}", artifact.resources().len());
    println!("envelope: {:?}", artifact.resource_envelope());

    let bytes = artifact.to_bytes()?;
    let decoded = Artifact::from_bytes(&bytes)?;
    if decoded.digest() != artifact.digest() {
        return Err(
            "the artifact did not survive a to_bytes/from_bytes round trip. Fix: repair the encoder that changed the canonical bytes, then rerun this example."
                .into(),
        );
    }
    println!("round trip: {} bytes, digest unchanged", bytes.len());

    let again = compile_artifact(&program)?;
    if again.digest() != artifact.digest() {
        return Err(
            "two compilations of the same program produced different artifacts. Fix: repair the pass that reads outside the request, then rerun this example."
                .into(),
        );
    }
    println!("recompile: digest unchanged");
    Ok(())
}

/// Compile `program` as the single node of a one-node graph.
///
/// Every storage buffer the program declares becomes an external value carrying
/// that buffer's own element type, count, and access, so the graph contract
/// cannot disagree with the program it wraps. Workgroup-local memory is not an
/// external value: it is scratch the program allocates per dispatch, and the
/// graph wire format has no way to name it.
fn compile_artifact(program: &Program) -> Result<Artifact, Box<dyn std::error::Error>> {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }
        graph.add_external_value(
            buffer.name(),
            ValueContract {
                dtype: buffer.element(),
                shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                access: buffer.access(),
                lifetime: ValueLifetime::Invocation,
            },
        )?;
    }
    graph.add_node("main", program.clone(), Vec::new(), Vec::new())?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 0, 0, 1),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
    )
    .validate()?;
    Ok(compile(&request)?)
}

/// Lowercase hex of a digest, for a line a reader can compare between runs.
fn hex(digest: Digest) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.0 {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}
