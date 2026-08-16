//! The canonical compiler artifact fixtures, owned by the workspace.
//!
//! Every crate downstream of `vyre-megakernel` that tests admission, packaging,
//! or artifact reporting needs the same input: a one-node graph over a few
//! external values, compiled under fixed facts and a bounded search. Four crates
//! each carried their own copy of that setup, and `vyre-megakernel` and
//! `vyre-runtime` carried it byte-identically because their assertions compare
//! digests that only agree while the two copies agree.
//!
//! The copies also had to keep restating which buffer declaration a value
//! contract implies, so a fixture whose declared element count disagreed with its
//! contract shape was accepted by whichever crate wrote it and rejected by the
//! next one to copy it. [`decl_for`] states that mapping once.
//!
//! Shared the same way as `tests/support/preferred_dispatch_backend_contract.rs`:
//! each consumer includes this file with `#[path]`.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, Artifact, ArtifactNodeId, ArtifactValueId, CompileRequest, DeviceFacts, Digest,
    ExternalFacts, SearchBudget, TargetEntryPoint, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};

/// A single-dimension external value contract.
pub(crate) fn contract(
    dtype: DataType,
    count: u32,
    access: BufferAccess,
    lifetime: ValueLifetime,
) -> ValueContract {
    ValueContract {
        dtype,
        shape: vec![ShapeDim::Known(u64::from(count))],
        access,
        lifetime,
    }
}

/// Element count a fixture value's contract declares.
fn declared_count(name: &str, contract: &ValueContract) -> u32 {
    match contract.shape.as_slice() {
        [ShapeDim::Known(count)] => {
            u32::try_from(*count).expect("fixture element count must fit u32")
        }
        shape => panic!("fixture value `{name}` must declare one known dimension, got {shape:?}"),
    }
}

/// Buffer declaration a fixture value's contract implies.
///
/// The mapping is the contract rather than a convenience. A read-only value is a
/// read buffer, a write-only value is an output buffer the caller reads back, and
/// a read-write value is retained storage. Uniform and workgroup memory are not
/// graph externals, so those contracts are rejected here instead of silently
/// becoming global storage.
fn decl_for(name: &str, slot: u32, contract: &ValueContract) -> BufferDecl {
    let decl = match &contract.access {
        BufferAccess::ReadOnly => BufferDecl::read(name, slot, contract.dtype.clone()),
        BufferAccess::WriteOnly => BufferDecl::output(name, slot, contract.dtype.clone()),
        BufferAccess::ReadWrite => BufferDecl::read_write(name, slot, contract.dtype.clone()),
        access => {
            panic!("fixture value `{name}` declares {access:?}, which is not a graph external")
        }
    };
    decl.with_count(declared_count(name, contract))
}

/// One-node graph over `values`, bound to Program slots in the order given.
pub(crate) fn graph_over(
    node: &str,
    workgroup_size: [u32; 3],
    values: &[(&str, ValueContract)],
) -> ProgramGraph {
    let mut graph = ProgramGraph::new();
    let mut decls = Vec::with_capacity(values.len());
    for (slot, (name, contract)) in values.iter().enumerate() {
        let slot = u32::try_from(slot).expect("fixture slot must fit u32");
        graph
            .add_external_value(*name, contract.clone())
            .expect("fixture resource must be valid");
        decls.push(decl_for(name, slot, contract));
    }
    graph
        .add_node(
            node,
            Program::wrapped(decls, workgroup_size, Vec::new()),
            Vec::new(),
            Vec::new(),
        )
        .expect("fixture node must be valid");
    graph
}

/// Compile a fixture graph under the facts and bounded search every fixture uses.
///
/// `facts_seed` fills the external facts digest, which participates in the
/// request digest, so two fixtures that differ only in seed compile to different
/// artifact identities.
pub(crate) fn compile_graph(graph: ProgramGraph, facts_seed: u8) -> Artifact {
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([facts_seed; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .expect("fixture request must validate");
    compile(&request).expect("fixture request must compile")
}

/// The canonical read-only single-input artifact.
///
/// `vyre-megakernel` and `vyre-runtime` both assert against this exact artifact,
/// including its digest, so it is one function rather than two copies that have
/// to stay identical for those assertions to keep meaning what they say.
pub(crate) fn neutral_artifact(workgroup_size: [u32; 3]) -> Artifact {
    compile_graph(
        graph_over(
            "entry",
            workgroup_size,
            &[(
                "input",
                contract(
                    DataType::U32,
                    8,
                    BufferAccess::ReadOnly,
                    ValueLifetime::Invocation,
                ),
            )],
        ),
        0,
    )
}

/// The single-binding entry point the target payload fixtures attach.
pub(crate) fn entry_point() -> TargetEntryPoint {
    TargetEntryPoint {
        name: "entry".into(),
        node: ArtifactNodeId(0),
        workgroup_size: [8, 1, 1],
        grid_size: [4, 1, 1],
        dynamic_shared_bytes: 64,
        resource_bindings: vec![TargetResourceBinding {
            resource: ArtifactValueId(0),
            group: 0,
            slot: 3,
            memory: TargetResourceMemory::Global,
            access: TargetResourceAccess::ReadOnly,
        }],
    }
}
