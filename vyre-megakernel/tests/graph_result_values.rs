//! Which graph values a completion carries, derived once for every executor.
//!
//! `returned_graph_values` answers "what does a completed execution hand back",
//! and `writable_graph_values` answers "what does this node write". For a graph
//! lifted from one Program the two answers are the same set, and an executor
//! that derives either of them by hand is how a retained read-write value came
//! to be rejected as undeclared by the code that had just written it. These
//! cases hold the two derivations against each other over every buffer access
//! a Program can declare.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ValueLifetime,
};
use vyre_megakernel::{returned_graph_values, writable_graph_values};

/// Path of the frozen access enum this suite claims to cover.
const ACCESS_SOURCE: &str = "vyre-spec/src/buffer_access.rs";

/// Every access a Program buffer can declare, with the variant name the frozen
/// enum states it under.
const ACCESS_MATRIX: &[(&str, BufferAccess)] = &[
    ("ReadOnly", BufferAccess::ReadOnly),
    ("ReadWrite", BufferAccess::ReadWrite),
    ("Uniform", BufferAccess::Uniform),
    ("WriteOnly", BufferAccess::WriteOnly),
    ("Workgroup", BufferAccess::Workgroup),
];

/// A single-node program declaring `buffers` and storing one word.
fn program(buffers: Vec<BufferDecl>) -> Program {
    Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![Node::let_bind("word", Expr::u32(7))],
    )
}

/// One buffer of `access`, sized so a backend-allocated declaration is legal.
fn declaration(access: BufferAccess) -> BufferDecl {
    match access {
        BufferAccess::Workgroup => BufferDecl::workgroup("scratch", 1, DataType::U32),
        access => BufferDecl::storage("slot", 0, access, DataType::U32).with_count(1),
    }
}

fn graph_of(buffers: Vec<BufferDecl>) -> ProgramGraph {
    ProgramGraph::from_program("node", program(buffers))
        .expect("single-node graph. Fix: keep the fixture buffers host-bindable and sized.")
}

fn written_values(graph: &ProgramGraph) -> BTreeSet<vyre_foundation::ir::GraphValueId> {
    writable_graph_values(&graph.nodes()[0])
        .into_iter()
        .collect()
}

#[test]
fn every_declarable_access_returns_exactly_what_its_node_writes() {
    for (name, access) in ACCESS_MATRIX {
        let graph = graph_of(vec![declaration(access.clone())]);
        assert_eq!(
            returned_graph_values(&graph),
            written_values(&graph),
            "access `{name}` disagrees between the returned-value and written-value derivations"
        );
    }
}

#[test]
fn a_retained_read_write_buffer_is_returned() {
    let graph = graph_of(vec![
        BufferDecl::read_write("state", 0, DataType::U32).with_count(1)
    ]);
    let value = graph.nodes()[0].inputs[0].value;
    assert_eq!(
        graph
            .values()
            .iter()
            .find(|candidate| candidate.id == value)
            .map(|candidate| candidate.contract.lifetime),
        Some(ValueLifetime::Retained)
    );
    assert_eq!(returned_graph_values(&graph), BTreeSet::from([value]));
}

#[test]
fn a_read_only_buffer_is_not_returned() {
    let graph = graph_of(vec![
        BufferDecl::read("input", 0, DataType::U32).with_count(1)
    ]);
    assert!(returned_graph_values(&graph).is_empty());
}

#[test]
fn an_output_buffer_and_the_state_beside_it_are_both_returned() {
    let graph = graph_of(vec![
        BufferDecl::read("input", 0, DataType::U32).with_count(1),
        BufferDecl::read_write("state", 1, DataType::U32).with_count(1),
        BufferDecl::output("result", 2, DataType::U32).with_count(1),
    ]);
    assert_eq!(returned_graph_values(&graph), written_values(&graph));
    assert_eq!(returned_graph_values(&graph).len(), 2);
}

#[test]
fn a_pipeline_live_out_read_write_buffer_is_returned_as_an_output() {
    let graph = graph_of(vec![BufferDecl::read_write("carried", 0, DataType::U32)
        .with_count(1)
        .with_pipeline_live_out(true)]);
    let value = graph.nodes()[0].outputs[0];
    assert_eq!(
        graph
            .values()
            .iter()
            .find(|candidate| candidate.id == value)
            .map(|candidate| candidate.contract.lifetime),
        Some(ValueLifetime::Output)
    );
    assert_eq!(returned_graph_values(&graph), BTreeSet::from([value]));
}

#[test]
fn the_access_matrix_names_every_frozen_access_variant() {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(ACCESS_SOURCE),
    )
    .expect("frozen access enum source. Fix: keep ACCESS_SOURCE pointing at the enum.");
    let body = source
        .split_once("pub enum BufferAccess {")
        .expect("enum declaration. Fix: update the marker this suite scans for.")
        .1
        .split_once('}')
        .expect("enum body. Fix: keep the enum body brace-delimited.")
        .0;
    let declared = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("///") && !line.starts_with('#'))
        .map(|line| line.trim_end_matches(',').to_string())
        .collect::<BTreeSet<_>>();
    let covered = ACCESS_MATRIX
        .iter()
        .map(|(name, _)| (*name).to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        declared, covered,
        "the returned-value suite covers a different access set than `{ACCESS_SOURCE}` declares"
    );
}
