//! Differential parity for `flows_to_alias_only` against the registered base `flows_to`.
//!
//! Both build ONE forward-reach hop over a program graph; they differ only in the
//! edge-kind mask: `flows_to` uses FLOWS_TO_MASK (all dataflow edges) while
//! `flows_to_alias_only` uses ALIAS_PROPAGATION_MASK (a strict subset. ASSIGNMENT,
//! ALIAS, MUT_REF, PHI). `flows_to_alias_only` was an orphan builder (registry-coverage
//! closure gate `adversarial_registry_closure.rs`). On a graph with one ALIAS edge and
//! one CALL_ARG edge (in FLOWS_TO_MASK but NOT the alias mask), the two must diverge:
//! dataflow reaches both neighbors, aliasing reaches only the alias neighbor. This pins
//! that with real frontier bytes (the exact FP class the alias/dataflow split fixed:
//! `strdup` is a CALL_ARG dataflow, NOT an alias).
#![cfg(feature = "security")]
#![forbid(unsafe_code)]

use vyre_libs::security::flows_to::{flows_to, flows_to_alias_only};
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::edge_kind;
use vyre_reference::value::Value;

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// One forward-reach hop from frontier {0} over the 3-node graph
///   0 --ASSIGNMENT--> 1   (an aliasing edge)
///   0 --CALL_ARG-----> 2  (a dataflow edge that is NOT an alias)
/// Returns the resulting `fout` frontier word.
fn one_hop(program: &vyre::ir::Program) -> u32 {
    // CSR: node 0 has edges [0,2); nodes 1,2 have none.
    let pg_nodes = [0u32, 0, 0];
    let pg_edge_offsets = [0u32, 2, 2, 2];
    let pg_edge_targets = [1u32, 2];
    let pg_edge_kind_mask = [edge_kind::ASSIGNMENT, edge_kind::CALL_ARG];
    let pg_node_tags = [0u32, 0, 0];
    let fin = [0b001u32]; // {0}
    let fout = [0b001u32]; // accumulator starts at the input frontier

    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack(&pg_nodes)),
            Value::from(pack(&pg_edge_offsets)),
            Value::from(pack(&pg_edge_targets)),
            Value::from(pack(&pg_edge_kind_mask)),
            Value::from(pack(&pg_node_tags)),
            Value::from(pack(&fin)),
            Value::from(pack(&fout)),
        ],
    )
    .expect("flows_to reach program must execute under reference_eval");
    unpack(&outputs[0].to_bytes())[0]
}

#[test]
fn dataflow_reaches_call_arg_neighbor_but_aliasing_does_not() {
    let shape = ProgramGraphShape::new(3, 2);
    let dataflow = one_hop(&flows_to(shape, "fin", "fout"));
    let shape_alias = ProgramGraphShape::new(3, 2);
    let alias = one_hop(&flows_to_alias_only(shape_alias, "fin", "fout"));

    // flows_to traverses BOTH edges: {0} ∪ {1 (assignment), 2 (call_arg)} = {0,1,2}.
    assert_eq!(
        dataflow, 0b111,
        "flows_to must reach the ALIAS neighbor (1) AND the CALL_ARG neighbor (2)"
    );
    // flows_to_alias_only traverses ONLY the alias edge: {0} ∪ {1} = {0,1}.
    assert_eq!(
        alias, 0b011,
        "flows_to_alias_only must reach the ALIAS neighbor (1) but NOT the CALL_ARG neighbor (2)"
    );
    // The distinguishing bit: node 2 is reached by dataflow, not by aliasing.
    assert_ne!(dataflow & 0b100, 0, "dataflow reaches node 2 via CALL_ARG");
    assert_eq!(
        alias & 0b100,
        0,
        "aliasing must NOT reach node 2 (CALL_ARG is not an alias edge)"
    );
}
