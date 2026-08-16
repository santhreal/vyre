//! Contracts for `vyre_runtime::resident_work_queue::advanced::parallel_dfa`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_foundation::ir::{Expr, Node};
use vyre_runtime::resident_work_queue::advanced::parallel_dfa::dfa_byte_scanner_parallel_composition;

use vyre_foundation::ir::MemoryOrdering;

#[test]
fn parallel_dfa_fragment_has_prefix_barriers_and_output() {
    let nodes = dfa_byte_scanner_parallel_composition();
    assert!(
        nodes
            .iter()
            .filter(|node| matches!(
                node,
                Node::Barrier {
                    ordering: MemoryOrdering::SeqCst
                }
            ))
            .count()
            >= 2,
        "prefix composition must synchronize scratch-table stages"
    );
    assert!(
        stores_buffer(&nodes, "out_state_by_lane"),
        "fragment must publish per-lane states"
    );
}

fn stores_buffer(nodes: &[Node], name: &str) -> bool {
    let mut found = false;
    vyre_foundation::visit::for_each_node(nodes, |node| {
        if let Node::Store { buffer, .. } = node {
            found = found || buffer.as_str() == name;
        }
    });
    found
}
