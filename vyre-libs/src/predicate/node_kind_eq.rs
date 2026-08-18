//! `node_kind_eq`  -  `NodeSet = { v : nodes[v] == kind }`.

use vyre_foundation::ir::Program;

use crate::label::nodeset_filter::{nodeset_filter_program, NodeSetFilter};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::predicate::node_kind_eq";

/// Build a Program: `NodeSet = { v : nodes[v] == kind }`.
#[must_use]
pub fn node_kind_eq(nodes: &str, nodeset_out: &str, node_count: u32, kind: u32) -> Program {
    nodeset_filter_program(
        OP_ID,
        nodes,
        nodeset_out,
        node_count,
        NodeSetFilter::Eq(kind),
    )
}

const EXPECTED_NODE_KIND_EQ_OUTPUT_BYTES: [u8; 4] = [5, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || node_kind_eq("nodes", "nodeset", 4, crate::predicate::node_kind::CALL),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[2, 1, 2, 4]), // nodes: CALL, VARIABLE, CALL, LITERAL
                to_bytes(&[0]),          // nodeset_out
            ]]
        }),
        Some(|| {
            // nodes 0 and 2 (CALL)
            vec![vec![EXPECTED_NODE_KIND_EQ_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["identity"])
}

#[cfg(test)]
mod tests {
    use crate::predicate::node_kind;
    use vyre_reference::composition_witness::{
        node_kind_eq_witness as cpu_ref, node_kind_eq_witness_into as cpu_ref_into,
    };
    #[test]
    fn filters_by_kind() {
        let got = cpu_ref(
            &[
                node_kind::CALL,
                node_kind::VARIABLE,
                node_kind::CALL,
                node_kind::LITERAL,
            ],
            node_kind::CALL,
        );
        assert_eq!(got, vec![0b0101]);
    }

    #[test]
    fn cpu_ref_into_reuses_nodeset_buffer() {
        let mut out = Vec::with_capacity(4);
        let ptr = out.as_ptr();
        cpu_ref_into(
            &[
                node_kind::CALL,
                node_kind::VARIABLE,
                node_kind::CALL,
                node_kind::LITERAL,
            ],
            node_kind::CALL,
            &mut out,
        );
        assert_eq!(out, vec![0b0101]);
        assert_eq!(out.as_ptr(), ptr);
    }
}
