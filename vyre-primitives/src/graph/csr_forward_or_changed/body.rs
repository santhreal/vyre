use std::sync::Arc;

use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{Expr, Node};

use super::layout::OP_ID;
use crate::graph::program_graph::ProgramGraphShape;

/// Build one in-place forward expansion pass over an accumulating frontier.
#[must_use]
pub fn csr_forward_or_changed_body(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
) -> Vec<Node> {
    csr_forward_or_changed_body_prefixed(shape, frontier_out, changed_var, edge_kind_mask, "")
}

fn local(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}_{name}")
    }
}

/// Build one traversal pass with caller-provided local-name prefixing for
/// repeated inlining under validators that disallow shadowing.
#[must_use]
pub fn csr_forward_or_changed_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    let src = local(prefix, "src");

    // The per-source neighbor expansion is the ONE canonical CSR edge-scan with an
    // IDENTITY frontier index (a single bitset, no per-query base) and a local
    // `assign(changed_var, 1)` on each newly-set bit. Sharing the builder keeps this
    // serial single-workgroup path from drifting out of step with the parallel/batch
    // paths. Byte-identical to the previous hand-written per_source body.
    let per_source = crate::graph::edge_scan::csr_edge_scan_nodes(
        shape,
        frontier_out,
        Expr::var(src.as_str()),
        |word| word,
        || vec![Node::assign(changed_var, Expr::u32(1))],
        edge_kind_mask,
        prefix,
    );

    vec![Node::if_then(
        Expr::eq(Expr::local_x(), Expr::u32(0)),
        vec![Node::loop_for(
            src.as_str(),
            Expr::u32(0),
            Expr::u32(shape.node_count),
            per_source,
        )],
    )]
}

/// Wrap one traversal pass as a child Region of `parent_op_id`.
#[must_use]
pub fn csr_forward_or_changed_child(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
) -> Node {
    csr_forward_or_changed_child_prefixed(
        parent_op_id,
        shape,
        frontier_out,
        changed_var,
        edge_kind_mask,
        "",
    )
}

/// Wrap a traversal pass with a local-name prefix for repeated inlining.
#[must_use]
pub fn csr_forward_or_changed_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(csr_forward_or_changed_body_prefixed(
            shape,
            frontier_out,
            changed_var,
            edge_kind_mask,
            local_prefix,
        )),
    }
}
