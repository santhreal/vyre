//! Hybrid sparse/dense traversal step: one program whose device-resident
//! frontier popcount picks CSR row expansion or a dense reverse-row scan.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::frontier_plan::ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE;
use super::mode_selection::dense_cutover_nodes;
use super::HYBRID_OP_ID;
/// Build the GPU Program for one adaptive sparse/dense step.
///
/// Each invocation uses the device-resident `frontier_popcount[0]` to choose
/// the path. Below `dense_threshold_pct`, invocation `src` expands the CSR row
/// for one active source node. At or above the threshold, invocation `dst`
/// scans the dense reverse-adjacency row for one destination node.
///
/// This is intentionally a single primitive contract: callers can keep
/// `frontier_in`, `frontier_popcount`, CSR buffers, dense rows, and
/// `frontier_out` resident across fixpoint iterations, eliminating the old
/// CPU branch/readback boundary from the release path.
#[must_use]
pub fn adaptive_sparse_dense_step(
    frontier_in: &str,
    frontier_out: &str,
    frontier_popcount: &str,
    edge_offsets: &str,
    edge_targets: &str,
    edge_kind_mask: &str,
    adj_rows_dense: &str,
    node_count: u32,
    edge_count: u32,
    allow_mask: u32,
    dense_threshold_pct: u32,
) -> Program {
    let (words, adj_count) = match super::dense_step::validate_dense_adj_counts(
        HYBRID_OP_ID,
        frontier_out,
        node_count,
        "adaptive_sparse_dense_step",
    ) {
        Ok(counts) => counts,
        Err(trap) => return trap,
    };
    let Some(offset_count) = node_count.checked_add(1) else {
        return trap_program(
            HYBRID_OP_ID,
            Some((frontier_out, DataType::U32)),
            "Fix: adaptive_sparse_dense_step CSR offset count overflows u32. Partition the graph."
                .to_string(),
        );
    };
    let physical_edge_count = edge_count.max(1);

    let lane = Expr::InvocationId { axis: 0 };
    let dense_cutover = dense_cutover_nodes(node_count, dense_threshold_pct);
    let dense_body = super::dense_step::dense_reverse_scan_body(
        frontier_in,
        frontier_out,
        adj_rows_dense,
        &lane,
        words,
        "dense_",
    );

    let composer = crate::builder::csr::CsrTraversalComposer::new(
        HYBRID_OP_ID,
        "adaptive_sparse_dense_step",
        node_count,
    )
    .with_buffers(crate::builder::csr::CsrBuffers::new(
        edge_offsets,
        edge_targets,
        Some(edge_kind_mask),
    ))
    .with_allow_mask(allow_mask)
    .with_prefix("sparse");

    let sparse_expand =
        composer.emit_edge_expand(frontier_out, lane.clone(), |word| word, Vec::new);
    let word_idx_name = composer.local_name("word_idx");
    let bit_mask_name = composer.local_name("bit_mask");
    let src_word_name = composer.local_name("src_word");

    let sparse_body = crate::graph::frontier_bits::when_bit_set(
        frontier_in,
        &lane,
        Some(word_idx_name.as_str()),
        src_word_name.as_str(),
        bit_mask_name.as_str(),
        |word| word,
        sparse_expand,
    );

    let body = vec![
        Node::let_bind(
            "frontier_popcount_total",
            Expr::load(frontier_popcount, Expr::u32(0)),
        ),
        Node::if_then_else(
            Expr::ge(
                Expr::var("frontier_popcount_total"),
                Expr::u32(dense_cutover),
            ),
            dense_body,
            sparse_body,
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(frontier_out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage(frontier_popcount, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(1),
            BufferDecl::storage(edge_offsets, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(offset_count),
            BufferDecl::storage(edge_targets, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(physical_edge_count),
            BufferDecl::storage(edge_kind_mask, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(physical_edge_count),
            BufferDecl::storage(adj_rows_dense, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(adj_count),
        ],
        ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            HYBRID_OP_ID,
            vec![Node::if_then(
                Expr::lt(lane.clone(), Expr::u32(node_count)),
                body,
            )],
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitset::bitset_words;

    #[test]
    fn emitted_hybrid_program_has_device_selector_and_both_graph_layouts() {
        let p = adaptive_sparse_dense_step(
            "fin", "fout", "count", "offs", "tgts", "kinds", "adj", 64, 7, 1, 25,
        );
        assert_eq!(p.workgroup_size, ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(
            names,
            vec!["fin", "fout", "count", "offs", "tgts", "kinds", "adj"]
        );
        let find = |name: &str| p.buffers.iter().find(|b| b.name() == name).unwrap().count;
        let words = bitset_words(64);
        assert_eq!(find("fin"), words);
        assert_eq!(find("fout"), words);
        assert_eq!(find("count"), 1);
        assert_eq!(find("offs"), 65);
        assert_eq!(find("tgts"), 7);
        assert_eq!(find("kinds"), 7);
        assert_eq!(find("adj"), 64 * words);
    }

    #[test]
    fn emitted_hybrid_program_sparse_branch_probes_frontier_in_and_expands_to_frontier_out() {
        let p = adaptive_sparse_dense_step(
            "fin", "fout", "count", "offs", "tgts", "kinds", "adj", 64, 7, 1, 25,
        );
        let rendered = format!("{p:?}");
        assert!(rendered.contains("fin"), "must probe input frontier in sparse branch");
        assert!(rendered.contains("fout"), "must expand out-edges into output frontier");
        assert!(rendered.contains("count"), "must load popcount for selector");
    }
}
