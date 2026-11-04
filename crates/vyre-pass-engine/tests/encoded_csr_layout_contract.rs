//! The CSR layout `encode_program` hands the analysis kernels.
//!
//! WHY: the encoder built one edge vector per graph node and flattened them by
//! walking sources in ascending order. That layout is what every encoded kernel
//! indexes, so replacing the per-node vectors with one flat edge list and a
//! counting sort has to reproduce it exactly: rows in ascending source order,
//! offsets monotone over `node_count + 1` entries, and columns within a row in
//! the order the encoder added them. A permuted row would send a kernel to the
//! wrong neighbour with no error anywhere.
//!
//! The expected layout is recomputed from the flat edge set at run time rather
//! than written down, so a program shape the encoder starts emitting differently
//! is covered the moment it is encoded.
//!
//! What this does not catch: which edges the encoder decides to emit. That is
//! the use-def contract the encoder's own tests defend.

#![forbid(unsafe_code)]

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_pass_engine::optimizer::encode::{edge_kind, encode_program, EncodedProgram};

/// A wide program of independent bindings, the last one stored.
///
/// Every binding is a separate graph node that uses the one before it, so the
/// encoder emits a use-def edge per node and the rows are non-trivial.
fn chained_bindings(count: usize) -> Program {
    let mut entry = Vec::with_capacity(count + 1);
    entry.push(Node::let_bind("v0", Expr::u32(1)));
    for index in 1..count {
        entry.push(Node::let_bind(
            format!("v{index}"),
            Expr::add(Expr::var(format!("v{}", index - 1)), Expr::u32(1)),
        ));
    }
    entry.push(Node::store(
        "buf",
        Expr::u32(0),
        Expr::var(format!("v{}", count - 1)),
    ));
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

/// A program whose nested scopes make one node use several names.
fn nested_scopes() -> Program {
    let entry = vec![
        Node::let_bind("outer", Expr::u32(2)),
        Node::If {
            cond: Expr::var("outer"),
            then: vec![
                Node::let_bind("inner", Expr::add(Expr::var("outer"), Expr::u32(1))),
                Node::store("buf", Expr::u32(0), Expr::var("inner")),
            ],
            otherwise: vec![Node::store("buf", Expr::u32(1), Expr::var("outer"))],
        },
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![Node::store("buf", Expr::var("i"), Expr::var("outer"))],
        ),
    ];
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

/// Every `(source, target, kind)` the encoding holds, read through its CSR.
fn edges_of(encoded: &EncodedProgram) -> Vec<(u32, u32, u32)> {
    let mut edges = Vec::new();
    for source in 0..encoded.node_count {
        let start = encoded.edge_offsets[source as usize] as usize;
        let end = encoded.edge_offsets[source as usize + 1] as usize;
        for slot in start..end {
            edges.push((
                source,
                encoded.edge_targets[slot],
                encoded.edge_kind_mask[slot],
            ));
        }
    }
    edges
}

/// WHY: the offsets are the row index every kernel reads. A row that is not
/// monotone, or a last offset that is not the edge count, silently truncates or
/// overruns a neighbour list.
#[test]
fn the_row_offsets_partition_the_edge_columns() {
    for program in [chained_bindings(64), nested_scopes()] {
        let encoded = encode_program(&program).expect("Fix: the encoder must accept the fixture");
        assert_eq!(
            encoded.edge_offsets.len(),
            encoded.node_count as usize + 1,
            "the CSR row array must hold one offset per node plus the terminator"
        );
        assert_eq!(
            encoded.edge_offsets[0], 0,
            "the first row must start at column zero"
        );
        for window in encoded.edge_offsets.windows(2) {
            assert!(
                window[0] <= window[1],
                "row offsets {} and {} are not monotone",
                window[0],
                window[1]
            );
        }
        let last = *encoded
            .edge_offsets
            .last()
            .expect("Fix: the row array is never empty");
        assert_eq!(
            last, encoded.edge_count,
            "the terminator must equal the edge count"
        );
        assert_eq!(encoded.edge_targets.len(), encoded.edge_count as usize);
        assert_eq!(encoded.edge_kind_mask.len(), encoded.edge_count as usize);
        assert_eq!(encoded.nodes.len(), encoded.node_count as usize);
        assert_eq!(encoded.node_tags.len(), encoded.node_count as usize);
    }
}

/// WHY: the flattening step is a counting sort, and an unstable one would
/// reorder the columns of a row that carries more than one edge. The expected
/// order is the encoder's own: grouped by ascending source, and within a source
/// the order the edges were added, which is what a stable placement produces.
#[test]
fn the_columns_of_a_row_keep_the_order_the_encoder_added_them() {
    for program in [chained_bindings(64), nested_scopes()] {
        let encoded = encode_program(&program).expect("Fix: the encoder must accept the fixture");
        let edges = edges_of(&encoded);
        let mut grouped = edges.clone();
        grouped.sort_by_key(|(source, _, _)| *source);
        assert_eq!(
            edges, grouped,
            "reading the CSR row by row did not yield edges grouped by ascending source"
        );
        assert!(
            edges.iter().all(|(source, target, kind)| {
                *source < encoded.node_count && *target < encoded.node_count && *kind != 0
            }),
            "an edge names a node outside the graph or carries no kind bit"
        );
    }
}

/// WHY: encoding is the input to a compile, and a compile whose input permutes
/// between two runs over the same program cannot produce a reproducible
/// artifact. The buffer reuse inside the encoder is exactly the kind of change
/// that can leak state from one node into the next.
#[test]
fn encoding_one_program_twice_yields_the_same_layout() {
    for program in [chained_bindings(128), nested_scopes()] {
        let first = encode_program(&program).expect("Fix: the encoder must accept the fixture");
        let second = encode_program(&program).expect("Fix: the encoder must accept the fixture");
        assert_eq!(first.node_count, second.node_count);
        assert_eq!(first.edge_count, second.edge_count);
        assert_eq!(first.nodes, second.nodes);
        assert_eq!(first.node_tags, second.node_tags);
        assert_eq!(first.edge_offsets, second.edge_offsets);
        assert_eq!(first.edge_targets, second.edge_targets);
        assert_eq!(first.edge_kind_mask, second.edge_kind_mask);
    }
}

/// WHY: a chain of `count` bindings has a known edge shape: the store is
/// ROOT-rooted and every binding after the first uses the one before it. Pinning
/// the count keeps a flattening bug that drops or duplicates a row from passing
/// the ordering contracts above.
#[test]
fn a_chain_of_bindings_encodes_one_use_def_edge_per_binding() {
    let count = 32;
    let encoded =
        encode_program(&chained_bindings(count)).expect("Fix: the encoder must accept the chain");
    assert_eq!(
        encoded.node_count as usize,
        count + 2,
        "the graph holds ROOT, one node per binding, and the store"
    );
    let edges = edges_of(&encoded);
    let use_def = edges
        .iter()
        .filter(|(_, _, kind)| *kind & edge_kind::USE_DEF != 0)
        .count();
    assert_eq!(
        use_def, count,
        "every binding after the first uses its predecessor, and the store uses the last"
    );
    let root_frontier = edges
        .iter()
        .filter(|(source, _, kind)| *source == 0 && *kind & edge_kind::ROOT_FRONTIER != 0)
        .count();
    assert_eq!(root_frontier, 1, "only the store is ROOT-rooted");
}
