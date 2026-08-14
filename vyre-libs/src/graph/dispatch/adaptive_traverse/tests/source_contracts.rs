//! The adaptive traversal release path dispatches the Programs that
//! `vyre-primitives` builds, not a local re-implementation of them.
//!
//! This file used to prove that by `include_str!`ing `upload.rs` and
//! `resident_steps.rs` and asserting the text contained
//! `primitive_adaptive_sparse_dense_step(`. That freezes a spelling, not a
//! contract: renaming the import alias broke the test while a genuine fork of
//! the traversal kernel under the same name would have passed it. The file was
//! also never declared in `tests/mod.rs`, so it never compiled and could not
//! fail on anything.
//!
//! The contract is now asserted where it is observable: drive the real resident
//! step against the recording dispatcher, take the wire fingerprint of every
//! Program it launches, and require it to equal the fingerprint of the Program
//! the `vyre-primitives` builder produces for the same arguments. A local fork
//! changes the fingerprint even when it keeps the name.
//!
//! The queue-driven half of the family is pinned the same way, one level up, by
//! `graph::csr_frontier_queue_programs::tests::resident_family_ir_fingerprints_are_byte_identical`,
//! which covers the `adaptive` site across all twelve roles. Naming those
//! builders here would be a second copy of that table.

use super::super::*;
use super::recording_dispatcher::{traversal_graph, RecordingResidentDispatcher};
use vyre_primitives::graph::adaptive_traverse::{
    adaptive_four_russians_dense_step, adaptive_sparse_dense_step,
};

/// The sparse/dense resident step launches clear, popcount, then the traversal
/// Program, and the traversal Program is the primitive builder's output.
#[test]
fn the_sparse_dense_release_path_dispatches_the_primitive_traversal_program() {
    let dispatcher = RecordingResidentDispatcher::default();
    let graph = ResidentAdaptiveTraversalGraph {
        ..traversal_graph()
    };
    let mut scratch = AdaptiveTraversalResidentScratch::default();
    let mut frontier_out = Vec::new();

    adaptive_traverse_resident_graph_step_with_scratch_into(
        &dispatcher,
        &graph,
        &[1, 0],
        u32::MAX,
        25,
        &mut scratch,
        &mut frontier_out,
    )
    .expect("Fix: recording dispatcher should complete the sparse/dense resident sequence");

    let expected = adaptive_sparse_dense_step(
        "frontier_in",
        "frontier_out",
        "frontier_popcount",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "adj_rows_dense",
        graph.node_count,
        graph.edge_count,
        u32::MAX,
        25,
    )
    .fingerprint();

    let launched = dispatcher.last_step_programs();
    assert_eq!(
        launched.len(),
        3,
        "sparse/dense step must launch clear, popcount, and traverse"
    );
    assert_eq!(
        launched[2], expected,
        "the traversal Program the release path launches must be the one \
         vyre-primitives::graph::adaptive_traverse::adaptive_sparse_dense_step builds"
    );
}

/// The Four-Russians dense step launches exactly the primitive builder's
/// Program and nothing else.
#[test]
fn the_four_russians_release_path_dispatches_the_primitive_dense_program() {
    let dispatcher = RecordingResidentDispatcher::default();
    let graph = ResidentAdaptiveFourRussiansDenseGraph {
        node_count: 33,
        words: 2,
        layout_hash: 11,
        lut_handle: 203,
    };
    let mut scratch = AdaptiveTraversalResidentScratch::default();
    let mut frontier_out = Vec::new();

    adaptive_traverse_resident_graph_four_russians_dense_step_with_scratch_into(
        &dispatcher,
        &graph,
        &[1, 0],
        &mut scratch,
        &mut frontier_out,
    )
    .expect("Fix: recording dispatcher should complete the Four-Russians dense step");

    let expected = adaptive_four_russians_dense_step(
        "frontier_in",
        "four_russians_tile_lut",
        "frontier_out",
        graph.node_count,
    )
    .fingerprint();

    assert_eq!(
        dispatcher.last_step_programs(),
        vec![expected],
        "the Four-Russians dense step must launch exactly the Program \
         vyre-primitives::graph::adaptive_traverse::adaptive_four_russians_dense_step builds"
    );
}
