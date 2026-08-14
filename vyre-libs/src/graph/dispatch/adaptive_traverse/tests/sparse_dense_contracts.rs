use super::super::*;
use super::recording_dispatcher::{traversal_graph, RecordingResidentDispatcher};

#[test]
fn sparse_dense_resident_step_does_not_upload_popcount_zero_seed() {
    let dispatcher = RecordingResidentDispatcher::default();
    let graph = ResidentAdaptiveTraversalGraph {
        node_count: 1,
        edge_count: 0,
        max_row_degree: 0,
        words: 1,
        ..traversal_graph()
    };
    let mut scratch = AdaptiveTraversalResidentScratch::default();
    let mut frontier_out = Vec::new();

    adaptive_traverse_resident_graph_step_with_scratch_into(
        &dispatcher,
        &graph,
        &[1],
        u32::MAX,
        25,
        &mut scratch,
        &mut frontier_out,
    )
    .expect("Fix: recording dispatcher should complete resident sparse/dense sequence");

    let scratch_handles = scratch
        .handles
        .expect("Fix: sparse/dense resident step must allocate frontier/popcount handles");
    assert_eq!(
        dispatcher.last_upload_handles(),
        vec![scratch_handles[0]],
        "sparse/dense traversal must upload only frontier input; output and popcount are initialized on device"
    );
    assert_eq!(frontier_out, vec![0]);
}

#[test]
fn sparse_dense_resident_program_cache_reuses_same_shape_graphs() {
    let dispatcher = RecordingResidentDispatcher::default();
    let graph_a = ResidentAdaptiveTraversalGraph {
        ..traversal_graph()
    };
    let graph_b = ResidentAdaptiveTraversalGraph {
        layout_hash: 99,
        handles: [201, 202, 203, 204],
        ..traversal_graph()
    };
    let mut scratch = AdaptiveTraversalResidentScratch::default();
    let mut frontier_out = Vec::new();

    adaptive_traverse_resident_graph_step_with_scratch_into(
        &dispatcher,
        &graph_a,
        &[1, 0],
        u32::MAX,
        25,
        &mut scratch,
        &mut frontier_out,
    )
    .expect("Fix: first adaptive resident step should dispatch");
    adaptive_traverse_resident_graph_step_with_scratch_into(
        &dispatcher,
        &graph_b,
        &[1, 0],
        u32::MAX,
        25,
        &mut scratch,
        &mut frontier_out,
    )
    .expect("Fix: second adaptive resident step should dispatch");

    assert_eq!(
        scratch.plan_cache_snapshot(),
        AdaptiveTraversalPlanCacheSnapshot {
            entries: 3,
            hits: 3,
            misses: 3,
        },
        "adaptive resident programs must be cached by shape/options, not resident graph contents"
    );
}
