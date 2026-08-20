use crate::graph::csr_closure_inputs::{graphs, CsrClosureInputs, CsrGraphView};
use crate::graph::persistent_bfs::{cpu_ref, try_cpu_ref};

fn forward_reach(graph: CsrGraphView<'_>, seed: &[u32], max_iters: u32) -> Vec<u32> {
    let (out, _) = cpu_ref(CsrClosureInputs::allow_all(graph, max_iters), seed);
    out
}
#[test]
fn checked_reference_surfaces_bad_frontier_width() {
    let offsets = vec![0u32; 65];
    let err = try_cpu_ref(
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 64,
                edge_offsets: &offsets,
                edge_targets: &[],
                edge_kind_mask: &[],
            },
            1,
        ),
        &[1],
    )
    .expect_err("short persistent BFS seed frontier must fail through substrate wrapper");

    assert!(
        err.contains("frontier"),
        "Fix: persistent BFS checked wrapper must preserve primitive frontier diagnostics, got: {err}"
    );
}

#[test]
fn expand_chain_saturates() {
    let (out, changed) = cpu_ref(CsrClosureInputs::allow_all(graphs::CHAIN_4, 8), &[0b0001]);
    assert_eq!(out, vec![0b1111]);
    assert_eq!(changed, 1);
}

#[test]
fn empty_seed_yields_empty_with_no_change() {
    let (out, changed) = cpu_ref(CsrClosureInputs::allow_all(graphs::CHAIN_4, 4), &[0u32]);
    assert_eq!(out, vec![0u32]);
    assert_eq!(changed, 0);
}

#[test]
fn saturated_seed_reports_no_change() {
    let (out, changed) = cpu_ref(CsrClosureInputs::allow_all(graphs::CHAIN_4, 4), &[0b1111]);
    assert_eq!(out, vec![0b1111]);
    assert_eq!(changed, 0);
}

#[test]
fn max_iters_bound_honored() {
    let (out, _) = cpu_ref(CsrClosureInputs::allow_all(graphs::CHAIN_4, 1), &[0b0001]);
    assert_eq!(out[0] & 0b1111, 0b0011);
}

#[test]
fn allow_mask_filters_all_edges() {
    let (out, changed) = cpu_ref(
        CsrClosureInputs {
            graph: graphs::CHAIN_4,
            allow_mask: 0b0010,
            max_iters: 4,
        },
        &[0b0001],
    );
    assert_eq!(out, vec![0b0001]);
    assert_eq!(changed, 0);
}

#[test]
fn forward_reach_saturates_chain() {
    let out = forward_reach(graphs::CHAIN_4, &[0b0001], u32::MAX);
    assert_eq!(out, vec![0b1111]);
}

#[test]
fn self_loop_terminates() {
    let off = vec![0, 1, 1];
    let tgt = vec![0];
    let msk = vec![1];
    let (out, _) = cpu_ref(
        CsrClosureInputs::allow_all(
            CsrGraphView {
                node_count: 2,
                edge_offsets: &off,
                edge_targets: &tgt,
                edge_kind_mask: &msk,
            },
            50,
        ),
        &[0b01],
    );
    assert_eq!(out, vec![0b01]);
}
