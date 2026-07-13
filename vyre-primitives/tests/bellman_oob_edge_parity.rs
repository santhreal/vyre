//! GPU-IR vs CPU-ref parity for `math::bellman_shortest_path` on OUT-OF-RANGE edges.
//!
//! The Bellman-Ford relaxation loads `dist[u]` and atomic-min-writes `next_dist[v]`
//! where `u = src[e]` and `v = dst[e]` are DATA (edge endpoints); nothing validates
//! them `< n_nodes`. The CPU reference SKIPS any edge with an out-of-range endpoint
//! (`if u >= n || v >= n { continue }`). The GPU IR must gate the relaxation on the
//! same `u < n_nodes && v < n_nodes` bound, otherwise (bug fixed 2026-07-10) an edge
//! with an OOB SOURCE loads `dist[u]` out of bounds (0 in the interpreter, UB on real
//! hardware), and since `0 != u32::MAX` it SPURIOUSLY relaxes a valid `next_dist[v]`,
//! diverging from the CPU ref; an edge with an OOB DEST additionally OOB atomic-WRITES
//! `next_dist[v]` (memory corruption on real hardware). This is the gather/test_bit
//! parity class. Pins the fix against `reference_eval`.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use vyre_primitives::math::bellman_shortest_path::{bellman_shortest_path, cpu_ref};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

const INF: u32 = u32::MAX;

fn gpu_dist(
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    n_nodes: u32,
    max_iterations: u32,
) -> Vec<u32> {
    let program = bellman_shortest_path(
        "src",
        "dst",
        "weight",
        "dist",
        "next_dist",
        "changed",
        n_nodes,
        src.len() as u32,
        max_iterations,
    );
    // Buffer binding order: dist(0), next_dist(1), changed(2), src(3), dst(4), weight(5).
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(dist_init)),
            Value::from(pack(dist_init)), // next_dist starts as a copy of dist
            Value::from(pack(&[0u32])),   // changed
            Value::from(pack(src)),
            Value::from(pack(dst)),
            Value::from(pack(weight)),
        ],
    )
    .expect("bellman reference evaluation must succeed");
    unpack(&outputs[0].to_bytes()) // final dist
}

#[test]
fn clean_single_hop_graph_matches_cpu_ref() {
    // Single-hop from the source (0->1, 0->2): converges in one relaxation round, so
    // this sanity check is insensitive to how many fixpoint rounds a single
    // reference_eval dispatch performs (multi-hop chains need host re-dispatch, which
    // reference_eval does not model (orthogonal to the OOB-edge gate under test)).
    let src = [0u32, 0];
    let dst = [1u32, 2];
    let weight = [5u32, 9];
    let dist_init = [0u32, INF, INF];
    let (cpu, _) = cpu_ref(&src, &dst, &weight, &dist_init, 3, 4);
    assert_eq!(
        cpu,
        vec![0, 5, 9],
        "cpu_ref sanity: node 0 relaxes 1 and 2 directly"
    );
    assert_eq!(
        gpu_dist(&src, &dst, &weight, &dist_init, 3, 4),
        cpu,
        "clean single-hop GPU-IR must match cpu_ref"
    );
}

#[test]
fn out_of_range_source_edge_does_not_spuriously_relax() {
    // Single-hop graph (converges in one round). Valid edge 0->1 (w=5) sets dist[1]=5.
    // Edge 7->1 has an OUT-OF-RANGE source (7 >= n_nodes==3): the cpu_ref skips it. The
    // pre-fix GPU loaded dist[7] (OOB -> 0 in the interpreter) and, since 0 != INF,
    // relaxed next_dist[1] to 0+1 = 1 (a spurious shortcut. The fix gates it out).
    let src = [0u32, 7];
    let dst = [1u32, 1];
    let weight = [5u32, 1];
    let dist_init = [0u32, INF, INF];
    let (cpu, _) = cpu_ref(&src, &dst, &weight, &dist_init, 3, 4);
    assert_eq!(
        cpu,
        vec![0, 5, INF],
        "cpu_ref skips the OOB-source edge -> dist[1]=5"
    );
    let gpu = gpu_dist(&src, &dst, &weight, &dist_init, 3, 4);
    assert_eq!(
        gpu, cpu,
        "OOB-source edge must be skipped (no spurious relaxation from an OOB dist load): \
         GPU={gpu:?} cpu={cpu:?}"
    );
    assert_ne!(
        gpu,
        vec![0, 1, INF],
        "must NOT show the pre-fix spurious shortcut dist[1]=1 from the OOB dist[7] load"
    );
}

#[test]
fn out_of_range_dest_edge_matches_cpu_ref() {
    // Single-hop graph. Edge 0 -> 9 has an OUT-OF-RANGE dest. cpu_ref skips it; the
    // fixed GPU also skips (no OOB atomic write to next_dist[9]). In reference_eval an
    // OOB store is dropped, so this asserts the contract; on real hardware the gate is
    // what prevents the out-of-bounds atomic write.
    let src = [0u32, 0];
    let dst = [1u32, 9];
    let weight = [5u32, 2];
    let dist_init = [0u32, INF, INF];
    let (cpu, _) = cpu_ref(&src, &dst, &weight, &dist_init, 3, 4);
    assert_eq!(cpu, vec![0, 5, INF]);
    assert_eq!(
        gpu_dist(&src, &dst, &weight, &dist_init, 3, 4),
        cpu,
        "OOB-dest edge must be skipped, matching cpu_ref"
    );
}
