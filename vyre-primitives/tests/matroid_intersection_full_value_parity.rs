//! Value parity for `math::matroid_intersection_full`: the IR PROGRAM, run through
//! `reference_eval`, must match its own `cpu_ref` (one Edmonds augmentation) on the updated
//! `set_x`.
//!
//! WHY: matroid_intersection_full is unregistered and its only tests exercise `cpu_ref`; the
//! actual augmenting-path BFS IR was never run through reference_eval (no validity OR value
//! check) (the same gap the union_find find-walk and tensor_scc seed-mask bugs fell through).
//!
//! The algorithm is a SINGLE-THREADED Edmonds augmentation: a level-synchronous BFS over the
//! exchange graph, then a non-idempotent `set_x[node] = 1 - set_x[node]` toggle along the chosen
//! augmenting path. Two properties are locked here:
//!
//! 1. DISPATCH-GRID INVARIANCE. The reference interpreter infers its grid from buffer SHAPES, so
//!    an `n`-element buffer spawns `n` invocations, and a real GPU dispatched with more than one
//!    invocation behaves the same. An unguarded serial body would have every lane redundantly run
//!    the whole program, and the non-idempotent toggle would RACE (two lanes flip the same slot).
//!    The builder guards the entire body to `InvocationId == 0` so exactly one lane executes it.
//! 2. AUGMENTING-PATH PARITY. When several augmenting paths exist the choice must be deterministic
//!    and IDENTICAL between IR and `cpu_ref`. A FIFO queue is a sequential-only construct the
//!    bitmap GPU kernel cannot realize, so both sides share one level-synchronous rule:
//!    `parent[v]` = lowest-id previous-level predecessor, and the path ends at the MINIMUM-id sink
//!    on the EARLIEST BFS level (level 0 = the source set, i.e. a node that is both source and
//!    sink is a length-0 path).
#![cfg(all(feature = "all-lego", feature = "cpu-parity"))]

use vyre_primitives::math::matroid_intersection_full::{cpu_ref, matroid_intersection_full};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Run one augmentation of the IR and return the updated `set_x`. `min_dispatch` forces a grid
/// FLOOR: passing a value above `n` proves the lane-0 guard holds the result invariant to how many
/// invocations the dispatch spawns (an unguarded serial kernel would race and diverge here).
fn run_ir_dispatch(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: u32,
    min_dispatch: u32,
) -> Vec<u32> {
    let program = matroid_intersection_full(
        "exchange_adj",
        "sources",
        "sinks",
        "set_x",
        "parent",
        "frontier",
        "next_frontier",
        "visited",
        "any_change",
        "path_out",
        "path_len",
        n,
        1, // max_augmentations = 1, matching cpu_ref's single Edmonds augmentation
    );
    let zeros_n = vec![0u32; n as usize];
    let zero1 = vec![0u32];
    // Buffer declaration order: exchange_adj(0), sources(1), sinks(2), set_x(3), parent(4),
    // frontier(5), next_frontier(6), visited(7), any_change(8), path_out(9), path_len(10),
    // target_node_buf(11, builder-internal). All scratch/output seeded to zero.
    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(exchange_adj)),
            Value::from(pack(sources)),
            Value::from(pack(sinks)),
            Value::from(pack(set_x)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zero1)),
        ],
        min_dispatch,
    )
    .expect("matroid_intersection_full reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "set_x")
        .expect("Fix: matroid_intersection_full must declare output `set_x`");
    unpack(&outputs[index].to_bytes())[..n as usize].to_vec()
}

/// Run one augmentation at the default (buffer-inferred) grid.
fn run_ir(exchange_adj: &[u32], sources: &[u32], sinks: &[u32], set_x: &[u32], n: u32) -> Vec<u32> {
    run_ir_dispatch(exchange_adj, sources, sinks, set_x, n, 0)
}

#[test]
fn matroid_ir_matches_cpu_ref_one_augmentation() {
    let mut state = 0x1357_9BDFu32;
    let mut augmenting_cases = 0u32;
    for case in 0..400u32 {
        let n = 3 + (next_u32(&mut state) % 5); // 3..=7
        let nn = (n * n) as usize;
        let exchange_adj: Vec<u32> = (0..nn).map(|_| next_u32(&mut state) % 100 / 60).collect(); // ~40% ones
        let sources: Vec<u32> = (0..n).map(|_| next_u32(&mut state) % 100 / 70).collect(); // ~30%
        let sinks: Vec<u32> = (0..n).map(|_| next_u32(&mut state) % 100 / 70).collect();
        let set_x: Vec<u32> = (0..n).map(|_| next_u32(&mut state) % 2).collect();

        let ir = run_ir(&exchange_adj, &sources, &sinks, &set_x, n);
        let cpu = cpu_ref(&exchange_adj, &sources, &sinks, &set_x, n as usize);
        if ir != set_x {
            augmenting_cases += 1;
        }
        assert_eq!(
            ir, cpu,
            "case {case}: matroid IR set_x {ir:?} != cpu_ref {cpu:?} \
             (n={n}, adj={exchange_adj:?}, sources={sources:?}, sinks={sinks:?}, set_x={set_x:?})"
        );
    }
    // Guard against a vacuous pass: a meaningful fraction of cases must actually augment (toggle
    // set_x), otherwise the differential is only exercising the no-path no-op branch.
    assert!(
        augmenting_cases > 20,
        "only {augmenting_cases}/400 cases augmented, the differential is not exercising the \
         augmenting-path toggle; strengthen the input distribution"
    );
}

#[test]
fn matroid_ir_matches_cpu_ref_direct_source_sink_edge() {
    // Minimal augmenting path: node 0 is a source, node 1 is a sink, edge 0->1. One augmentation
    // toggles set_x along [1,0]: both flip. set_x starts all-zero.
    let n = 2u32;
    let exchange_adj = vec![0u32, 1, 0, 0]; // 0->1
    let sources = vec![1u32, 0];
    let sinks = vec![0u32, 1];
    let set_x = vec![0u32, 0];
    let ir = run_ir(&exchange_adj, &sources, &sinks, &set_x, n);
    let cpu = cpu_ref(&exchange_adj, &sources, &sinks, &set_x, n as usize);
    assert_eq!(cpu, vec![1, 1], "sanity: cpu_ref toggles both path nodes");
    assert_eq!(
        ir, cpu,
        "matroid IR must toggle the augmenting path like cpu_ref"
    );
}

#[test]
fn matroid_ir_no_source_is_noop() {
    // No sources => no BFS => set_x unchanged.
    let n = 3u32;
    let exchange_adj = vec![0u32; 9];
    let sources = vec![0u32; 3];
    let sinks = vec![1u32, 1, 1];
    let set_x = vec![1u32, 0, 1];
    let ir = run_ir(&exchange_adj, &sources, &sinks, &set_x, n);
    assert_eq!(ir, set_x, "no source => set_x unchanged");
    assert_eq!(
        ir,
        cpu_ref(&exchange_adj, &sources, &sinks, &set_x, n as usize)
    );
}

#[test]
fn matroid_ir_is_invariant_to_dispatch_grid_size() {
    // REGRESSION for the lane-0 guard: the serial Edmonds body must produce the SAME set_x no
    // matter how many invocations the dispatch spawns. Before the guard, forcing a grid far larger
    // than `n` made every extra lane redundantly re-run the whole program, and the non-idempotent
    // `set_x = 1 - set_x` toggle raced to a wrong result. With the guard, only invocation 0 runs.
    let n = 2u32;
    let exchange_adj = vec![0u32, 1, 0, 0]; // 0->1
    let sources = vec![1u32, 0];
    let sinks = vec![0u32, 1];
    let set_x = vec![0u32, 0];
    let cpu = cpu_ref(&exchange_adj, &sources, &sinks, &set_x, n as usize);
    assert_eq!(
        cpu,
        vec![1, 1],
        "sanity: this case augments (both slots flip)"
    );
    // Sweep grid floors from the natural size up to 256 lanes: parity must hold at every size.
    for floor in [0u32, 1, 2, 4, 16, 64, 256] {
        let ir = run_ir_dispatch(&exchange_adj, &sources, &sinks, &set_x, n, floor);
        assert_eq!(
            ir, cpu,
            "dispatch floor {floor}: matroid IR set_x {ir:?} != cpu_ref {cpu:?}, the lane-0 \
             guard must make the serial kernel invariant to the dispatch grid size"
        );
    }
}
