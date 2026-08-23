//! Launch-geometry and dispatch-cut contracts for `math::scallop_join`.
//!
//! Two properties the builder must hold at every `w`, neither of which the
//! bit-exact parity test in `scallop_join_ir_parity.rs` can observe because it
//! runs a single-word 2x2 matrix:
//!
//! 1. **The grid covers every word.** A lane owns one relation CELL and walks
//!    the `w` contiguous `u32` words of that cell, so a grid sized for `n * n`
//!    lanes still covers `n * n * w` words. A mapping that sized the grid for
//!    cells and then indexed words per lane would leave `w - 1` of every `w`
//!    words unwritten, and an unwritten word keeps its seed value. The test
//!    drives `w = 8, n = 16` (2048 words) through the reference evaluator and
//!    requires every word to have moved off its seed and to match the oracle.
//!
//! 2. **A grid-sync fence is a dispatch cut, not a barrier instruction.** Above
//!    one workgroup of cells the builder emits `MemoryOrdering::GridSync`
//!    fences. No shading language has that instruction, so the fences must be
//!    hoistable to dispatch level and consumed by
//!    `grid_sync_split::split_on_grid_sync`. A fence that survives into a
//!    segment reaches an emitter that refuses it, and a fence nested in a loop
//!    cannot be cut at all.
#![cfg(all(feature = "math-kernels", feature = "fixpoint"))]

use vyre_foundation::ir::{MemoryOrdering, Node};
use vyre_foundation::transform::grid_sync_split::{
    contains_grid_sync, entry_sequence, loop_nested_grid_sync, split_on_grid_sync,
};
use vyre_libs::math::scallop_join::{
    scallop_join, scallop_join_dispatch_grid, SCALLOP_JOIN_WORKGROUP_SIZE,
};
use vyre_reference::composition_witness::scallop_join_fixpoint_witness as cpu_ref;
use vyre_reference::value::Value;

fn count_grid_sync(nodes: &[Node]) -> usize {
    let mut total = 0;
    let mut stack: Vec<&Node> = nodes.iter().collect();
    while let Some(node) = stack.pop() {
        if matches!(
            node,
            Node::LogicalBarrier {
                ordering: MemoryOrdering::GridSync
            }
        ) {
            total += 1;
        }
        for body in vyre_foundation::visit::child_bodies(node) {
            stack.extend(body);
        }
    }
    total
}

fn pack(data: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(data))
}

fn words(value: &Value) -> Vec<u32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Lanes the dispatch grid launches for an `n`-by-`n` relation.
fn launched_lanes(n: u32) -> u64 {
    let grid = scallop_join_dispatch_grid(n);
    u64::from(grid[0])
        * u64::from(grid[1])
        * u64::from(grid[2])
        * u64::from(SCALLOP_JOIN_WORKGROUP_SIZE[0])
        * u64::from(SCALLOP_JOIN_WORKGROUP_SIZE[1])
        * u64::from(SCALLOP_JOIN_WORKGROUP_SIZE[2])
}

#[test]
fn wide_grid_launches_one_lane_per_cell() {
    for n in [1u32, 2, 15, 16, 17, 64, 256] {
        let cells = u64::from(n) * u64::from(n);
        assert!(
            launched_lanes(n) >= cells,
            "n={n}: the grid launches {} lanes for {cells} cells, so some cell has no owner",
            launched_lanes(n)
        );
    }
}

#[test]
fn wide_fixpoint_writes_every_word_of_every_cell() {
    let n = 16u32;
    let w = 8u32;
    let cells = (n * n) as usize;
    let total_words = cells * w as usize;
    let max_iterations = 4u32;

    // One distinct low bit per cell, replicated across the cell's words. Every
    // cell is non-zero, so the zero-absorbing combine never short-circuits.
    let seed: Vec<u32> = (0..total_words)
        .map(|i| 1u32 << ((i / w as usize) % 31))
        .collect();
    // Every join rule cell carries every bit, so one transfer step drives each
    // reachable word to all-ones and no word can match its seed afterwards.
    let join_rules = vec![u32::MAX; total_words];

    let program = scallop_join(
        "state",
        "next",
        "join_rules",
        "changed",
        n,
        w,
        max_iterations,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(&seed),
            pack(&vec![0u32; total_words]),
            pack(&[0u32]),
            pack(&join_rules),
        ],
    )
    .expect("wide scallop_join reference evaluation must succeed");
    let got = words(&outputs[0]);

    let (want, _iterations) = cpu_ref(&seed, &join_rules, n, w, max_iterations);
    assert_eq!(
        want.len(),
        total_words,
        "the oracle must cover all {total_words} words"
    );

    // Non-vacuity, per word: the oracle moves every one of the 2048 words off
    // its seed, so an uncovered word is visible as a stale seed value rather
    // than hidden behind a cell that happened not to change.
    let unmoved: Vec<usize> = (0..total_words).filter(|&i| want[i] == seed[i]).collect();
    assert!(
        unmoved.is_empty(),
        "oracle left {} words at their seed, so the coverage check is vacuous there: {:?}",
        unmoved.len(),
        &unmoved[..unmoved.len().min(8)]
    );

    let stale: Vec<usize> = (0..total_words).filter(|&i| got[i] == seed[i]).collect();
    assert!(
        stale.is_empty(),
        "n={n} w={w}: {} of {total_words} words were never written, first at {:?}: the grid covers cells but not the words inside them",
        stale.len(),
        &stale[..stale.len().min(8)]
    );
    assert_eq!(
        got, want,
        "n={n} w={w}: wide lineage fixpoint must match the host oracle word for word"
    );
}

#[test]
fn grid_sync_fences_split_into_dispatch_segments() {
    // 1024 cells is four workgroups, so the builder takes the grid-sync path.
    let n = 32u32;
    let max_iterations = 3u32;
    for w in [1u32, 4] {
        let program = scallop_join(
            "state",
            "next",
            "join_rules",
            "changed",
            n,
            w,
            max_iterations,
        );
        assert!(
            contains_grid_sync(&program),
            "n={n} w={w}: a matrix wider than one workgroup must fence across the grid"
        );
        assert_eq!(
            loop_nested_grid_sync(&program),
            None,
            "n={n} w={w}: a loop-nested fence cannot be cut, so the iterations must be unrolled"
        );

        let fences = count_grid_sync(entry_sequence(&program));
        assert!(
            fences > 0,
            "n={n} w={w}: the fence count must be observable at dispatch level, not buried in a wrapper"
        );

        let segments = split_on_grid_sync(&program).expect("every fence must be hoistable");
        assert_eq!(
            segments.len(),
            fences + 1,
            "n={n} w={w}: {fences} dispatch-level fences must cut the program into {} segments",
            fences + 1
        );
        for (index, segment) in segments.iter().enumerate() {
            assert!(
                !contains_grid_sync(segment),
                "n={n} w={w}: segment {index} still carries a grid-sync fence, which no shading language can lower"
            );
        }
    }
}

#[test]
fn block_local_fixpoint_emits_no_grid_sync() {
    // 256 cells is exactly one workgroup: convergence is a workgroup barrier
    // loop, and cutting it into dispatches would be pure overhead.
    let program = scallop_join("state", "next", "join_rules", "changed", 16, 8, 4);
    assert!(
        !contains_grid_sync(&program),
        "a block-local matrix must converge inside one dispatch"
    );
}
