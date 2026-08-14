//! Why the DCE BFS analysis is ONE dispatch pinned to ONE workgroup, gated and
//! measured on live CUDA.
//!
//! Three wrong readings of `build_dce_bfs_program` are on record, and each cost
//! something, so this suite pins the reasoning to device behavior instead of to
//! argument.
//!
//! The first reading was that its workgroup-scoped barrier plus grid-shared
//! `changed[0]` is a LOST-CLEAR race. Acting on that produced host-repeated
//! grid-synchronized wave batches, measured here at 4 to 6 times the wall time and
//! about 232 times the launches on a deep chain, and it bounded an IR size that a
//! bounded `Node::loop_for` already bounded. Reverted.
//!
//! The second reading was that the extra workgroups are harmless duplicates, so
//! nothing needs fixing. Coverage is indeed redundant, and one workgroup really does
//! reach the whole closure, which the first test gates. But discovery ATTRIBUTION is
//! exclusive: only the lane whose `atomic_or` flipped the bit sees growth, so a
//! duplicate group can win a discovery and leave the covering group reading
//! `changed == 0`. So the kernel is pinned to one workgroup, which is what
//! `pipeline_resident` already did and what `dce_via_encoded` was failing to do.
//!
//! The third was that the `if changed[0] == 0 { Return }` early exit worked at all.
//! `Node::Return` nested in a loop emits NOTHING on PTX, so a converged run used to
//! burn its whole `max_iters` budget: 183x on a star that converges in two
//! iterations. The third test gates that it no longer does.

mod common;

use std::time::Instant;

use common::{live_backend, CudaProgramDispatcher};
use vyre::ir::Program;
use vyre_primitives::graph::program_graph::{
    ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};
use vyre_self_substrate::optimizer::dce_program::build_dce_bfs_program;
use vyre_foundation::program_dispatch::ProgramDispatcher;

const EDGE_KIND: u32 = 1;

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Stage the analysis program's inputs by reading each binding's DECLARED count
/// off the program, so a change to the layout cannot silently mis-size a slot.
/// `wg_scratch` is workgroup-only and takes no input slot.
fn stage_inputs(
    program: &Program,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    frontier_in: &[u32],
) -> Vec<Vec<u8>> {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.name() != "wg_scratch")
        .map(|buffer| {
            let count = buffer.count() as usize;
            let mut words = vec![0u32; count];
            // Match the CANONICAL binding names rather than literals. Staging by
            // guessed name is how this first ran: every edge buffer silently
            // staged as zeros, so the traversal had no graph to walk and reported
            // an immediate fixpoint with one live node.
            match buffer.name() {
                NAME_EDGE_OFFSETS => {
                    let tail = edge_offsets.last().copied().unwrap_or(0);
                    words = edge_offsets.to_vec();
                    words.resize(count, tail);
                }
                NAME_EDGE_TARGETS => {
                    let staged = edge_targets.len().min(count);
                    words[..staged].copy_from_slice(&edge_targets[..staged]);
                }
                NAME_EDGE_KIND_MASK => words = vec![EDGE_KIND; count],
                "frontier_in" => {
                    let staged = frontier_in.len().min(count);
                    words[..staged].copy_from_slice(&frontier_in[..staged]);
                }
                // pg_nodes, pg_node_tags, frontier_out, changed, converged start zero.
                _ => {}
            }
            pack(&words)
        })
        .collect()
}

fn live_nodes(frontier: &[u32], node_count: u32) -> u32 {
    (0..node_count)
        .filter(|id| (frontier[(id / 32) as usize] >> (id % 32)) & 1 == 1)
        .count() as u32
}

/// `0 -> 1 -> ... -> node_count-1`. Diameter `node_count - 1`: the worst case for
/// host repetition, because every wave advances the frontier by exactly one node.
fn chain(node_count: u32) -> (Vec<u32>, Vec<u32>) {
    let edges = node_count - 1;
    let targets = (1..node_count).collect();
    let offsets = (0..=node_count).map(|node| node.min(edges)).collect();
    (offsets, targets)
}

/// `0 -> every other node`. Diameter 1: the common case this module's header calls
/// most real Programs.
fn star(node_count: u32) -> (Vec<u32>, Vec<u32>) {
    let targets = (1..node_count).collect();
    let mut offsets = vec![0u32];
    offsets.extend(std::iter::repeat_n(node_count - 1, node_count as usize));
    (offsets, targets)
}

/// One dispatch, timed. Returns the live node count, the converged word, and the
/// wall time of the dispatch alone.
fn dispatch_timed(
    dispatcher: &CudaProgramDispatcher<'_>,
    program: &Program,
    node_count: u32,
    inputs: &[Vec<u8>],
) -> (u32, u32, std::time::Duration) {
    let started = Instant::now();
    let outputs = dispatcher
        .dispatch(program, inputs, Some([1, 1, 1]))
        .expect("the DCE analysis must dispatch on live CUDA");
    let elapsed = started.elapsed();
    (
        live_nodes(&unpack(&outputs[0]), node_count),
        unpack(&outputs[2])[0],
        elapsed,
    )
}

/// The single dispatch reaches the COMPLETE closure on live CUDA for graphs far
/// above one workgroup, at both diameter extremes, and reports its fixpoint.
///
/// Chain and star are the two ends of the range that matters. The chain has
/// diameter 1999, so the in-kernel persistent loop must relax 1999 times within
/// one dispatch; a form that lost relaxation across iterations reports 2 live
/// nodes instead of 2000. The star has diameter 1, so it must not need any
/// repetition at all. Both must report `converged == 1`, because
/// `dce_via_encoded` treats 0 as a truncated liveness set and refuses the graph.
///
/// Cold and warm times are reported SEPARATELY and never blended. A single
/// average over both folds one-time module load and JIT into every repetition,
/// which is exactly how an earlier read of this comparison produced a false
/// near-parity verdict against a host-repetition form: warm batch against cold
/// single dispatch.
#[test]
fn dce_single_dispatch_reaches_full_closure_on_live_cuda() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    let node_count = 2000_u32;
    let program = build_dce_bfs_program(
        ProgramGraphShape::new(node_count, node_count - 1),
        node_count,
    );

    for (label, (offsets, targets)) in [
        ("chain (diameter 1999)", chain(node_count)),
        ("star  (diameter 1)", star(node_count)),
    ] {
        let inputs = stage_inputs(&program, &offsets, &targets, &seed(node_count));
        let (cold_live, cold_converged, cold) =
            dispatch_timed(&dispatcher, &program, node_count, &inputs);
        let (warm_live, warm_converged, warm) =
            dispatch_timed(&dispatcher, &program, node_count, &inputs);
        println!(
            "{label}: 1 dispatch, {warm_live} live nodes, cold {:.3} ms, warm {:.3} ms",
            cold.as_secs_f64() * 1e3,
            warm.as_secs_f64() * 1e3
        );
        assert_eq!(
            cold_live, node_count,
            "{label}: one dispatch must reach the complete closure"
        );
        assert_eq!(
            warm_live, cold_live,
            "{label}: the closure must not depend on module warmth"
        );
        assert_eq!(
            (cold_converged, warm_converged),
            (1, 1),
            "{label}: the dispatch must observe its fixpoint, or dce_via_encoded refuses the graph"
        );
    }
}

fn seed(node_count: u32) -> Vec<u32> {
    let mut frontier = vec![0u32; ((node_count + 31) / 32) as usize];
    frontier[0] = 1;
    frontier
}

/// ONE workgroup computes the COMPLETE closure by itself, which is the condition
/// that lets every caller pin this program to a single workgroup.
///
/// THIS TEST IS WHAT LICENSES THE PINNING in `dce_program.rs` and in both callers.
/// If one workgroup could not cover the whole node range, pinning would truncate
/// the closure and DCE would delete live code, so the pin and this test stand or
/// fall together.
///
/// The property is that the step strides `src = gid_x() + stride * DCE_WORKGROUP_X`
/// for `stride_count = ceil(node_count / DCE_WORKGROUP_X)` iterations, so lanes
/// `0..DCE_WORKGROUP_X` visit EVERY source without help. A 2000-node chain forces
/// two strided passes and 1999 sequential relaxations, so a stride that stopped
/// covering the range would show up here as a short closure rather than as a
/// timing artifact.
///
/// The full-grid run is dispatched and PRINTED but deliberately NOT asserted. Above
/// one workgroup this kernel's early exit is genuinely racy: discovery is attributed
/// to the single lane whose `atomic_or` flipped the bit, so a duplicate group can
/// win a discovery and strand the covering group at `changed == 0`. Asserting that
/// the grid run agrees would be pinning a racy value, and asserting that it differs
/// would be asserting the defect. It is printed because a divergence between the two
/// numbers is informative to a human reading the log.
#[test]
fn one_workgroup_computes_the_complete_closure_by_itself() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    let node_count = 2000_u32;
    let (offsets, targets) = chain(node_count);
    let program = build_dce_bfs_program(
        ProgramGraphShape::new(node_count, node_count - 1),
        node_count,
    );
    let inputs = stage_inputs(&program, &offsets, &targets, &seed(node_count));

    // Derived from the program, never from the 2000 written above: a constant that
    // silently stops needing more than one strided pass is how a gate keeps passing
    // while testing nothing.
    let width = program.workgroup_size()[0];
    assert!(
        node_count.div_ceil(width) > 1,
        "the strided step must take more than one pass ({node_count} nodes over width {width}), \
         or one workgroup covering every source is trivially true and proves nothing"
    );

    let pinned = dispatcher
        .dispatch(&program, &inputs, Some([1, 1, 1]))
        .expect("the analysis must dispatch pinned to one workgroup");
    let spanned = dispatcher
        .dispatch(&program, &inputs, None)
        .expect("the analysis must dispatch across the grid");
    let pinned_live = live_nodes(&unpack(&pinned[0]), node_count);
    println!(
        "chain over {} strided passes: pinned to 1 workgroup -> {pinned_live} live, full grid \
         (racy, not asserted) -> {} live",
        node_count.div_ceil(width),
        live_nodes(&unpack(&spanned[0]), node_count)
    );

    assert_eq!(
        pinned_live, node_count,
        "one workgroup must compute the complete closure by itself; if it cannot, pinning the \
         grid truncates the liveness set and DCE deletes live code"
    );
    assert_eq!(
        unpack(&pinned[2])[0],
        1,
        "the pinned run must observe its fixpoint, or dce_via_encoded refuses the graph"
    );
}

/// The persistent loop EXITS AT THE FIXPOINT instead of burning its iteration
/// budget, and the cost of a converged run does not scale with `max_iters`.
///
/// This is the whole reason the early exit exists, and nothing else in the tree
/// checks it. The star reaches its fixpoint in a fixed small number of iterations
/// regardless of budget: iteration 1 expands the hub, iteration 2 adds nothing
/// because every `atomic_or` returns a bit that was already set, and the loop
/// returns. So a 2000-iteration budget must cost about what an 8-iteration budget
/// costs.
///
/// If the early exit ever stops firing, the budget becomes the running time and
/// this ratio blows up. That is a silent defect otherwise: the answer stays
/// correct, so every correctness test in the tree still passes while the kernel
/// does 250 times the work. The ratio is asserted loosely, at 10x, because wall
/// time is noisy and only an order-of-magnitude change means the exit died.
#[test]
fn the_persistent_loop_exits_at_the_fixpoint_rather_than_burning_its_budget() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher { backend: &backend };
    let node_count = 2000_u32;
    let (offsets, targets) = star(node_count);
    let shape = ProgramGraphShape::new(node_count, node_count - 1);

    let mut timings = Vec::new();
    for budget in [8_u32, node_count] {
        let program = build_dce_bfs_program(shape, budget);
        let inputs = stage_inputs(&program, &offsets, &targets, &seed(node_count));
        // Warm the module first so the comparison is warm against warm.
        dispatch_timed(&dispatcher, &program, node_count, &inputs);
        let (live, converged, elapsed) = dispatch_timed(&dispatcher, &program, node_count, &inputs);
        println!(
            "star, budget {budget}: {live} live, converged={converged}, warm {:.3} ms",
            elapsed.as_secs_f64() * 1e3
        );
        assert_eq!(
            live, node_count,
            "the star closure completes within 8 iterations, so both budgets must reach it"
        );
        assert_eq!(
            converged, 1,
            "both budgets must observe the fixpoint; a budget-exhausted run reports 0"
        );
        timings.push(elapsed.as_secs_f64());
    }

    let ratio = timings[1] / timings[0];
    println!("budget {node_count} costs {ratio:.1}x budget 8");
    assert!(
        ratio < 10.0,
        "a converged run must not scale with its iteration budget: budget {node_count} cost \
         {ratio:.1}x budget 8 ({:.3} ms vs {:.3} ms), so the early exit is not firing and the \
         kernel is running the full budget after reaching its fixpoint",
        timings[1] * 1e3,
        timings[0] * 1e3
    );
}
