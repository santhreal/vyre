//! Tensor-network contraction order via shortest-path on the contraction-cost graph.
//!
//! Extends `tensor_network_fusion_order`. Instead of a greedy heuristic,
//! we frame the search for the optimal contraction order of a Region chain as
//! finding the shortest path in a state graph where:
//! - Node = subset of contracted tensors (represented as an integer bitset or ID).
//! - Edge = contracting two adjacent sub-networks.
//! - Weight = FLOP cost of that specific contraction step.
//!
//! We dispatch `crate::math::bellman_shortest_path` to find the
//! globally optimal sequence of pairwise fusions.

use crate::math::bellman_shortest_path::{bellman_shortest_path, BellmanBuffers, BellmanExtents};
use vyre_foundation::ir::Program;

use crate::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Canonical self-substrate op ID for the Bellman TN order.
pub const OP_ID: &str = "vyre-libs::self_substrate::bellman_tn_order";

/// Caller-owned GPU dispatch scratch for Bellman tensor-network ordering.
#[derive(Debug, Default)]
pub struct BellmanTnOrderGpuScratch {
    inputs: Vec<Vec<u8>>,
    changed: Vec<u32>,
}

/// Compile a Program that finds the optimal tensor-network contraction
/// order by running Bellman-Ford over the state space of contractions.
///
/// `n_nodes` is the number of possible contraction states (e.g. `2^N` for N
/// tensors) and `n_edges` the number of valid contraction transitions, both named
/// by [`BellmanExtents`]. The output `dist` buffer will contain the minimum cost
/// to reach each state.
///
/// The composition it adds is the telemetry counter; the program is the
/// primitive's, over the caller's binding record and extents unchanged.
#[must_use]
pub fn bellman_tn_order_program(buffers: BellmanBuffers<'_>, extents: BellmanExtents) -> Program {
    use crate::telemetry::{bellman_tn_order_calls, bump};
    bump(&bellman_tn_order_calls);
    bellman_shortest_path(buffers, extents)
}

/// The binding names this crate's own GPU dispatch wrapper uses.
///
/// The dispatch uploads and reads back by binding index, so these names are only
/// labels to it. It still names each field, because the convergence-flag width
/// below is looked up by `changed`, and a program whose flag is labelled
/// something else would silently fall back to a one-word upload.
const DISPATCH_BINDINGS: BellmanBuffers<'static> = BellmanBuffers::CANONICAL;

/// GPU dispatch wrapper for the Bellman-Ford-based contraction-order
/// solver. Returns the converged minimum-distance vector.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed edge or distance
/// buffers.
#[allow(clippy::too_many_arguments)]
pub fn bellman_tn_order_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    n_nodes: u32,
    max_iterations: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    bellman_tn_order_via_into(
        dispatcher,
        policy,
        src,
        dst,
        weight,
        dist_init,
        n_nodes,
        max_iterations,
        &mut out,
    )?;
    Ok(out)
}

/// GPU dispatch wrapper for the Bellman-Ford contraction-order solver into
/// caller-owned output storage.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed edge or distance buffers.
#[allow(clippy::too_many_arguments)]
pub fn bellman_tn_order_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    n_nodes: u32,
    max_iterations: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = BellmanTnOrderGpuScratch::default();
    bellman_tn_order_via_with_scratch_into(
        dispatcher,
        policy,
        src,
        dst,
        weight,
        dist_init,
        n_nodes,
        max_iterations,
        &mut scratch,
        out,
    )
}

/// GPU dispatch wrapper for the Bellman-Ford contraction-order solver into
/// caller-owned dispatch and output storage.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed edge or distance buffers.
#[allow(clippy::too_many_arguments)]
pub fn bellman_tn_order_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    n_nodes: u32,
    max_iterations: u32,
    scratch: &mut BellmanTnOrderGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if n_nodes == 0 {
        if !dist_init.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: bellman_tn_order_via n_nodes=0 requires empty dist_init, got {} entries.",
                dist_init.len()
            )));
        }
        out.clear();
        return Ok(());
    }
    if max_iterations == 0 {
        if dist_init.len() != n_nodes as usize {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: bellman_tn_order_via expected dist_init length {n_nodes}, got {}.",
                dist_init.len()
            )));
        }
        out.clear();
        out.extend_from_slice(dist_init);
        return Ok(());
    }
    if src.len() != dst.len() || src.len() != weight.len() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: bellman_tn_order_via requires equal edge buffer lengths, got src={}, dst={}, weight={}.",
        src.len(),
        dst.len(),
        weight.len()
    )));
    }
    if dist_init.len() != n_nodes as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: bellman_tn_order_via expected dist_init length {n_nodes}, got {}.",
            dist_init.len()
        )));
    }
    for (idx, (&u, &v)) in src.iter().zip(dst.iter()).enumerate() {
        if u >= n_nodes || v >= n_nodes {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: bellman_tn_order_via edge {idx} has endpoint ({u}->{v}) outside n_nodes {n_nodes}."
        )));
        }
    }
    let n_edges = u32::try_from(src.len()).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: bellman_tn_order_via edge count {} exceeds u32 index space.",
            src.len()
        ))
    })?;
    if n_edges == 0 {
        out.clear();
        out.extend_from_slice(dist_init);
        return Ok(());
    }
    let program = bellman_tn_order_program(
        DISPATCH_BINDINGS,
        BellmanExtents {
            n_nodes,
            n_edges,
            max_iterations,
        },
    );
    // Size the convergence-flag upload from what the program DECLARES, never
    // from an assumed single word. Above one workgroup width
    // `bellman_shortest_path` routes to the grid fixpoint, which writes
    // `changed[iteration]` and therefore declares one word per iteration; a
    // hardcoded one-word upload would hand that form an under-sized binding.
    let changed_words = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == DISPATCH_BINDINGS.changed)
        .map_or(1, vyre_foundation::ir::BufferDecl::count)
        .max(1) as usize;
    scratch.changed.clear();
    scratch.changed.resize(changed_words, 0);
    ensure_input_slots(&mut scratch.inputs, 6);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], dist_init);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], dist_init);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], &scratch.changed);
    write_u32_slice_le_bytes(&mut scratch.inputs[3], src);
    write_u32_slice_le_bytes(&mut scratch.inputs[4], dst);
    write_u32_slice_le_bytes(&mut scratch.inputs[5], weight);
    // Dispatch SPAN, not edge count. The relaxation needs a lane per EDGE but the
    // fixpoint's compare-and-publish step needs a lane per NODE, and `vyre-driver`
    // spans the largest declared non-shared binding for an atomic-carrying
    // program. Sizing the grid off `n_edges` alone leaves every node past the
    // launch width with no lane to publish it whenever `n_nodes` exceeds
    // `n_edges`, silently freezing those distances at their seed values.
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: bellman_tn_order_via expected at least the dist output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], n_nodes as usize, "bellman_tn_order_via", out)
        .map_err(|error| SemanticExecutionError::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::test_parity_oracles::NeverDispatches;
    use vyre_reference::composition_witness::bellman_shortest_path_witness as reference_bellman_shortest_path;

    /// Terse binding names, for the tests that only care about the program.
    const FIXTURE: BellmanBuffers<'static> = BellmanBuffers::TERSE;

    /// The dispatch wrapper's own names, for the tests that compare against it.
    const VERBOSE_FIXTURE: BellmanBuffers<'static> = DISPATCH_BINDINGS;

    fn extents(n_nodes: u32, n_edges: u32, max_iterations: u32) -> BellmanExtents {
        BellmanExtents {
            n_nodes,
            n_edges,
            max_iterations,
        }
    }

    /// One refinement stage over the shared edge buffers.
    ///
    /// The stages differ only in their per-stage state buffers. Sharing the edge
    /// names is the point of the multi-region test: three regions read one graph
    /// and refine three separate distance vectors.
    fn stage<'a>(dist: &'a str, next_dist: &'a str, changed: &'a str) -> BellmanBuffers<'a> {
        BellmanBuffers {
            src: FIXTURE.src,
            dst: FIXTURE.dst,
            weight: FIXTURE.weight,
            dist,
            next_dist,
            changed,
        }
    }

    /// The composition must emit the primitive's program, byte for byte.
    ///
    /// `bellman_tn_order_program` adds a telemetry counter and nothing else. This
    /// pins that on the wire encoding, on both sides of the fixpoint routing
    /// threshold, so a re-derivation creeping back into the forwarder fails here
    /// rather than at a GPU parity test on one shape.
    #[test]
    fn composition_emits_the_primitive_program_unchanged() {
        for (n_nodes, n_edges, max_iterations) in [(4, 4, 10), (300, 4, 6), (8, 12, 5)] {
            let extents = extents(n_nodes, n_edges, max_iterations);
            let wire = |program: &Program| {
                vyre_foundation::serial::wire::encode::to_wire(program)
                    .expect("Fix: a bellman program must encode to the wire form.")
            };
            assert_eq!(
                wire(&bellman_tn_order_program(FIXTURE, extents)),
                wire(&bellman_shortest_path(FIXTURE, extents)),
                "Fix: the tensor-order composition must forward to the primitive unchanged."
            );
        }
    }

    struct BellmanDispatcher;

    impl SemanticExecutor for BellmanDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                assert_eq!(inputs.len(), 6);
                let dist = crate::dispatch_buffers::read_u32s(&inputs[0]);
                let next_dist = crate::dispatch_buffers::read_u32s(&inputs[1]);
                let changed = crate::dispatch_buffers::read_u32s(&inputs[2]);
                let src = crate::dispatch_buffers::read_u32s(&inputs[3]);
                let dst = crate::dispatch_buffers::read_u32s(&inputs[4]);
                let weight = crate::dispatch_buffers::read_u32s(&inputs[5]);
                assert_eq!(dist, next_dist);
                // The invariant is a CLEARED flag buffer of whatever width the routed
                // program declares, not a one-word buffer. Pinning `vec![0]` here would
                // re-encode the assumption the grid route invalidates.
                assert!(
                !changed.is_empty() && changed.iter().all(|&word| word == 0),
                "Fix: the consumer must upload a non-empty, fully cleared convergence-flag buffer, got {changed:?}."
            );
                let (out, _) = reference_bellman_shortest_path(
                    &src,
                    &dst,
                    &weight,
                    &dist,
                    dist.len() as u32,
                    10,
                );
                Ok(vec![u32_slice_to_le_bytes(&out)])
            })();
            let mut ordered = ordered?;
            let output_count = request.logical().graph().nodes()[0].outputs.len();
            if ordered.len() < output_count {
                ordered.resize(output_count, Vec::new());
            }
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    /// Every name the caller supplies must reach a declared buffer.
    ///
    /// The old form asserted the literal `"dist"` was present, which passes for any
    /// program that happens to declare that name and says nothing about the other
    /// five. Reading the names out of the record instead means a field that stops
    /// being forwarded fails here.
    #[test]
    fn every_supplied_binding_name_is_declared() {
        let program = bellman_tn_order_program(FIXTURE, extents(8, 12, 5));
        let declared: Vec<&str> = program.buffers().iter().map(|b| b.name()).collect();

        for name in [
            FIXTURE.dist,
            FIXTURE.next_dist,
            FIXTURE.changed,
            FIXTURE.src,
            FIXTURE.dst,
            FIXTURE.weight,
        ] {
            assert!(
                declared.contains(&name),
                "Fix: the program must declare the buffer named `{name}`; it declared {declared:?}."
            );
        }
        assert_eq!(
            declared.len(),
            6,
            "Fix: bellman_tn_order_program must expose exactly the six bindings it is given."
        );
    }

    /// The consumer must upload a convergence-flag buffer as wide as the program declares.
    ///
    /// A fixed one-word scratch binding would let later iterations write beyond
    /// the supplied resource, so this checks the semantic input ABI directly.
    #[test]
    fn consumer_packs_declared_flag_width() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct CapturingDispatcher {
            changed_words: AtomicUsize,
            n_nodes: usize,
        }

        impl SemanticExecutor for CapturingDispatcher {
            fn execute(
                &self,
                request: &vyre_megakernel::SemanticExecutionRequest<'_>,
            ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError>
            {
                let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
                let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                    self.changed_words.store(
                        crate::dispatch_buffers::read_u32s(&inputs[2]).len(),
                        Ordering::Relaxed,
                    );
                    Ok(vec![u32_slice_to_le_bytes(&vec![0_u32; self.n_nodes])])
                })();
                let mut ordered = ordered?;
                let output_count = request.logical().graph().nodes()[0].outputs.len();
                if ordered.len() < output_count {
                    ordered.resize(output_count, Vec::new());
                }
                crate::test_parity_oracles::semantic_output(request, ordered)
            }
        }

        let n_nodes = 300_u32;
        let max_iterations = 4_u32;
        let src = vec![0_u32, 1, 2, 3];
        let dst = vec![1_u32, 2, 3, 299];
        let weight = vec![1_u32, 1, 1, 1];
        let mut dist_init = vec![u32::MAX; n_nodes as usize];
        dist_init[0] = 0;

        let declared =
            bellman_tn_order_program(VERBOSE_FIXTURE, extents(n_nodes, 4, max_iterations));
        let declared_changed = declared
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: the program must declare its convergence-flag buffer.")
            .count();
        assert_eq!(declared_changed, max_iterations);

        let dispatcher = CapturingDispatcher {
            changed_words: AtomicUsize::new(0),
            n_nodes: n_nodes as usize,
        };
        bellman_tn_order_via(
            &dispatcher,
            &crate::test_parity_oracles::policy(),
            &src,
            &dst,
            &weight,
            &dist_init,
            n_nodes,
            max_iterations,
        )
        .expect("Fix: the consumer must dispatch a 300-state ordering problem.");

        assert_eq!(
            dispatcher.changed_words.load(Ordering::Relaxed),
            declared_changed as usize,
            "Fix: the uploaded convergence-flag buffer must be as wide as the program declares."
        );
    }

    #[test]
    fn test_multi_stage_order_refining() {
        // Build a Program with 3 separate Bellman regions.
        let p1 = bellman_tn_order_program(stage("dist1", "nd1", "c1"), extents(4, 4, 5));
        let p2 = bellman_tn_order_program(stage("dist2", "nd2", "c2"), extents(4, 4, 5));
        let p3 = bellman_tn_order_program(stage("dist3", "nd3", "c3"), extents(4, 4, 5));

        let final_p =
            crate::test_parity_oracles::wrap_program_sequence(&[&p1, &p2, &p3], [256, 1, 1]);
        crate::solvers::test_helpers::assert_min_region_count(&final_p, 3);
    }

    #[test]
    fn bellman_tn_order_via_dispatches_primitive() {
        let src = vec![0, 1, 2, 0];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![10, 20, 30, 100];
        let dist_init = vec![0, u32::MAX, u32::MAX, u32::MAX];

        let out = bellman_tn_order_via(
            &BellmanDispatcher,
            &crate::test_parity_oracles::policy(),
            &src,
            &dst,
            &weight,
            &dist_init,
            4,
            10,
        )
        .unwrap();

        assert_eq!(out, vec![0, 10, 30, 60]);
    }

    #[test]
    fn bellman_tn_order_via_into_reuses_output() {
        let src = vec![0, 1, 2, 0];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![10, 20, 30, 100];
        let dist_init = vec![0, u32::MAX, u32::MAX, u32::MAX];
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();

        bellman_tn_order_via_into(
            &BellmanDispatcher,
            &crate::test_parity_oracles::policy(),
            &src,
            &dst,
            &weight,
            &dist_init,
            4,
            10,
            &mut out,
        )
        .unwrap();

        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out, vec![0, 10, 30, 60]);
    }

    #[test]
    fn bellman_tn_order_via_with_scratch_reuses_dispatch_and_output_storage() {
        let src = vec![0, 1, 2, 0];
        let dst = vec![1, 2, 3, 3];
        let weight = vec![10, 20, 30, 100];
        let dist_init = vec![0, u32::MAX, u32::MAX, u32::MAX];
        let mut scratch = BellmanTnOrderGpuScratch::default();
        let mut out = Vec::with_capacity(4);

        bellman_tn_order_via_with_scratch_into(
            &BellmanDispatcher,
            &crate::test_parity_oracles::policy(),
            &src,
            &dst,
            &weight,
            &dist_init,
            4,
            10,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_capacity = out.capacity();

        bellman_tn_order_via_with_scratch_into(
            &BellmanDispatcher,
            &crate::test_parity_oracles::policy(),
            &src,
            &dst,
            &weight,
            &dist_init,
            4,
            10,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        crate::solvers::test_helpers::assert_scratch_capacities_preserved(
            &scratch.inputs,
            &input_capacities,
        );
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(out, vec![0, 10, 30, 60]);
    }

    #[test]
    fn bellman_tn_order_via_rejects_bad_edge_shape() {
        let err = bellman_tn_order_via(
            &BellmanDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0],
            &[],
            &[1],
            &[0],
            1,
            10,
        )
        .unwrap_err();

        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }

    #[test]
    fn bellman_tn_order_via_empty_edges_returns_initial_dist_without_dispatch() {
        let mut out = Vec::with_capacity(8);
        bellman_tn_order_via_into(
            &NeverDispatches(
                "Fix: empty Bellman edge set must not submit a zero-work GPU dispatch",
            ),
            &crate::test_parity_oracles::policy(),
            &[],
            &[],
            &[],
            &[0, u32::MAX],
            2,
            10,
            &mut out,
        )
        .expect("Fix: empty Bellman edge set must return the initial distances");
        assert_eq!(out, vec![0, u32::MAX]);
    }
}
