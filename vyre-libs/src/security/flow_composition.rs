//! The one reachability-plus-sanitizer-projection skeleton behind every
//! security flow op.
//!
//! A family member supplies a [`FlowPredicate`] (which edge kinds its walk
//! follows, and in which direction) plus the buffers it wants written. Nothing
//! else. The traversal, the sanitizer projection, the sink intersection, the
//! any-reduce, the region tagging, and the fusion boundary all live here, so no
//! member can drift away from the one it is paired with:
//!
//! ```text
//!   clean   = source AND NOT sanitizer        (sanitizer projection, optional)
//!   reach   = traverse(clean, direction, edge_mask)
//!   alive   = reach AND NOT sanitizer         (sanitizer projection, optional)
//!   hits    = alive AND sink                  (sink projection, optional)
//!   any_hit = bitset_any(hits)                (sink projection, optional)
//! ```

use crate::bitset::and::bitset_and;
#[cfg(test)]
use crate::bitset::and::cpu_ref as bitset_and_cpu_ref;
use crate::bitset::and_not::bitset_and_not;
#[cfg(test)]
use crate::bitset::and_not::cpu_ref as bitset_and_not_cpu_ref;
use crate::bitset::any::bitset_any;
use crate::bitset::bitset_words;
use crate::graph::csr_backward_traverse::csr_backward_traverse;
#[cfg(test)]
use crate::graph::csr_forward_traverse::cpu_ref as csr_forward_cpu_ref;
use crate::graph::csr_forward_traverse::csr_forward_traverse;
use crate::graph::program_graph::ProgramGraphShape;
use crate::predicate::edge_kind;
use vyre_foundation::composition::{
    reparent_program_children, tag_program, trap_program, wrap_anonymous_region,
};
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::DataType;
use vyre_foundation::ir::Program;

use crate::security::flows_to::{FLOWS_TO_MASK, OP_ID as FLOWS_TO_OP_ID};

/// Iteration ceiling every flow-family op registers.
///
/// AUDIT_2026-04-24 F-FT-03 / F-TF-03 / F-BBC-01 / F-DT-01 raised this from 64.
/// Kernel-scale call graphs and dominance trees run hundreds of steps deep and
/// the old ceiling truncated them into false negatives. A fixpoint driver stops
/// as soon as the frontier stops growing, so a high ceiling costs nothing on a
/// shallow graph.
pub(crate) const FLOW_MAX_ITERATIONS: u32 = 4096;

/// Which way the shared walk runs.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlowDirection {
    /// Along CSR edges: from a node to its successors.
    Forward,
    /// Against CSR edges: from a node to its predecessors.
    Backward,
}

/// The one thing a family member owns.
#[derive(Clone, Copy)]
pub(crate) struct FlowPredicate {
    direction: FlowDirection,
    edge_mask: u32,
}

impl FlowPredicate {
    /// Successor reachability restricted to `edge_mask`.
    pub(crate) const fn forward(edge_mask: u32) -> Self {
        Self {
            direction: FlowDirection::Forward,
            edge_mask,
        }
    }

    /// Predecessor reachability restricted to `edge_mask`.
    pub(crate) const fn backward(edge_mask: u32) -> Self {
        Self {
            direction: FlowDirection::Backward,
            edge_mask,
        }
    }

    fn traverse(self, shape: ProgramGraphShape, source: &str, reach: &str) -> Program {
        match self.direction {
            FlowDirection::Forward => csr_forward_traverse(shape, source, reach, self.edge_mask),
            FlowDirection::Backward => csr_backward_traverse(shape, source, reach, self.edge_mask),
        }
    }
}

/// Scratch and output buffers for the optional sanitizer projection.
#[derive(Clone, Copy)]
pub(crate) struct SanitizerProjection<'a> {
    /// Read-only bitset of sanitizer-tagged nodes.
    pub(crate) sanitizer: &'a str,
    /// Receives `source AND NOT sanitizer`, the frontier the walk starts from.
    pub(crate) clean: &'a str,
    /// Receives `reach AND NOT sanitizer`, the frontier the sink sees.
    pub(crate) alive: &'a str,
}

/// Scratch and output buffers for the optional sink projection.
#[derive(Clone, Copy)]
pub(crate) struct SinkProjection<'a> {
    /// Read-only bitset of sink-tagged nodes.
    pub(crate) sink: &'a str,
    /// Receives the reached-and-sink intersection.
    pub(crate) hits: &'a str,
    /// Receives a 1-word witness: nonzero iff any sink node was reached.
    pub(crate) out_scalar: &'a str,
}

/// Everything the skeleton needs to emit one family member.
///
/// Illegal states are unrepresentable: a projection either brings all of its
/// buffers or is absent, so the builder has no partially-specified branch to
/// trap on.
#[derive(Clone, Copy)]
pub(crate) struct SecurityFlowOptions<'a> {
    op_id: &'static str,
    shape: ProgramGraphShape,
    predicate: FlowPredicate,
    source_buf: &'a str,
    reach_buf: &'a str,
    sanitizer: Option<SanitizerProjection<'a>>,
    sink: Option<SinkProjection<'a>>,
}

impl<'a> SecurityFlowOptions<'a> {
    /// One reachability step, no projections.
    pub(crate) const fn reach(
        op_id: &'static str,
        shape: ProgramGraphShape,
        predicate: FlowPredicate,
        source_buf: &'a str,
        reach_buf: &'a str,
    ) -> Self {
        Self {
            op_id,
            shape,
            predicate,
            source_buf,
            reach_buf,
            sanitizer: None,
            sink: None,
        }
    }

    /// One dataflow reachability step projected onto a sink set.
    pub(crate) const fn hit(
        op_id: &'static str,
        shape: ProgramGraphShape,
        source_buf: &'a str,
        reach_buf: &'a str,
        sink: SinkProjection<'a>,
    ) -> Self {
        Self {
            op_id,
            shape,
            predicate: FlowPredicate::forward(FLOWS_TO_MASK),
            source_buf,
            reach_buf,
            sanitizer: None,
            sink: Some(sink),
        }
    }

    /// One dataflow reachability step, sanitizer-gated on both ends, projected
    /// onto a sink set.
    pub(crate) const fn sanitized_hit(
        op_id: &'static str,
        shape: ProgramGraphShape,
        source_buf: &'a str,
        reach_buf: &'a str,
        sanitizer: SanitizerProjection<'a>,
        sink: SinkProjection<'a>,
    ) -> Self {
        Self {
            op_id,
            shape,
            predicate: FlowPredicate::forward(FLOWS_TO_MASK),
            source_buf,
            reach_buf,
            sanitizer: Some(sanitizer),
            sink: Some(sink),
        }
    }
}

pub(crate) fn fuse_security_flow(op_id: &'static str, parts: &[Program], output: &str) -> Program {
    let fused = match fuse_programs(parts) {
        Ok(fused) => fused,
        Err(error) => {
            return trap_program(
                op_id,
                Some((output, DataType::U32)),
                format!("Fix: security flow composition failed to fuse: {error}"),
            );
        }
    };
    Program::wrapped(
        fused.buffers().to_vec(),
        fused.workgroup_size(),
        vec![wrap_anonymous_region(
            op_id,
            reparent_program_children(&fused, op_id),
        )],
    )
}

/// Emit one family member.
pub(crate) fn security_flow_program(options: SecurityFlowOptions<'_>) -> Program {
    crate::security::assert_security_inputs(
        options.op_id,
        options.shape.node_count,
        &[
            ("source_buf", options.source_buf),
            ("reach_buf", options.reach_buf),
        ],
    );
    let words = bitset_words(options.shape.node_count);
    let mut parts = Vec::new();
    let walk_from = match options.sanitizer {
        Some(projection) => {
            parts.push(bitset_and_not(
                options.source_buf,
                projection.sanitizer,
                projection.clean,
                words,
            ));
            projection.clean
        }
        None => options.source_buf,
    };
    // The traversal region is named for whoever owns the walk: the entry point
    // itself when the walk is the whole op, `flows_to` when it is one stage of
    // a larger composition that owns the outer region.
    let walk_owner = if options.sink.is_some() {
        FLOWS_TO_OP_ID
    } else {
        options.op_id
    };
    let traverse = tag_program(
        walk_owner,
        options
            .predicate
            .traverse(options.shape, walk_from, options.reach_buf),
    );
    let Some(sink) = options.sink else {
        if parts.is_empty() {
            return traverse;
        }
        parts.push(traverse);
        return fuse_security_flow(options.op_id, &parts, options.reach_buf);
    };
    parts.push(traverse);
    let hit_from = match options.sanitizer {
        Some(projection) => {
            parts.push(bitset_and_not(
                options.reach_buf,
                projection.sanitizer,
                projection.alive,
                words,
            ));
            projection.alive
        }
        None => options.reach_buf,
    };
    parts.push(bitset_and(hit_from, sink.sink, sink.hits, words));
    parts.push(bitset_any(sink.hits, sink.out_scalar, words));
    fuse_security_flow(options.op_id, &parts, sink.out_scalar)
}

// ---------------------------------------------------------------------------
// Registration fixtures. The conformance harness feeds these to every family
// member, so they live here rather than being retyped per file.
// ---------------------------------------------------------------------------

fn pack(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

/// Linear chain `0 → 1 → 2 → 3` over ASSIGNMENT edges, frontier seeded at {0}.
/// `fout` seeds as the accumulator so the convergence lens grows monotonically.
pub(crate) fn forward_reach_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        pack(&[0, 0, 0, 0]),               // pg_nodes
        pack(&[0, 1, 2, 3, 3]),            // pg_edge_offsets
        pack(&[1, 2, 3]),                  // pg_edge_targets
        pack(&[edge_kind::ASSIGNMENT; 3]), // pg_edge_kind_mask
        pack(&[0, 0, 0, 0]),               // pg_node_tags
        pack(&[0b0001]),                   // frontier_in = {0}
        pack(&[0b0001]),                   // frontier_out accumulator seed
    ]]
}

/// One forward hop from {0} writes {1} into the accumulator. A no-op that
/// leaves the accumulator at {0} fails this oracle.
pub(crate) fn forward_reach_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![pack(&[0b0011])]]
}

/// Diamond dominance tree `0 → {1, 2} → 3`, frontier seeded at {3}.
pub(crate) fn dominance_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        pack(&[0, 0, 0, 0]),              // pg_nodes
        pack(&[0, 2, 3, 4, 4]),           // pg_edge_offsets
        pack(&[1, 2, 3, 3]),              // pg_edge_targets
        pack(&[edge_kind::DOMINANCE; 4]), // pg_edge_kind_mask
        pack(&[0, 0, 0, 0]),              // pg_node_tags
        pack(&[0b1000]),                  // frontier_in = {3}
        pack(&[0b1000]),                  // frontier_out accumulator seed
    ]]
}

/// One backward hop from {3} lights up nodes 1 and 2; the seed survives.
pub(crate) fn dominance_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![pack(&[0b1110])]]
}

/// The forward chain again, with source {0} and the sink tag on {1}.
pub(crate) fn dataflow_hit_fixture_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![
        pack(&[0, 0, 0, 0]),               // pg_nodes
        pack(&[0, 1, 2, 3, 3]),            // pg_edge_offsets
        pack(&[1, 2, 3]),                  // pg_edge_targets
        pack(&[edge_kind::ASSIGNMENT; 3]), // pg_edge_kind_mask
        pack(&[0, 0, 0, 0]),               // pg_node_tags
        pack(&[0b0001]),                   // source = {0}
        pack(&[0b0001]),                   // reach accumulator seed
        pack(&[0b0010]),                   // sink = {1}
        pack(&[0b0000]),                   // hits
        pack(&[0b0000]),                   // out_scalar
    ]]
}

/// Reach grows to {0, 1}, the sink at {1} is hit, the witness reads 1.
pub(crate) fn dataflow_hit_fixture_expected() -> Vec<Vec<Vec<u8>>> {
    vec![vec![pack(&[0b0011]), pack(&[0b0010]), pack(&[0b0001])]]
}

// ---------------------------------------------------------------------------
// CPU oracles. Same composition as the emitted skeleton, one implementation.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn dataflow_reach_step_cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    source: &[u32],
) -> Vec<u32> {
    csr_forward_cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        source,
        FLOWS_TO_MASK,
    )
}

#[cfg(test)]
pub(crate) fn any_dataflow_hit_cpu_ref(reach: &[u32], sink: &[u32]) -> u32 {
    let hits = bitset_and_cpu_ref(reach, sink);
    u32::from(hits.iter().any(|word| *word != 0))
}

/// One dataflow hop from `source`, intersected with `sink`: 1 if anything
/// landed, 0 otherwise.
#[cfg(test)]
pub(crate) fn dataflow_hit_cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    source: &[u32],
    sink: &[u32],
) -> u32 {
    let reach = dataflow_reach_step_cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        source,
    );
    any_dataflow_hit_cpu_ref(&reach, sink)
}

#[cfg(test)]
pub(crate) fn sanitized_dataflow_hit_cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    source: &[u32],
    sink: &[u32],
    sanitizer: &[u32],
) -> u32 {
    let clean = bitset_and_not_cpu_ref(source, sanitizer);
    let reach = dataflow_reach_step_cpu_ref(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        &clean,
    );
    let alive = bitset_and_not_cpu_ref(&reach, sanitizer);
    any_dataflow_hit_cpu_ref(&alive, sink)
}

#[cfg(test)]
pub(crate) fn linear_dataflow(node_count: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; (node_count + 1) as usize];
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    for i in 0..node_count.saturating_sub(1) {
        offsets[i as usize + 1] = offsets[i as usize] + 1;
        targets.push(i + 1);
        masks.push(edge_kind::ASSIGNMENT);
    }
    let penultimate = offsets[node_count as usize - 1];
    if let Some(last) = offsets.last_mut() {
        *last = penultimate;
    }
    (offsets, targets, masks)
}

/// Diamond dominance tree `0 → {1, 2} → 3` as a CSR triple.
#[cfg(test)]
pub(crate) fn diamond_dominance_tree() -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    (
        4,
        vec![0, 2, 3, 4, 4],
        vec![1, 2, 3, 3],
        vec![edge_kind::DOMINANCE; 4],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameterized_reach_builder_matches_flows_to_public_wrapper() {
        let shape = ProgramGraphShape::new(4, 3);
        let expected = crate::security::flows_to::flows_to(shape, "fin", "fout");
        let actual = security_flow_program(SecurityFlowOptions::reach(
            crate::security::flows_to::OP_ID,
            shape,
            FlowPredicate::forward(FLOWS_TO_MASK),
            "fin",
            "fout",
        ));

        assert_eq!(actual.fingerprint(), expected.fingerprint());
    }

    #[test]
    fn parameterized_hit_builder_matches_flows_to_to_sink_public_wrapper() {
        let shape = ProgramGraphShape::new(4, 3);
        let expected = crate::security::flows_to_to_sink::flows_to_to_sink(
            shape,
            "source",
            "sink",
            "reach",
            "hits",
            "out_scalar",
        );
        let actual = security_flow_program(SecurityFlowOptions::hit(
            crate::security::flows_to_to_sink::OP_ID,
            shape,
            "source",
            "reach",
            SinkProjection {
                sink: "sink",
                hits: "hits",
                out_scalar: "out_scalar",
            },
        ));

        assert_eq!(actual.fingerprint(), expected.fingerprint());
    }

    #[test]
    fn parameterized_sanitized_builder_matches_public_wrapper() {
        let shape = ProgramGraphShape::new(4, 3);
        let expected = crate::security::flows_to_with_sanitizer::flows_to_with_sanitizer(
            shape,
            "source",
            "sink",
            "sanitizer",
            "clean",
            "reach",
            "alive",
            "hits",
            "out_scalar",
        );
        let actual = security_flow_program(SecurityFlowOptions::sanitized_hit(
            crate::security::flows_to_with_sanitizer::OP_ID,
            shape,
            "source",
            "reach",
            SanitizerProjection {
                sanitizer: "sanitizer",
                clean: "clean",
                alive: "alive",
            },
            SinkProjection {
                sink: "sink",
                hits: "hits",
                out_scalar: "out_scalar",
            },
        ));

        assert_eq!(actual.fingerprint(), expected.fingerprint());
    }

    #[test]
    fn backward_predicate_walks_against_the_csr_edges() {
        let shape = ProgramGraphShape::new(4, 4);
        let expected =
            crate::security::bounded_by_comparison::bounded_by_comparison(shape, "fin", "fout");
        let actual = security_flow_program(SecurityFlowOptions::reach(
            crate::security::bounded_by_comparison::OP_ID,
            shape,
            FlowPredicate::backward(edge_kind::DOMINANCE),
            "fin",
            "fout",
        ));

        assert_eq!(actual.fingerprint(), expected.fingerprint());
    }
}
