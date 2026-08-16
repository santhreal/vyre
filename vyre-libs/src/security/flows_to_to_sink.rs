//! `flows_to_to_sink`  -  composite source→sink reachability primitive.
//!
//! The "does taint reach a sink node" pattern is the single most
//! common composition in every taint-style generic query dialect rule:
//!
//! ```text
//!   reach    = csr_forward_traverse(source, FLOWS_TO_MASK)
//!   hits     = reach AND sink
//!   any_hit  = bitset_any(hits) → u32
//! ```
//!
//! Earlier lowering paths emitted this composition
//! inline at every call site (~25 lines of boilerplate per call,
//! plus a fresh accumulator buffer per invocation). Centralising it
//! in `crate::security::flow_composition` as one fused Region:
//!
//! * cuts per-call lowering surface from ~5 sub-programs
//!   merged via `merge_programs` to one helper invocation;
//! * gives the optimizer one Region with a stable op id to fuse,
//!   cache, and CSE across rules;
//! * eliminates the "did you remember to compose all three steps"
//!   foot-gun that the audit caught when `flows_to_via` and
//!   `flows_to_not_via` silently shared the same emitted Program.
//!
//! Soundness: identical to the one BFS step `flows_to` provides  -
//! [`MayOver`](vyre_spec::soundness::Soundness::MayOver) on a single
//! step, `Exact` when iterated to fixpoint with sanitizer gating.

use vyre_foundation::ir::Program;
use crate::graph::program_graph::ProgramGraphShape;

#[cfg(test)]
use crate::security::flow_composition::dataflow_hit_cpu_ref;
use crate::security::flow_composition::{
    dataflow_hit_fixture_expected, dataflow_hit_fixture_inputs, security_flow_program,
    SecurityFlowOptions, SinkProjection,
};

pub(crate) const OP_ID: &str = "vyre-libs::security::flows_to_to_sink";

/// One BFS step from `source_buf` along dataflow edges, intersected
/// with `sink_buf`, reduced to a single u32 stored in `out_scalar_buf`.
///
/// Buffers:
/// * `source_buf`   -  read-only bitset of source-tagged nodes.
/// * `sink_buf`     -  read-only bitset of sink-tagged nodes.
/// * `reach_buf`    -  read-write scratch bitset for the BFS step result.
/// * `hits_buf`     -  read-write scratch bitset for the AND result.
/// * `out_scalar_buf`  -  read-write 1-word output: nonzero iff any
///   sink node was reached.
#[must_use]
pub fn flows_to_to_sink(
    shape: ProgramGraphShape,
    source_buf: &str,
    sink_buf: &str,
    reach_buf: &str,
    hits_buf: &str,
    out_scalar_buf: &str,
) -> Program {
    security_flow_program(SecurityFlowOptions::hit(
        OP_ID,
        shape,
        source_buf,
        reach_buf,
        SinkProjection {
            sink: sink_buf,
            hits: hits_buf,
            out_scalar: out_scalar_buf,
        },
    ))
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || flows_to_to_sink(ProgramGraphShape::new(4, 3), "source", "sink", "reach", "hits", "out_scalar"),
        Some(dataflow_hit_fixture_inputs),
        Some(dataflow_hit_fixture_expected),
    )
    .with_category("security")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::flow_composition::linear_dataflow;

    #[test]
    fn one_hop_source_reaches_sink_returns_one() {
        let (off, tgt, msk) = linear_dataflow(4);
        let source = [0b0001u32]; // node 0
        let sink = [0b0010u32]; // node 1 (one hop away)
        let result = dataflow_hit_cpu_ref(4, &off, &tgt, &msk, &source, &sink);
        assert_eq!(result, 1);
    }

    #[test]
    fn two_hops_unreachable_in_one_step_returns_zero() {
        let (off, tgt, msk) = linear_dataflow(4);
        let source = [0b0001u32]; // node 0
        let sink = [0b0100u32]; // node 2 (two hops away  -  not reached in one step)
        let result = dataflow_hit_cpu_ref(4, &off, &tgt, &msk, &source, &sink);
        assert_eq!(result, 0);
    }

    #[test]
    fn empty_source_returns_zero() {
        let (off, tgt, msk) = linear_dataflow(4);
        let source = [0u32];
        let sink = [0b0010u32];
        let result = dataflow_hit_cpu_ref(4, &off, &tgt, &msk, &source, &sink);
        assert_eq!(result, 0);
    }

    #[test]
    fn empty_sink_returns_zero() {
        let (off, tgt, msk) = linear_dataflow(4);
        let source = [0b0001u32];
        let sink = [0u32];
        let result = dataflow_hit_cpu_ref(4, &off, &tgt, &msk, &source, &sink);
        assert_eq!(result, 0);
    }
}
