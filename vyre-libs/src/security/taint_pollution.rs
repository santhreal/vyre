//! `taint_pollution`  -  "did taint reach a label-tagged node?"
//!
//! The CodeQL `globalAllowingExtras` shape compressed to one Region. Same
//! reach-then-intersect-then-any-reduce composition as
//! [`crate::security::flows_to_to_sink::flows_to_to_sink`]; the sink predicate
//! is a family-tagged node set rather than a sink-tagged one.

use vyre_foundation::ir::Program;
use vyre_primitives::graph::program_graph::ProgramGraphShape;

#[cfg(test)]
use crate::security::flow_composition::dataflow_hit_cpu_ref;
use crate::security::flow_composition::{
    dataflow_hit_fixture_expected, dataflow_hit_fixture_inputs, security_flow_program,
    SecurityFlowOptions, SinkProjection,
};

pub(crate) const OP_ID: &str = "vyre-libs::security::taint_pollution";

/// Build a one-step taint-pollution Program: source → reach
/// (FLOWS_TO_MASK) → AND with label-tagged sink set → any-reduce.
#[must_use]
pub fn taint_pollution(
    shape: ProgramGraphShape,
    source_buf: &str,
    label_set: &str,
    reach_buf: &str,
    hits_buf: &str,
    out_scalar: &str,
) -> Program {
    security_flow_program(SecurityFlowOptions::hit(
        OP_ID,
        shape,
        source_buf,
        reach_buf,
        SinkProjection {
            sink: label_set,
            hits: hits_buf,
            out_scalar,
        },
    ))
}

/// Soundness marker for [`taint_pollution`].
pub struct TaintPollution;
impl vyre_spec::soundness::SoundnessTagged for TaintPollution {
    fn soundness(&self) -> vyre_spec::soundness::Soundness {
        vyre_spec::soundness::Soundness::MayOver
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || taint_pollution(ProgramGraphShape::new(4, 3), "source", "label_set", "reach", "hits", "out_scalar"),
        Some(dataflow_hit_fixture_inputs),
        Some(dataflow_hit_fixture_expected),
    )
    .with_category("security")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_primitives::predicate::edge_kind;

    #[test]
    fn one_hop_to_labeled_returns_one() {
        // 0 -> 1, label = {1}
        let off = vec![0u32, 1, 1];
        let tgt = vec![1u32];
        let msk = vec![edge_kind::ASSIGNMENT];
        assert_eq!(
            dataflow_hit_cpu_ref(2, &off, &tgt, &msk, &[0b01], &[0b10]),
            1
        );
    }

    #[test]
    fn no_label_hit_returns_zero() {
        let off = vec![0u32, 1, 1];
        let tgt = vec![1u32];
        let msk = vec![edge_kind::ASSIGNMENT];
        assert_eq!(dataflow_hit_cpu_ref(2, &off, &tgt, &msk, &[0b01], &[0]), 0);
    }

    #[test]
    fn empty_source_returns_zero() {
        let off = vec![0u32, 1, 1];
        let tgt = vec![1u32];
        let msk = vec![edge_kind::ASSIGNMENT];
        assert_eq!(
            dataflow_hit_cpu_ref(2, &off, &tgt, &msk, &[0], &[0xFFFF]),
            0
        );
    }

    #[test]
    fn unreachable_label_returns_zero() {
        // 0 -> 1, label = {0}  -  source 0 doesn't taint itself.
        let off = vec![0u32, 1, 1];
        let tgt = vec![1u32];
        let msk = vec![edge_kind::ASSIGNMENT];
        assert_eq!(
            dataflow_hit_cpu_ref(2, &off, &tgt, &msk, &[0b01], &[0b01]),
            0
        );
    }
}
