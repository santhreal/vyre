//! Contracts for `vyre_driver::pipeline_fusion`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::arm_independence::{
    can_dispatch_concurrently, ArmBindingSummary, ArmIndependenceVerdict,
};
use vyre_driver::pipeline_fusion::{
    decide_cross_pipeline_fusion, CrossPipelineConflict, CrossPipelineFusionDecision,
};

fn summary(reads: &[u32], writes: &[u32]) -> ArmBindingSummary {
    ArmBindingSummary {
        reads: reads.iter().copied().collect(),
        writes: writes.iter().copied().collect(),
    }
}

#[test]
fn disjoint_pipelines_fuse() {
    let a = summary(&[0, 1], &[2]);
    let b = summary(&[3, 4], &[5]);
    assert_eq!(
        decide_cross_pipeline_fusion(&a, &b),
        CrossPipelineFusionDecision::Fuse
    );
}

#[test]
fn write_write_conflict_keeps_separate() {
    let a = summary(&[0], &[2]);
    let b = summary(&[1], &[2]);
    assert_eq!(
        decide_cross_pipeline_fusion(&a, &b),
        CrossPipelineFusionDecision::KeepSeparate {
            reason: CrossPipelineConflict::WriteWriteConflict,
        }
    );
}

#[test]
fn read_after_write_keeps_separate() {
    let a = summary(&[0], &[2]);
    let b = summary(&[2], &[3]);
    assert_eq!(
        decide_cross_pipeline_fusion(&a, &b),
        CrossPipelineFusionDecision::KeepSeparate {
            reason: CrossPipelineConflict::ReadAfterWrite,
        }
    );
}

#[test]
fn read_only_share_same_slot_fuses() {
    // Two pipelines reading the same slot is always safe to fuse.
    let a = summary(&[0, 1], &[2]);
    let b = summary(&[0, 1], &[3]);
    assert_eq!(
        decide_cross_pipeline_fusion(&a, &b),
        CrossPipelineFusionDecision::Fuse
    );
}
