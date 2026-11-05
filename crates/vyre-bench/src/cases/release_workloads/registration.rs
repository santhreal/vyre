//! Suite membership, GPU requirement shape, and inventory registration for the
//! release bench cases.

use super::callgraph_reachability::CallgraphReachabilityStep;
use super::families::{
    ALIAS_REACHING_DEF, CONDITION_EVAL_BATCH, C_AST_TRAVERSAL, EGRAPH_SATURATION, ENTROPY_WINDOW,
    IFDS_WITNESS, MEGAKERNEL_QUEUE, OFFSET_COUNT_AGGREGATION, QUANTIFIED_LOOPS,
    STRING_BITMAP_SCATTER,
};
use super::metadata_condition::MetadataConditionBatch;
use super::sparse_compaction::SparseOutputCompactionCount;
use crate::api::case::{BenchCase, BenchRequirements};

pub(super) const RELEASE_SUITES: &[crate::api::suite::SuiteKind] = &[
    crate::api::suite::SuiteKind::Release,
    crate::api::suite::SuiteKind::Gpu,
    crate::api::suite::SuiteKind::Deep,
    crate::api::suite::SuiteKind::Honest,
];

pub(super) fn gpu_requirements(input_bytes: u64) -> BenchRequirements {
    BenchRequirements {
        needs_gpu: true,
        needs_network: false,
        min_vram_bytes: None,
        min_input_bytes: Some(input_bytes),
        feature_set: vec!["release-workload".to_string()],
    }
}

inventory::submit! {
    &SparseOutputCompactionCount as &'static dyn BenchCase
}

inventory::submit! {
    &CallgraphReachabilityStep as &'static dyn BenchCase
}

inventory::submit! {
    &MetadataConditionBatch as &'static dyn BenchCase
}

inventory::submit! {
    &CONDITION_EVAL_BATCH as &'static dyn BenchCase
}

inventory::submit! {
    &STRING_BITMAP_SCATTER as &'static dyn BenchCase
}

inventory::submit! {
    &OFFSET_COUNT_AGGREGATION as &'static dyn BenchCase
}

inventory::submit! {
    &ENTROPY_WINDOW as &'static dyn BenchCase
}

inventory::submit! {
    &QUANTIFIED_LOOPS as &'static dyn BenchCase
}

inventory::submit! {
    &ALIAS_REACHING_DEF as &'static dyn BenchCase
}

inventory::submit! {
    &IFDS_WITNESS as &'static dyn BenchCase
}

inventory::submit! {
    &C_AST_TRAVERSAL as &'static dyn BenchCase
}

inventory::submit! {
    &MEGAKERNEL_QUEUE as &'static dyn BenchCase
}

inventory::submit! {
    &EGRAPH_SATURATION as &'static dyn BenchCase
}
