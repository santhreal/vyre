//! Read-only, backend-neutral analyses on `KernelDescriptor`.
//!
//! These analyses run after verified lowering and before emission. They report
//! descriptor facts and candidate target strategies without changing program
//! semantics or descriptor structure. Concrete emission strategy lives in the
//! owning emitter or driver.

pub(crate) mod affine_access_map;
pub(crate) mod resource_bounds;

pub(crate) mod access_kind;
pub mod alias_facts;
pub(crate) mod bank_conflict;
/// Shared candidate-plan data structures.
pub mod candidate_plan;
pub(crate) mod coalesce;
pub(crate) mod common_subexpr;
pub(crate) mod const_buffer_promote;
pub(crate) mod dead_op;
pub(crate) mod def_use;
pub(crate) mod facts;
pub(crate) mod layout_aos_to_soa;
pub(crate) mod load_counts;
pub(crate) mod op_histogram;
pub(crate) mod reaching_def_facts;
pub(crate) mod shared_mem_promote;
pub(crate) mod shared_store_race;
pub mod structured_walk;
pub(crate) mod texture_promote;
pub(crate) mod value_range;
pub mod vec_pack;
pub(crate) mod workgroup_uniform;

use crate::operand_class::operand_is_result_reference;
use crate::{KernelBody, KernelOp, KernelOpKind};
use rustc_hash::FxHashMap;

/// Result id to the op that produces it, for one body.
///
/// Public because it appears in [`structured_walk::StructuredVisitor::visit_op`],
/// which backends implement.
pub type ProducerMap<'a> = FxHashMap<u32, &'a KernelOp>;

pub(crate) fn producer_map(body: &KernelBody) -> ProducerMap<'_> {
    let mut producers = FxHashMap::with_capacity_and_hasher(body.ops.len(), Default::default());
    for op in &body.ops {
        for result in op.result_ids() {
            producers.insert(result, op);
        }
    }
    producers
}

/// The `u32` a `Literal` op publishes, read through its pool index.
fn literal_op_u32(body: &KernelBody, producer: &KernelOp) -> Option<u32> {
    if producer.kind != KernelOpKind::Literal {
        return None;
    }
    producer
        .operands
        .first()
        .and_then(|index| pool_entry_u32(body, *index))
}

/// The `u32` at a literal-pool index, or `None` for another literal type.
pub(crate) fn pool_entry_u32(body: &KernelBody, pool_index: u32) -> Option<u32> {
    match body.literals.get(pool_index as usize) {
        Some(crate::LiteralValue::U32(value)) => Some(*value),
        _ => None,
    }
}

/// The constant `u32` behind an operand, however it was written.
///
/// WHY: an index operand carries a constant in one of two encodings. Either a
/// `Literal` op produced it, in which case that op's first operand is the pool
/// index, or the operand id is itself a pool index. Bank-conflict and coalescing
/// classification each resolved both encodings with its own copy of this, so a
/// third encoding would have had to be added twice.
pub(crate) fn constant_u32_operand(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    operand_id: u32,
) -> Option<u32> {
    producers
        .get(&operand_id)
        .and_then(|producer| literal_op_u32(body, producer))
        .or_else(|| pool_entry_u32(body, operand_id))
}

pub(crate) fn body_result_ids(body: &KernelBody) -> rustc_hash::FxHashSet<u32> {
    let mut results = rustc_hash::FxHashSet::default();
    for op in &body.ops {
        results.extend(op.result_ids());
    }
    for child in &body.child_bodies {
        results.extend(body_result_ids(child));
    }
    results
}

pub(crate) fn body_refs_only(body: &KernelBody, produced: &rustc_hash::FxHashSet<u32>) -> bool {
    body.ops.iter().all(|op| {
        op.operands.iter().enumerate().all(|(position, operand)| {
            !operand_is_result_reference(&op.kind, position) || produced.contains(operand)
        })
    }) && body
        .child_bodies
        .iter()
        .all(|child| body_refs_only(child, produced))
}
/// Greatest common divisor for positive unsigned integers.
pub(crate) fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Child-body indices referenced by a structured control-flow op's operands.
///
/// Every placement analysis and every descriptor walk calls this instead of
/// re-deriving the skip offsets. The offsets themselves come from
/// [`crate::op_facts::facts_for`], which is the crate's only enumeration of
/// `KernelOpKind` and has no wildcard arm: a new variant that carries a nested
/// body fails to compile until someone states where its child indices begin,
/// rather than silently stopping every analysis from descending into it.
pub fn child_body_operands<'a>(
    kind: &KernelOpKind,
    operands: &'a [u32],
) -> impl Iterator<Item = u32> + 'a {
    let start = crate::op_facts::facts_for(kind)
        .child_body_start
        .unwrap_or(operands.len());
    operands.iter().skip(start).copied()
}

// Re-exports for the common case: a one-call combined audit.
pub use access_kind::AccessKind;
pub use affine_access_map::{
    AffineAccessMap, AffineMapError, ConsumerAbiRequirement, DimExtent, SliceSpec, StrideExpr,
};
pub use bank_conflict::{analyze as analyze_bank_conflict, BankConflictReport};
pub use bank_conflict::{
    derive_shared_access_profiles, evaluate_mitigation_candidate, select_bank_conflict_strategy,
    AccessPhase, AccessPhaseProfile, BankConflictMitigation, MitigationEvaluation,
    PhaseConflictReport, SharedBindingAccessProfile, SharedPermutationBlock, TargetBankGeometry,
};
pub use bank_conflict::{BankAccessSite, BankConflictKind, ConflictSeverity};
pub use coalesce::{analyze as analyze_coalesce, CoalescenceReport};
pub use coalesce::{AccessPattern, AccessSite};
pub use coalesce::{CoalescenceRewrite, CoalescenceWarning};
pub use common_subexpr::{analyze as analyze_common_subexpr, CommonSubexprReport};
pub use common_subexpr::{analyze_body, analyze_body_shallow, EquivalenceGroup};
pub use const_buffer_promote::ConstBufferCandidate;
pub use const_buffer_promote::{analyze as analyze_const_buffer_promote, ConstBufferPlan};
pub use dead_op::{analyze as analyze_dead_op, DeadOpReport};
pub use def_use::{
    analyze as analyze_def_use, dead_by_no_use, DefUseReport, PerBodyChains, UseSite,
};
pub use facts::AnalysisFacts;
pub use layout_aos_to_soa::LayoutCandidate;
pub use layout_aos_to_soa::{analyze as analyze_layout_aos_to_soa, LayoutTransformPlan};
pub use op_histogram::{analyze as analyze_op_histogram, OpHistogram};
pub use reaching_def_facts::import_descriptor_reaching_defs;
pub use reaching_def_facts::{resolve_copy_alias, ReachingDefFactSet};
pub use resource_bounds::{
    verify_candidate_legality, CandidateLegalityReport, LegalityCheck, ResourceBounds,
    RetainedFallbacks, TailHandling, TargetResourceLimits,
};
pub use shared_mem_promote::PromotionCandidate;
pub use shared_mem_promote::{analyze as analyze_shared_mem_promote, PromotionPlan};
pub use shared_store_race::{
    analyze as analyze_shared_store_race, SharedStoreLegality, SharedStoreRaceReport,
    SharedStoreRaceSite,
};
pub use texture_promote::TextureCandidate;
pub use texture_promote::{analyze as analyze_texture_promote, TexturePromotionPlan};
pub use value_range::{analyze as analyze_value_range, IntRange, ValueRangeReport};
pub use workgroup_uniform::{analyze as analyze_workgroup_uniform, WorkgroupUniformReport};
pub use workgroup_uniform::{BranchEmitHint, BranchHint};
pub use workgroup_uniform::{BranchSite, BranchUniformity};
