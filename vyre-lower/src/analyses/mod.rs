//! Read-only, backend-neutral analyses on `KernelDescriptor`.
//!
//! These analyses run after verified lowering and before emission. They report
//! descriptor facts and candidate target strategies without changing program
//! semantics or descriptor structure. Concrete emission strategy lives in the
//! owning emitter or driver.

pub mod access_kind;
pub mod alias_facts;
pub mod alias_import;
pub mod bank_conflict;
/// Shared candidate-plan data structures.
pub mod candidate_plan;
pub mod coalesce;
pub mod common_subexpr;
pub mod const_buffer_promote;
pub mod dead_op;
pub mod def_use;
pub mod layout_aos_to_soa;
pub(crate) mod load_counts;
pub mod op_histogram;
pub mod reaching_def_facts;
pub mod reaching_def_import;
pub mod shared_mem_promote;
pub mod structured_walk;
pub mod texture_promote;
pub mod value_range;
pub mod vec_pack;
pub mod workgroup_uniform;

use crate::operand_class::operand_is_result_reference;
use crate::{KernelBody, KernelOp, KernelOpKind};
use rustc_hash::FxHashMap;

pub(crate) type ProducerMap<'a> = FxHashMap<u32, &'a KernelOp>;

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

/// Child-body indices referenced by a structured control-flow op's operands.
///
/// ONE owner for the per-op-kind child-body start-offset table; every
/// placement analysis imports this instead of re-deriving the skip offsets.
pub fn child_body_operands<'a>(
    kind: &KernelOpKind,
    operands: &'a [u32],
) -> impl Iterator<Item = u32> + 'a {
    let start = match kind {
        KernelOpKind::StructuredIfThen | KernelOpKind::StructuredIfThenElse => 1,
        KernelOpKind::StructuredForLoop { .. } => 2,
        KernelOpKind::StructuredBlock | KernelOpKind::Region { .. } => 0,
        _ => operands.len(),
    };
    operands.iter().skip(start).copied()
}

// Re-exports for the common case: a one-call combined audit.
pub use access_kind::AccessKind;
pub use bank_conflict::{analyze as analyze_bank_conflict, BankConflictReport};
pub use coalesce::{analyze as analyze_coalesce, CoalescenceReport};
pub use common_subexpr::{analyze as analyze_common_subexpr, CommonSubexprReport};
pub use const_buffer_promote::{analyze as analyze_const_buffer_promote, ConstBufferPlan};
pub use dead_op::{analyze as analyze_dead_op, DeadOpReport};
pub use def_use::{
    analyze as analyze_def_use, dead_by_no_use, DefUseReport, PerBodyChains, UseSite,
};
pub use layout_aos_to_soa::{analyze as analyze_layout_aos_to_soa, LayoutTransformPlan};
pub use op_histogram::{analyze as analyze_op_histogram, OpHistogram};
pub use reaching_def_facts::import_descriptor_reaching_defs;
pub use shared_mem_promote::{analyze as analyze_shared_mem_promote, PromotionPlan};
pub use texture_promote::{analyze as analyze_texture_promote, TexturePromotionPlan};
pub use value_range::{analyze as analyze_value_range, IntRange, ValueRangeReport};
pub use workgroup_uniform::{analyze as analyze_workgroup_uniform, WorkgroupUniformReport};

#[cfg(test)]
mod dedup_guard {
    use std::path::{Path, PathBuf};

    // Fails if a second `child_body_operands` copy reappears anywhere under
    // src/analyses/ (the table must live only in this module).
    #[test]
    fn child_body_operands_has_single_owner() {
        let dir =
            vyre_test_support::monorepo::vyre_workspace_root().join("vyre-lower/src/analyses");
        let mut hits = Vec::new();
        visit(&dir, &mut hits);
        hits.sort();
        assert_eq!(
            hits,
            vec![dir.join("mod.rs")],
            "Fix: child_body_operands must have exactly ONE owner (analyses/mod.rs); a copy reappeared: {hits:?}"
        );
    }

    fn visit(dir: &Path, hits: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, hits);
            } else if path.extension().is_some_and(|e| e == "rs")
                && std::fs::read_to_string(&path)
                    .unwrap()
                    .contains("fn child_body_operands")
            {
                hits.push(path);
            }
        }
    }
}
