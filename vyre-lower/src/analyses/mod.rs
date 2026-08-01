//! Substrate-aware-but-backend-agnostic analyses on `KernelDescriptor`.
//!
//! Source-of-truth: `SEPARATION_AUDIT_2026-05-01.md` section S3 +
//! `PERF_ROADMAP_2026-05-01.md` section B.3.
//!
//! Each analysis here operates on a `KernelDescriptor` post-lowering
//! and pre-emission. Any rewrite they produce is consumed by every
//! emitter, so analyses that work this layer pay off across all
//! substrates with one implementation.
//!
//! Substrate-specific emission patterns live in their respective
//! emitter crates instead.

pub mod access_kind;
pub mod alias_facts;
pub mod alias_import;
pub mod bank_conflict;
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
pub mod texture_promote;
pub mod value_range;
pub mod vec_pack;
pub mod workgroup_uniform;

use crate::operand_semantics::operand_is_result_reference;
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
pub(crate) fn child_body_operands<'a>(
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

#[cfg(test)]
mod dedup_guard {
    use std::path::{Path, PathBuf};

    // Fails if a second `child_body_operands` copy reappears anywhere under
    // src/analyses/ (the table must live only in this module).
    #[test]
    fn child_body_operands_has_single_owner() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/analyses");
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
