//! Backend-neutral structural lowering rewrites for `KernelDescriptor`.
//!
//! Lowering rewrites are bounded structural transforms driven by analysis facts.
//! They preserve `Program` semantics and execute between verified lowering and
//! backend emission.
//!
//! Non-goals: Layer-1 semantic IR transformations belong in `vyre-foundation::optimizer`;
//! concrete backend emission strategies belong in `vyre-emit-*`.

mod canonicalize;
mod const_buffer_promote;
mod dead_op;
mod registry;
mod vector_memory;

pub use canonicalize::canonicalize_for_emit;
pub use const_buffer_promote::rewrite_const_buffer_promote;
pub use dead_op::rewrite_dead_ops;
pub use registry::{
    all_registered_contracts, classify_rule, lowering_owned_rules, LoweringRewriteRule,
    RewriteApplicabilityContract, RewriteOwnership, ALL_REWRITE_RULES,
};
pub use vector_memory::{rewrite_vector_memory, rewrite_vector_memory_with_alias_facts};

use crate::analyses::AnalysisFacts;
use crate::KernelDescriptor;

/// Apply all verified profitable lowering-owned structural rewrites in canonical sequence.
///
/// 1. Promotes qualified read-only global buffers to constant memory, when the
///    caller stated a constant capacity.
/// 2. Canonicalizes verified adjacent global load and store chains into vec2/vec4 vector memory transactions.
/// 3. Eliminates unreferenced dead pure operations.
/// 4. Orders pure same-body SSA producers before consumers.
#[must_use]
pub fn apply_lowering_rewrites(desc: &KernelDescriptor, facts: &AnalysisFacts) -> KernelDescriptor {
    let with_const = match facts.constant_buffer_bytes() {
        Some(budget) => rewrite_const_buffer_promote(desc, budget),
        None => desc.clone(),
    };
    let with_vec = rewrite_vector_memory(&with_const);
    let with_dce = rewrite_dead_ops(&with_vec);
    canonicalize_for_emit(&with_dce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
    use crate::{BindingSlot, BindingVisibility, KernelOpKind, LiteralValue, MemoryClass};
    use vyre_foundation::ir::{BinOp, DataType};

    #[test]
    fn apply_lowering_rewrites_executes_pipeline_and_verifies() {
        let desc = descriptor("full_pipeline_test")
            .slot(BindingSlot {
                slot: 0,
                name: "readonly_table".into(),
                element_type: DataType::F32,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadOnly,
                element_count: Some(128),
            })
            .slot(global_rw(1, DataType::F32, "output"))
            .dispatch(64, 1, 1)
            .body(body().literals([LiteralValue::U32(0)]).ops([
                lit(0, 0), // index 0 (result 0)
                lit(0, 1), // dead literal (result 1)
                op(KernelOpKind::LoadGlobal, [0, 0], 2),
                op(KernelOpKind::LoadGlobal, [0, 0], 3),
                op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 4),
                effect(KernelOpKind::StoreGlobal, [1, 0, 4]),
            ]))
            .build();

        let facts = AnalysisFacts::none().with_constant_buffer_bytes(
            core::num::NonZeroU32::new(64 * 1024).expect("positive capacity"),
        );
        let result = apply_lowering_rewrites(&desc, &facts);

        // Constant promotion: slot 0 should be Constant, loads should be LoadConstant.
        assert_eq!(result.bindings.slots[0].memory_class, MemoryClass::Constant);
        assert_eq!(result.body.ops[1].kind, KernelOpKind::LoadConstant);
        assert_eq!(result.body.ops[2].kind, KernelOpKind::LoadConstant);

        // Dead op elimination: result 1 (dead literal) should be stripped.
        let results: Vec<u32> = result.body.ops.iter().filter_map(|op| op.result).collect();
        assert!(!results.contains(&1));

        // Verification check.
        assert!(crate::verify(&result).is_ok());
    }
}
