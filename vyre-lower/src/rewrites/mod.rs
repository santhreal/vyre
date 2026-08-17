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
pub use const_buffer_promote::{
    rewrite_const_buffer_promote, rewrite_const_buffer_promote_with_budget,
};
pub use dead_op::rewrite_dead_ops;
pub use registry::{
    all_registered_contracts, classify_rule, lowering_owned_rules, LoweringRewriteRule,
    RewriteApplicabilityContract, RewriteOwnership, ALL_REWRITE_RULES,
};
pub use vector_memory::{rewrite_vector_memory, rewrite_vector_memory_with_alias_facts};

use crate::KernelDescriptor;

/// Apply all verified profitable lowering-owned structural rewrites in canonical sequence.
///
/// 1. Promotes qualified read-only global buffers to constant memory.
/// 2. Canonicalizes verified adjacent global load and store chains into vec2/vec4 vector memory transactions.
/// 3. Eliminates unreferenced dead pure operations.
/// 4. Orders pure same-body SSA producers before consumers.
#[must_use]
pub fn apply_lowering_rewrites(desc: &KernelDescriptor) -> KernelDescriptor {
    let with_const = rewrite_const_buffer_promote(desc);
    let with_vec = rewrite_vector_memory(&with_const);
    let with_dce = rewrite_dead_ops(&with_vec);
    canonicalize_for_emit(&with_dce)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::lit;
    use crate::{
        BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelOp,
        KernelOpKind, LiteralValue, MemoryClass,
    };
    use vyre_foundation::ir::{BinOp, DataType};

    #[test]
    fn apply_lowering_rewrites_executes_pipeline_and_verifies() {
        let desc = KernelDescriptor {
            id: "full_pipeline_test".into(),
            bindings: BindingLayout {
                slots: vec![
                    BindingSlot {
                        slot: 0,
                        name: "readonly_table".into(),
                        element_type: DataType::F32,
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadOnly,
                        element_count: Some(128),
                    },
                    BindingSlot {
                        slot: 1,
                        name: "output".into(),
                        element_type: DataType::F32,
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadWrite,
                        element_count: Some(128),
                    },
                ],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 0), // index 0 (result 0)
                    lit(0, 1), // dead literal (result 1)
                    KernelOp {
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 0],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 0],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![2, 3],
                        result: Some(4),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![1, 0, 4],
                        result: None,
                    },
                ],
                literals: vec![LiteralValue::U32(0)],
                child_bodies: vec![],
            },
        };

        let result = apply_lowering_rewrites(&desc);

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
