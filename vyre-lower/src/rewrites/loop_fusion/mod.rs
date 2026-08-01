//! Conservative loop fusion for adjacent disjoint-write loops.

use super::dataflow_facts::resolve_reaching_def_id as resolve;
use super::write_targets_are_independent;
use crate::descriptor::Name;
use crate::rewrites::literal_pool_splice::LiteralPoolSplice;
use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

/// Fuse adjacent `StructuredForLoop`s with identical bounds and
/// provably disjoint writes.
#[must_use]
pub fn loop_fusion(desc: &KernelDescriptor) -> KernelDescriptor {
    loop_fusion_with_optional_dataflow_facts(desc, None, None)
}

#[must_use]
pub fn loop_fusion_with_alias_facts(
    desc: &KernelDescriptor,
    alias_facts: &crate::analyses::alias_facts::AliasFactSet,
) -> KernelDescriptor {
    loop_fusion_with_optional_dataflow_facts(desc, Some(alias_facts), None)
}

#[must_use]
pub fn loop_fusion_with_external_alias_facts(
    desc: &KernelDescriptor,
    alias_facts: &crate::analyses::alias_import::AliasFactSet,
) -> KernelDescriptor {
    loop_fusion_with_alias_facts(desc, alias_facts)
}

#[must_use]
pub fn loop_fusion_with_dataflow_facts(
    desc: &KernelDescriptor,
    alias_facts: &crate::analyses::alias_facts::AliasFactSet,
    reaching_defs: &crate::analyses::reaching_def_facts::ReachingDefFactSet,
) -> KernelDescriptor {
    loop_fusion_with_optional_dataflow_facts(desc, Some(alias_facts), Some(reaching_defs))
}

#[must_use]
pub fn loop_fusion_with_dataflow_analysis_facts(
    desc: &KernelDescriptor,
    alias_facts: &crate::analyses::alias_import::AliasFactSet,
    reaching_defs: &crate::analyses::reaching_def_import::ReachingDefFactSet,
) -> KernelDescriptor {
    loop_fusion_with_dataflow_facts(desc, alias_facts, reaching_defs)
}

fn loop_fusion_with_optional_dataflow_facts(
    desc: &KernelDescriptor,
    alias_facts: Option<&crate::analyses::alias_facts::AliasFactSet>,
    reaching_defs: Option<&crate::analyses::reaching_def_facts::ReachingDefFactSet>,
) -> KernelDescriptor {
    let mut out = desc.clone();
    out.body = fuse_body(out.body, alias_facts, reaching_defs);
    out
}

fn fuse_body(
    mut body: KernelBody,
    alias_facts: Option<&crate::analyses::alias_facts::AliasFactSet>,
    reaching_defs: Option<&crate::analyses::reaching_def_facts::ReachingDefFactSet>,
) -> KernelBody {
    body.child_bodies = body
        .child_bodies
        .into_iter()
        .map(|child| fuse_body(child, alias_facts, reaching_defs))
        .collect();
    let old_ops = std::mem::take(&mut body.ops);
    let mut new_ops = Vec::with_capacity(old_ops.len());
    let mut index = 0;
    while index < old_ops.len() {
        if index + 1 < old_ops.len() {
            if let Some(fused) = try_fuse_pair(
                &mut body,
                &old_ops[index],
                &old_ops[index + 1],
                alias_facts,
                reaching_defs,
            ) {
                new_ops.push(fused);
                index += 2;
                continue;
            }
        }
        new_ops.push(old_ops[index].clone());
        index += 1;
    }
    body.ops = new_ops;
    body
}

fn try_fuse_pair(
    body: &mut KernelBody,
    left: &KernelOp,
    right: &KernelOp,
    alias_facts: Option<&crate::analyses::alias_facts::AliasFactSet>,
    reaching_defs: Option<&crate::analyses::reaching_def_facts::ReachingDefFactSet>,
) -> Option<KernelOp> {
    let (left_loop_var, left_body) = loop_parts(left)?;
    let (right_loop_var, right_body) = loop_parts(right)?;
    if left_loop_var != right_loop_var || left.operands[0..2] != right.operands[0..2] {
        return None;
    }
    let left_child = body.child_bodies.get(left_body as usize)?.clone();
    let right_child = body.child_bodies.get(right_body as usize)?.clone();
    if !simple_disjoint_write_bodies(&left_child, &right_child, alias_facts, reaching_defs) {
        return None;
    }
    let fused_body_index = body.child_bodies.len() as u32;
    let mut fused_child = left_child;
    // `right_child`'s ops index into ITS OWN literal pool, and that pool does
    // not survive the fusion  -  only `fused_child`'s (left's) pool does.
    // Relocate the ops so their pool operands point at equivalent entries in
    // the surviving pool instead of at whatever left's pool happens to hold at
    // the same index.
    //
    // Latent rather than live at the moment: `simple_disjoint_write_bodies`
    // admits only a body whose single op is a `StoreGlobal`/`StoreShared`, and
    // a store carries no literal-pool operand, so today there is nothing to
    // remap and no pool entry to copy. It is written through the splice anyway
    // so that relaxing that gate to admit a compute prologue, the obvious next
    // extension, cannot silently begin resolving right's pool indices against
    // left's pool. That failure mode does not trip `verify` whenever the index
    // happens to be in range; it just emits the wrong constant.
    //
    // DO NOT "simplify" this back to `fused_child.ops.extend(right_ops)` on the
    // grounds that the splice currently remaps nothing. The routing IS what
    // makes widening the legality gate safe, and whoever adds a compute
    // prologue will be thinking about aliasing, not about literal pools.
    let KernelBody {
        ops: right_ops,
        literals: right_literals,
        child_bodies: right_children,
    } = right_child;
    debug_assert!(
        right_children.is_empty(),
        "fusion legality requires childless bodies, so no child-body index \
         operand can need rebasing here"
    );
    let mut splice = LiteralPoolSplice::new(&right_literals, &mut fused_child.literals);
    let relocated = splice.relocate_all(right_ops);
    fused_child.ops.extend(relocated);
    body.child_bodies.push(fused_child);
    let mut fused = left.clone();
    fused.operands[2] = fused_body_index;
    Some(fused)
}

fn loop_parts(op: &KernelOp) -> Option<(&Name, u32)> {
    match &op.kind {
        KernelOpKind::StructuredForLoop { loop_var } if op.operands.len() == 3 => {
            Some((loop_var, op.operands[2]))
        }
        _ => None,
    }
}

fn simple_disjoint_write_bodies(
    left: &KernelBody,
    right: &KernelBody,
    alias_facts: Option<&crate::analyses::alias_facts::AliasFactSet>,
    reaching_defs: Option<&crate::analyses::reaching_def_facts::ReachingDefFactSet>,
) -> bool {
    if !left.child_bodies.is_empty() || !right.child_bodies.is_empty() {
        return false;
    }
    let Some(left_write) = single_write_target(left, reaching_defs) else {
        return false;
    };
    let Some(right_write) = single_write_target(right, reaching_defs) else {
        return false;
    };
    write_targets_are_independent(left_write, right_write, alias_facts)
}

fn single_write_target(
    body: &KernelBody,
    reaching_defs: Option<&crate::analyses::reaching_def_facts::ReachingDefFactSet>,
) -> Option<(u8, u32, u32)> {
    if body.ops.len() != 1 {
        return None;
    }
    match body.ops[0].kind {
        KernelOpKind::StoreGlobal if body.ops[0].operands.len() == 3 => Some((
            0,
            body.ops[0].operands[0],
            resolve(body.ops[0].operands[1], reaching_defs),
        )),
        KernelOpKind::StoreShared if body.ops[0].operands.len() == 3 => Some((
            1,
            body.ops[0].operands[0],
            resolve(body.ops[0].operands[1], reaching_defs),
        )),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
    };

    use super::*;

    #[test]
    fn fuses_adjacent_disjoint_store_loops() {
        let desc = KernelDescriptor {
            id: "fusion".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 0],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![store_body(0), shared_store_body(0, 0)],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(4)],
            },
        };
        let out = loop_fusion(&desc);
        let loops = out
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(loops, 1);
    }

    #[test]
    fn does_not_fuse_different_global_bindings_without_external_no_alias_fact() {
        let desc = KernelDescriptor {
            id: "fusion_cross_binding_conservative".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 0],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![store_body(0), store_body(1)],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(4)],
            },
        };
        let conservative = loop_fusion(&desc);
        let conservative_loops = conservative
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(conservative_loops, 2);

        let mut facts = crate::analyses::alias_facts::AliasFactSet::default();
        facts.insert_no_alias(crate::analyses::alias_facts::NoAliasFact {
            left_binding: 0,
            left_index: 0,
            right_binding: 1,
            right_index: 0,
        });
        let alias_aware = loop_fusion_with_alias_facts(&desc, &facts);
        let alias_aware_loops = alias_aware
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(alias_aware_loops, 1);
    }

    fn store_body(binding: u32) -> KernelBody {
        KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![binding, 0, 0],
                result: None,
            }],
            child_bodies: vec![],
            literals: vec![],
        }
    }

    #[test]
    fn fuses_same_binding_loops_when_analysis_proves_indices_no_alias() {
        let desc = KernelDescriptor {
            id: "fusion_alias".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 0],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![store_body_with_index(0, 10), store_body_with_index(0, 11)],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(4)],
            },
        };
        let conservative = loop_fusion(&desc);
        let conservative_loops = conservative
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(conservative_loops, 2);
        let mut facts = crate::analyses::alias_facts::AliasFactSet::default();
        facts.insert_no_alias(crate::analyses::alias_facts::NoAliasFact {
            left_binding: 0,
            left_index: 10,
            right_binding: 0,
            right_index: 11,
        });
        let alias_aware = loop_fusion_with_alias_facts(&desc, &facts);
        let alias_aware_loops = alias_aware
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(alias_aware_loops, 1);
    }

    #[test]
    fn fuses_same_slot_global_and_shared_loops_without_alias_facts() {
        let desc = KernelDescriptor {
            id: "fusion_address_space".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 0],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        operands: vec![0, 1, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![store_body_with_index(0, 10), shared_store_body(0, 10)],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(4)],
            },
        };
        let out = loop_fusion(&desc);
        let loops = out
            .body
            .ops
            .iter()
            .filter(|op| matches!(op.kind, KernelOpKind::StructuredForLoop { .. }))
            .count();
        assert_eq!(loops, 1);
    }

    fn store_body_with_index(binding: u32, index: u32) -> KernelBody {
        KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![binding, index, 0],
                result: None,
            }],
            child_bodies: vec![],
            literals: vec![],
        }
    }

    fn shared_store_body(binding: u32, index: u32) -> KernelBody {
        KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::StoreShared,
                operands: vec![binding, index, 0],
                result: None,
            }],
            child_bodies: vec![],
            literals: vec![],
        }
    }
}
