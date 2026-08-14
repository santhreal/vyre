//! Common-subexpression detection on `KernelDescriptor`.
//!
//! Detects pairs of ops that compute the same value  -  same `KernelOpKind`,
//! same operand list  -  and could share a single result instead of
//! recomputing.
//!
//! Returns groups of equivalent ops. The descriptor CSE rewrite picks a
//! canonical op per group and rewrites every subsequent reference to point at
//! the canonical.
//!
//! ## Soundness note
//!
//! Most ops are keyed by exact operand order. A small allow-list of operations
//! with bit-exact symmetric semantics is keyed with sorted binary operands so
//! `xor(x, y)` and `xor(y, x)` share a group. Arithmetic add/mul are not
//! normalized here because this descriptor layer does not carry enough
//! dtype/FP-mode context to prove bit-identical results across all backends.

use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use vyre_foundation::ir::BinOp;

/// One set of operations with identical value semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EquivalenceGroup {
    /// Op-indices that all compute the same value. The first element
    /// is the canonical op (chosen by lowest op-index); the rest are
    /// the duplicates that could be eliminated.
    pub op_indices: Vec<usize>,
}

/// Common-subexpression analysis for one kernel body.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommonSubexprReport {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Equivalent operation groups in deterministic body order.
    pub groups: Vec<EquivalenceGroup>,
}

impl CommonSubexprReport {
    /// Number of ops that could be eliminated if every group is
    /// canonicalized: total ops in groups minus number of groups.
    #[must_use]
    pub fn ops_eliminable(&self) -> usize {
        self.groups
            .iter()
            .map(|g| g.op_indices.len().saturating_sub(1))
            .sum()
    }
}

/// Analyze a descriptor and all nested bodies for common subexpressions.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> CommonSubexprReport {
    analyze_body(desc.id.clone(), &desc.body)
}

/// Analyze one body and its descendants for common subexpressions.
#[must_use]
pub fn analyze_body(kernel_id: String, body: &KernelBody) -> CommonSubexprReport {
    analyze_body_impl(kernel_id, body, true)
}

/// Analyze only the operations directly owned by one body.
#[must_use]
pub fn analyze_body_shallow(kernel_id: String, body: &KernelBody) -> CommonSubexprReport {
    analyze_body_impl(kernel_id, body, false)
}

fn analyze_body_impl(
    kernel_id: String,
    body: &KernelBody,
    include_children: bool,
) -> CommonSubexprReport {
    let mut buckets: FxHashMap<OpKey, Vec<usize>> = FxHashMap::default();
    let mut next_index = 0usize;
    if include_children {
        walk_body(body, &mut buckets, &mut next_index);
    } else {
        walk_ops(body, &mut buckets, &mut next_index);
    }

    let groups = buckets
        .into_iter()
        .filter(|(_, idxs)| idxs.len() >= 2)
        .map(|(_, op_indices)| EquivalenceGroup { op_indices })
        .collect();

    CommonSubexprReport { kernel_id, groups }
}

fn walk_body(
    body: &KernelBody,
    buckets: &mut FxHashMap<OpKey, Vec<usize>>,
    next_index: &mut usize,
) {
    walk_ops(body, buckets, next_index);
    for child in &body.child_bodies {
        walk_body(child, buckets, next_index);
    }
}

fn walk_ops(body: &KernelBody, buckets: &mut FxHashMap<OpKey, Vec<usize>>, next_index: &mut usize) {
    for op in &body.ops {
        let op_index = *next_index;
        *next_index = next_index.saturating_add(1);
        // Side-effect ops (stores, barriers, etc.) are NEVER candidates
        // for CSE  -  repeating them is the user's intent, not redundancy.
        if !is_eligible(&op.kind) {
            continue;
        }
        let key = OpKey::from_op(op);
        buckets.entry(key).or_default().push(op_index);
    }
}

fn is_eligible(kind: &KernelOpKind) -> bool {
    matches!(
        kind,
        KernelOpKind::Literal
            | KernelOpKind::LocalInvocationId
            | KernelOpKind::GlobalInvocationId
            | KernelOpKind::WorkgroupId
            | KernelOpKind::SubgroupLocalId
            | KernelOpKind::SubgroupSize
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::Fma
            | KernelOpKind::Select
            | KernelOpKind::Cast { .. }
            | KernelOpKind::BufferLength
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpKey {
    kind: KernelOpKind,
    operands: SmallVec<[u32; 4]>,
}

impl OpKey {
    fn from_op(op: &KernelOp) -> Self {
        let mut operands = SmallVec::from_slice(&op.operands);
        if let KernelOpKind::BinOpKind(bin_op) = &op.kind {
            normalize_commutative_operands(*bin_op, &mut operands);
        }
        Self {
            kind: op.kind.clone(),
            operands,
        }
    }
}

fn normalize_commutative_operands(bin_op: BinOp, operands: &mut SmallVec<[u32; 4]>) {
    if operands.len() != 2 || !is_bit_exact_commutative_binop(bin_op) {
        return;
    }
    if operands[0] > operands[1] {
        operands.swap(0, 1);
    }
}

fn is_bit_exact_commutative_binop(bin_op: BinOp) -> bool {
    matches!(
        bin_op,
        BinOp::BitAnd
            | BinOp::BitOr
            | BinOp::BitXor
            | BinOp::Eq
            | BinOp::Ne
            | BinOp::And
            | BinOp::Or
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{binop, body, descriptor, global_wo, lit, op, store_global};
    use crate::LiteralValue;
    use vyre_foundation::ir::{BinOp, DataType};

    #[test]
    fn empty_kernel_no_groups() {
        let desc = descriptor("k").build();
        let r = analyze(&desc);
        assert!(r.groups.is_empty());
        assert_eq!(r.ops_eliminable(), 0);
    }

    #[test]
    fn two_identical_literals_form_group() {
        let desc = descriptor("dup_lit")
            .body(
                body()
                    .literals([LiteralValue::U32(7)])
                    .op(lit(0, 0))
                    .op(lit(0, 1)),
            )
            .build();
        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].op_indices, vec![0, 1]);
        assert_eq!(r.ops_eliminable(), 1);
    }

    #[test]
    fn distinct_literal_pool_indices_are_distinct() {
        let desc = descriptor("two_lits")
            .body(
                body()
                    .literals([LiteralValue::U32(7), LiteralValue::U32(8)])
                    .op(lit(0, 0))
                    .op(lit(1, 1)),
            )
            .build();
        let r = analyze(&desc);
        assert!(r.groups.is_empty());
    }

    #[test]
    fn duplicate_binop_with_same_operands_grouped() {
        let desc = descriptor("dup_add")
            .body(
                body()
                    .literals([LiteralValue::U32(3), LiteralValue::U32(4)])
                    .op(lit(0, 0))
                    .op(lit(1, 1))
                    .op(binop(BinOp::Add, 0, 1, 2))
                    .op(binop(BinOp::Add, 0, 1, 3)),
            )
            .build();
        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].op_indices, vec![2, 3]);
    }

    #[test]
    fn arithmetic_commutative_swap_not_grouped_without_type_context() {
        // Add may be integer, wrapping, or floating point at this layer. Keep
        // it order-sensitive until descriptor ops carry enough semantic context
        // to prove bit-identical results for every backend.
        let desc = descriptor("comm")
            .body(
                body()
                    .literals([LiteralValue::U32(3), LiteralValue::U32(4)])
                    .op(lit(0, 0))
                    .op(lit(1, 1))
                    .op(binop(BinOp::Add, 0, 1, 2))
                    .op(binop(BinOp::Add, 1, 0, 3)),
            )
            .build();
        let r = analyze(&desc);
        assert!(
            r.groups.is_empty(),
            "descriptor CSE must not normalize arithmetic add without dtype context"
        );
    }

    #[test]
    fn bit_exact_commutative_swap_is_grouped() {
        let desc = descriptor("comm_bitxor")
            .body(
                body()
                    .literals([LiteralValue::U32(3), LiteralValue::U32(4)])
                    .op(lit(0, 0))
                    .op(lit(1, 1))
                    .op(binop(BinOp::BitXor, 0, 1, 2))
                    .op(binop(BinOp::BitXor, 1, 0, 3)),
            )
            .build();
        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].op_indices, vec![2, 3]);
    }

    #[test]
    fn store_ops_not_grouped_even_if_identical() {
        // Two identical stores must NOT be CSE'd: they are side effects.
        let desc = descriptor("double_store")
            .slot(global_wo(0, DataType::U32, "out"))
            .body(
                body()
                    .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                    .op(lit(0, 0))
                    .op(lit(1, 1))
                    .op(store_global(0, 0, 1))
                    .op(store_global(0, 0, 1)),
            )
            .build();
        let r = analyze(&desc);
        let store_groups = r
            .groups
            .iter()
            .filter(|g| {
                g.op_indices
                    .iter()
                    .any(|&i| matches!(desc.body.ops[i].kind, KernelOpKind::StoreGlobal))
            })
            .count();
        assert_eq!(store_groups, 0);
    }

    #[test]
    fn three_identical_literals_eliminate_two() {
        let desc = descriptor("three")
            .body(
                body()
                    .literals([LiteralValue::U32(42)])
                    .op(lit(0, 0))
                    .op(lit(0, 1))
                    .op(lit(0, 2)),
            )
            .build();
        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].op_indices, vec![0, 1, 2]);
        assert_eq!(r.ops_eliminable(), 2); // 3 ops, keep 1, eliminate 2
    }

    #[test]
    fn local_invocation_id_calls_grouped() {
        // Two LocalInvocationId calls are equivalent (constant per
        // thread, same in any order).
        let desc = descriptor("tid_dup")
            .dispatch(64, 1, 1)
            .body(body().op(op(KernelOpKind::LocalInvocationId, [], 0)).op(op(
                KernelOpKind::LocalInvocationId,
                [],
                1,
            )))
            .build();
        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
    }

    #[test]
    fn sibling_child_body_indices_are_monotonic_not_overlapping() {
        let child = body().op(lit(0, 1)).build();
        let desc = descriptor("siblings")
            .body(
                body()
                    .literals([LiteralValue::U32(9)])
                    .op(lit(0, 0))
                    .children([child.clone(), child]),
            )
            .build();

        let r = analyze(&desc);
        assert_eq!(r.groups.len(), 1);
        assert_eq!(
            r.groups[0].op_indices,
            vec![0, 1, 2],
            "sibling children must receive distinct preorder op indices"
        );
    }

    #[test]
    fn shallow_analysis_excludes_child_bodies() {
        let tree = body()
            .literals([LiteralValue::U32(9)])
            .op(lit(0, 0))
            .child(body().op(lit(0, 1)))
            .build();

        let recursive = analyze_body("recursive".into(), &tree);
        let shallow = analyze_body_shallow("shallow".into(), &tree);
        assert_eq!(recursive.groups.len(), 1);
        assert!(shallow.groups.is_empty());
    }
}
