//! Shared store race legality analysis implementation.

use super::report::{SharedStoreLegality, SharedStoreRaceReport, SharedStoreRaceSite};
use crate::analyses::structured_walk::ArmDescent;
use crate::analyses::ProducerMap;
use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass};
use vyre_foundation::ir::BinOp;

/// Analyze a kernel descriptor for multi-invocation constant-index `StoreShared` data races.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> SharedStoreRaceReport {
    let invocations: u64 = u64::from(desc.dispatch.workgroup_size[0])
        .saturating_mul(u64::from(desc.dispatch.workgroup_size[1]))
        .saturating_mul(u64::from(desc.dispatch.workgroup_size[2]));

    let mut collector = SharedStoreCollector {
        desc,
        is_single_invocation: invocations <= 1,
        sites: Vec::new(),
    };

    walk_body_with_guards(&desc.body, ArmDescent::Enter, &mut collector, false);

    SharedStoreRaceReport {
        kernel_id: desc.id.clone(),
        sites: collector.sites,
    }
}

struct SharedStoreCollector<'a> {
    desc: &'a KernelDescriptor,
    is_single_invocation: bool,
    sites: Vec<SharedStoreRaceSite>,
}

fn walk_body_with_guards<'a>(
    body: &'a KernelBody,
    arms: ArmDescent,
    collector: &mut SharedStoreCollector<'a>,
    in_single_invocation_guard: bool,
) {
    let producers = crate::analyses::producer_map(body);
    for (local_idx, op) in body.ops.iter().enumerate() {
        match &op.kind {
            KernelOpKind::StoreShared => {
                let binding_slot = op.operands.first().copied().unwrap_or(0);
                let is_shared = collector.desc.bindings.slots.iter().any(|b| {
                    b.slot == binding_slot && matches!(b.memory_class, MemoryClass::Shared)
                });
                if is_shared {
                    let index_op = op.operands.get(1).copied().unwrap_or(0);
                    let legality = if collector.is_single_invocation || in_single_invocation_guard {
                        SharedStoreLegality::RaceFreeSingleInvocation
                    } else if is_thread_varying_index(body, &producers, index_op) {
                        SharedStoreLegality::RaceFreeDistinctIndices
                    } else {
                        SharedStoreLegality::IllegalMultiInvocationConstantStore
                    };

                    collector.sites.push(SharedStoreRaceSite {
                        op_index: local_idx,
                        binding_slot,
                        legality,
                    });
                }
            }
            KernelOpKind::Atomic { .. } => {
                let binding_slot = op.operands.first().copied().unwrap_or(0);
                let is_shared = collector.desc.bindings.slots.iter().any(|b| {
                    b.slot == binding_slot && matches!(b.memory_class, MemoryClass::Shared)
                });
                if is_shared {
                    collector.sites.push(SharedStoreRaceSite {
                        op_index: local_idx,
                        binding_slot,
                        legality: SharedStoreLegality::RaceFreeAtomic,
                    });
                }
            }
            KernelOpKind::StructuredIfThen => {
                if arms == ArmDescent::Enter {
                    let cond_op = op.operands.first().copied().unwrap_or(0);
                    let child_body_idx = op.operands.get(1).copied().unwrap_or(0);
                    let guard = in_single_invocation_guard
                        || is_single_invocation_cond(body, &producers, cond_op);
                    if let Some(child) = body.child_bodies.get(child_body_idx as usize) {
                        walk_body_with_guards(child, arms, collector, guard);
                    }
                }
            }
            KernelOpKind::StructuredIfThenElse => {
                if arms == ArmDescent::Enter {
                    let cond_op = op.operands.first().copied().unwrap_or(0);
                    let then_idx = op.operands.get(1).copied().unwrap_or(0);
                    let else_idx = op.operands.get(2).copied().unwrap_or(0);
                    let guard_then = in_single_invocation_guard
                        || is_single_invocation_cond(body, &producers, cond_op);
                    if let Some(child) = body.child_bodies.get(then_idx as usize) {
                        walk_body_with_guards(child, arms, collector, guard_then);
                    }
                    if let Some(child) = body.child_bodies.get(else_idx as usize) {
                        walk_body_with_guards(child, arms, collector, in_single_invocation_guard);
                    }
                }
            }
            _ => {
                for child_idx in crate::analyses::child_body_operands(&op.kind, &op.operands) {
                    if let Some(child) = body.child_bodies.get(child_idx as usize) {
                        walk_body_with_guards(child, arms, collector, in_single_invocation_guard);
                    }
                }
            }
        }
    }
}

fn is_literal_zero(body: &KernelBody, op: &KernelOp) -> bool {
    if op.kind != KernelOpKind::Literal {
        return false;
    }
    let Some(pool_idx) = op.operands.first().copied() else {
        return false;
    };
    match body.literals.get(pool_idx as usize) {
        Some(LiteralValue::U32(v)) => *v == 0,
        Some(LiteralValue::I32(v)) => *v == 0,
        Some(LiteralValue::F32(v)) => *v == 0.0,
        _ => false,
    }
}
fn is_single_invocation_cond(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    cond_op_id: u32,
) -> bool {
    let Some(producer) = producers.get(&cond_op_id).copied() else {
        return false;
    };
    match &producer.kind {
        KernelOpKind::BinOpKind(BinOp::Eq) => {
            let left_op = producer.operands.first().copied().unwrap_or(0);
            let right_op = producer.operands.get(1).copied().unwrap_or(0);
            let left = producers.get(&left_op).copied();
            let right = producers.get(&right_op).copied();
            let is_local_id = |op: Option<&KernelOp>| {
                matches!(
                    op.map(|k| &k.kind),
                    Some(KernelOpKind::LocalInvocationId | KernelOpKind::GlobalInvocationId)
                )
            };
            (is_local_id(left) && right.is_some_and(|r| is_literal_zero(body, r)))
                || (is_local_id(right) && left.is_some_and(|l| is_literal_zero(body, l)))
        }
        _ => false,
    }
}

fn is_thread_varying_index(
    body: &KernelBody,
    producers: &ProducerMap<'_>,
    index_op_id: u32,
) -> bool {
    let Some(producer) = producers.get(&index_op_id).copied() else {
        return false;
    };
    match &producer.kind {
        KernelOpKind::LocalInvocationId
        | KernelOpKind::GlobalInvocationId
        | KernelOpKind::SubgroupLocalId => true,
        KernelOpKind::Literal => false,
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd | BinOp::Mul) => producer
            .operands
            .iter()
            .any(|&operand| is_thread_varying_index(body, producers, operand)),
        _ => false,
    }
}
