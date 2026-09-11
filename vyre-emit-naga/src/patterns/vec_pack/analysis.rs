//! Analysis pass: walk a `KernelDescriptor`, identify groups of
//! adjacent loads/stores that can fuse into one packed vector op.
//!
//! ## Algorithm (phase 1)
//!
//! 1. Scan the op stream linearly.
//! 2. When we hit a `LoadGlobal` or `StoreGlobal`, look at the next
//!    1, 2, or 3 ops. If they are loads/stores of the same kind, on
//!    the same buffer, with indices that differ by exactly +1 (i.e.
//!    consecutive offsets), and the same target dtype, they form a
//!    pack group.
//! 3. Greedy maximal grouping: prefer Vec4 over Vec2 when possible.
//! 4. Skip any group that crosses a side-effecting op on the same
//!    buffer (RAW/WAR hazard).
//!
//! Index-difference detection is conservative  -  phase 1 only proves
//! `+1` increment when the index expressions both decompose as
//! `Add(<same base>, LiteralU32(<k>))` and `Add(<same base>, LiteralU32(<k+1>))`,
//! OR when one is `<base>` and the other is `Add(<base>, LiteralU32(1))`.
//! Anything more exotic (multi-term polynomials, runtime base) falls
//! through to "not packable"  -  phase 2 may upgrade.

use super::plan::{PackGroup, PackKind, PackingPlan};
use vyre_foundation::ir::BinOp;
use vyre_lower::analyses::AccessKind;
use vyre_lower::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};

/// Run vec-packing analysis.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> PackingPlan {
    let mut groups = Vec::new();
    walk_body(&desc.body, &mut groups);
    PackingPlan {
        kernel_id: desc.id.clone(),
        groups,
    }
}

fn walk_body(body: &KernelBody, groups: &mut Vec<PackGroup>) {
    // Phase 1: collect every Load/Store op with its decomposed index.
    // Upper bound on accesses is body.ops.len() (one record per op at
    // most). Pre-sizing avoids three-to-four reallocations on the
    // typical megakernel body (dozens to hundreds of ops).
    let mut accesses: Vec<AccessRecord> = Vec::with_capacity(body.ops.len());
    for (op_idx, op) in body.ops.iter().enumerate() {
        if let Some(kind) = access_kind(&op.kind) {
            if op.operands.len() < 2 {
                continue;
            }
            let buffer_slot = op.operands[0];
            let index_id = op.operands[1];
            if let Some((base, offset)) = decompose_index(body, index_id) {
                accesses.push(AccessRecord {
                    op_index: op_idx,
                    kind,
                    buffer_slot,
                    base,
                    offset,
                });
            }
        }
        // Recurse into structured children.
        if matches!(
            op.kind,
            KernelOpKind::StructuredIfThen | KernelOpKind::StructuredForLoop { .. }
        ) {
            if let Some(child_id) = op.operands.last() {
                if let Some(child) = body.child_bodies.get(*child_id as usize) {
                    walk_body(child, groups);
                }
            }
        }
    }

    // Phase 2: greedily group consecutive accesses (in collection order)
    // that match same kind + same buffer + same base + offsets that
    // form `k, k+1, k+2, ...`. Max group size 4 (Vec4).
    //
    // Hazard barrier: a store between loads to the same buffer breaks
    // the chain (RAW). We track per-buffer "last-store-position" and
    // refuse to grow a group across one.
    let mut i = 0;
    while i < accesses.len() {
        let start = &accesses[i];
        let mut size = 1;
        while size < 4 && i + size < accesses.len() {
            let prev = &accesses[i + size - 1];
            let next = &accesses[i + size];
            if next.kind != start.kind
                || next.buffer_slot != start.buffer_slot
                || next.base != start.base
                || next.offset != prev.offset + 1
            {
                break;
            }
            // Hazard check: the group steps over every op between prev and
            // next, so one it cannot be reordered against stops it here.
            if hazard_between(
                body,
                &accesses,
                start.kind,
                i + size - 1,
                i + size,
                start.buffer_slot,
            ) {
                break;
            }
            size += 1;
        }
        if size >= 2 {
            let pack = match size {
                2 => PackKind::Vec2,
                3 => PackKind::Vec3,
                _ => PackKind::Vec4,
            };
            groups.push(PackGroup {
                start_op_index: accesses[i].op_index,
                end_op_index: accesses[i + size - 1].op_index,
                kind: start.kind,
                binding_slot: start.buffer_slot,
                pack,
            });
            i += size;
        } else {
            i += 1;
        }
    }
}

#[derive(Debug)]
struct AccessRecord {
    op_index: usize,
    kind: AccessKind,
    buffer_slot: u32,
    base: u32,
    offset: u32,
}

fn access_kind(kind: &KernelOpKind) -> Option<AccessKind> {
    match kind {
        KernelOpKind::LoadGlobal => Some(AccessKind::Load),
        KernelOpKind::StoreGlobal => Some(AccessKind::Store),
        _ => None,
    }
}

/// Whether an op the group cannot be reordered against sits strictly between
/// two consecutive members.
///
/// Members are adjacent entries of `accesses`, which records every global load
/// and store in the body, so no other global load or store can sit between
/// them. What can sit between them is a barrier, an atomic, a vector access, a
/// branch, or a cross-lane op, and fusing the group moves both members to one
/// point in the stream past all of it. The version before this one scanned
/// `accesses` instead of the op stream and therefore never reported anything:
/// the only records in that window were the two members it was called with.
fn hazard_between(
    body: &KernelBody,
    accesses: &[AccessRecord],
    group_kind: AccessKind,
    prev_idx: usize,
    next_idx: usize,
    buffer_slot: u32,
) -> bool {
    let first = accesses[prev_idx].op_index + 1;
    let last = accesses[next_idx].op_index;
    body.ops[first..last]
        .iter()
        .any(|op| !reorder_safe_over_group(op, group_kind, buffer_slot))
}

/// Whether a fused access of `group_kind` on `buffer_slot` may cross `op`.
///
/// A load group may cross a read of its own binding, because two reads of one
/// buffer commute. A store group may cross neither a read nor a write of it: an
/// intervening read would observe a value the fused store writes later, and an
/// intervening write would land out of order at an index this analysis cannot
/// prove distinct. Neither kind may cross a barrier, an atomic, an async
/// transaction, a branch, a loop-carrier boundary, or a cross-lane op.
///
/// Every kind whose verdict differs from the one `vyre_lower::facts_for`
/// implies has an arm of its own, and every remaining kind is graded by that
/// fact table, which is the workspace's only enumeration of `KernelOpKind`. A
/// kind added there without a fact stops `vyre-lower` from compiling, and a
/// cross-lane kind added to its removable arm has to be named below too,
/// because the fact table reads "no memory effect" and this analysis also
/// needs "no ordering meaning".
fn reorder_safe_over_group(op: &KernelOp, group_kind: AccessKind, buffer_slot: u32) -> bool {
    /// Whether `op` names `buffer_slot` as the binding it accesses.
    ///
    /// Every global memory op carries the binding slot as operand 0.
    fn same_slot(op: &KernelOp, buffer_slot: u32) -> bool {
        op.operands.first() == Some(&buffer_slot)
    }

    match op.kind {
        // Reads of the group's own binding. A load group commutes with them; a
        // store group would change what they observe.
        KernelOpKind::LoadGlobal | KernelOpKind::VectorLoadGlobal { .. } => {
            group_kind == AccessKind::Load || !same_slot(op, buffer_slot)
        }

        // Writes and read-modify-writes of the group's own binding.
        KernelOpKind::StoreGlobal
        | KernelOpKind::VectorStoreGlobal { .. }
        | KernelOpKind::Atomic { .. } => !same_slot(op, buffer_slot),

        // Workgroup-shared memory is a separate address space and never
        // aliases a global binding, so a global group crosses it freely. The
        // barrier that makes a shared write visible is graded by the fact
        // table, which retains it.
        KernelOpKind::LoadShared | KernelOpKind::StoreShared => true,

        // Cross-lane communication and a loop-carrier read produce a value and
        // write nothing, so the fact table grades them removable, but a fused
        // access still may not cross them: a shuffle reads a neighbour lane
        // whose access this fusion moves, and a carrier read is an iteration
        // boundary.
        KernelOpKind::SubgroupBallot
        | KernelOpKind::SubgroupShuffle
        | KernelOpKind::SubgroupBroadcast
        | KernelOpKind::SubgroupReduce { .. }
        | KernelOpKind::LoopCarrier { .. } => false,

        // A region names a child body this analysis does not walk, which is
        // why the fact table retains it. The fused access is emitted on the
        // same side of the region as the scalar accesses it replaces, so it
        // crosses one freely.
        KernelOpKind::Region { .. } => true,

        // Ordering, control flow, iteration boundaries, opaque effects, and
        // pure value production. A kind the fact table retains carries a
        // nested body, a memory effect or a backend contract, and a fused
        // access crosses none of those whatever binding it names. A kind it
        // reports removable when its results are unused computes or reads a
        // value and nothing else.
        _ => !vyre_lower::facts_for(&op.kind).retained_effect,
    }
}

/// Decompose an index expression into `(base_operand_id, constant_offset)`.
/// `Some((base, 0))` means the index IS the base.
/// `Some((base, k))` means `Add(base, LiteralU32(k))`.
/// `None` means the index isn't recognizable in phase 1.
fn decompose_index(body: &KernelBody, operand_id: u32) -> Option<(u32, u32)> {
    let producer = body.ops.iter().rfind(|op| op.result == Some(operand_id))?;
    match producer.kind {
        KernelOpKind::BinOpKind(BinOp::Add) => {
            if producer.operands.len() != 2 {
                return None;
            }
            let lhs_const = literal_u32_value(body, producer.operands[0]);
            let rhs_const = literal_u32_value(body, producer.operands[1]);
            match (lhs_const, rhs_const) {
                (Some(k), None) => Some((producer.operands[1], k)),
                (None, Some(k)) => Some((producer.operands[0], k)),
                _ => None,
            }
        }
        // Anything else: treat the operand id itself as the "base"
        // with constant offset 0. This lets us match the case
        // (prev = base, next = Add(base, 1)).
        _ => Some((operand_id, 0)),
    }
}

fn literal_u32_value(body: &KernelBody, operand_id: u32) -> Option<u32> {
    let producer = body.ops.iter().rfind(|op| op.result == Some(operand_id))?;
    if producer.kind != KernelOpKind::Literal {
        return None;
    }
    let pool_idx = producer.operands.first()?;
    match body.literals.get(*pool_idx as usize)? {
        LiteralValue::U32(v) => Some(*v),
        _ => None,
    }
}
