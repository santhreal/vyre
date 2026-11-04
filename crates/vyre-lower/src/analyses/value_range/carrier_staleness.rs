//! Position from which a named-carrier snapshot stops describing its slot.
//!
//! A source-level variable mutated inside a nested body is lowered to a named
//! carrier, and the SSA id that seeded the slot keeps reading as a known
//! constant long after the slot holds something else. This module owns the
//! op index past which each such id must no longer be believed. It derives no
//! ranges.

use rustc_hash::FxHashMap;

use crate::descriptor::Name;
use crate::operand_class::{classify_operand, OperandClass};
use crate::{KernelBody, KernelOpKind};

/// For each id that snapshots a named carrier slot, the op index from which
/// that snapshot may be stale.
///
/// # Why this exists
///
/// The lowering represents a source-level variable mutated inside a nested
/// body as a **named carrier**: a `LoopCarrierInit { name }` seeds the slot
/// from a pre-construct SSA value, `LoopCarrierEnd { name }` commits a new
/// value to it, and `LoopCarrier { name }` reads it back. The read is a
/// fresh operand-less op, so it never carries a derived range and the
/// analysis is already conservative about it.
///
/// The **seed** is the hazard. It is an ordinary SSA id, frequently a
/// `Literal`, so `ranges` knows it exactly. Before the mutating construct
/// runs, that range genuinely describes the variable, and collapsing a
/// guard on it is a sound optimization worth keeping. After the construct
/// runs, the variable may hold something else entirely while the seed id
/// still reads as a known constant. Anything that treats the seed as the
/// variable's current value past that point miscompiles.
///
/// So the answer has to be position-sensitive, which is why this returns an
/// op index per id rather than just deleting the range. Deleting it would
/// also disable the sound pre-construct collapse: on the shape that
/// motivated this analysis (`let end = 0; if (end == 0) { .. end = a .. }`)
/// the guard reads the seed one op before the construct that mutates it.
///
/// # Conservatism
///
/// Fail-closed in three ways. A construct is treated as mutating a carrier
/// if a `LoopCarrierEnd` for that name appears **anywhere** in its subtree,
/// at any depth, without asking whether that write is reachable. Every id
/// tied to a name is invalidated from the **earliest** mutation of that
/// name, whatever the order of the tying ops. And a value committed by
/// `LoopCarrierEnd` is tied to the name just as a seed is, since a later
/// write to the same slot makes it equally stale.
pub(super) fn carrier_snapshot_invalidations(body: &KernelBody) -> FxHashMap<u32, usize> {
    // Earliest op index at which each carrier name can be written.
    let mut earliest_write: FxHashMap<Name, usize> = FxHashMap::default();
    let mut note = |name: &Name, index: usize| {
        earliest_write
            .entry(name.clone())
            .and_modify(|slot| *slot = (*slot).min(index))
            .or_insert(index);
    };

    for (index, op) in body.ops.iter().enumerate() {
        match &op.kind {
            // A write at this level: reads after it are stale.
            KernelOpKind::LoopCarrierEnd { name } => note(name, index),
            // A construct whose body may write: reads after the whole
            // construct are stale. Its own operands (the condition, the
            // loop bounds) are evaluated before the body runs, so `index`
            // itself is still safe and the +1 lands in `invalidated_from`.
            _ => {
                for (pos, &operand) in op.operands.iter().enumerate() {
                    if classify_operand(&op.kind, pos) != OperandClass::ChildBodyIdx {
                        continue;
                    }
                    let Some(child) = body.child_bodies.get(operand as usize) else {
                        continue;
                    };
                    for name in carrier_writes_in_subtree(child) {
                        note(&name, index);
                    }
                }
            }
        }
    }

    if earliest_write.is_empty() {
        return FxHashMap::default();
    }

    let mut invalidated: FxHashMap<u32, usize> = FxHashMap::default();
    for op in &body.ops {
        let (name, tied_id) = match (&op.kind, op.operands.first()) {
            (KernelOpKind::LoopCarrierInit { name }, Some(&seed)) => (name, seed),
            (KernelOpKind::LoopCarrierEnd { name }, Some(&committed)) => (name, committed),
            _ => continue,
        };
        let Some(&write_index) = earliest_write.get(name) else {
            continue;
        };
        let from = write_index.saturating_add(1);
        invalidated
            .entry(tied_id)
            .and_modify(|slot| *slot = (*slot).min(from))
            .or_insert(from);
    }
    invalidated
}

/// Every carrier name written by a `LoopCarrierEnd` in `body` or any
/// descendant body.
fn carrier_writes_in_subtree(body: &KernelBody) -> Vec<Name> {
    let mut out = Vec::new();
    collect_carrier_writes(body, &mut out);
    out
}

fn collect_carrier_writes(body: &KernelBody, out: &mut Vec<Name>) {
    for op in &body.ops {
        if let KernelOpKind::LoopCarrierEnd { name } = &op.kind {
            out.push(name.clone());
        }
    }
    for child in &body.child_bodies {
        collect_carrier_writes(child, out);
    }
}
