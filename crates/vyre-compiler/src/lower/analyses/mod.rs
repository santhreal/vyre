//! Read-only, backend-neutral analyses on `KernelDescriptor`.
//!
//! These analyses run after verified lowering and before emission. They report
//! descriptor facts and candidate target strategies without changing program
//! semantics or descriptor structure. Concrete emission strategy lives in the
//! owning emitter or driver.

pub(crate) mod access_kind;
pub(crate) mod bank_conflict;
pub mod structured_walk;

use crate::lower::{KernelBody, KernelOp, KernelOpKind};
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
        Some(crate::lower::LiteralValue::U32(value)) => Some(*value),
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
/// [`crate::lower::op_facts::facts_for`], which is the crate's only enumeration of
/// `KernelOpKind` and has no wildcard arm: a new variant that carries a nested
/// body fails to compile until someone states where its child indices begin,
/// rather than silently stopping every analysis from descending into it.
pub fn child_body_operands<'a>(
    kind: &KernelOpKind,
    operands: &'a [u32],
) -> impl Iterator<Item = u32> + 'a {
    let start = crate::lower::op_facts::facts_for(kind)
        .nested_bodies
        .map_or(operands.len(), |bodies| bodies.start);
    operands.iter().skip(start).copied()
}

// Analyses read by emitters and drivers.
pub use access_kind::AccessKind;
pub use bank_conflict::{
    derive_shared_access_profiles, evaluate_mitigation_candidate, select_bank_conflict_strategy,
    AccessPhase, AccessPhaseProfile, BankConflictMitigation, ConflictSeverity,
    MitigationEvaluation, PhaseConflictReport, SharedBindingAccessProfile, SharedPermutationBlock,
    TargetBankGeometry, CANDIDATE_MITIGATIONS,
};
