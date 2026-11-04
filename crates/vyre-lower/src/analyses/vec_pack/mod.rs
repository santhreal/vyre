//! B1 substrate: adjacent-load packing analysis.
//!
//! Detects chains of `LoadGlobal` ops on the same binding slot whose
//! normalized indices share one base and have consecutive offsets
//! (`base+i`, `base+i+1`, `base+i+2`, `base+i+3`). Such chains are
//! candidates for vec2/vec4 packed loads  -  one wide
//! transaction instead of N narrow ones, saving (N-1) memory
//! request slots and improving coalescing.
//!
//! Pure analysis on a [`KernelDescriptor`]. The actual rewrite
//! (collapse N adjacent Loads into one wide Load + N AccessIndex
//! projections) is downstream work in `vyre-lower::rewrites`. This
//! substrate just produces the per-body chain inventory.

use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};
use rustc_hash::FxHashMap;
use vyre_foundation::ir::BinOp;

type LoadsBySlotAndBase = FxHashMap<(u32, Option<u32>), Vec<(u32, usize)>>;

/// One detected adjacent-load chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VecPackChain {
    /// The binding slot all loads in the chain target.
    pub slot: u32,
    /// Op indices in the body, in chain order. Length is the
    /// chain length (always >= 2  -  single loads are not a chain).
    pub op_indices: Vec<usize>,
    /// Starting literal index or constant offset. Subsequent loads target
    /// `start_index + 1`, `+ 2`, ...
    pub start_index: u32,
}

impl VecPackChain {
    /// The width of the packed load this chain enables (2, 3, or
    /// 4 depending on chain length, capped at 4).
    #[must_use]
    pub fn pack_width(&self) -> u32 {
        let len = self.op_indices.len() as u32;
        len.min(4)
    }
}

/// Per-body inventory of vec-pack chains.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VecPackReport {
    /// All chains in the body, sorted by `(slot, start_index)`.
    pub chains: Vec<VecPackChain>,
    /// Total ops eliminated if every chain were packed at its
    /// maximum width.
    pub total_ops_eliminated: u32,
}

impl VecPackReport {
    /// True iff at least one chain was detected.
    #[must_use]
    pub fn has_chains(&self) -> bool {
        !self.chains.is_empty()
    }
}

/// Analyse `desc.body` and return the vec-pack chain inventory.
///
/// O(ops + candidates log candidates) per body. Pure: no allocation outside
/// analysis-local tables and the returned report.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> VecPackReport {
    let mut report = VecPackReport::default();
    walk_body(&desc.body, &mut report);
    report.chains.sort_by_key(|c| (c.slot, c.start_index));
    report.total_ops_eliminated = report
        .chains
        .iter()
        .map(|c| (c.op_indices.len() as u32).saturating_sub(1))
        .sum();
    report
}

fn walk_body(body: &KernelBody, report: &mut VecPackReport) {
    detect_chains_in_body(body, report);
    for child in &body.child_bodies {
        walk_body(child, report);
    }
}

fn detect_chains_in_body(body: &KernelBody, report: &mut VecPackReport) {
    let indices = index_expr_by_result(body);
    let mut by_slot_and_base =
        LoadsBySlotAndBase::with_capacity_and_hasher(body.ops.len(), Default::default());

    for (op_idx, op) in body.ops.iter().enumerate() {
        let Some((slot, index)) = load_with_index_expr(op, &indices) else {
            continue;
        };
        by_slot_and_base
            .entry((slot, index.base_result))
            .or_default()
            .push((index.offset, op_idx));
    }

    for ((slot, _base_result), mut candidates) in by_slot_and_base {
        candidates.sort_unstable_by_key(|(offset, op_idx)| (*offset, *op_idx));
        let mut run_start = 0usize;
        while run_start < candidates.len() {
            let mut run_end = run_start + 1;
            while run_end < candidates.len()
                && candidates[run_end].0 == candidates[run_end - 1].0.saturating_add(1)
            {
                run_end += 1;
            }

            if run_end - run_start >= 2 {
                report.chains.push(VecPackChain {
                    slot,
                    op_indices: candidates[run_start..run_end]
                        .iter()
                        .map(|(_, op_idx)| *op_idx)
                        .collect(),
                    start_index: candidates[run_start].0,
                });
            }
            run_start = run_end;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IndexExpr {
    pub(crate) base_result: Option<u32>,
    pub(crate) offset: u32,
}

pub(crate) fn index_expr_by_result(body: &KernelBody) -> FxHashMap<u32, IndexExpr> {
    let mut out = FxHashMap::with_capacity_and_hasher(body.ops.len(), Default::default());
    for op in &body.ops {
        let Some(result) = op.result else {
            continue;
        };
        if let Some(expr) = literal_index_expr(op, body).or_else(|| add_index_expr(op, &out)) {
            out.insert(result, expr);
        } else {
            out.insert(
                result,
                IndexExpr {
                    base_result: Some(result),
                    offset: 0,
                },
            );
        }
    }
    out
}

pub(crate) fn literal_index_expr(op: &KernelOp, body: &KernelBody) -> Option<IndexExpr> {
    if !matches!(op.kind, KernelOpKind::Literal) {
        return None;
    }
    let pool_idx = *op.operands.first()?;
    let value = match body.literals.get(pool_idx as usize)? {
        LiteralValue::U32(val) => *val,
        LiteralValue::I32(val) if *val >= 0 => *val as u32,
        _ => return None,
    };
    Some(IndexExpr {
        base_result: None,
        offset: value,
    })
}

pub(crate) fn add_index_expr(
    op: &KernelOp,
    indices: &FxHashMap<u32, IndexExpr>,
) -> Option<IndexExpr> {
    if !matches!(
        op.kind,
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd)
    ) {
        return None;
    }
    let lhs = indices.get(op.operands.first()?)?;
    let rhs = indices.get(op.operands.get(1)?)?;
    let base_result = match (lhs.base_result, rhs.base_result) {
        (None, None) => None,
        (Some(base), None) | (None, Some(base)) => Some(base),
        (Some(lhs_base), Some(rhs_base)) if lhs_base == rhs_base => Some(lhs_base),
        (Some(_), Some(_)) => return None,
    };
    Some(IndexExpr {
        base_result,
        offset: lhs.offset.checked_add(rhs.offset)?,
    })
}

/// Returns `Some((slot, index))` when `op` is a `LoadGlobal` whose index
/// operand resolves to a normalized expression.
fn load_with_index_expr(
    op: &KernelOp,
    indices: &FxHashMap<u32, IndexExpr>,
) -> Option<(u32, IndexExpr)> {
    if !matches!(op.kind, KernelOpKind::LoadGlobal) {
        return None;
    }
    let slot = *op.operands.first()?;
    let index_op_id = *op.operands.get(1)?;
    indices.get(&index_op_id).map(|index| (slot, *index))
}
