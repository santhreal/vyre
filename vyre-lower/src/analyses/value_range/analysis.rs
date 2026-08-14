//! The value-range walk.
//!
//! Owns the forward pass that assigns each result id an [`IntRange`] from its
//! producer: literal singletons, the interval arithmetic for the binary
//! operators that have one, and the union at a structured join. Absence of a
//! range always means "not derived", never "unbounded".

use rustc_hash::FxHashMap;

use super::carrier_staleness::carrier_snapshot_invalidations;
use super::{IntRange, ValueRangeReport};
use crate::{KernelBody, KernelDescriptor, KernelOpKind, LiteralValue};
use vyre_foundation::ir::BinOp;

/// Analyze the descriptor's top-level body for integer value ranges.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> ValueRangeReport {
    analyze_body(&desc.body)
}

/// Analyze one body for integer value ranges.
#[must_use]
pub fn analyze_body(body: &KernelBody) -> ValueRangeReport {
    let mut ranges: FxHashMap<u32, IntRange> = FxHashMap::default();

    // Phase 1a: seed from Literal ops.
    for op in &body.ops {
        if matches!(op.kind, KernelOpKind::Literal) {
            if let (Some(rid), Some(&pool_idx)) = (op.result, op.operands.first()) {
                if let Some(lit) = body.literals.get(pool_idx as usize) {
                    let r = match lit {
                        LiteralValue::U32(v) => Some(IntRange::singleton(*v as i64)),
                        LiteralValue::I32(v) => Some(IntRange::singleton(*v as i64)),
                        LiteralValue::Bool(true) => Some(IntRange::singleton(1)),
                        LiteralValue::Bool(false) => Some(IntRange::singleton(0)),
                        _ => None,
                    };
                    if let Some(r) = r {
                        ranges.insert(rid, r);
                    }
                }
            }
        }
    }

    // Phase 1b: propagate through Min/Max BinOps where both operands
    // have known ranges. The result range is the union  -  Min(a, b)
    // could be either, so the result is in [min(a.min, b.min),
    // min(a.max, b.max)] for Min, but a tighter union is safe.
    for op in &body.ops {
        if let KernelOpKind::BinOpKind(bin_op) = &op.kind {
            if op.operands.len() < 2 {
                continue;
            }
            let lhs = ranges.get(&op.operands[0]).copied();
            let rhs = ranges.get(&op.operands[1]).copied();
            let Some(rid) = op.result else { continue };
            if let (Some(l), Some(r)) = (lhs, rhs) {
                let derived = match bin_op {
                    BinOp::Min => Some(IntRange {
                        min: l.min.min(r.min),
                        max: l.max.min(r.max),
                    }),
                    BinOp::Max => Some(IntRange {
                        min: l.min.max(r.min),
                        max: l.max.max(r.max),
                    }),
                    BinOp::Add | BinOp::WrappingAdd => {
                        // Result range is [l.min+r.min, l.max+r.max].
                        // Use checked_add to bail on overflow rather
                        // than silently wrap (which would produce a
                        // false-narrow range).
                        match (l.min.checked_add(r.min), l.max.checked_add(r.max)) {
                            (Some(min), Some(max)) => Some(IntRange { min, max }),
                            _ => None,
                        }
                    }
                    BinOp::Sub | BinOp::WrappingSub => {
                        // Result range is [l.min-r.max, l.max-r.min].
                        // Subtraction of a range flips the bounds.
                        match (l.min.checked_sub(r.max), l.max.checked_sub(r.min)) {
                            (Some(min), Some(max)) => Some(IntRange { min, max }),
                            _ => None,
                        }
                    }
                    BinOp::Mul => mul_range(l, r),
                    BinOp::BitAnd => {
                        // x & mask: result is in [0, max_possible].
                        // The max_possible is the smaller of the two
                        // operand maxes  -  neither operand can
                        // contribute bits the other doesn't have set.
                        // Conservative: refuse on negatives (sign bit
                        // makes the range non-trivial).
                        if l.min < 0 || r.min < 0 {
                            None
                        } else {
                            Some(IntRange {
                                min: 0,
                                max: l.max.min(r.max),
                            })
                        }
                    }
                    BinOp::BitOr => {
                        // x | y: each bit is ≥ either input's bit, so
                        // result.min ≥ max(l.min, r.min). Conservative
                        // upper bound: l.max | r.max (no bit can appear
                        // that wasn't in some operand's max). Refuse on
                        // negatives.
                        if l.min < 0 || r.min < 0 {
                            None
                        } else {
                            Some(IntRange {
                                min: l.min.max(r.min),
                                max: l.max | r.max,
                            })
                        }
                    }
                    BinOp::Shl if r.is_singleton() && r.min >= 0 && r.min < 32 => {
                        // x << k for known k: result range scales by 2^k.
                        // Use checked_shl to bail on overflow. l can be
                        // negative  -  Shl on negatives is well-defined
                        // arithmetic-shift in Rust (multiplies by 2^k).
                        let k = r.min as u32;
                        match (l.min.checked_shl(k), l.max.checked_shl(k)) {
                            (Some(min), Some(max)) => Some(IntRange { min, max }),
                            _ => None,
                        }
                    }
                    BinOp::Shr if r.is_singleton() && r.min >= 0 && r.min < 32 => {
                        // x >> k: arithmetic right shift on i64.
                        // Result range is [l.min >> k, l.max >> k]
                        // (shifting preserves order for the same shift).
                        let k = r.min as u32;
                        Some(IntRange {
                            min: l.min >> k,
                            max: l.max >> k,
                        })
                    }
                    _ => None,
                };
                if let Some(d) = derived {
                    ranges.insert(rid, d);
                }
            }
        }
    }

    ValueRangeReport {
        ranges,
        invalidated_from: carrier_snapshot_invalidations(body),
    }
}

/// Range of `l * r` accounting for sign  -  the result range is the
/// min/max of the four corner products (l.min*r.min, l.min*r.max,
/// l.max*r.min, l.max*r.max). Bails on overflow.
pub(super) fn mul_range(l: IntRange, r: IntRange) -> Option<IntRange> {
    let corners = [
        l.min.checked_mul(r.min),
        l.min.checked_mul(r.max),
        l.max.checked_mul(r.min),
        l.max.checked_mul(r.max),
    ];
    let [Some(a), Some(b), Some(c), Some(d)] = corners else {
        return None;
    };
    let min = a.min(b).min(c).min(d);
    let max = a.max(b).max(c).max(d);
    Some(IntRange { min, max })
}
