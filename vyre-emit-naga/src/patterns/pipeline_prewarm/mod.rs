//! Pipeline pre-warm hint.
//!
//! Source-of-truth: `PERF_ROADMAP_2026-05-01.md` section B item B4.
//!
//! First-dispatch pipeline reflection is sync-blocking on wgpu  -  the
//! host has to wait for the driver to compile + reflect the shader
//! before issuing the first dispatch. Pre-warming during the
//! canonicalize / lower phase moves that cost off the dispatch path.
//!
//! This module computes a `PrewarmHint` indicating whether a kernel
//! is large/complex enough to merit pre-warm, plus the suggested
//! warm-up time budget. The host's pre-warm executor consumes this
//! to decide which kernels to dispatch a no-op for during canonicalize.

use serde::{Deserialize, Serialize};
use vyre_lower::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

/// Pipeline prewarming recommendation for one kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrewarmHint {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// True if pre-warm is recommended.
    pub should_prewarm: bool,
    /// Estimated first-dispatch reflection cost in microseconds (very
    /// rough  -  anchored on op-count + binding-count proxies).
    pub estimated_first_dispatch_us: u32,
    /// Reason  -  useful for logging.
    pub reason: String,
}

/// Op-count threshold above which pre-warm is recommended. Below this
/// the reflection cost is small enough that pre-warm doesn't pay back.
pub const PREWARM_OP_THRESHOLD: u32 = 50;

/// Analyze first-dispatch compilation cost and recommend prewarming.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> PrewarmHint {
    let op_count = count_ops(&desc.body);
    let binding_count = desc.bindings.slots.len() as u32;
    // Crude cost model: ~10us baseline + 1us per op + 50us per binding
    // (driver work scales steeply with binding count).
    let estimated_us = 10 + op_count + 50 * binding_count;

    let (should_prewarm, reason) = if op_count >= PREWARM_OP_THRESHOLD {
        (
            true,
            format!("op-count {op_count} ≥ {PREWARM_OP_THRESHOLD}"),
        )
    } else if binding_count >= 4 {
        (true, format!("binding-count {binding_count} ≥ 4"))
    } else {
        (
            false,
            format!(
                "small kernel ({op_count} ops, {binding_count} bindings)  -  pre-warm not worth it"
            ),
        )
    };

    PrewarmHint {
        kernel_id: desc.id.clone(),
        should_prewarm,
        estimated_first_dispatch_us: estimated_us,
        reason,
    }
}

fn count_ops(body: &KernelBody) -> u32 {
    let mut total: u32 = body.ops.len() as u32;
    for op in &body.ops {
        if has_child_body(op) {
            for child in &body.child_bodies {
                total = total.saturating_add(count_ops(child));
            }
            break; // child bodies counted once
        }
    }
    total
}

fn has_child_body(op: &KernelOp) -> bool {
    matches!(
        op.kind,
        KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;
    use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit};
    use vyre_lower::{BindingSlot, KernelDescriptor, KernelOpKind, LiteralValue};

    fn binding(slot: u32) -> BindingSlot {
        global_rw(slot, DataType::U32, &format!("b{slot}"))
    }

    fn small_kernel() -> KernelDescriptor {
        descriptor("small")
            .slot(binding(0))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([lit(0, 0), effect(KernelOpKind::Return, [])])
                    .literal(LiteralValue::U32(0)),
            )
            .build()
    }

    #[test]
    fn small_kernel_does_not_warrant_prewarm() {
        let h = analyze(&small_kernel());
        assert!(!h.should_prewarm);
        assert!(h.reason.contains("not worth it"));
    }

    #[test]
    fn many_op_kernel_warrants_prewarm() {
        let mut ops = Vec::with_capacity(60);
        for i in 0..60 {
            ops.push(lit(0, i));
        }
        let kernel = descriptor("big").body(body().ops(ops).literal(LiteralValue::U32(0))).build();
        let h = analyze(&kernel);
        assert!(h.should_prewarm);
        assert!(h.reason.contains("op-count"));
    }

    #[test]
    fn many_binding_kernel_warrants_prewarm() {
        let kernel = descriptor("many_bindings").slots((0..6).map(binding)).build();
        let h = analyze(&kernel);
        assert!(h.should_prewarm);
        assert!(h.reason.contains("binding-count"));
    }

    #[test]
    fn estimated_us_grows_with_op_and_binding_counts() {
        let small = analyze(&small_kernel());
        // 10 baseline + 2 ops + 50 * 1 binding = 62us
        assert_eq!(small.estimated_first_dispatch_us, 62);
    }

    #[test]
    fn empty_kernel_estimated_at_baseline() {
        let kernel = descriptor("empty").dispatch(64, 1, 1).build();
        let h = analyze(&kernel);
        assert_eq!(h.estimated_first_dispatch_us, 10); // baseline only
        assert!(!h.should_prewarm);
    }

    #[test]
    fn threshold_constant_is_documented_value() {
        assert_eq!(PREWARM_OP_THRESHOLD, 50);
    }

    #[test]
    fn kernel_id_echoed_in_hint() {
        let h = analyze(&small_kernel());
        assert_eq!(h.kernel_id, "small");
    }
}
