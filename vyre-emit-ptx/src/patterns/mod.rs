//! PTX-specific emit-time patterns.
//!
//! These rewrites operate at PTX emit time on the lowered
//! KernelDescriptor and produce PTX that takes advantage of
//! CUDA-specific features. They live in this crate because they are
//! specific to NVIDIA hardware; equivalent patterns for naga live in
//! `vyre-emit-naga::patterns`.

pub mod instruction_scheduling;
pub mod ldmatrix_cp_async;
pub mod predicated_execution;
pub mod tensor_core_fragment;
pub mod vec_load_fusion;
mod vec_memory_fusion;
pub mod vec_store_fusion;

use serde::{Deserialize, Serialize};
use std::fmt;
use vyre_lower::pattern_audit::PatternAudit;
use vyre_lower::KernelDescriptor;

use crate::ComputeCapability;

/// Unified PTX-side audit: runs every shipped pattern against the
/// descriptor and returns the combined report. Mirror of
/// `vyre_lower::audit::audit` but for PTX-specific patterns.
///
/// `target` controls capability-gated patterns (tensor cores require
/// sm_70+; ldmatrix.cp.async requires sm_80+).
#[must_use]
pub fn audit(desc: &KernelDescriptor, target: ComputeCapability) -> PtxAuditReport {
    PtxAuditReport {
        kernel_id: desc.id.clone(),
        target,
        predication: predicated_execution::analyze(desc),
        vec_load: vec_load_fusion::analyze(desc),
        vec_store: vec_store_fusion::analyze(desc),
        async_copy: ldmatrix_cp_async::analyze(desc, target),
        tensor_core: tensor_core_fragment::analyze(desc, target),
        scheduling: instruction_scheduling::analyze(desc),
    }
}

/// Combined PTX-pattern report. One `pub` field per shipped pattern.
/// Callers can drill into individual reports for details, or use
/// `total_candidates()` for a single-number "is anything actionable"
/// signal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtxAuditReport {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Target compute capability.
    pub target: ComputeCapability,
    /// Predicated-execution opportunities.
    pub predication: predicated_execution::PredicationPlan,
    /// Vector-load fusion opportunities.
    pub vec_load: vec_load_fusion::FusionPlan,
    /// Vector-store fusion opportunities.
    pub vec_store: vec_store_fusion::FusionPlan,
    /// Asynchronous-copy opportunities.
    pub async_copy: ldmatrix_cp_async::AsyncCopyPlan,
    /// Matrix-fragment opportunities.
    pub tensor_core: tensor_core_fragment::TensorCorePlan,
    /// Instruction-scheduling findings.
    pub scheduling: instruction_scheduling::SchedulingHints,
}

impl PatternAudit for PtxAuditReport {
    const FINDING_NOUN: &'static str = "candidates";

    fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    fn finding_count(&self) -> usize {
        self.predication.candidates.len()
            + self.vec_load.candidates.len()
            + self.vec_store.candidates.len()
            + self.async_copy.candidates.len()
            + self.tensor_core.candidates.len()
            + self.scheduling.long_chains.len()
    }

    fn write_target_tag(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        write!(out, "ptx sm_{}_{}", self.target.major, self.target.minor)
    }

    fn write_breakdown(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        write!(
            out,
            "{}p, {}vl, {}vs, {}ac, {}tc, {}sched",
            self.predication.candidates.len(),
            self.vec_load.candidates.len(),
            self.vec_store.candidates.len(),
            self.async_copy.candidates.len(),
            self.tensor_core.candidates.len(),
            self.scheduling.long_chains.len(),
        )
    }
}

impl PtxAuditReport {
    /// Sum of actionable findings across all patterns. `0` means no
    /// PTX-specific optimizations apply to this kernel.
    pub fn total_candidates(&self) -> usize {
        PatternAudit::finding_count(self)
    }

    /// Whether any pattern fired.
    pub fn has_any(&self) -> bool {
        PatternAudit::has_any(self)
    }

    /// One-line human-readable summary suitable for log lines.
    pub fn format_short(&self) -> String {
        PatternAudit::format_short(self)
    }

    /// True iff no PTX-specific optimization opportunities found.
    pub fn is_clean(&self) -> bool {
        PatternAudit::is_clean(self)
    }

    /// Identity element for [`Self::merge`]  -  empty report. The `target`
    /// defaults to SM_70 (the broadest-compatibility floor); merging
    /// reports with different targets is allowed but the aggregate
    /// keeps the seed's target.
    pub fn zero() -> Self {
        Self {
            kernel_id: String::new(),
            target: ComputeCapability::SM_70,
            predication: predicated_execution::PredicationPlan {
                kernel_id: String::new(),
                candidates: vec![],
            },
            vec_load: vec_load_fusion::FusionPlan { candidates: vec![] },
            vec_store: vec_store_fusion::FusionPlan { candidates: vec![] },
            async_copy: ldmatrix_cp_async::AsyncCopyPlan {
                kernel_id: String::new(),
                target_supports_cp_async: false,
                target_supports_ldmatrix: false,
                candidates: vec![],
            },
            tensor_core: tensor_core_fragment::TensorCorePlan {
                kernel_id: String::new(),
                target_sm: String::new(),
                candidates: vec![],
            },
            scheduling: instruction_scheduling::SchedulingHints {
                kernel_id: String::new(),
                long_chains: vec![],
                total_op_count: 0,
            },
        }
    }

    /// Aggregate another report's findings into this one. Concatenates
    /// every candidate vector + long_chains. Useful for corpus-level
    /// rollups.
    pub fn merge(&mut self, other: PtxAuditReport) {
        self.predication
            .candidates
            .extend(other.predication.candidates);
        self.vec_load.candidates.extend(other.vec_load.candidates);
        self.vec_store.candidates.extend(other.vec_store.candidates);
        self.async_copy
            .candidates
            .extend(other.async_copy.candidates);
        self.tensor_core
            .candidates
            .extend(other.tensor_core.candidates);
        self.scheduling
            .long_chains
            .extend(other.scheduling.long_chains);
        self.scheduling.total_op_count = self
            .scheduling
            .total_op_count
            .saturating_add(other.scheduling.total_op_count);
    }
}

impl std::fmt::Display for PtxAuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_short(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;
    use vyre_lower::descriptor_builder::{body, descriptor, global_rw, lit, op};
    use vyre_lower::{KernelOpKind, LiteralValue};

    #[test]
    fn empty_kernel_yields_zero_candidates() {
        let desc = descriptor("empty").build();
        let report = audit(&desc, ComputeCapability::SM_70);
        assert_eq!(report.kernel_id, "empty");
        assert_eq!(report.total_candidates(), 0);
        assert!(!report.has_any());
    }

    #[test]
    fn vec_load_chain_shows_up_in_audit() {
        let desc = descriptor("vload_chain")
            .slot(global_rw(0, DataType::U32, "buf"))
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        lit(1, 1),
                        op(KernelOpKind::LoadGlobal, [0, 0], 2),
                        op(KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add), [0, 1], 3),
                        op(KernelOpKind::LoadGlobal, [0, 3], 4),
                    ])
                    .literals([LiteralValue::U32(0), LiteralValue::U32(1)]),
            )
            .build();
        let report = audit(&desc, ComputeCapability::SM_70);
        assert!(report.has_any());
        assert_eq!(report.vec_load.candidates.len(), 1);
        assert_eq!(report.total_candidates(), 1);
    }

    #[test]
    fn ptx_audit_merge_aggregates_candidates() {
        let mut acc = PtxAuditReport::zero();
        // Merge two empty reports  -  both have no findings, so aggregate
        // stays empty.
        let desc = descriptor("k").dispatch(64, 1, 1).build();
        acc.merge(audit(&desc, ComputeCapability::SM_70));
        acc.merge(audit(&desc, ComputeCapability::SM_70));
        assert_eq!(acc.total_candidates(), 0);
    }

    #[test]
    fn format_short_and_is_clean_on_empty() {
        let desc = descriptor("k").build();
        let r = audit(&desc, ComputeCapability::SM_80);
        assert!(r.is_clean());
        let s = r.format_short();
        assert!(s.contains("k (ptx sm_8_0)"));
        assert!(s.contains("0 candidates"));
    }

    #[test]
    fn audit_carries_target_through() {
        let desc = descriptor("k").build();
        let r80 = audit(&desc, ComputeCapability::SM_80);
        let r90 = audit(&desc, ComputeCapability::SM_90);
        assert_eq!(r80.target, ComputeCapability::SM_80);
        assert_eq!(r90.target, ComputeCapability::SM_90);
    }
}
