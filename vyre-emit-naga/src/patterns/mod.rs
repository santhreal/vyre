//! Naga-specific emit-time patterns.
//!
//! These rewrites operate at emit time on the lowered KernelDescriptor
//! and produce naga IR that takes advantage of substrate-specific
//! features. They live in this crate because they are specific to the
//! naga backend (and the wgpu/Vulkan/WebGPU shaders it targets);
//! equivalent patterns for CUDA live in `vyre-emit-ptx::patterns`.

pub mod pipeline_prewarm;
pub mod vec_pack;

use serde::{Deserialize, Serialize};
use std::fmt;
use vyre_lower::pattern_audit::PatternAudit;
use vyre_lower::KernelDescriptor;

/// Unified naga-side pattern audit. Runs every shipped naga pattern
/// against the descriptor and bundles the reports. Mirror of
/// `vyre_emit_ptx::patterns::audit` and `vyre_lower::audit::audit`,
/// but for naga-specific patterns (vec packing, pipeline prewarm).
///
/// Finding totals, the clean/any predicates, and the one-line summary come
/// from [`PatternAudit`]; import that trait to reach them.
#[must_use]
pub fn audit(desc: &KernelDescriptor) -> NagaAuditReport {
    NagaAuditReport {
        kernel_id: desc.id.clone(),
        vec_pack: vec_pack::analysis::analyze(desc),
        prewarm: pipeline_prewarm::analyze(desc),
    }
}

/// Combined naga-pattern report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NagaAuditReport {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Vector-packing opportunities.
    pub vec_pack: vec_pack::plan::PackingPlan,
    /// Pipeline prewarming recommendation.
    pub prewarm: pipeline_prewarm::PrewarmHint,
}

impl PatternAudit for NagaAuditReport {
    const FINDING_NOUN: &'static str = "candidates";

    fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    fn finding_count(&self) -> usize {
        self.vec_pack.groups.len() + usize::from(self.prewarm.should_prewarm)
    }

    fn write_target_tag(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        out.write_str("naga")
    }

    fn write_breakdown(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        write!(
            out,
            "{} vec_pack, prewarm={}",
            self.vec_pack.groups.len(),
            self.prewarm.should_prewarm
        )
    }
}

impl NagaAuditReport {
    /// Identity element for `merge`  -  empty report. Useful as the
    /// seed of a corpus fold.
    pub fn zero() -> Self {
        Self {
            kernel_id: String::new(),
            vec_pack: vec_pack::plan::PackingPlan {
                kernel_id: String::new(),
                groups: vec![],
            },
            prewarm: pipeline_prewarm::PrewarmHint {
                kernel_id: String::new(),
                should_prewarm: false,
                estimated_first_dispatch_us: 0,
                reason: String::new(),
            },
        }
    }

    /// Aggregate another report's findings into this one. Concatenates
    /// candidate vectors; ORs `should_prewarm`. Useful for corpus-level
    /// "how many naga-specific opportunities are there in this kernel
    /// suite?" rollups.
    pub fn merge(&mut self, other: NagaAuditReport) {
        self.vec_pack.groups.extend(other.vec_pack.groups);
        self.prewarm.should_prewarm |= other.prewarm.should_prewarm;
    }
}

impl std::fmt::Display for NagaAuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_short(f)
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use vyre_lower::descriptor_builder::descriptor;

    #[test]
    fn empty_kernel_yields_zero_candidates() {
        let desc = descriptor("empty").build();
        let report = audit(&desc);
        assert_eq!(report.kernel_id, "empty");
        assert_eq!(report.finding_count(), 0);
        assert!(!report.has_any());
    }

    #[test]
    fn merge_aggregates_findings() {
        let mut acc = NagaAuditReport::zero();
        let desc = descriptor("k").dispatch(64, 1, 1).build();
        let r1 = audit(&desc);
        let r2 = audit(&desc);
        acc.merge(r1);
        acc.merge(r2);
        // No findings on empty kernels  -  sums to 0.
        assert_eq!(acc.finding_count(), 0);
    }

    #[test]
    fn format_short_and_is_clean_on_empty() {
        let desc = descriptor("k").build();
        let r = audit(&desc);
        assert!(r.is_clean());
        let s = r.format_short();
        assert!(s.contains("k (naga)"));
        assert!(s.contains("0 candidates"));
    }

    #[test]
    fn nonempty_kernel_audit_doesnt_panic() {
        let report = audit(&crate::tests::single_store_desc("k"));
        assert_eq!(report.kernel_id, "k");
        // 3-op, 1-binding kernel sits below every naga pattern threshold
        // (vec_pack needs Load/Store fusion groups, prewarm needs
        // ops >= 50 or bindings >= 4).
        // The contract this test enforces is "audit returns cleanly on
        // a real kernel without panicking", not a non-zero candidate count.
        assert_eq!(report.finding_count(), 0);
    }
}
