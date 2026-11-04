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
/// `vyre_emit_ptx::patterns::audit` and `vyre_lower::audit`,
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
