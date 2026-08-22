//! SPIR-V-specific emit-time patterns.
//!
//! These analyses walk the `KernelDescriptor` and produce
//! Vulkan/SPIR-V-specific reports (capability declarations,
//! descriptor-set normalization candidates, etc.) that emitters and
//! pipeline builders consume to make correct dispatch decisions.

pub mod subgroup_capabilities;
pub mod workgroup_size_validation;

use serde::{Deserialize, Serialize};
use std::fmt;
use vyre_lower::pattern_audit::PatternAudit;
use vyre_lower::{KernelDescriptor, SubgroupCapabilities};

/// Unified SPIR-V-side pattern audit. Runs every shipped SPIR-V
/// pattern against the descriptor and bundles the reports. Mirror of
/// `vyre_emit_naga::patterns::audit` and `vyre_emit_ptx::patterns::audit`.
///
/// Finding totals, the clean/any predicates, and the one-line summary come
/// from [`PatternAudit`]; import that trait to reach them.
#[must_use]
pub fn audit(desc: &KernelDescriptor) -> SpirvAuditReport {
    SpirvAuditReport {
        kernel_id: desc.id.clone(),
        subgroup: subgroup_capabilities::analyze(desc),
        workgroup_validation: workgroup_size_validation::analyze(desc),
    }
}

/// Combined SPIR-V-pattern report.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpirvAuditReport {
    /// Audited kernel identifier.
    pub kernel_id: String,
    /// Subgroup capabilities required by the kernel.
    pub subgroup: subgroup_capabilities::SubgroupCapabilityReport,
    /// Workgroup-size validation result.
    pub workgroup_validation: workgroup_size_validation::ValidationReport,
}

impl PatternAudit for SpirvAuditReport {
    const FINDING_NOUN: &'static str = "findings";

    fn kernel_id(&self) -> &str {
        &self.kernel_id
    }

    fn finding_count(&self) -> usize {
        self.required_capability_count() + self.workgroup_validation.violations.len()
    }

    fn write_target_tag(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        out.write_str("spirv")
    }

    fn write_breakdown(&self, out: &mut dyn fmt::Write) -> fmt::Result {
        write!(
            out,
            "{} subgroup caps, {} wg violations",
            self.required_capability_count(),
            self.workgroup_validation.violations.len()
        )
    }
}

impl SpirvAuditReport {
    /// Subgroup capability bits the kernel requires.
    fn required_capability_count(&self) -> usize {
        let caps = &self.subgroup.capabilities;
        usize::from(caps.basic)
            + usize::from(caps.ballot)
            + usize::from(caps.shuffle)
            + usize::from(caps.arithmetic)
    }

    /// Identity element for `merge`  -  no required caps, no
    /// violations, baseline workgroup limits.
    pub fn zero() -> Self {
        Self {
            kernel_id: String::new(),
            subgroup: subgroup_capabilities::SubgroupCapabilityReport {
                kernel_id: String::new(),
                capabilities: SubgroupCapabilities {
                    basic: false,
                    ballot: false,
                    shuffle: false,
                    arithmetic: false,
                },
            },
            workgroup_validation: workgroup_size_validation::ValidationReport {
                kernel_id: String::new(),
                workgroup_size: [1, 1, 1],
                limits: workgroup_size_validation::VULKAN_BASELINE,
                violations: vec![],
            },
        }
    }

    /// Aggregate another report's findings: ORs each subgroup
    /// capability bit, concatenates workgroup violations. Workgroup
    /// size + limits are kept from the SEED (merging mismatched
    /// dispatches doesn't make geometric sense).
    pub fn merge(&mut self, other: SpirvAuditReport) {
        let dst = &mut self.subgroup.capabilities;
        let src = &other.subgroup.capabilities;
        dst.basic |= src.basic;
        dst.ballot |= src.ballot;
        dst.shuffle |= src.shuffle;
        dst.arithmetic |= src.arithmetic;
        self.workgroup_validation
            .violations
            .extend(other.workgroup_validation.violations);
    }
}

impl std::fmt::Display for SpirvAuditReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.write_short(f)
    }
}
