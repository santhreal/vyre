//! Vulkan workgroup-size limit validation.
//!
//! SPIR-V compute shaders for Vulkan are subject to per-device
//! `VkPhysicalDeviceLimits`:
//!
//! - `maxComputeWorkGroupSize[3]`  -  per-dimension limit. Standard
//!   minimum is `[1024, 1024, 64]`; many drivers go higher.
//! - `maxComputeWorkGroupInvocations`  -  total threads per workgroup
//!   (the product of the three dims). Standard minimum is `1024`.
//!
//! This pattern checks `desc.dispatch.workgroup_size` against the
//! Vulkan-baseline limits AND a configurable per-device profile.
//! Returns a `ValidationReport` with each violation as a separate
//! entry so callers can route them individually.
//!
//! Detection-only: emit happens regardless. The host pipeline
//! builder consults this report to decide whether to refuse the
//! dispatch, fall back to a smaller workgroup_size override, or
//! raise the device requirement bar.

use serde::{Deserialize, Serialize};
use vyre_lower::{
    validate_workgroup_size, KernelDescriptor, WorkgroupLimitViolation, WorkgroupLimits,
};

/// Vulkan-baseline limits: every conformant Vulkan implementation must
/// support at least these values.
pub const VULKAN_BASELINE: WorkgroupLimits = WorkgroupLimits {
    max_size: [1024, 1024, 64],
    max_invocations: 1024,
};

/// Workgroup-size validation result for one kernel descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Validated kernel identifier.
    pub kernel_id: String,
    /// Requested `[x, y, z]` workgroup size.
    pub workgroup_size: [u32; 3],
    /// Target-neutral limits used for validation.
    pub limits: WorkgroupLimits,
    /// Every constraint violated by the requested size.
    pub violations: Vec<WorkgroupLimitViolation>,
}

impl ValidationReport {
    /// Return whether the workgroup size satisfies all limits.
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }

    /// Return the requested total invocations per workgroup.
    pub fn invocations(&self) -> u32 {
        self.workgroup_size[0]
            .saturating_mul(self.workgroup_size[1])
            .saturating_mul(self.workgroup_size[2])
    }
}

/// Validate against the Vulkan baseline ([`VULKAN_BASELINE`]).
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> ValidationReport {
    analyze_against(desc, VULKAN_BASELINE)
}

/// Validate against a target-neutral device limit profile.
#[must_use]
pub fn analyze_against(desc: &KernelDescriptor, limits: WorkgroupLimits) -> ValidationReport {
    let workgroup_size = desc.dispatch.workgroup_size;
    ValidationReport {
        kernel_id: desc.id.clone(),
        workgroup_size,
        limits,
        violations: validate_workgroup_size(workgroup_size, limits),
    }
}
