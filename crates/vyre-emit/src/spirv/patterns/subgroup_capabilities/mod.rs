//! Subgroup capability detection.
//!
//! Vulkan/SPIR-V requires the pipeline / device to declare which
//! subgroup feature flags are used. Walking the descriptor for
//! subgroup ops tells the host which `VkSubgroupFeatureFlagBits` to
//! enable.
//!
//! The mapping (per Vulkan 1.3 spec):
//! - `SubgroupBallot` → `VK_SUBGROUP_FEATURE_BALLOT_BIT`
//! - `SubgroupShuffle` → `VK_SUBGROUP_FEATURE_SHUFFLE_BIT`
//! - `SubgroupAdd` → `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT`
//! - `SubgroupLocalId` / `SubgroupSize` → `VK_SUBGROUP_FEATURE_BASIC_BIT`

use serde::{Deserialize, Serialize};
use vyre_lower::{required_subgroup_capabilities, KernelDescriptor, SubgroupCapabilities};

/// Subgroup capabilities discovered for one kernel descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubgroupCapabilityReport {
    /// Audited kernel identifier.
    pub kernel_id: String,
    /// Capabilities required by the kernel body.
    pub capabilities: SubgroupCapabilities,
}

/// Analyze a kernel descriptor for required subgroup capabilities.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> SubgroupCapabilityReport {
    SubgroupCapabilityReport {
        kernel_id: desc.id.clone(),
        capabilities: required_subgroup_capabilities(desc),
    }
}
