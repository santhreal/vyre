//! Target-neutral capability contracts used at emission boundaries.
//!
//! Lowering discovers what a descriptor requires; emitters compare those
//! requirements with an explicitly supplied target profile. Runtime and driver
//! policy does not participate in either operation.

use serde::{Deserialize, Serialize};

use crate::{KernelBody, KernelDescriptor, KernelOpKind};

/// Compute workgroup limits exposed by an emission target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkgroupLimits {
    /// Maximum workgroup size along each `[x, y, z]` axis.
    pub max_size: [u32; 3],
    /// Maximum product of the three workgroup dimensions.
    pub max_invocations: u32,
}

/// A target-neutral workgroup limit violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkgroupLimitViolation {
    /// A workgroup dimension is zero.
    ZeroDimension {
        /// Zero-based workgroup axis.
        axis: u8,
    },
    /// A workgroup dimension exceeds the target limit for that axis.
    DimensionExceeded {
        /// Zero-based workgroup axis.
        axis: u8,
        /// Requested dimension.
        actual: u32,
        /// Maximum supported dimension.
        limit: u32,
    },
    /// The product of the dimensions exceeds the target invocation limit.
    InvocationsExceeded {
        /// Requested invocation count, saturated to `u32::MAX` on overflow.
        actual: u32,
        /// Maximum supported invocation count.
        limit: u32,
    },
}

impl std::fmt::Display for WorkgroupLimitViolation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::ZeroDimension { axis } => {
                write!(formatter, "workgroup dimension {axis} is zero")
            }
            Self::DimensionExceeded {
                axis,
                actual,
                limit,
            } => write!(
                formatter,
                "workgroup dimension {axis} requests {actual}, target limit is {limit}"
            ),
            Self::InvocationsExceeded { actual, limit } => write!(
                formatter,
                "workgroup requests {actual} invocations, target limit is {limit}"
            ),
        }
    }
}

/// Validate a workgroup size against target-neutral compute limits.
#[must_use]
pub fn validate_workgroup_size(
    workgroup_size: [u32; 3],
    limits: WorkgroupLimits,
) -> Vec<WorkgroupLimitViolation> {
    let mut violations = Vec::new();
    for (axis, actual) in workgroup_size.into_iter().enumerate() {
        if actual == 0 {
            violations.push(WorkgroupLimitViolation::ZeroDimension { axis: axis as u8 });
        } else if actual > limits.max_size[axis] {
            violations.push(WorkgroupLimitViolation::DimensionExceeded {
                axis: axis as u8,
                actual,
                limit: limits.max_size[axis],
            });
        }
    }

    let actual = workgroup_size
        .into_iter()
        .fold(1_u32, u32::saturating_mul);
    if actual > limits.max_invocations {
        violations.push(WorkgroupLimitViolation::InvocationsExceeded {
            actual,
            limit: limits.max_invocations,
        });
    }
    violations
}

/// Subgroup features required by a descriptor or supported by a target.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
pub struct SubgroupCapabilities {
    /// Subgroup identity and size builtins.
    pub basic: bool,
    /// Subgroup ballot operations.
    pub ballot: bool,
    /// Subgroup shuffle and broadcast operations.
    pub shuffle: bool,
    /// Subgroup arithmetic reductions.
    pub arithmetic: bool,
}

impl SubgroupCapabilities {
    /// Return whether the set contains any subgroup feature.
    #[must_use]
    pub const fn any(self) -> bool {
        self.basic || self.ballot || self.shuffle || self.arithmetic
    }

    /// Return the number of distinct subgroup features in the set.
    #[must_use]
    pub const fn count(self) -> u32 {
        self.basic as u32 + self.ballot as u32 + self.shuffle as u32 + self.arithmetic as u32
    }

    /// Return the first required feature absent from this supported set.
    ///
    /// The order is stable so emitters can expose deterministic diagnostics.
    #[must_use]
    pub const fn first_missing(self, required: Self) -> Option<&'static str> {
        if required.basic && !self.basic {
            Some("subgroup.basic")
        } else if required.ballot && !self.ballot {
            Some("subgroup.ballot")
        } else if required.shuffle && !self.shuffle {
            Some("subgroup.shuffle")
        } else if required.arithmetic && !self.arithmetic {
            Some("subgroup.arithmetic")
        } else {
            None
        }
    }
}

/// Target capabilities that affect substrate emission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmissionTargetCapabilities {
    /// Compute workgroup limits.
    pub workgroup: WorkgroupLimits,
    /// Supported subgroup features.
    pub subgroup: SubgroupCapabilities,
}

/// Discover subgroup features required by a lowered kernel descriptor.
#[must_use]
pub fn required_subgroup_capabilities(desc: &KernelDescriptor) -> SubgroupCapabilities {
    let mut required = SubgroupCapabilities::default();
    scan_body(&desc.body, &mut required);
    required
}

fn scan_body(body: &KernelBody, required: &mut SubgroupCapabilities) {
    for op in &body.ops {
        match op.kind {
            KernelOpKind::SubgroupLocalId | KernelOpKind::SubgroupSize => required.basic = true,
            KernelOpKind::SubgroupBallot => required.ballot = true,
            KernelOpKind::SubgroupShuffle | KernelOpKind::SubgroupBroadcast => {
                required.shuffle = true;
            }
            KernelOpKind::SubgroupReduce { .. } => required.arithmetic = true,
            KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. } => {
                for child_id in &op.operands {
                    if let Some(child) = body.child_bodies.get(*child_id as usize) {
                        scan_body(child, required);
                    }
                }
            }
            _ => {}
        }
    }
}
