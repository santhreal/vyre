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

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_lower::{
        BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
    };

    fn empty_desc() -> KernelDescriptor {
        KernelDescriptor {
            id: "empty".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        }
    }

    #[test]
    fn empty_kernel_no_capabilities() {
        let r = analyze(&empty_desc());
        assert!(!r.capabilities.any());
        assert_eq!(r.capabilities.count(), 0);
    }

    #[test]
    fn ballot_op_sets_ballot_capability() {
        let mut desc = empty_desc();
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        });
        desc.body.literals.push(LiteralValue::Bool(true));
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupBallot,
            operands: vec![0],
            result: Some(1),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.ballot);
        assert!(!r.capabilities.shuffle);
        assert!(!r.capabilities.arithmetic);
        assert!(!r.capabilities.basic);
        assert_eq!(r.capabilities.count(), 1);
    }

    #[test]
    fn add_op_sets_arithmetic_capability() {
        let mut desc = empty_desc();
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        });
        desc.body.literals.push(LiteralValue::U32(5));
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupReduce {
                op: vyre_lower::SubgroupReduceOp::Add,
            },
            operands: vec![0],
            result: Some(1),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.arithmetic);
        assert!(!r.capabilities.ballot);
    }

    #[test]
    fn shuffle_op_sets_shuffle_capability() {
        let mut desc = empty_desc();
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![1],
            result: Some(1),
        });
        desc.body.literals.push(LiteralValue::U32(7));
        desc.body.literals.push(LiteralValue::U32(3));
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupShuffle,
            operands: vec![0, 1],
            result: Some(2),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.shuffle);
    }

    #[test]
    fn local_id_sets_basic_capability() {
        let mut desc = empty_desc();
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupLocalId,
            operands: vec![],
            result: Some(0),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.basic);
    }

    #[test]
    fn size_sets_basic_capability() {
        let mut desc = empty_desc();
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupSize,
            operands: vec![],
            result: Some(0),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.basic);
    }

    #[test]
    fn multi_op_kernel_sets_multiple_capabilities() {
        let mut desc = empty_desc();
        desc.body.literals.push(LiteralValue::Bool(true));
        desc.body.literals.push(LiteralValue::U32(5));
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupBallot,
            operands: vec![0],
            result: Some(1),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![1],
            result: Some(2),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupReduce {
                op: vyre_lower::SubgroupReduceOp::Add,
            },
            operands: vec![2],
            result: Some(3),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::SubgroupLocalId,
            operands: vec![],
            result: Some(4),
        });
        let r = analyze(&desc);
        assert!(r.capabilities.ballot);
        assert!(r.capabilities.arithmetic);
        assert!(r.capabilities.basic);
        assert!(!r.capabilities.shuffle);
        assert_eq!(r.capabilities.count(), 3);
    }

    #[test]
    fn kernel_id_echoed_in_report() {
        let r = analyze(&empty_desc());
        assert_eq!(r.kernel_id, "empty");
    }

    #[test]
    fn capability_count_helper() {
        let mut caps = SubgroupCapabilities::default();
        assert_eq!(caps.count(), 0);
        caps.ballot = true;
        assert_eq!(caps.count(), 1);
        caps.basic = true;
        caps.shuffle = true;
        caps.arithmetic = true;
        assert_eq!(caps.count(), 4);
    }

    #[test]
    fn nested_subgroup_op_inside_if_detected() {
        // if (cond) { subgroup_ballot }
        let mut desc = empty_desc();
        desc.body.literals.push(LiteralValue::Bool(true));
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        });
        desc.body.ops.push(KernelOp {
            kind: KernelOpKind::StructuredIfThen,
            operands: vec![0, 0],
            result: None,
        });
        desc.body.child_bodies.push(KernelBody {
            ops: vec![KernelOp {
                kind: KernelOpKind::SubgroupBallot,
                operands: vec![0],
                result: Some(1),
            }],
            child_bodies: vec![],
            literals: vec![],
        });
        let r = analyze(&desc);
        assert!(r.capabilities.ballot);
    }
}
