//! Target-neutral emission capability contracts.

use vyre_lower::{
    required_subgroup_capabilities, validate_workgroup_size, BindingLayout, Dispatch, KernelBody,
    KernelDescriptor, KernelOp, KernelOpKind, SubgroupCapabilities, WorkgroupLimitViolation,
    WorkgroupLimits,
};

fn descriptor(body: KernelBody) -> KernelDescriptor {
    KernelDescriptor {
        id: "target-contract".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body,
    }
}

#[test]
fn positive_plain_descriptor_requires_no_subgroup_capability() {
    let required = required_subgroup_capabilities(&descriptor(KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![],
    }));
    assert_eq!(required, SubgroupCapabilities::default());
    assert_eq!(SubgroupCapabilities::default().first_missing(required), None);
}

#[test]
fn negative_missing_capability_order_is_stable() {
    let supported = SubgroupCapabilities::default();
    let required = SubgroupCapabilities {
        basic: true,
        ballot: true,
        shuffle: true,
        arithmetic: true,
    };
    assert_eq!(supported.first_missing(required), Some("subgroup.basic"));
}

#[test]
fn boundary_workgroup_at_every_limit_is_admitted() {
    let limits = WorkgroupLimits {
        max_size: [8, 4, 2],
        max_invocations: 64,
    };
    assert!(validate_workgroup_size([8, 4, 2], limits).is_empty());
}

#[test]
fn adversarial_workgroup_overflow_saturates_and_reports_deterministically() {
    let limits = WorkgroupLimits {
        max_size: [u32::MAX; 3],
        max_invocations: 1024,
    };
    assert_eq!(
        validate_workgroup_size([u32::MAX; 3], limits),
        vec![WorkgroupLimitViolation::InvocationsExceeded {
            actual: u32::MAX,
            limit: 1024,
        }]
    );
}

#[test]
fn nested_subgroup_requirements_are_discovered_below_structured_regions() {
    let child = KernelBody {
        ops: vec![KernelOp {
            kind: KernelOpKind::SubgroupBroadcast,
            operands: vec![],
            result: None,
        }],
        child_bodies: vec![],
        literals: vec![],
    };
    let required = required_subgroup_capabilities(&descriptor(KernelBody {
        ops: vec![KernelOp {
            kind: KernelOpKind::StructuredBlock,
            operands: vec![0],
            result: None,
        }],
        child_bodies: vec![child],
        literals: vec![],
    }));
    assert!(required.shuffle);
}
