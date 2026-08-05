//! Naga emission target-capability admission contracts.

use vyre_lower::{
    BindingLayout, Dispatch, EmissionTargetCapabilities, KernelBody, KernelDescriptor, KernelOp,
    KernelOpKind, SubgroupCapabilities, WorkgroupLimits,
};

fn descriptor(workgroup_size: [u32; 3], ops: Vec<KernelOp>) -> KernelDescriptor {
    KernelDescriptor {
        id: "capability-fixture".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch { workgroup_size },
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals: vec![],
        },
    }
}

fn target(workgroup: WorkgroupLimits, subgroup: SubgroupCapabilities) -> EmissionTargetCapabilities {
    EmissionTargetCapabilities {
        workgroup,
        subgroup,
    }
}

#[test]
fn positive_supported_target_emits_the_requested_module() {
    let desc = descriptor([64, 1, 1], vec![]);
    let module = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [1024, 1024, 64],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect("supported descriptor must emit");
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
}

#[test]
fn negative_unsupported_subgroup_capability_has_stable_error() {
    let desc = descriptor(
        [64, 1, 1],
        vec![KernelOp {
            kind: KernelOpKind::SubgroupLocalId,
            operands: vec![],
            result: Some(0),
        }],
    );
    let err = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [1024, 1024, 64],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect_err("missing subgroup support must reject emission");
    assert_eq!(err.to_string(), "unsupported emission capability `subgroup.basic`");
}

#[test]
fn boundary_workgroup_equal_to_target_limit_is_supported() {
    let desc = descriptor([1024, 1, 1], vec![]);
    let module = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [1024, 1, 1],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect("exact workgroup boundary must emit");
    assert_eq!(module.entry_points[0].workgroup_size, [1024, 1, 1]);
}

#[test]
fn adversarial_oversized_workgroup_returns_stable_error_without_emitting() {
    let desc = descriptor([u32::MAX, u32::MAX, u32::MAX], vec![]);
    let err = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [u32::MAX; 3],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect_err("invocation product overflow must fail closed");
    assert_eq!(
        err.to_string(),
        "unsupported emission capability `workgroup`: workgroup requests 4294967295 invocations, target limit is 1024"
    );
}
