//! SPIR-V target-capability admission and exact-artifact contracts.

use vyre_lower::{
    BindingLayout, Dispatch, EmissionTargetCapabilities, KernelBody, KernelDescriptor, KernelOp,
    KernelOpKind, SubgroupCapabilities, WorkgroupLimits,
};

fn descriptor(workgroup_size: [u32; 3], ops: Vec<KernelOp>) -> KernelDescriptor {
    KernelDescriptor {
        id: "spirv-capability-fixture".into(),
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

fn baseline_target() -> EmissionTargetCapabilities {
    target(
        WorkgroupLimits {
            max_size: [1024, 1024, 64],
            max_invocations: 1024,
        },
        SubgroupCapabilities {
            basic: true,
            ballot: true,
            shuffle: true,
            arithmetic: true,
        },
    )
}

#[test]
fn positive_capability_path_preserves_exact_spirv_artifact() {
    let desc = descriptor([64, 1, 1], vec![]);
    let pinned = vyre_emit_spirv::emit(&desc).expect("pinned fixture must emit");
    let admitted = vyre_emit_spirv::emit_with_capabilities(&desc, &baseline_target())
        .expect("supported target must emit");
    assert_eq!(admitted, pinned);
    assert_eq!(admitted[0], vyre_emit_spirv::SPIRV_MAGIC);
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
    let err = vyre_emit_spirv::emit_with_capabilities(
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
    assert_eq!(
        err.to_string(),
        "naga emission failed: unsupported emission capability `subgroup.basic`"
    );
}

#[test]
fn boundary_workgroup_equal_to_target_limit_preserves_exact_words() {
    let desc = descriptor([1024, 1, 1], vec![]);
    let pinned = vyre_emit_spirv::emit(&desc).expect("boundary fixture must emit");
    let admitted = vyre_emit_spirv::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [1024, 1, 1],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect("exact target boundary must emit");
    assert_eq!(admitted, pinned);
}

#[test]
fn adversarial_oversized_workgroup_fails_closed_with_stable_error() {
    let desc = descriptor([u32::MAX, u32::MAX, u32::MAX], vec![]);
    let err = vyre_emit_spirv::emit_with_capabilities(
        &desc,
        &target(
            WorkgroupLimits {
                max_size: [u32::MAX; 3],
                max_invocations: 1024,
            },
            SubgroupCapabilities::default(),
        ),
    )
    .expect_err("overflowing invocation product must reject emission");
    assert_eq!(
        err.to_string(),
        "naga emission failed: unsupported emission capability `workgroup`: workgroup requests 4294967295 invocations, target limit is 1024"
    );
}
