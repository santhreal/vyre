//! WHY: closes the class "a registered op declares a workgroup no target
//! profile accepts", which took `(wgpu,
//! vyre-libs::reduce::multi_block_prefix_scan_inclusive_sum)` out of the
//! conformance certificate with `target workgroup extent 1024 exceeds profile
//! limit 256`. A workgroup extent is chosen when the op builds its program,
//! before any backend is known, so an extent above the least-capable registered
//! target is a compile-time fact and belongs in a test rather than in a device
//! refusal 300 ops into a proof run.
//!
//! The roster is both live registries: every op the operation registry carries a
//! builder for, against the profile of every backend that registers a target
//! compiler. Adding an op with a wider workgroup, or a backend with a narrower
//! limit, turns this red without anyone editing a list.
//!
//! What it does not catch: a workgroup the profile accepts but the live device
//! refuses. The profile is the authenticated limit a payload is admitted
//! against; a device that reports less is `validate_device_support`'s subject,
//! and a per-axis limit below the invocation limit would pass here while
//! failing an axis check. It also says nothing about whether the extent is a
//! good one for throughput.

use vyre_driver::BackendRegistration;
use vyre_foundation::ir::Program;
use vyre_registry_link::backend::live_backend_registry;
use vyre_registry_link::operation::live_operation_registry;

/// One registered target's authenticated workgroup limits.
struct TargetLimits {
    backend_id: &'static str,
    max_invocations: u32,
    max_extent: [u32; 3],
}

fn target_limits() -> Vec<TargetLimits> {
    let registrations =
        live_backend_registry().expect("Fix: the backend registry must start before it is judged.");
    let limits = registrations
        .iter()
        .filter(|registration| registration.target_compiler.is_some())
        .map(profile_limits)
        .collect::<Vec<_>>();
    assert!(
        !limits.is_empty(),
        "Fix: no linked backend registers a target compiler, so this test judges nothing. Link a concrete driver crate."
    );
    limits
}

fn profile_limits(registration: &BackendRegistration) -> TargetLimits {
    let compiler = registration.target_compiler().unwrap_or_else(|error| {
        panic!(
            "Fix: backend `{}` registers a target compiler that will not construct: {error}",
            registration.id
        )
    });
    let profile = compiler.profile();
    TargetLimits {
        backend_id: registration.id,
        max_invocations: profile.max_invocations_per_workgroup(),
        max_extent: profile.max_workgroup_size(),
    }
}

fn declared_invocations(program: &Program) -> u64 {
    program
        .workgroup_size()
        .into_iter()
        .map(u64::from)
        .product::<u64>()
}

#[test]
fn every_registered_op_declares_a_workgroup_every_target_profile_admits() {
    let limits = target_limits();
    let mut refused = Vec::new();
    for operation in live_operation_registry().iter() {
        let Some(program) = operation.program() else {
            continue;
        };
        let declared = program.workgroup_size();
        let invocations = declared_invocations(&program);
        for target in &limits {
            if target.max_invocations > 0 && invocations > u64::from(target.max_invocations) {
                refused.push(format!(
                    "({}, {}): declares {invocations} invocations per workgroup, profile admits {}",
                    target.backend_id, operation.id, target.max_invocations
                ));
            }
            for (axis, (extent, limit)) in declared.into_iter().zip(target.max_extent).enumerate() {
                if extent > limit {
                    refused.push(format!(
                        "({}, {}): declares workgroup_size[{axis}] = {extent}, profile admits {limit}",
                        target.backend_id, operation.id
                    ));
                }
            }
        }
    }
    assert!(
        refused.is_empty(),
        "Fix: an op declares a workgroup its target profile refuses, so the payload is rejected at admission instead of executing. Size the block to the portable extent, or record the op as unsupported on that backend.\n{}",
        refused.join("\n")
    );
}

#[test]
fn every_target_profile_admits_the_portable_workgroup_extent() {
    for target in target_limits() {
        assert!(
            target.max_invocations >= vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS,
            "Fix: backend `{}` admits {} invocations per workgroup, below the portable extent {} that shared-crate ops size their cooperative blocks to. Raise the profile, or lower `PORTABLE_WORKGROUP_INVOCATIONS` and resize every op that reads it.",
            target.backend_id,
            target.max_invocations,
            vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS
        );
        assert!(
            target.max_extent[0] >= vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS,
            "Fix: backend `{}` admits workgroup_size[0] = {}, below the portable extent {}. A 1D cooperative block of that width is what shared-crate ops declare.",
            target.backend_id,
            target.max_extent[0],
            vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS
        );
    }
}
