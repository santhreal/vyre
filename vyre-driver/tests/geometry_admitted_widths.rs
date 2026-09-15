//! A device profile reports which workgroup widths it admits, as facts.
//!
//! WHY: launch geometry has one selection owner, `vyre-megakernel`. A driver
//! that returned widths in preference order would be a second cost model, so
//! these tests assert legality and ascending order and never a winner. The
//! class closed here is a profile admitting a width its own limits reject, and
//! a preference order creeping back into the fact.

use vyre_driver::{DeviceProfile, DeviceTimingQuality};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::{CooperativeWidth, ElementPolicy, GeometryRequirements, Uniformity};

fn test_profile_1024() -> DeviceProfile {
    DeviceProfile {
        backend: "test-profile-1024",
        supports_subgroup_ops: true,
        supports_indirect_dispatch: true,
        supports_distributed_collectives: true,
        supports_cooperative_launch: true,
        per_launch_overhead_ns: 2_000,
        persistent_setup_overhead_ns: 10_000,
        supports_specialization_constants: true,
        supports_f16: true,
        supports_bf16: true,
        supports_trap_propagation: true,
        supports_tensor_cores: true,
        has_mul_high: true,
        has_dual_issue_fp32_int32: true,
        has_subgroup_shuffle: true,
        has_shared_memory: true,
        max_native_int_width: 64,
        max_workgroup_size: [1024, 1024, 64],
        max_invocations_per_workgroup: 1024,
        max_shared_memory_bytes: 48 * 1024,
        max_storage_buffer_binding_size: 1 << 30,
        subgroup_size: 32,
        compute_units: 80,
        regs_per_thread_max: 255,
        max_registers_per_compute_unit: 65_536,
        max_invocations_per_compute_unit: 2_048,
        l1_cache_bytes: 128 * 1024,
        l2_cache_bytes: 32 * 1024 * 1024,
        mem_bw_gbps: 900,
        timing_quality: DeviceTimingQuality::HardwareCounters,
        supports_device_timestamps: true,
        supports_hardware_counters: true,
        ideal_unroll_depth: 8,
        ideal_vector_pack_bits: 128,
        ideal_workgroup_tile: [32, 32, 1],
        shared_memory_bank_count: 32,
        shared_memory_bank_width_bytes: 4,
    }
}

fn test_profile_256() -> DeviceProfile {
    DeviceProfile {
        backend: "test-profile-256",
        supports_subgroup_ops: true,
        supports_indirect_dispatch: true,
        supports_distributed_collectives: false,
        supports_cooperative_launch: false,
        per_launch_overhead_ns: 5_000,
        persistent_setup_overhead_ns: 25_000,
        supports_specialization_constants: false,
        supports_f16: false,
        supports_bf16: false,
        supports_trap_propagation: true,
        supports_tensor_cores: false,
        has_mul_high: true,
        has_dual_issue_fp32_int32: false,
        has_subgroup_shuffle: false,
        has_shared_memory: true,
        max_native_int_width: 32,
        max_workgroup_size: [256, 256, 64],
        max_invocations_per_workgroup: 256,
        max_shared_memory_bytes: 16 * 1024,
        max_storage_buffer_binding_size: 1 << 27,
        subgroup_size: 32,
        compute_units: 16,
        regs_per_thread_max: 64,
        max_registers_per_compute_unit: 16_384,
        max_invocations_per_compute_unit: 512,
        l1_cache_bytes: 32 * 1024,
        l2_cache_bytes: 2 * 1024 * 1024,
        mem_bw_gbps: 200,
        timing_quality: DeviceTimingQuality::HostEnqueueWait,
        supports_device_timestamps: false,
        supports_hardware_counters: false,
        ideal_unroll_depth: 4,
        ideal_vector_pack_bits: 128,
        ideal_workgroup_tile: [16, 16, 1],
        shared_memory_bank_count: 16,
        shared_memory_bank_width_bytes: 4,
    }
}

/// Every reported width is legal on the reporting profile, and the profile
/// reports every legal power of two rather than one preferred answer.
#[test]
fn the_widths_a_profile_admits_are_bounded_by_its_own_limits() {
    let agnostic = GeometryRequirements::cooperative(CooperativeWidth::Agnostic);

    for profile in [test_profile_1024(), test_profile_256()] {
        let widths = profile.admissible_workgroup_widths(&agnostic);
        let limit = profile
            .max_invocations_per_workgroup
            .min(profile.max_workgroup_size[0]);
        assert!(
            widths.contains(&limit),
            "profile `{}` must admit its own limit {limit}: {widths:?}",
            profile.backend
        );
        assert!(
            widths.iter().all(|width| *width <= limit && *width > 0),
            "profile `{}` admitted a width outside 1..={limit}: {widths:?}",
            profile.backend
        );
    }

    let wide = test_profile_1024().admissible_workgroup_widths(&agnostic);
    for width in [1_u32, 32, 64, 128, 256, 512, 1024] {
        assert!(wide.contains(&width), "{width} missing from {wide:?}");
    }
    assert!(
        !test_profile_256()
            .admissible_workgroup_widths(&agnostic)
            .contains(&512),
        "a 256-invocation profile must not admit 512"
    );
}

/// A fact carries no order. Ascending is the only order a legality list may
/// have, and a mutation that ranks widest-first turns this red.
#[test]
fn admitted_widths_are_reported_ascending_and_carry_no_preference() {
    for profile in [test_profile_1024(), test_profile_256()] {
        for requirements in [
            GeometryRequirements::cooperative(CooperativeWidth::Agnostic),
            GeometryRequirements::cooperative(CooperativeWidth::AtLeast(64)),
            GeometryRequirements::cooperative(CooperativeWidth::Exactly(64)),
        ] {
            let widths = profile.admissible_workgroup_widths(&requirements);
            assert!(
                !widths.is_empty(),
                "profile `{}` admits nothing for {requirements:?}",
                profile.backend
            );
            assert!(
                widths.windows(2).all(|pair| pair[0] < pair[1]),
                "profile `{}` reported {widths:?} out of ascending order",
                profile.backend
            );
        }
    }
}

/// An exact width is a semantic invariant: the profile admits it or admits
/// nothing, and never substitutes a width the operation did not ask for.
#[test]
fn an_exact_width_the_profile_cannot_reach_admits_nothing() {
    let narrow = test_profile_256();

    for exact in [0_u32, 512, 1024, u32::MAX] {
        let requirements = GeometryRequirements::cooperative(CooperativeWidth::Exactly(exact))
            .with_element_policy(ElementPolicy::Scalar);
        assert!(
            narrow.admissible_workgroup_widths(&requirements).is_empty(),
            "a 256-invocation profile must reject Exactly({exact})"
        );
    }

    let reachable = GeometryRequirements::cooperative(CooperativeWidth::Exactly(256));
    assert_eq!(
        narrow.admissible_workgroup_widths(&reachable),
        vec![256],
        "an admitted exact width is the only width reported"
    );

    let floor_above_limit = GeometryRequirements::cooperative(CooperativeWidth::AtLeast(512));
    assert!(
        narrow
            .admissible_workgroup_widths(&floor_above_limit)
            .is_empty(),
        "a floor above the profile limit admits nothing"
    );
}

/// Each neutral constraint the profile cannot satisfy admits no width at all.
/// Fail-closed is the contract: an unsatisfiable requirement never degrades to
/// a legal-looking width.
#[test]
fn every_unsatisfiable_schedule_constraint_admits_no_width() {
    let profile = test_profile_256();
    let exact_subgroup =
        GeometryRequirements::agnostic().with_subgroup_width(CooperativeWidth::Exactly(32));
    assert!(!profile
        .admissible_workgroup_widths(&exact_subgroup)
        .is_empty());

    for requirements in [
        GeometryRequirements::agnostic().with_subgroup_width(CooperativeWidth::Exactly(64)),
        GeometryRequirements::agnostic().with_subgroup_width(CooperativeWidth::AtLeast(64)),
        GeometryRequirements::agnostic().with_cooperative_launch(),
        GeometryRequirements::agnostic().with_memory_ordering(MemoryOrdering::GridSync),
        GeometryRequirements::agnostic().with_element_policy(ElementPolicy::Multiple(0)),
    ] {
        assert!(
            profile
                .admissible_workgroup_widths(&requirements)
                .is_empty(),
            "unsupported schedule constraint must admit no width: {requirements:?}"
        );
    }

    let mut without_shared_memory = profile;
    without_shared_memory.has_shared_memory = false;
    assert!(without_shared_memory
        .admissible_workgroup_widths(&GeometryRequirements::agnostic().with_min_shared_bytes(4))
        .is_empty());

    let mut beyond_shared_memory = profile;
    beyond_shared_memory.max_shared_memory_bytes = 1_024;
    assert!(beyond_shared_memory
        .admissible_workgroup_widths(&GeometryRequirements::agnostic().with_min_shared_bytes(4_096))
        .is_empty());

    let mut without_subgroup_facts = profile;
    without_subgroup_facts.subgroup_size = 0;
    assert!(without_subgroup_facts
        .admissible_workgroup_widths(
            &GeometryRequirements::agnostic().with_subgroup_uniformity(Uniformity::SubgroupUniform)
        )
        .is_empty());
}
