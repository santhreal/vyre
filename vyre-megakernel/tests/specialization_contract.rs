//! A guard set is unambiguous and complete, or it is rejected.
//!
//! WHY: specialization used to be parameter substitution. A caller pinned a
//! symbolic dimension, one artifact came out, and nothing recorded which inputs
//! it was correct for. Two guards could then admit the same request and
//! selection would depend on iteration order, or no guard would admit it and a
//! consumer would run whichever payload it found. Both are wrong answers rather
//! than failures, so both are proved here instead of documented.
//!
//! These cases close the class rather than the incident: the overlap proof is
//! exercised on intervals, member sets, and identities; the coverage proof is
//! exercised on a tiled guard whose tail it must expose; and each capability and
//! resource axis is shown to read its own device fact, so an axis wired to the
//! wrong field cannot pass.

use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::specialization::{
    AxisDomain, AxisValue, GuardTerm, RemainderKind, SpecializationAxis, SpecializationContract,
    TargetCapabilityAxis, TargetResourceAxis, VariantGuard, MAX_COVERAGE_CELLS,
    SPECIALIZATION_SCHEMA_VERSION,
};
use vyre_megakernel::DeviceFacts;

#[path = "support/specialization_fixtures.rs"]
mod specialization_fixtures;

use specialization_fixtures::{contract_over as contract, in_range, tokens};

#[test]
fn a_contract_states_the_schema_it_was_written_under() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 8 });
    assert_eq!(
        stated.schema_version(),
        SPECIALIZATION_SCHEMA_VERSION,
        "Fix: a contract must record the schema it was stated under, so a stale one is rejected \
         rather than read with today's meaning."
    );
}

#[test]
fn a_domain_that_admits_nothing_is_rejected() {
    let cases: Vec<(&str, AxisDomain)> = vec![
        ("inverted", AxisDomain::Interval { low: 9, high: 4 }),
        (
            "empty members",
            AxisDomain::Members {
                members: BTreeSet::new(),
            },
        ),
        (
            "empty identities",
            AxisDomain::Identities {
                identities: BTreeSet::new(),
            },
        ),
    ];
    for (label, domain) in cases {
        let mut axes = BTreeMap::new();
        axes.insert(tokens(), domain);
        let error = SpecializationContract::new(axes)
            .expect_err("Fix: a domain admitting nothing must not declare.");
        assert_eq!(
            error.diagnostic.code.as_str(),
            "MKC033_INVALID_SPECIALIZATION_CONTRACT",
            "Fix: the {label} domain must be rejected as an invalid contract: {error}"
        );
    }
}

#[test]
fn an_axis_and_a_domain_must_agree_on_what_a_value_is() {
    let mut axes = BTreeMap::new();
    axes.insert(
        tokens(),
        AxisDomain::Identities {
            identities: BTreeSet::from([vyre_megakernel::Digest([7; 32])]),
        },
    );
    let error = SpecializationContract::new(axes)
        .expect_err("Fix: a scalar axis must not declare content identities.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC033_INVALID_SPECIALIZATION_CONTRACT",
        "Fix: a dimension axis reads an extent, so an identity domain over it is unreadable: \
         {error}"
    );
}

#[test]
fn an_empty_contract_is_rejected() {
    let error = SpecializationContract::new(BTreeMap::new())
        .expect_err("Fix: a contract declaring no axis must not declare.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC033_INVALID_SPECIALIZATION_CONTRACT",
        "Fix: a contract with no axis specializes on nothing and must say so: {error}"
    );
}

#[test]
fn a_guard_reading_an_undeclared_axis_is_rejected() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 64 });
    let guard = VariantGuard::new(
        vec![GuardTerm::InRange {
            axis: SpecializationAxis::LaunchBatch,
            low: 1,
            high: 4,
        }],
        0,
    );
    let error = stated
        .validate_guard(&guard)
        .expect_err("Fix: a guard over an undeclared axis must not validate.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC034_INVALID_VARIANT_GUARD",
        "Fix: an axis the contract does not declare cannot be read at selection time: {error}"
    );
}

#[test]
fn a_guard_outside_its_declared_domain_is_rejected() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 64 });
    let error = stated
        .validate_guard(&in_range(128, 256, 0))
        .expect_err("Fix: a guard admitting nothing in the domain must not validate.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC034_INVALID_VARIANT_GUARD",
        "Fix: a variant no request can reach is dead bytes and must be reported: {error}"
    );
}

#[test]
fn conjoined_terms_that_cannot_hold_at_once_are_rejected() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let guard = VariantGuard::new(
        vec![
            GuardTerm::InRange {
                axis: tokens(),
                low: 1,
                high: 8,
            },
            GuardTerm::InRange {
                axis: tokens(),
                low: 64,
                high: 128,
            },
        ],
        0,
    );
    let error = stated
        .validate_guard(&guard)
        .expect_err("Fix: two disjoint ranges conjoined must not validate.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC034_INVALID_VARIANT_GUARD",
        "Fix: a conjunction admitting no value selects nothing: {error}"
    );
}

#[test]
fn two_guards_that_can_meet_at_one_precedence_are_rejected() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let error = stated
        .prove(
            &[in_range(1, 128, 0), in_range(64, 256, 0)],
            RemainderKind::Generic,
        )
        .expect_err("Fix: overlapping guards at one precedence must not prove.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC035_GUARD_OVERLAP",
        "Fix: two guards admitting one request at one precedence make selection depend on \
         iteration order: {error}"
    );
}

#[test]
fn two_guards_that_can_meet_at_distinct_precedence_are_accepted() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let proof = stated
        .prove(
            &[in_range(1, 128, 0), in_range(64, 256, 1)],
            RemainderKind::Unsupported,
        )
        .expect("Fix: distinct precedence resolves an overlap.");
    assert!(
        proof.is_complete(),
        "Fix: two ranges covering the whole domain leave no gap, so the remainder may be \
         unsupported: {} of {} cells covered",
        proof.covered(),
        proof.cells()
    );
}

#[test]
fn disjoint_guards_may_share_one_precedence() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let proof = stated
        .prove(
            &[in_range(1, 128, 0), in_range(129, 256, 0)],
            RemainderKind::Unsupported,
        )
        .expect("Fix: guards that cannot meet need no precedence separation.");
    assert!(
        proof.is_complete(),
        "Fix: two adjoining ranges cover the domain: {} of {} cells covered",
        proof.covered(),
        proof.cells()
    );
}

#[test]
fn guards_over_disjoint_member_sets_may_share_one_precedence() {
    let stated = contract(AxisDomain::Members {
        members: BTreeSet::from([1, 2, 4, 8]),
    });
    let members = |values: [u64; 2]| {
        VariantGuard::new(
            vec![GuardTerm::OneOf {
                axis: tokens(),
                members: BTreeSet::from(values),
            }],
            0,
        )
    };
    let proof = stated
        .prove(
            &[members([1, 2]), members([4, 8])],
            RemainderKind::Unsupported,
        )
        .expect("Fix: disjoint member sets are provably exclusive.");
    assert_eq!(
        proof.cells(),
        4,
        "Fix: a member domain cuts one cell per member so coverage is exact."
    );
    assert!(
        proof.is_complete(),
        "Fix: the member sets cover the domain."
    );
}

#[test]
fn a_gap_is_a_failure_only_when_the_remainder_is_unsupported() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let guards = [in_range(1, 128, 0)];
    let proof = stated
        .prove(&guards, RemainderKind::Generic)
        .expect("Fix: a generic remainder serves what the guards leave.");
    assert_eq!(
        proof.gaps(),
        1,
        "Fix: the uncovered upper range must be reported as a gap the remainder serves."
    );
    assert_eq!(
        proof.remainder(),
        RemainderKind::Generic,
        "Fix: a proof records what serves its gaps."
    );
    let error = stated
        .prove(&guards, RemainderKind::Unsupported)
        .expect_err("Fix: an unsupported remainder must not hide a gap.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC036_GUARD_COVERAGE_GAP",
        "Fix: declaring the remainder unsupported asserts the guards are complete: {error}"
    );
    assert!(
        error.diagnostic.message.contains("tokens"),
        "Fix: the diagnostic must name the axis and value nothing serves: {error}"
    );
}

#[test]
fn a_tiled_guard_does_not_cover_its_own_tail() {
    let stated = contract(AxisDomain::Interval { low: 1, high: 256 });
    let tiled = VariantGuard::new(
        vec![GuardTerm::DivisibleBy {
            axis: tokens(),
            divisor: 64,
        }],
        0,
    );
    let proof = stated
        .prove(&[tiled.clone()], RemainderKind::Generic)
        .expect("Fix: a generic remainder serves the tail.");
    assert!(
        proof.gaps() > 0,
        "Fix: a variant admitting only multiples of 64 leaves every other extent uncovered, and \
         a coverage proof that reports none is proving nothing."
    );
    let error = stated
        .prove(&[tiled], RemainderKind::Unsupported)
        .expect_err("Fix: a tiled guard alone must not claim the whole domain.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC036_GUARD_COVERAGE_GAP",
        "Fix: the tail of a tiled schedule is exactly the ragged case that must be named: {error}"
    );
}

#[test]
fn coverage_refuses_to_answer_beyond_its_cell_bound() {
    let mut axes = BTreeMap::new();
    axes.insert(
        tokens(),
        AxisDomain::Members {
            members: (0..u64::try_from(MAX_COVERAGE_CELLS).expect("Fix: the bound must fit u64.")
                + 1)
                .collect(),
        },
    );
    let stated =
        SpecializationContract::new(axes).expect("Fix: a large member domain is declarable.");
    let error = stated
        .prove(&[], RemainderKind::Generic)
        .expect_err("Fix: an undecidable coverage question must fail closed.");
    assert_eq!(
        error.diagnostic.code.as_str(),
        "MKC036_GUARD_COVERAGE_GAP",
        "Fix: a coverage answer nobody can compute must be refused, not assumed: {error}"
    );
}

#[test]
fn a_guard_rejects_a_fact_the_caller_did_not_state() {
    let guard = in_range(1, 64, 0);
    let stated: BTreeMap<SpecializationAxis, AxisValue> = BTreeMap::new();
    assert!(
        !guard
            .admits_facts(&stated)
            .expect("Fix: a satisfiable guard evaluates."),
        "Fix: a missing fact must not act as a wildcard; a variant selected on an unstated fact \
         is selected on a guess."
    );
    let stated = BTreeMap::from([(tokens(), AxisValue::Scalar(32))]);
    assert!(
        guard
            .admits_facts(&stated)
            .expect("Fix: a satisfiable guard evaluates."),
        "Fix: a stated fact inside the guard's range must be admitted."
    );
}

#[test]
fn a_guard_rejects_a_value_of_the_wrong_kind() {
    let guard = in_range(1, 64, 0);
    let stated = BTreeMap::from([(
        tokens(),
        AxisValue::Identity(vyre_megakernel::Digest([3; 32])),
    )]);
    assert!(
        !guard
            .admits_facts(&stated)
            .expect("Fix: a satisfiable guard evaluates."),
        "Fix: a content identity is not an extent and must not satisfy a range term."
    );
}

/// Device facts with every capability off and every extent at a distinct value.
fn bare_device() -> DeviceFacts {
    DeviceFacts::new(BackendCapabilities::default(), 64)
        .with_compute_units(3)
        .with_concurrent_queues(5)
        .with_subgroup_size(7)
        .with_cache_capacity(9)
        .with_occupancy(11, 13)
}

#[test]
fn every_capability_axis_reads_its_own_device_fact() {
    let bare = bare_device();
    for axis in TargetCapabilityAxis::ALL {
        assert!(
            !axis.read(bare),
            "Fix: `{}` must read false from facts that grant nothing.",
            axis.name()
        );
        let granted = grant(*axis, bare);
        assert!(
            axis.read(granted),
            "Fix: `{}` must read the fact it names.",
            axis.name()
        );
        for other in TargetCapabilityAxis::ALL {
            if other == axis {
                continue;
            }
            assert!(
                !other.read(granted),
                "Fix: granting `{}` also turned `{}` on, so the two axes read one field and a \
                 variant guarded on either would be selected for both.",
                axis.name(),
                other.name()
            );
        }
    }
}

/// Facts granting exactly one capability.
fn grant(axis: TargetCapabilityAxis, base: DeviceFacts) -> DeviceFacts {
    let mut capabilities = BackendCapabilities::default();
    let mut device = base;
    match axis {
        TargetCapabilityAxis::SubgroupOps => capabilities.supports_subgroup_ops = true,
        TargetCapabilityAxis::IndirectDispatch => capabilities.supports_indirect_dispatch = true,
        TargetCapabilityAxis::SpecializationConstants => {
            capabilities.supports_specialization_constants = true;
        }
        TargetCapabilityAxis::DistributedCollectives => {
            capabilities.supports_distributed_collectives = true;
        }
        TargetCapabilityAxis::MulHigh => capabilities.has_mul_high = true,
        TargetCapabilityAxis::DualIssueFp32Int32 => capabilities.has_dual_issue_fp32_int32 = true,
        TargetCapabilityAxis::TensorCoreInt => capabilities.has_tensor_core_int = true,
        TargetCapabilityAxis::NativeF16 => capabilities.has_native_f16 = true,
        TargetCapabilityAxis::SubgroupShuffle => capabilities.has_subgroup_shuffle = true,
        TargetCapabilityAxis::SharedMemory => capabilities.has_shared_memory = true,
        TargetCapabilityAxis::TranscendentalPolynomialEmit => {
            capabilities.has_transcendental_polynomial_emit = true;
        }
        TargetCapabilityAxis::TensorCores => capabilities.supports_tensor_cores = true,
        TargetCapabilityAxis::CooperativeLaunch => {
            device = device.with_cooperative_launch(true);
        }
        TargetCapabilityAxis::DeviceTimestamps => {
            device = device.with_device_timestamps(true);
        }
        TargetCapabilityAxis::SpatialPartitioning => {
            device = device.with_spatial_partitioning(true);
        }
    }
    DeviceFacts::new(capabilities, device.max_invocations_per_workgroup())
        .with_cooperative_launch(device.supports_cooperative_launch())
        .with_device_timestamps(device.supports_device_timestamps())
        .with_spatial_partitioning(device.supports_spatial_partitioning())
}

#[test]
fn every_resource_axis_reads_a_distinct_device_extent() {
    let facts = bare_device();
    let mut seen: BTreeMap<u64, TargetResourceAxis> = BTreeMap::new();
    for axis in TargetResourceAxis::ALL {
        let extent = axis.read(facts);
        assert!(
            extent > 0,
            "Fix: `{}` read zero from facts that state every extent, so it is wired to a field \
             the fixture does not set.",
            axis.name()
        );
        if let Some(previous) = seen.insert(extent, *axis) {
            panic!(
                "Fix: `{}` and `{}` both read {extent}, so two axes name one extent and a guard \
                 on either selects for both.",
                previous.name(),
                axis.name()
            );
        }
    }
}
