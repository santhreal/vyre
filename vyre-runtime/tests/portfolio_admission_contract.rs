//! A runtime selects from a guarded artifact set, and never for it.
//!
//! WHY: a runtime holding several artifacts for one graph has to pick one, and
//! the only legal way to pick is to evaluate the guards the compiler
//! authenticated. Reading a shape and choosing a launch, or preferring whichever
//! member ran fastest last time, is schedule selection happening after the
//! artifact froze it.
//!
//! So these cases prove what admission refuses: a set whose authenticated target
//! is another device or objective, a set one of whose members lacks the required
//! payload format, and a stated workload outside the domain the contract
//! declares, whichever remainder the set carries. They also prove what selection
//! does: evaluate guards in the set's own order, fall to the declared remainder
//! inside the domain, and return a member that was compiled and scored before
//! admission.

use std::collections::BTreeMap;

use vyre_megakernel::specialization::{
    AxisDomain, AxisValue, GuardTerm, PortfolioEnvelope, RemainderKind, SpecializationAxis,
    SpecializationContract, VariantGuard,
};
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, ArtifactNodeId, ArtifactValueId, Digest, TargetPayload,
    TargetPayloadFormat, TargetProfile, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};
use vyre_runtime::artifact_admission::{admit_portfolio, ArtifactAdmissionError};

#[path = "../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{compile_graph, entry_over, single_input_graph};

/// The axis every fixture guard reads.
fn tokens() -> SpecializationAxis {
    SpecializationAxis::SymbolicDimension {
        dimension: "tokens".to_string(),
    }
}

fn fixture_contract() -> SpecializationContract {
    let mut axes = BTreeMap::new();
    axes.insert(tokens(), AxisDomain::Interval { low: 1, high: 64 });
    SpecializationContract::new(axes).expect("Fix: the fixture contract must be declarable.")
}

fn in_range(low: u64, high: u64) -> VariantGuard {
    VariantGuard::new(
        vec![GuardTerm::InRange {
            axis: tokens(),
            low,
            high,
        }],
        0,
    )
}

fn facts(extent: u64) -> BTreeMap<SpecializationAxis, AxisValue> {
    BTreeMap::from([(tokens(), AxisValue::Scalar(extent))])
}

/// One artifact over a named single-value graph.
///
/// Members of one set must agree on the graph they came from, so every member
/// here is compiled from the same graph. `seed` fills the external facts
/// digest, which participates in the request identity, so each member is a
/// distinct artifact and a selection that returned the wrong one is visible.
fn member(seed: u8) -> Artifact {
    compile_graph(single_input_graph([64, 1, 1]), seed)
}

fn format(identity: &str, version: u16) -> TargetPayloadFormat {
    TargetPayloadFormat::new(identity, version).expect("Fix: the fixture format must be valid.")
}

fn payload(neutral: &Artifact, payload_format: TargetPayloadFormat) -> TargetPayload {
    let profile = TargetProfile::new(
        payload_format.identity(),
        u64::from(payload_format.version()),
        [64, 1, 1],
        64,
        1_024,
        0,
    )
    .expect("Fix: the fixture profile must be valid.");
    TargetPayload::new(
        neutral,
        payload_format,
        profile,
        vec![entry_over(
            neutral,
            "entry",
            ArtifactNodeId(0),
            vec![TargetResourceBinding {
                resource: ArtifactValueId(0),
                group: 0,
                slot: 0,
                memory: TargetResourceMemory::Global,
                access: TargetResourceAccess::ReadOnly,
            }],
        )],
        vec![1, 2, 3],
    )
    .expect("Fix: the fixture payload must be valid.")
}

fn attachment(neutral: Artifact, payload_format: TargetPayloadFormat) -> ArtifactEnvelope {
    let mut envelope = ArtifactEnvelope::new(neutral);
    let attached = payload(envelope.neutral(), payload_format);
    envelope
        .attach_target_payload(attached)
        .expect("Fix: the fixture payload must attach to the artifact it names.");
    envelope
}

/// A set sealed for `identity` whose guards cover the declared domain.
fn sealed(
    identity: Digest,
    remainder: RemainderKind,
    remainder_format: TargetPayloadFormat,
) -> Vec<u8> {
    set(identity, remainder, remainder_format, in_range(33, 64))
}

/// A set whose guards leave part of the declared domain to the remainder.
fn gapped(identity: Digest, remainder_format: TargetPayloadFormat) -> Vec<u8> {
    set(
        identity,
        RemainderKind::Generic,
        remainder_format,
        in_range(33, 48),
    )
}

/// A two-variant set over the fixture contract, sealed for `identity`.
fn set(
    identity: Digest,
    remainder: RemainderKind,
    remainder_format: TargetPayloadFormat,
    high: VariantGuard,
) -> Vec<u8> {
    let required = format("test.portfolio-target", 1);
    let mut envelope = PortfolioEnvelope::new(fixture_contract(), remainder, identity);
    envelope
        .attach_variant(in_range(1, 32), attachment(member(0), required.clone()))
        .expect("Fix: the low variant must attach.");
    envelope
        .attach_variant(high, attachment(member(1), required))
        .expect("Fix: the high variant must attach.");
    if remainder == RemainderKind::Generic {
        envelope
            .attach_remainder(attachment(member(2), remainder_format))
            .expect("Fix: the generic remainder must attach.");
    }
    envelope
        .to_bytes()
        .expect("Fix: the fixture set must encode.")
}

fn code(error: &ArtifactAdmissionError) -> String {
    error.diagnostic().code.as_str().to_string()
}

const IDENTITY: Digest = Digest([3; 32]);

#[test]
fn a_stated_extent_selects_the_variant_whose_guard_admits_it() {
    let required = format("test.portfolio-target", 1);
    let bytes = sealed(IDENTITY, RemainderKind::Generic, required.clone());
    let admitted = admit_portfolio(&bytes, &required, IDENTITY)
        .expect("Fix: a set compiled for this target must admit.");

    assert_eq!(
        admitted.variants().len(),
        2,
        "Fix: every attached variant must be admitted, or a request the set covers is served by \
         the remainder."
    );
    let low = admitted
        .select(&facts(8))
        .expect("Fix: an extent the low guard admits must select a variant.");
    let high = admitted
        .select(&facts(48))
        .expect("Fix: an extent the high guard admits must select a variant.");
    assert_eq!(
        low.neutral().digest(),
        admitted.variants()[0].1.neutral().digest(),
        "Fix: selection must return the member whose guard admits the facts, in the set's own \
         evaluation order."
    );
    assert_eq!(
        high.neutral().digest(),
        admitted.variants()[1].1.neutral().digest(),
        "Fix: the second guard must serve the extent only it admits."
    );
}

/// An extent inside the declared domain that no guard admits is the remainder's.
#[test]
fn an_in_domain_extent_no_guard_admits_falls_to_the_declared_remainder() {
    let required = format("test.portfolio-target", 1);
    let bytes = gapped(IDENTITY, required.clone());
    let admitted = admit_portfolio(&bytes, &required, IDENTITY)
        .expect("Fix: a set compiled for this target must admit.");

    let served = admitted
        .select(&facts(56))
        .expect("Fix: a set declaring a generic remainder must serve an uncovered extent.");
    assert_eq!(
        served.neutral().digest(),
        admitted
            .remainder()
            .expect("Fix: a generic set must admit its remainder.")
            .neutral()
            .digest(),
        "Fix: an extent no guard admits must be served by the remainder, not by whichever variant \
         happens to be first."
    );
}

/// A workload outside the declared domain is refused for every remainder kind.
///
/// WHY: a generic remainder is compiled over the declared domain, so serving a
/// workload beyond it would run a schedule that launches over fewer points than
/// the workload holds. The runtime is the last place that can refuse it, because
/// nothing downstream reads the contract again.
#[test]
fn a_workload_outside_the_declared_domain_is_refused_for_every_remainder_kind() {
    for kind in RemainderKind::ALL {
        let required = format("test.portfolio-target", 1);
        let bytes = sealed(IDENTITY, *kind, required.clone());
        let admitted = admit_portfolio(&bytes, &required, IDENTITY)
            .expect("Fix: a set compiled for this target must admit.");

        let Err(error) = admitted.select(&facts(4096)) else {
            panic!(
                "Fix: an extent outside the declared domain must be refused under a {} \
                 remainder, not served by a member compiled for a smaller extent.",
                kind.name()
            )
        };
        assert_eq!(
            code(&error),
            "MKC039_UNSUPPORTED_WORKLOAD",
            "Fix: the refusal must name the workload the set does not serve: {error}"
        );
    }
}

#[test]
fn a_set_compiled_for_another_target_is_not_admitted() {
    let required = format("test.portfolio-target", 1);
    let bytes = sealed(IDENTITY, RemainderKind::Generic, required.clone());

    let error = admit_portfolio(&bytes, &required, Digest([9; 32]))
        .expect_err("Fix: an authenticated target the set was not compiled for must be refused.");
    assert_eq!(
        code(&error),
        "MKC038_TARGET_IDENTITY_MISMATCH",
        "Fix: admission must refuse a set before evaluating a guard against the wrong device: \
         {error}"
    );
}

#[test]
fn one_member_without_the_required_format_refuses_the_whole_set() {
    let required = format("test.portfolio-target", 1);
    let bytes = sealed(
        IDENTITY,
        RemainderKind::Generic,
        format("test.other-target", 1),
    );

    let error = admit_portfolio(&bytes, &required, IDENTITY).expect_err(
        "Fix: a set is admitted whole; one member without the required format must refuse it.",
    );
    assert_eq!(
        code(&error),
        "MKC021_INCOMPATIBLE_TARGET_PAYLOAD",
        "Fix: the refusal must name the payload format the member does not carry: {error}"
    );
}

#[test]
fn corrupted_set_bytes_do_not_authenticate() {
    let required = format("test.portfolio-target", 1);
    let mut bytes = sealed(IDENTITY, RemainderKind::Generic, required.clone());
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;

    let error = admit_portfolio(&bytes, &required, IDENTITY)
        .expect_err("Fix: edited set bytes must fail their own digest.");
    assert_eq!(
        code(&error),
        "MKC016_DIGEST_MISMATCH",
        "Fix: a set is one authenticated product, so any edit inside it must be refused: {error}"
    );
}
