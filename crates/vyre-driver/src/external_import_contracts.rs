//! The external-import contract every concrete driver's importer answers.
//!
//! Pre-allocation refusal, zero-copy admission, exact schedule execution and
//! device-loss invalidation are stated by
//! [`ExternalResourceImporter`](crate::external_import::ExternalResourceImporter), not by
//! any one backend, so the assertions belong here too. A concrete driver names
//! the handles and the combination its own policy refuses, and runs the set.

use crate::external_import::{
    ExternalImportDescriptor, ExternalImportPolicy, ExternalResourceImporter,
};
use crate::semantic_resource_abi::{
    ExternalMemoryKind, ImageDimensions, ImageFormat, ResourceAbiError, ResourcePermittedUsages,
    ResourceTransitionSchedule, ResourceUsageTransition, TimelineSyncProtocol,
};

/// What the shared contract needs a backend to supply.
#[derive(Clone, Debug)]
pub struct ExternalImportContractCase<H> {
    /// A handle whose combination with `refused_format` and `refused_usages`
    /// the backend policy refuses.
    pub refused_handle: H,
    /// The format of the refused combination.
    pub refused_format: ImageFormat,
    /// The usages of the refused combination.
    pub refused_usages: ResourcePermittedUsages,
    /// The external memory class the refusal names.
    pub refused_memory_kind: ExternalMemoryKind,
    /// A handle the backend admits, used for the pitch and admission cases.
    pub admitted_handle: H,
    /// A second admitted handle, used for the device-loss case.
    pub device_loss_handle: H,
    /// The timeline protocol this backend synchronizes an admitted import on.
    pub sync_protocol: TimelineSyncProtocol,
}

/// The dimensions and pitch every admitted case in this contract imports at.
const ADMITTED_WIDTH: u32 = 1920;
/// Row count of the admitted surface.
const ADMITTED_HEIGHT: u32 = 1080;

/// Run every backend-neutral external-import contract against `P`.
///
/// Proves, in order: an unsupported combination is refused before allocation
/// and names the memory class; a short unaligned pitch is refused and states
/// the aligned pitch it needed; an admitted import records exact dimensions and
/// pitch as zero-copy; a transition schedule executes exactly with no copies
/// and no device-wide waits; and device loss invalidates the record together
/// with its dependent view and artifact, after which the schedule is refused.
///
/// # Panics
///
/// Panics when the importer under test breaks any of those contracts.
pub fn assert_external_import_contract<P>(case: ExternalImportContractCase<P::Handle>)
where
    P: ExternalImportPolicy,
    P::Handle: Clone + core::fmt::Debug,
{
    assert_refuses_unsupported_combination::<P>(&case);
    assert_refuses_unaligned_pitch::<P>(&case);
    assert_admits_zero_copy_and_executes_schedule::<P>(&case);
    assert_device_loss_invalidates_dependents::<P>(&case);
}

/// The refused combination is named before any allocation happens.
fn assert_refuses_unsupported_combination<P>(case: &ExternalImportContractCase<P::Handle>)
where
    P: ExternalImportPolicy,
    P::Handle: Clone,
{
    let importer = ExternalResourceImporter::<P>::new(1);
    let descriptor = ExternalImportDescriptor {
        resource_id: 101,
        format: case.refused_format,
        dimensions: ImageDimensions::d2(1024, 1024),
        row_pitch_bytes: 1024 * 4,
        handle: case.refused_handle.clone(),
        permitted_usages: case.refused_usages,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let refusal = importer
        .authenticate_import(&descriptor)
        .expect_err("an unsupported combination is refused before allocation");

    assert_eq!(
        refusal,
        ResourceAbiError::UnsupportedZeroCopyNegotiation {
            resource_id: 101,
            format: case.refused_format,
            memory_kind: case.refused_memory_kind,
        }
    );
}

/// A pitch that is both short and unaligned is refused, and the refusal states
/// the aligned pitch the import needed.
fn assert_refuses_unaligned_pitch<P>(case: &ExternalImportContractCase<P::Handle>)
where
    P: ExternalImportPolicy,
    P::Handle: Clone,
{
    let importer = ExternalResourceImporter::<P>::new(1);
    let descriptor = ExternalImportDescriptor {
        resource_id: 102,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(100, 100),
        row_pitch_bytes: 400,
        handle: case.admitted_handle.clone(),
        permitted_usages: ResourcePermittedUsages::SAMPLED,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    let refusal = importer
        .authenticate_import(&descriptor)
        .expect_err("an unaligned pitch is refused before allocation");

    assert_eq!(
        refusal,
        ResourceAbiError::InvalidDimensionsOrPitch {
            resource_id: 102,
            provided_pitch: 400,
            required_pitch: 512,
        }
    );
}

/// An admitted import is zero-copy at the exact dimensions and pitch, and its
/// transition schedule executes with no copies and no device-wide waits.
///
/// # Panics
///
/// Panics when the descriptor is refused or the schedule does not execute.
/// Admission and execution are the contract this case exists to prove, so a
/// backend that fails either has failed the suite and reporting it as a
/// recoverable error would let a caller continue past a broken import path.
fn assert_admits_zero_copy_and_executes_schedule<P>(case: &ExternalImportContractCase<P::Handle>)
where
    P: ExternalImportPolicy,
    P::Handle: Clone,
{
    let importer = ExternalResourceImporter::<P>::new(2);
    let descriptor = ExternalImportDescriptor {
        resource_id: 201,
        format: ImageFormat::Rgba8Unorm,
        dimensions: ImageDimensions::d2(ADMITTED_WIDTH, ADMITTED_HEIGHT),
        row_pitch_bytes: ADMITTED_WIDTH * 4,
        handle: case.admitted_handle.clone(),
        permitted_usages: ResourcePermittedUsages::SAMPLED
            .union(ResourcePermittedUsages::COLOR_ATTACHMENT),
        sync_protocol: case.sync_protocol,
    };

    let record = importer
        .import_external_resource(descriptor)
        .expect("Fix: admit this descriptor in the policy under test; the zero-copy case observes nothing without an imported record");

    assert_eq!(record.resource_id, 201);
    assert_eq!(record.device_id, 2);
    assert!(record.is_zero_copy);
    assert_eq!(record.row_pitch_bytes, ADMITTED_WIDTH * 4);

    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(201, ResourceUsageTransition::storage_to_color_attachment());
    schedule.add_wait(case.sync_protocol);
    schedule.add_signal(case.sync_protocol);

    let report = importer
        .registry()
        .execute_transition_schedule(&schedule)
        .expect("Fix: accept a schedule against an admitted resource in the registry under test; the transition counts are unobservable otherwise");

    assert_eq!(report.transitions_executed, 1);
    assert_eq!(report.barriers_emitted, 1);
    assert_eq!(report.timeline_waits_executed, 1);
    assert_eq!(report.timeline_signals_executed, 1);
    assert_eq!(report.copy_count, 0);
    assert_eq!(report.device_wide_waits, 0);
}

/// Device loss invalidates the record, its dependent view and its dependent
/// artifact, and every later schedule against it is refused.
///
/// # Panics
///
/// Panics when the import, the dependent view registration, or the dependent
/// artifact registration is refused. Invalidation cannot be observed without
/// all three in place, so a failure there leaves the case proving nothing.
fn assert_device_loss_invalidates_dependents<P>(case: &ExternalImportContractCase<P::Handle>)
where
    P: ExternalImportPolicy,
    P::Handle: Clone,
{
    let importer = ExternalResourceImporter::<P>::new(3);
    let descriptor = ExternalImportDescriptor {
        resource_id: 301,
        format: ImageFormat::Rgba16Float,
        dimensions: ImageDimensions::d2(ADMITTED_WIDTH, ADMITTED_HEIGHT),
        row_pitch_bytes: ADMITTED_WIDTH * 8,
        handle: case.device_loss_handle.clone(),
        permitted_usages: ResourcePermittedUsages::STORAGE,
        sync_protocol: TimelineSyncProtocol::ImplicitQueue,
    };

    importer
        .import_external_resource(descriptor)
        .expect("Fix: admit this descriptor in the policy under test; device loss invalidates nothing without an imported record");

    importer
        .registry()
        .register_dependent_view(301, 4001)
        .expect("Fix: register a dependent view against an admitted record in the registry under test; view invalidation is unobservable otherwise");
    importer
        .registry()
        .register_dependent_artifact(301, 5001)
        .expect("Fix: register a dependent artifact against an admitted record in the registry under test; artifact invalidation is unobservable otherwise");

    let report = importer.invalidate_on_device_loss();
    assert_eq!(report.device_id, 3);
    assert_eq!(report.invalidated_resources, vec![301]);
    assert_eq!(report.invalidated_views, vec![4001]);
    assert_eq!(report.invalidated_artifacts, vec![5001]);

    let mut schedule = ResourceTransitionSchedule::new();
    schedule.add_transition(301, ResourceUsageTransition::storage_to_color_attachment());
    let refusal = importer
        .registry()
        .execute_transition_schedule(&schedule)
        .expect_err("an invalidated resource refuses its schedule");

    assert_eq!(
        refusal,
        ResourceAbiError::ResourceInvalidated { resource_id: 301 }
    );
}
