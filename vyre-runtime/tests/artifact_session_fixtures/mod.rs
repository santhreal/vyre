//! Shared test fixtures for artifact session and materializer contracts.
//!
//! Provides a recording materializer, neutral artifact payload builder,
//! and fixture backend registration shared across session state machine,
//! typed resource ingestion, and workspace contracts.


use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use vyre_driver::materialize::{DeviceSpec, MaterializerDevice};
use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BackendRegistration, Device,
    ResidentOwner, Resource, VyreBackend,
};
use vyre_megakernel::{
    AbiAccess, Artifact, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_test_support::fixture_instance::FixtureInstance;

/// A materializer that records resident allocations and releases while producing neutral instances.
pub struct SessionFixtureMaterializer {
    /// Underlying materializer device.
    pub device: MaterializerDevice,
    /// Resident resource owner.
    pub owner: ResidentOwner,
    /// Next resident handle identifier.
    pub next: AtomicU64,
    /// Log of byte counts allocated for resident resources.
    pub allocated: Mutex<Vec<usize>>,
    /// Log of resident resources freed.
    pub freed: Mutex<Vec<Resource>>,
}

impl SessionFixtureMaterializer {
    /// Acquire a new recording materializer device with the given backend, device, and format names.
    pub fn new(backend: &'static str, device: &str, format: &'static str) -> Arc<Self> {
        Arc::new(Self {
            device: MaterializerDevice::acquire(DeviceSpec {
                backend,
                device: device.to_string(),
                format_extension: format,
                format_version: 1,
                profile: TargetProfile::new(format, 1, [64, 1, 1], 64, 1_024, 0)
                    .expect("fixture target profile must be valid"),
            })
            .expect("fixture materializer device must acquire"),
            owner: ResidentOwner::new().expect("resident owner must be acquired"),
            next: AtomicU64::new(0),
            allocated: Mutex::new(Vec::new()),
            freed: Mutex::new(Vec::new()),
        })
    }
}

impl ArtifactMaterializer for SessionFixtureMaterializer {
    fn device(&self) -> &dyn Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn ArtifactInstance>, BackendError> {
        Ok(FixtureInstance::neutral(
            artifact,
            payload,
            self.device.identity(),
        ))
    }

    fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.allocated
            .lock()
            .expect("allocation log must not be poisoned")
            .push(byte_len);
        let id = self.next.fetch_add(1, Ordering::AcqRel);
        Ok(Resource::Resident(self.owner.handle(id)))
    }

    fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
        self.freed
            .lock()
            .expect("free log must not be poisoned")
            .push(resource);
        Ok(())
    }

    fn upload_resident(&self, _resource: &Resource, _bytes: &[u8]) -> Result<(), BackendError> {
        Ok(())
    }

    fn upload_resident_at(
        &self,
        _resource: &Resource,
        _offset_bytes: usize,
        _bytes: &[u8],
    ) -> Result<(), BackendError> {
        Ok(())
    }
}

/// Backend factory that returns UnsupportedFeature for tests that only admit artifacts.
pub fn fixture_backend_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    Err(BackendError::UnsupportedFeature {
        name: "raw Program backend".to_string(),
        backend: "fixture-artifact".to_string(),
    })
}

/// Empty supported ops set for fixture registrations.
pub fn fixture_supported_ops() -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    static OPS: std::sync::LazyLock<std::collections::HashSet<vyre_foundation::ir::OpId>> =
        std::sync::LazyLock::new(std::collections::HashSet::new);
    &OPS
}

/// Construct a minimal backend registration for fixture sessions.
pub const fn fixture_backend_registration(id: &'static str) -> BackendRegistration {
    BackendRegistration {
        id,
        target_id: vyre_foundation::operation::TargetId::expect_valid(id),
        payload_format: None,
        reference_oracle: false,
        factory: fixture_backend_factory,
        supported_ops: fixture_supported_ops,
        semantic_operations: fixture_supported_ops,
        target_compiler: None,
        materializer: None,
    }
}

/// Construct a target payload for an artifact by binding each entry's inputs and outputs.
pub fn fixture_target_payload(
    artifact: &Artifact,
    format: &str,
    payload_bytes: Vec<u8>,
) -> TargetPayload {
    let entries = artifact
        .abi()
        .entries
        .iter()
        .map(|entry| {
            let recorded = artifact
                .geometry()
                .iter()
                .find(|record| record.node == entry.node)
                .expect("the artifact must record geometry for every entry");
            let resource_bindings = entry
                .inputs
                .iter()
                .chain(entry.outputs.iter())
                .enumerate()
                .map(|(slot, resource)| {
                    let access = artifact
                        .abi()
                        .resources
                        .iter()
                        .find(|r| r.value == *resource)
                        .map(|r| match r.access {
                            AbiAccess::ReadOnly | AbiAccess::Uniform => {
                                TargetResourceAccess::ReadOnly
                            }
                            AbiAccess::WriteOnly => TargetResourceAccess::WriteOnly,
                            AbiAccess::ReadWrite => TargetResourceAccess::ReadWrite,
                        })
                        .unwrap_or(TargetResourceAccess::ReadWrite);
                    TargetResourceBinding {
                        resource: *resource,
                        group: 0,
                        slot: u32::try_from(slot).expect("resource slot index must fit in u32"),
                        memory: TargetResourceMemory::Global,
                        access,
                    }
                })
                .collect();
            TargetEntryPoint {
                name: format!("entry{}", entry.node.0),
                node: entry.node,
                workgroup_size: recorded.workgroup_size,
                grid_size: recorded.grid,
                dynamic_shared_bytes: recorded.dynamic_shared_bytes,
                resource_bindings,
            }
        })
        .collect();

    TargetPayload::new(
        artifact,
        TargetPayloadFormat::new(format, 1).expect("target payload format must be valid"),
        TargetProfile::new(format, 1, [64, 1, 1], 64, 1_024, 0)
            .expect("target profile must be valid"),
        entries,
        payload_bytes,
    )
    .expect("target payload must seal")
}
