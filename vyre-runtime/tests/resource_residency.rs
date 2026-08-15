//! Immutable resource, compiled artifact, and mutable-state residency contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use vyre_driver::{
    ArtifactInstance, BackendError, BindingSet, DeviceIdentity, ResidentOwner, Resource, Submission,
};
use vyre_megakernel::Digest;
use vyre_runtime::resource_residency::{
    ArtifactInstanceBinding, ImmutableResourceUpload, MutableStateSpec, ResidentResourceDevice,
    ResourceAdmissionStatus, ResourceResidency, ResourceResidencyError, ResourceSetAdmission,
    ResourceSetKey,
};

#[derive(Debug)]
struct RecordingDevice {
    owner: ResidentOwner,
    next_id: AtomicU64,
    allocation_calls: AtomicUsize,
    free_calls: AtomicUsize,
    fail_allocation_at: AtomicUsize,
    fail_upload_many: AtomicBool,
    fail_upload_at: AtomicBool,
    buffers: Mutex<BTreeMap<u64, Vec<u8>>>,
}

impl RecordingDevice {
    fn new() -> Self {
        Self {
            owner: ResidentOwner::new().expect("Fix: test resident owner must be available"),
            next_id: AtomicU64::new(1),
            allocation_calls: AtomicUsize::new(0),
            free_calls: AtomicUsize::new(0),
            fail_allocation_at: AtomicUsize::new(0),
            fail_upload_many: AtomicBool::new(false),
            fail_upload_at: AtomicBool::new(false),
            buffers: Mutex::new(BTreeMap::new()),
        }
    }

    fn fail_on_allocation(&self, call: usize) {
        self.fail_allocation_at.store(call, Ordering::SeqCst);
    }

    fn resident_count(&self) -> usize {
        self.buffers
            .lock()
            .expect("Fix: test device lock must remain usable")
            .len()
    }

    fn bytes(&self, resource: &Resource) -> Vec<u8> {
        let id = self
            .resolve(resource)
            .expect("Fix: test resource must resolve");
        self.buffers
            .lock()
            .expect("Fix: test device lock must remain usable")[&id]
            .clone()
    }

    fn overwrite(&self, resource: &Resource, bytes: &[u8]) {
        let id = self
            .resolve(resource)
            .expect("Fix: test resource must resolve");
        let mut buffers = self
            .buffers
            .lock()
            .expect("Fix: test device lock must remain usable");
        let target = buffers
            .get_mut(&id)
            .expect("Fix: test resource must remain allocated");
        assert_eq!(target.len(), bytes.len());
        target.copy_from_slice(bytes);
    }

    fn resolve(&self, resource: &Resource) -> Result<u64, BackendError> {
        match resource {
            Resource::Resident(handle) => self.owner.resolve(*handle, "recording residency device"),
            Resource::Borrowed(_) => Err(injected("borrowed resource is not resident")),
        }
    }
}

impl ResidentResourceDevice for RecordingDevice {
    fn allocate(&self, byte_len: usize) -> Result<Resource, BackendError> {
        let call = self.allocation_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_allocation_at.load(Ordering::SeqCst) == call {
            return Err(injected("injected resident allocation failure"));
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        self.buffers
            .lock()
            .map_err(|_| injected("recording device lock poisoned"))?
            .insert(id, vec![0xa5; byte_len]);
        Ok(Resource::Resident(self.owner.handle(id)))
    }

    fn upload_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError> {
        if self.fail_upload_many.load(Ordering::SeqCst) {
            return Err(injected("injected resident batch upload failure"));
        }
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| injected("recording device lock poisoned"))?;
        for (resource, bytes) in uploads {
            let id = self.resolve(resource)?;
            let target = buffers
                .get_mut(&id)
                .ok_or_else(|| injected("resident upload target is absent"))?;
            if target.len() != bytes.len() {
                return Err(injected("resident upload length mismatch"));
            }
            target.copy_from_slice(bytes);
        }
        Ok(())
    }

    fn upload_at(
        &self,
        resource: &Resource,
        offset: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        if self.fail_upload_at.load(Ordering::SeqCst) {
            return Err(injected("injected resident ranged upload failure"));
        }
        let id = self.resolve(resource)?;
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| injected("recording device lock poisoned"))?;
        let target = buffers
            .get_mut(&id)
            .ok_or_else(|| injected("resident ranged upload target is absent"))?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| injected("resident ranged upload overflow"))?;
        let destination = target
            .get_mut(offset..end)
            .ok_or_else(|| injected("resident ranged upload exceeds target"))?;
        destination.copy_from_slice(bytes);
        Ok(())
    }

    fn free(&self, resource: Resource) -> Result<(), BackendError> {
        let id = self.resolve(&resource)?;
        let removed = self
            .buffers
            .lock()
            .map_err(|_| injected("recording device lock poisoned"))?
            .remove(&id);
        if removed.is_none() {
            return Err(injected("resident free target is absent"));
        }
        self.free_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn injected(detail: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!("Fix: {detail}"),
    }
}

fn key(seed: u8) -> ResourceSetKey {
    ResourceSetKey {
        source_digest: [seed; 32],
        artifact_digest: [seed.wrapping_add(1); 32],
    }
}

fn immutable_resource<'a>(name: &'a str, bytes: &'a [u8]) -> ImmutableResourceUpload<'a> {
    ImmutableResourceUpload {
        name,
        bytes,
        blake3: *blake3::hash(bytes).as_bytes(),
    }
}

struct FixtureInstance {
    device: DeviceIdentity,
}

impl ArtifactInstance for FixtureInstance {
    fn artifact(&self) -> Digest {
        Digest([2; 32])
    }

    fn payload(&self) -> Digest {
        Digest([3; 32])
    }

    fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    fn submit(&self, _bindings: BindingSet) -> Result<Box<dyn Submission>, BackendError> {
        Err(BackendError::UnsupportedFeature {
            name: "resource residency fixture submission".to_string(),
            backend: "fixture".to_string(),
        })
    }
}

fn artifact_fixture(generation: u64) -> Arc<dyn ArtifactInstance> {
    Arc::new(FixtureInstance {
        device: DeviceIdentity {
            backend: "fixture",
            device: "fixture-device".to_string(),
            generation,
        },
    })
}

/// Proves cold admission uploads once and an exact warm key reuses immutable resources and artifacts.
#[test]
fn cold_then_warm_resource_set_admission_reuses_every_resident_binding() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8, 2, 3, 4];
    let instance = artifact_fixture(1);
    let first = residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(1),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: vec![ArtifactInstanceBinding::new(
                "decode",
                Arc::clone(&instance),
                23,
            )],
        })
        .expect("Fix: cold resource set must admit");
    assert_eq!(first.status, ResourceAdmissionStatus::Cold);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        27
    );
    let resource = residency
        .immutable_resource(key(1), "immutable_resource")
        .expect("Fix: resident immutable_resource must bind");
    assert_eq!(device.bytes(&resource), bytes);
    assert!(Arc::ptr_eq(
        &instance,
        &residency
            .artifact(key(1), "decode")
            .expect("Fix: artifact instance must bind")
    ));

    let second = residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(1),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: vec![ArtifactInstanceBinding::new(
                "decode",
                Arc::clone(&instance),
                23,
            )],
        })
        .expect("Fix: exact warm resource_set must reuse");
    assert_eq!(second.status, ResourceAdmissionStatus::Warm);
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 1);
}

/// WHY: device-loss recovery must replace only the native generation, not resource_set identity.
#[test]
fn recovered_artifact_instance_replaces_stale_generation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device, 1024);
    let bytes = [1_u8, 2, 3, 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(1),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: vec![ArtifactInstanceBinding::new(
                "decode",
                artifact_fixture(1),
                23,
            )],
        })
        .unwrap();

    residency
        .replace_artifact_instance(key(1), "decode", artifact_fixture(2))
        .unwrap();
    assert_eq!(
        residency
            .artifact(key(1), "decode")
            .unwrap()
            .device()
            .generation,
        2
    );
}

/// Prevents unverified immutable bytes from reaching a backend allocation.
#[test]
fn immutable_resource_digest_mismatch_fails_before_allocation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let error = residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(2),
            immutable_resources: vec![ImmutableResourceUpload {
                name: "immutable_resource",
                bytes: &[1, 2, 3],
                blake3: [0; 32],
            }],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: bad immutable_resource digest must fail");
    assert!(matches!(
        error,
        ResourceResidencyError::ImmutableResourceDigestMismatch { .. }
    ));
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 0);
}

/// Prevents OOM admission from performing a partial device allocation.
#[test]
fn resource_set_oom_admission_fails_before_allocation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 3);
    let bytes = [1_u8, 2, 3, 4];
    assert_eq!(
        residency
            .admit_resource_set(ResourceSetAdmission {
                key: key(3),
                immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
                artifacts: Vec::new(),
            })
            .expect_err("Fix: over-budget resource_set must fail"),
        ResourceResidencyError::OutOfMemory {
            context: "resource-set admission",
            used: 0,
            requested: 4,
            budget: 3,
        }
    );
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 0);
}

/// Proves a late allocation failure frees every earlier immutable_resource and commits no accounting.
#[test]
fn partial_immutable_resource_allocation_rolls_back_completely() {
    let device = Arc::new(RecordingDevice::new());
    device.fail_on_allocation(2);
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let first = [1_u8; 4];
    let second = [2_u8; 4];
    let error = residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(4),
            immutable_resources: vec![
                immutable_resource("first", &first),
                immutable_resource("second", &second),
            ],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: second allocation failure must roll back first");
    assert!(matches!(error, ResourceResidencyError::Backend { .. }));
    assert_eq!(device.resident_count(), 0);
    assert_eq!(device.free_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Proves a failed batched upload frees every allocated immutable_resource and leaves no warm entry.
#[test]
fn failed_immutable_resource_upload_rolls_back_all_allocations() {
    let device = Arc::new(RecordingDevice::new());
    device.fail_upload_many.store(true, Ordering::SeqCst);
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    let error = residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(5),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: batch upload failure must roll back");
    assert!(matches!(error, ResourceResidencyError::Backend { .. }));
    assert_eq!(device.resident_count(), 0);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Proves concurrent states receive distinct identities and isolated zeroed state.
#[test]
fn concurrent_states_own_isolated_zero_initialized_state() {
    let device = Arc::new(RecordingDevice::new());
    let residency = Arc::new(ResourceResidency::with_device(device.clone(), 4096));
    let bytes = [9_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(6),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: concurrent fixture resource_set must admit");
    let joins = (0..8)
        .map(|_| {
            let residency = Arc::clone(&residency);
            thread::spawn(move || {
                residency
                    .start_state(
                        key(6),
                        &[MutableStateSpec {
                            name: "cache",
                            byte_len: 16,
                        }],
                    )
                    .expect("Fix: concurrent state must start")
            })
        })
        .collect::<Vec<_>>();
    let leases = joins
        .into_iter()
        .map(|join| join.join().expect("Fix: state thread must not panic"))
        .collect::<Vec<_>>();
    let mut ids = leases.iter().map(|lease| lease.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 8);
    for lease in &leases {
        let resource = residency
            .mutable_state(*lease, "cache")
            .expect("Fix: state cache must bind");
        assert_eq!(device.bytes(&resource), [0; 16]);
    }
    assert_eq!(
        residency
            .active_states(key(6))
            .expect("Fix: active count must read"),
        8
    );
    for lease in leases {
        residency
            .finish_state(lease)
            .expect("Fix: state must finish");
    }
    assert_eq!(
        residency
            .active_states(key(6))
            .expect("Fix: active count must read"),
        0
    );
}

/// Proves cancellation releases state and every old lease becomes unusable.
#[test]
fn cancellation_releases_state_and_invalidates_the_lease() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(7),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: cancellation fixture resource_set must admit");
    let lease = residency
        .start_state(
            key(7),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: cancellation fixture state must start");
    residency
        .cancel_state(lease)
        .expect("Fix: cancellation must release state");
    assert_eq!(
        residency
            .mutable_state(lease, "cache")
            .expect_err("Fix: cancelled lease must be stale"),
        ResourceResidencyError::StateLeaseNotFound { state: lease.id }
    );
    assert_eq!(device.resident_count(), 1);
}

/// Proves reset zeroes mutated bytes and invalidates the previous generation.
#[test]
fn cache_reset_zeroes_state_and_rotates_the_generation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(8),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: reset fixture resource_set must admit");
    let old = residency
        .start_state(
            key(8),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: reset fixture state must start");
    let resource = residency
        .mutable_state(old, "cache")
        .expect("Fix: reset fixture cache must bind");
    device.overwrite(&resource, &[9; 8]);
    let current = residency.reset_state(old).expect("Fix: reset must succeed");
    assert_eq!(current.id, old.id);
    assert_eq!(current.generation, 1);
    assert_eq!(device.bytes(&resource), [0; 8]);
    assert_eq!(
        residency
            .mutable_state(old, "cache")
            .expect_err("Fix: old generation must fail"),
        ResourceResidencyError::StaleStateLease {
            state: old.id,
            expected_generation: 1,
            actual_generation: 0,
        }
    );
}

/// Proves a failed reset destroys partial state instead of exposing stale cache bytes.
#[test]
fn failed_reset_removes_the_state_fail_closed() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(9),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: failed-reset fixture resource_set must admit");
    let lease = residency
        .start_state(
            key(9),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: failed-reset fixture state must start");
    device.fail_upload_at.store(true, Ordering::SeqCst);
    let error = residency
        .reset_state(lease)
        .expect_err("Fix: failed reset must remove state");
    assert!(matches!(error, ResourceResidencyError::Backend { .. }));
    assert_eq!(
        residency
            .active_states(key(9))
            .expect("Fix: active count must read"),
        0
    );
    assert_eq!(
        residency
            .mutable_state(lease, "cache")
            .expect_err("Fix: failed-reset lease must be gone"),
        ResourceResidencyError::StateLeaseNotFound { state: lease.id }
    );
}

/// Prevents resource-set eviction while a state still owns mutable state.
#[test]
fn eviction_requires_all_states_to_finish() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(10),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: eviction fixture resource_set must admit");
    let lease = residency
        .start_state(
            key(10),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: eviction fixture state must start");
    assert_eq!(
        residency
            .evict_resource_set(key(10))
            .expect_err("Fix: active resource set eviction must fail"),
        ResourceResidencyError::ResourceSetInUse {
            key: key(10),
            active_states: 1,
        }
    );
    residency
        .finish_state(lease)
        .expect("Fix: eviction fixture state must finish");
    residency
        .evict_resource_set(key(10))
        .expect("Fix: idle resource_set must evict");
    assert_eq!(device.resident_count(), 0);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Prevents cancelled state identities from being reused with freshly zeroed state.
#[test]
fn new_state_after_cancellation_has_new_identity_and_no_stale_bytes() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ResourceResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_resource_set(ResourceSetAdmission {
            key: key(11),
            immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: stale-state fixture resource_set must admit");
    let first = residency
        .start_state(
            key(11),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: first state must start");
    let first_resource = residency
        .mutable_state(first, "cache")
        .expect("Fix: first cache must bind");
    device.overwrite(&first_resource, &[0xff; 8]);
    residency
        .cancel_state(first)
        .expect("Fix: first state must cancel");
    let second = residency
        .start_state(
            key(11),
            &[MutableStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: second state must start");
    assert_ne!(second.id, first.id);
    let second_resource = residency
        .mutable_state(second, "cache")
        .expect("Fix: second cache must bind");
    assert_eq!(device.bytes(&second_resource), [0; 8]);
}

/// Prevents manager destruction from leaking live resource_set or state resources.
#[test]
fn dropping_manager_releases_resource_sets_and_active_states() {
    let device = Arc::new(RecordingDevice::new());
    {
        let residency = ResourceResidency::with_device(device.clone(), 1024);
        let bytes = [1_u8; 4];
        residency
            .admit_resource_set(ResourceSetAdmission {
                key: key(12),
                immutable_resources: vec![immutable_resource("immutable_resource", &bytes)],
                artifacts: Vec::new(),
            })
            .expect("Fix: drop fixture resource_set must admit");
        residency
            .start_state(
                key(12),
                &[MutableStateSpec {
                    name: "cache",
                    byte_len: 8,
                }],
            )
            .expect("Fix: drop fixture state must start");
        assert_eq!(device.resident_count(), 2);
    }
    assert_eq!(device.resident_count(), 0);
    assert_eq!(device.free_calls.load(Ordering::SeqCst), 2);
}
