//! Model, compiled artifact, and sequence-state residency contracts.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use vyre_driver::backend::{
    ArtifactInstance, BackendError, BindingSet, DeviceIdentity, ResidentOwner, Resource, Submission,
};
use vyre_megakernel::Digest;
use vyre_runtime::model_residency::{
    ArtifactInstanceBinding, ImmutableWeightUpload, ModelAdmission, ModelAdmissionStatus,
    ModelResidency, ModelResidencyError, ModelResidencyKey, ResidencyDevice, SequenceStateSpec,
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

impl ResidencyDevice for RecordingDevice {
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

fn key(seed: u8) -> ModelResidencyKey {
    ModelResidencyKey {
        checkpoint_digest: [seed; 32],
        artifact_digest: [seed.wrapping_add(1); 32],
    }
}

fn weight<'a>(name: &'a str, bytes: &'a [u8]) -> ImmutableWeightUpload<'a> {
    ImmutableWeightUpload {
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
            name: "model residency fixture submission".to_string(),
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

/// Proves cold admission uploads once and an exact warm key reuses weights and artifacts.
#[test]
fn cold_then_warm_model_admission_reuses_every_resident_binding() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8, 2, 3, 4];
    let instance = artifact_fixture(1);
    let first = residency
        .admit_model(ModelAdmission {
            key: key(1),
            weights: vec![weight("weight", &bytes)],
            artifacts: vec![ArtifactInstanceBinding::new(
                "decode",
                Arc::clone(&instance),
                23,
            )],
        })
        .expect("Fix: cold model must admit");
    assert_eq!(first.status, ModelAdmissionStatus::Cold);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        27
    );
    let resource = residency
        .weight(key(1), "weight")
        .expect("Fix: resident weight must bind");
    assert_eq!(device.bytes(&resource), bytes);
    assert!(Arc::ptr_eq(
        &instance,
        &residency
            .artifact(key(1), "decode")
            .expect("Fix: artifact instance must bind")
    ));

    let second = residency
        .admit_model(ModelAdmission {
            key: key(1),
            weights: vec![weight("weight", &bytes)],
            artifacts: vec![ArtifactInstanceBinding::new(
                "decode",
                Arc::clone(&instance),
                23,
            )],
        })
        .expect("Fix: exact warm model must reuse");
    assert_eq!(second.status, ModelAdmissionStatus::Warm);
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 1);
}

/// WHY: device-loss recovery must replace only the native generation, not model identity.
#[test]
fn recovered_artifact_instance_replaces_stale_generation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device, 1024);
    let bytes = [1_u8, 2, 3, 4];
    residency
        .admit_model(ModelAdmission {
            key: key(1),
            weights: vec![weight("weight", &bytes)],
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
fn weight_digest_mismatch_fails_before_allocation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let error = residency
        .admit_model(ModelAdmission {
            key: key(2),
            weights: vec![ImmutableWeightUpload {
                name: "weight",
                bytes: &[1, 2, 3],
                blake3: [0; 32],
            }],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: bad weight digest must fail");
    assert!(matches!(
        error,
        ModelResidencyError::WeightDigestMismatch { .. }
    ));
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 0);
}

/// Prevents OOM admission from performing a partial device allocation.
#[test]
fn model_oom_admission_fails_before_allocation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 3);
    let bytes = [1_u8, 2, 3, 4];
    assert_eq!(
        residency
            .admit_model(ModelAdmission {
                key: key(3),
                weights: vec![weight("weight", &bytes)],
                artifacts: Vec::new(),
            })
            .expect_err("Fix: over-budget model must fail"),
        ModelResidencyError::OutOfMemory {
            context: "model admission",
            used: 0,
            requested: 4,
            budget: 3,
        }
    );
    assert_eq!(device.allocation_calls.load(Ordering::SeqCst), 0);
}

/// Proves a late allocation failure frees every earlier weight and commits no accounting.
#[test]
fn partial_weight_allocation_rolls_back_completely() {
    let device = Arc::new(RecordingDevice::new());
    device.fail_on_allocation(2);
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let first = [1_u8; 4];
    let second = [2_u8; 4];
    let error = residency
        .admit_model(ModelAdmission {
            key: key(4),
            weights: vec![weight("first", &first), weight("second", &second)],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: second allocation failure must roll back first");
    assert!(matches!(error, ModelResidencyError::Backend { .. }));
    assert_eq!(device.resident_count(), 0);
    assert_eq!(device.free_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Proves a failed batched upload frees every allocated weight and leaves no warm entry.
#[test]
fn failed_weight_upload_rolls_back_all_allocations() {
    let device = Arc::new(RecordingDevice::new());
    device.fail_upload_many.store(true, Ordering::SeqCst);
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    let error = residency
        .admit_model(ModelAdmission {
            key: key(5),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect_err("Fix: batch upload failure must roll back");
    assert!(matches!(error, ModelResidencyError::Backend { .. }));
    assert_eq!(device.resident_count(), 0);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Proves concurrent sequences receive distinct identities and isolated zeroed state.
#[test]
fn concurrent_sequences_own_isolated_zero_initialized_state() {
    let device = Arc::new(RecordingDevice::new());
    let residency = Arc::new(ModelResidency::with_device(device.clone(), 4096));
    let bytes = [9_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(6),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: concurrent fixture model must admit");
    let joins = (0..8)
        .map(|_| {
            let residency = Arc::clone(&residency);
            thread::spawn(move || {
                residency
                    .start_sequence(
                        key(6),
                        &[SequenceStateSpec {
                            name: "cache",
                            byte_len: 16,
                        }],
                    )
                    .expect("Fix: concurrent sequence must start")
            })
        })
        .collect::<Vec<_>>();
    let leases = joins
        .into_iter()
        .map(|join| join.join().expect("Fix: sequence thread must not panic"))
        .collect::<Vec<_>>();
    let mut ids = leases.iter().map(|lease| lease.id).collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 8);
    for lease in &leases {
        let resource = residency
            .sequence_state(*lease, "cache")
            .expect("Fix: sequence cache must bind");
        assert_eq!(device.bytes(&resource), [0; 16]);
    }
    assert_eq!(
        residency
            .active_sequences(key(6))
            .expect("Fix: active count must read"),
        8
    );
    for lease in leases {
        residency
            .finish_sequence(lease)
            .expect("Fix: sequence must finish");
    }
    assert_eq!(
        residency
            .active_sequences(key(6))
            .expect("Fix: active count must read"),
        0
    );
}

/// Proves cancellation releases state and every old lease becomes unusable.
#[test]
fn cancellation_releases_state_and_invalidates_the_lease() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(7),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: cancellation fixture model must admit");
    let lease = residency
        .start_sequence(
            key(7),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: cancellation fixture sequence must start");
    residency
        .cancel_sequence(lease)
        .expect("Fix: cancellation must release state");
    assert_eq!(
        residency
            .sequence_state(lease, "cache")
            .expect_err("Fix: cancelled lease must be stale"),
        ModelResidencyError::SequenceNotFound { sequence: lease.id }
    );
    assert_eq!(device.resident_count(), 1);
}

/// Proves reset zeroes mutated bytes and invalidates the previous generation.
#[test]
fn cache_reset_zeroes_state_and_rotates_the_generation() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(8),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: reset fixture model must admit");
    let old = residency
        .start_sequence(
            key(8),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: reset fixture sequence must start");
    let resource = residency
        .sequence_state(old, "cache")
        .expect("Fix: reset fixture cache must bind");
    device.overwrite(&resource, &[9; 8]);
    let current = residency
        .reset_sequence(old)
        .expect("Fix: reset must succeed");
    assert_eq!(current.id, old.id);
    assert_eq!(current.generation, 1);
    assert_eq!(device.bytes(&resource), [0; 8]);
    assert_eq!(
        residency
            .sequence_state(old, "cache")
            .expect_err("Fix: old generation must fail"),
        ModelResidencyError::StaleSequenceLease {
            sequence: old.id,
            expected_generation: 1,
            actual_generation: 0,
        }
    );
}

/// Proves a failed reset destroys partial state instead of exposing stale cache bytes.
#[test]
fn failed_reset_removes_the_sequence_fail_closed() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(9),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: failed-reset fixture model must admit");
    let lease = residency
        .start_sequence(
            key(9),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: failed-reset fixture sequence must start");
    device.fail_upload_at.store(true, Ordering::SeqCst);
    let error = residency
        .reset_sequence(lease)
        .expect_err("Fix: failed reset must remove sequence");
    assert!(matches!(error, ModelResidencyError::Backend { .. }));
    assert_eq!(
        residency
            .active_sequences(key(9))
            .expect("Fix: active count must read"),
        0
    );
    assert_eq!(
        residency
            .sequence_state(lease, "cache")
            .expect_err("Fix: failed-reset lease must be gone"),
        ModelResidencyError::SequenceNotFound { sequence: lease.id }
    );
}

/// Prevents model eviction while a sequence still owns mutable state.
#[test]
fn eviction_requires_all_sequences_to_finish() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(10),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: eviction fixture model must admit");
    let lease = residency
        .start_sequence(
            key(10),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: eviction fixture sequence must start");
    assert_eq!(
        residency
            .evict_model(key(10))
            .expect_err("Fix: active model eviction must fail"),
        ModelResidencyError::ModelInUse {
            key: key(10),
            active_sequences: 1,
        }
    );
    residency
        .finish_sequence(lease)
        .expect("Fix: eviction fixture sequence must finish");
    residency
        .evict_model(key(10))
        .expect("Fix: idle model must evict");
    assert_eq!(device.resident_count(), 0);
    assert_eq!(
        residency.used_bytes().expect("Fix: accounting must read"),
        0
    );
}

/// Prevents cancelled sequence identities from being reused with freshly zeroed state.
#[test]
fn new_sequence_after_cancellation_has_new_identity_and_no_stale_bytes() {
    let device = Arc::new(RecordingDevice::new());
    let residency = ModelResidency::with_device(device.clone(), 1024);
    let bytes = [1_u8; 4];
    residency
        .admit_model(ModelAdmission {
            key: key(11),
            weights: vec![weight("weight", &bytes)],
            artifacts: Vec::new(),
        })
        .expect("Fix: stale-state fixture model must admit");
    let first = residency
        .start_sequence(
            key(11),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: first sequence must start");
    let first_resource = residency
        .sequence_state(first, "cache")
        .expect("Fix: first cache must bind");
    device.overwrite(&first_resource, &[0xff; 8]);
    residency
        .cancel_sequence(first)
        .expect("Fix: first sequence must cancel");
    let second = residency
        .start_sequence(
            key(11),
            &[SequenceStateSpec {
                name: "cache",
                byte_len: 8,
            }],
        )
        .expect("Fix: second sequence must start");
    assert_ne!(second.id, first.id);
    let second_resource = residency
        .sequence_state(second, "cache")
        .expect("Fix: second cache must bind");
    assert_eq!(device.bytes(&second_resource), [0; 8]);
}

/// Prevents manager destruction from leaking live model or sequence resources.
#[test]
fn dropping_manager_releases_models_and_active_sequences() {
    let device = Arc::new(RecordingDevice::new());
    {
        let residency = ModelResidency::with_device(device.clone(), 1024);
        let bytes = [1_u8; 4];
        residency
            .admit_model(ModelAdmission {
                key: key(12),
                weights: vec![weight("weight", &bytes)],
                artifacts: Vec::new(),
            })
            .expect("Fix: drop fixture model must admit");
        residency
            .start_sequence(
                key(12),
                &[SequenceStateSpec {
                    name: "cache",
                    byte_len: 8,
                }],
            )
            .expect("Fix: drop fixture sequence must start");
        assert_eq!(device.resident_count(), 2);
    }
    assert_eq!(device.resident_count(), 0);
    assert_eq!(device.free_calls.load(Ordering::SeqCst), 2);
}
