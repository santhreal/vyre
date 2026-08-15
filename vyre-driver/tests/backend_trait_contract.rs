//! The `VyreBackend` default and override contract.
//!
//! Two fixtures bracket the trait. `MinimalBackend` overrides nothing beyond
//! `id` and `dispatch`, so every capability query and lifecycle hook takes its
//! default body: this is the shape a new backend starts from, and a default that
//! is renamed, moved, or made required stops this file compiling. `FullBackend`
//! overrides every capability and every hook with a value observably different
//! from that default, so an override that silently returns the default value is
//! a failure rather than a coincidence.
//!
//! Object safety, `Send + Sync`, and the `Backend` blanket implementation are
//! part of the same contract and live here. The async adapter defaults belong to
//! `async_dispatch_contract`; what this file asserts about them is only that a
//! heavily overridden backend still inherits them.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use vyre_driver::backend::Backend;
use vyre_driver::{BackendError, CompiledPipeline, DispatchConfig, VyreBackend};
use vyre_foundation::ir::Program;

/// Overrides nothing beyond the two required methods.
struct MinimalBackend;

impl vyre_driver::backend::private::Sealed for MinimalBackend {}

impl VyreBackend for MinimalBackend {
    fn id(&self) -> &'static str {
        "minimal"
    }

    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![])
    }
}

/// Overrides every capability query and every lifecycle hook.
///
/// The hooks count their calls so a test can prove trait dispatch reached the
/// override rather than the default body.
struct FullBackend {
    prepare_calls: AtomicUsize,
    flush_calls: AtomicUsize,
    shutdown_calls: AtomicUsize,
    recover_calls: AtomicUsize,
}

impl FullBackend {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            prepare_calls: AtomicUsize::new(0),
            flush_calls: AtomicUsize::new(0),
            shutdown_calls: AtomicUsize::new(0),
            recover_calls: AtomicUsize::new(0),
        })
    }
}

impl vyre_driver::backend::private::Sealed for FullBackend {}

impl VyreBackend for FullBackend {
    fn id(&self) -> &'static str {
        "full"
    }
    fn version(&self) -> &'static str {
        "0.6.0-test"
    }
    fn dispatch(
        &self,
        _program: &Program,
        _inputs: &[Vec<u8>],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![])
    }

    fn supports_subgroup_ops(&self) -> bool {
        true
    }
    fn supports_f16(&self) -> bool {
        true
    }
    fn supports_bf16(&self) -> bool {
        true
    }
    fn supports_tensor_cores(&self) -> bool {
        true
    }
    fn supports_async_compute(&self) -> bool {
        true
    }
    fn supports_indirect_dispatch(&self) -> bool {
        true
    }
    fn is_distributed(&self) -> bool {
        true
    }
    fn max_workgroup_size(&self) -> [u32; 3] {
        [1024, 1024, 64]
    }
    fn max_storage_buffer_bytes(&self) -> u64 {
        1 << 40
    }

    fn prepare(&self) -> Result<(), BackendError> {
        self.prepare_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn flush(&self) -> Result<(), BackendError> {
        self.flush_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn shutdown(&self) -> Result<(), BackendError> {
        self.shutdown_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn device_lost(&self) -> bool {
        true
    }
    fn try_recover(&self) -> Result<(), BackendError> {
        self.recover_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn every_capability_default_is_conservative() {
    let backend: Arc<dyn VyreBackend> = Arc::new(MinimalBackend);
    assert_eq!(
        backend.version(),
        "unspecified",
        "Fix: a backend that did not override version must report `unspecified`, not invent one."
    );
    assert!(
        !backend.supports_subgroup_ops(),
        "Fix: default supports_subgroup_ops must be false."
    );
    assert!(
        !backend.supports_f16(),
        "Fix: default supports_f16 must be false."
    );
    assert!(
        !backend.supports_bf16(),
        "Fix: default supports_bf16 must be false."
    );
    assert!(
        !backend.supports_tensor_cores(),
        "Fix: default supports_tensor_cores must be false."
    );
    assert!(
        !backend.supports_async_compute(),
        "Fix: default supports_async_compute must be false."
    );
    assert!(
        !backend.supports_indirect_dispatch(),
        "Fix: default supports_indirect_dispatch must be false."
    );
    assert!(
        !backend.is_distributed(),
        "Fix: default is_distributed must be false."
    );
    assert_eq!(
        backend.max_workgroup_size(),
        [1, 1, 1],
        "Fix: default max_workgroup_size must be the scalar grid."
    );
    assert_eq!(
        backend.max_compute_workgroups_per_dimension(),
        1,
        "Fix: default max_compute_workgroups_per_dimension must claim one workgroup."
    );
    assert_eq!(
        backend.max_compute_invocations_per_workgroup(),
        1,
        "Fix: default max_compute_invocations_per_workgroup must claim one invocation."
    );
    assert_eq!(
        backend.max_storage_buffer_bytes(),
        0,
        "Fix: default max_storage_buffer_bytes must be zero, never a guessed limit."
    );
    assert!(
        backend.subgroup_size().is_none(),
        "Fix: a backend that never probed a subgroup width must report None."
    );
    assert!(
        !backend.device_lost(),
        "Fix: default device_lost must be false."
    );
}

#[test]
fn default_lifecycle_hooks_succeed_and_recovery_stays_opt_in() {
    let backend = MinimalBackend;
    backend
        .prepare()
        .expect("Fix: default prepare must return Ok so every backend author can rely on it.");
    backend
        .flush()
        .expect("Fix: default flush must return Ok so every backend author can rely on it.");
    backend
        .shutdown()
        .expect("Fix: default shutdown must return Ok so every backend author can rely on it.");
    let error = backend
        .try_recover()
        .expect_err("Fix: default try_recover must fail because recovery is opt-in.");
    assert!(
        matches!(error, BackendError::UnsupportedFeature { .. }),
        "Fix: default try_recover must refuse with UnsupportedFeature: {error:?}"
    );
}

#[test]
fn every_override_is_observably_different_from_its_default() {
    let minimal = MinimalBackend;
    let full = FullBackend::new();

    assert_ne!(minimal.version(), full.version());
    assert_ne!(
        minimal.supports_subgroup_ops(),
        full.supports_subgroup_ops()
    );
    assert_ne!(minimal.supports_f16(), full.supports_f16());
    assert_ne!(minimal.supports_bf16(), full.supports_bf16());
    assert_ne!(
        minimal.supports_tensor_cores(),
        full.supports_tensor_cores()
    );
    assert_ne!(
        minimal.supports_async_compute(),
        full.supports_async_compute()
    );
    assert_ne!(
        minimal.supports_indirect_dispatch(),
        full.supports_indirect_dispatch()
    );
    assert_ne!(minimal.is_distributed(), full.is_distributed());
    assert_ne!(minimal.max_workgroup_size(), full.max_workgroup_size());
    assert_ne!(
        minimal.max_storage_buffer_bytes(),
        full.max_storage_buffer_bytes()
    );
    assert_ne!(minimal.device_lost(), full.device_lost());

    assert!(full.supports_subgroup_ops());
    assert!(full.supports_f16());
    assert!(full.supports_bf16());
    assert!(full.supports_tensor_cores());
    assert!(full.supports_async_compute());
    assert!(full.supports_indirect_dispatch());
    assert!(full.is_distributed());
    assert_eq!(full.max_workgroup_size(), [1024, 1024, 64]);
    assert_eq!(full.max_storage_buffer_bytes(), 1u64 << 40);
    assert!(full.device_lost());
}

#[test]
fn overridden_lifecycle_hooks_receive_every_call() {
    let full = FullBackend::new();
    full.prepare().expect("Fix: the override must return Ok.");
    full.flush().expect("Fix: the override must return Ok.");
    full.shutdown().expect("Fix: the override must return Ok.");
    full.try_recover()
        .expect("Fix: an overriding backend must be able to accept recovery.");
    assert_eq!(full.prepare_calls.load(Ordering::Relaxed), 1);
    assert_eq!(full.flush_calls.load(Ordering::Relaxed), 1);
    assert_eq!(full.shutdown_calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        full.recover_calls.load(Ordering::Relaxed),
        1,
        "Fix: every lifecycle call must reach the override, not the default body."
    );
}

#[test]
fn blanket_backend_impl_exposes_the_backend_identity() {
    let minimal: &dyn Backend = &MinimalBackend;
    assert_eq!(minimal.id(), "minimal");
    assert_eq!(minimal.version(), "unspecified");
    assert!(
        !minimal.supported_ops().is_empty(),
        "Fix: the blanket Backend impl must report the core op set, not an empty one."
    );
}

/// Object safety is load-bearing: dispatch routes through `&dyn VyreBackend`.
/// A trait addition that breaks it stops this test compiling.
#[test]
fn the_driver_trait_surface_stays_object_safe() {
    let _minimal: Arc<dyn VyreBackend> = Arc::new(MinimalBackend);
    let _full: Arc<dyn VyreBackend> = FullBackend::new();
    let _backend: Option<Box<dyn Backend>> = None;
    let _pipeline: Option<Box<dyn CompiledPipeline>> = None;
}

#[test]
fn backends_are_send_and_sync_concretely_and_behind_dyn() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MinimalBackend>();
    assert_send_sync::<FullBackend>();
    assert_send_sync::<Arc<dyn VyreBackend>>();
}

/// A backend that overrides everything else still inherits the async adapter.
///
/// `async_dispatch_contract` owns what that adapter guarantees. What matters
/// here is that overriding fourteen other methods does not detach a backend from
/// the default, which is the shape every synchronous driver ships.
#[test]
fn a_fully_overriding_backend_still_inherits_the_async_default() {
    let full = FullBackend::new();
    let pending = full
        .dispatch_async(&Program::default(), &[], &DispatchConfig::default())
        .expect("Fix: the inherited dispatch_async must succeed when dispatch succeeds.");
    assert!(
        pending.is_ready(),
        "Fix: the inherited adapter must return a ready handle."
    );
    assert!(
        pending
            .await_result()
            .expect("Fix: the inherited adapter's result must be retrievable.")
            .is_empty(),
        "Fix: the inherited adapter must forward this backend's empty outputs verbatim."
    );
}
