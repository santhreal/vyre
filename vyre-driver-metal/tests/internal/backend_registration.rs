//! Acquisition, registry submission, and reported device profile.

use crate::*;

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[test]
fn non_apple_acquire_fails_actionably() {
    let error = match acquire() {
        Ok(_) => panic!("non-Apple builds must not fabricate a Metal backend"),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        message.contains("Apple Metal.framework") && message.contains("Fix:"),
        "non-Apple Metal acquisition must be actionable: {message}"
    );
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[test]
fn non_apple_build_does_not_register_fake_backend() {
    assert!(
        vyre_driver::registered_backends()
            .expect("valid backend registry")
            .iter()
            .all(|registration| registration.id != METAL_BACKEND_ID),
        "non-Apple builds must not submit a fake `metal` backend registration"
    );
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn apple_acquire_registers_dispatch_backend() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice for native dispatch.",
    );
    assert_eq!(backend.id(), METAL_BACKEND_ID);
    assert!(
        vyre_driver::registered_backends()
            .expect("valid backend registry")
            .iter()
            .any(|registration| registration.id == METAL_BACKEND_ID),
        "Fix: Apple Metal builds must submit a real backend registration."
    );
    assert!(
        vyre_driver::backend_dispatches(METAL_BACKEND_ID).expect("valid backend registry"),
        "Fix: Apple Metal registration must declare live dispatch capability."
    );
    assert_eq!(
        vyre_driver::backend_precedence(METAL_BACKEND_ID)
            .expect("valid backend registry"),
        25,
        "Fix: Metal precedence must stay above portable fallbacks only after live native dispatch exists."
    );
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn apple_device_profile_reports_live_metal_limits() {
    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before profile testing.",
    );
    let profile = backend.device_profile();

    assert_eq!(profile.backend, METAL_BACKEND_ID);
    assert!(profile.supports_subgroup_ops);
    assert_eq!(profile.subgroup_size, backend.subgroup_size().unwrap_or(0));
    assert_eq!(profile.max_workgroup_size, backend.max_workgroup_size());
    assert_eq!(
        profile.max_invocations_per_workgroup,
        backend.max_compute_invocations_per_workgroup()
    );
    assert!(
        profile.max_workgroup_size[0] > 0 && profile.max_invocations_per_workgroup > 0,
        "Fix: Metal profile must report nonzero live workgroup limits."
    );
    assert!(
        profile.max_shared_memory_bytes > 0 && profile.has_shared_memory,
        "Fix: Metal profile must expose threadgroup memory as a typed shared-memory capability."
    );
    assert_eq!(
        profile.max_storage_buffer_binding_size,
        backend.max_storage_buffer_bytes()
    );
    assert!(
        profile.max_storage_buffer_binding_size > 0,
        "Fix: Metal profile must expose the native maxBufferLength storage limit."
    );
    assert!(
        !profile.supports_specialization_constants,
        "Fix: Metal must not advertise function constants until lowering/runtime actually use them."
    );
    assert!(
        !profile.supports_indirect_dispatch,
        "Fix: Metal must not advertise indirect dispatch until the backend executes indirect dispatch nodes."
    );
    assert_eq!(
        profile.timing_quality,
        vyre_driver::DeviceTimingQuality::HostEnqueueWait,
        "Fix: Metal profile must classify timing as host enqueue/wait until device timestamps are implemented."
    );
    assert!(
        !profile.supports_device_timestamps && !profile.supports_hardware_counters,
        "Fix: Metal must not advertise timestamp or counter support until the runtime exposes real measurements."
    );
    assert!(profile.validation_capabilities().supports_subgroup_ops);
    assert_eq!(
        profile.adapter_caps().max_shared_memory_bytes,
        profile.max_shared_memory_bytes
    );
}
