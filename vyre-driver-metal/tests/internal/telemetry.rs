//! Backend metric snapshot contents.
//!
//! The poisoned-lock sentinel is asserted in `resident_table_metrics`, which the
//! library compiles as one of its own modules because the emitting function is
//! private to `runtime`.

use super::*;

use std::collections::BTreeMap;

use super::fixtures::stores_word;
use vyre_driver::DispatchConfig;

#[test]
fn apple_backend_metric_snapshot_exposes_cache_and_resident_counters() {
    let program = stores_word(233);

    let backend = acquire().expect(
        "Fix: Apple Metal builds must acquire the system default MTLDevice before backend metric testing.",
    );
    let resident = backend.allocate_resident(16).expect(
        "Fix: native Metal must allocate a resident buffer before metric snapshot testing.",
    );
    backend
        .upload_resident(&resident, &11u32.to_le_bytes())
        .expect("Fix: native Metal must upload resident bytes before metric snapshot testing.");
    let mut resident_readback = Vec::new();
    backend
        .download_resident_range_into(&resident, 0, 4, &mut resident_readback)
        .expect("Fix: native Metal must download resident bytes before metric snapshot testing.");
    assert_eq!(resident_readback, 11u32.to_le_bytes().to_vec());
    backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: first Metal metric-snapshot dispatch must compile and execute.");
    backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: second Metal metric-snapshot dispatch must hit the pipeline cache.");
    let mut changed_policy = DispatchConfig::default();
    changed_policy.workgroup_override = Some([2, 1, 1]);
    backend
        .dispatch(&program, &[], &changed_policy)
        .expect("Fix: Metal metric-snapshot policy probe must compile a distinct policy key.");
    let changed_program = stores_word(234);
    backend
        .dispatch(&changed_program, &[], &DispatchConfig::default())
        .expect("Fix: Metal metric-snapshot program probe must compile a distinct Program key.");

    let metrics = backend
        .backend_metric_snapshot()
        .into_iter()
        .collect::<BTreeMap<_, _>>();

    assert!(
        metrics.get("metal_pipeline_cache_hits").copied().unwrap_or(0) >= 1,
        "Fix: Metal backend metric snapshot must expose real pipeline cache hits for benchmark JSON."
    );
    assert!(
        metrics
            .get("metal_pipeline_cache_misses")
            .copied()
            .unwrap_or(0)
            >= 1,
        "Fix: Metal backend metric snapshot must expose real pipeline cache misses for benchmark JSON."
    );
    assert_eq!(
        metrics
            .get("metal_pipeline_cache_miss_empty_cache")
            .copied(),
        Some(1),
        "Fix: Metal metric snapshot must explain the first cold miss as an empty-cache miss."
    );
    assert_eq!(
        metrics
            .get("metal_pipeline_cache_miss_dispatch_policy_changed")
            .copied(),
        Some(1),
        "Fix: Metal metric snapshot must explain same-program policy changes as dispatch-policy cache misses."
    );
    assert_eq!(
        metrics
            .get("metal_pipeline_cache_miss_program_changed")
            .copied(),
        Some(1),
        "Fix: Metal metric snapshot must explain different Program digests as program-change cache misses."
    );
    assert_eq!(
        metrics
            .get("metal_pipeline_cache_miss_device_or_runtime_changed")
            .copied(),
        Some(0),
        "Fix: Metal metric snapshot must expose the device/runtime miss bucket even when a single backend instance cannot trigger it."
    );
    assert_eq!(
        metrics.get("metal_pipeline_cache_miss_key_absent").copied(),
        Some(0),
        "Fix: Metal metric snapshot must expose the fallback key-absent miss bucket for future key fields."
    );
    assert!(
        metrics
            .get("metal_buffer_allocation_count")
            .copied()
            .unwrap_or(0)
            >= 1,
        "Fix: Metal metric snapshot must expose buffer allocation count."
    );
    assert!(
        metrics
            .get("metal_buffer_allocation_bytes")
            .copied()
            .unwrap_or(0)
            >= 16,
        "Fix: Metal metric snapshot must expose buffer allocation bytes."
    );
    assert!(
        metrics
            .get("metal_host_to_device_copy_count")
            .copied()
            .unwrap_or(0)
            >= 1,
        "Fix: Metal metric snapshot must expose host-to-device copy count."
    );
    assert!(
        metrics
            .get("metal_host_to_device_bytes")
            .copied()
            .unwrap_or(0)
            >= 4,
        "Fix: Metal metric snapshot must expose host-to-device bytes."
    );
    assert!(
        metrics
            .get("metal_device_to_host_copy_count")
            .copied()
            .unwrap_or(0)
            >= 1,
        "Fix: Metal metric snapshot must expose device-to-host copy count."
    );
    assert!(
        metrics
            .get("metal_device_to_host_bytes")
            .copied()
            .unwrap_or(0)
            >= 4,
        "Fix: Metal metric snapshot must expose device-to-host bytes."
    );
    assert!(
        metrics
            .get("metal_output_readback_bytes")
            .copied()
            .unwrap_or(0)
            >= 4,
        "Fix: Metal metric snapshot must expose dispatch output readback bytes separately from resident downloads."
    );
    assert_eq!(
        metrics.get("metal_resident_buffer_count").copied(),
        Some(1),
        "Fix: Metal backend metric snapshot must expose live resident buffer count."
    );
    assert_eq!(
        metrics.get("metal_resident_bytes").copied(),
        Some(16),
        "Fix: Metal backend metric snapshot must expose logical resident bytes."
    );
    assert!(
        !metrics.contains_key("metal_resident_buffer_error"),
        "Fix: a healthy backend must not report the resident-table poison sentinel."
    );

    backend
        .free_resident(resident)
        .expect("Fix: native Metal must free metric-snapshot resident handles.");
}
