//! Backend metric snapshot contents, including the poisoned-lock sentinel.

#![cfg(feature = "device-tests")]

use crate::*;

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

/// When the `resident_buffers` Mutex is poisoned (a background thread panicked
/// while holding it), the resident-buffer metric rows must NOT silently vanish.
/// The pre-fix `if let Ok(table) = ...` arm discarded the `PoisonError`, leaving
/// two fewer entries in the snapshot and making "zero resident buffers"
/// indistinguishable from "poisoned backend".
///
/// This exercises `push_resident_table_metrics` directly against a genuinely
/// poisoned table. `MetalBackend::backend_metric_snapshot` needs a live
/// `MTLDevice` and offers no way to poison the lock it owns, so a test written
/// against the backend can only ever observe the healthy arm, which is what the
/// previous version of this test did while claiming to prove the poisoned one.
/// The emitting function is device-free, so the sentinel contract is provable
/// without a device.
#[test]
fn metric_snapshot_poisoned_mutex_is_loud() {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use crate::runtime::{push_resident_table_metrics, MetalResidentBufferTable};

    let table: MetalResidentBufferTable = Arc::new(Mutex::new(HashMap::new()));

    // Healthy arm: an empty table reports zero for both counters and emits no
    // error row. This is the value the poisoned arm must be distinguishable
    // from, so it is asserted here rather than assumed.
    let mut healthy = Vec::new();
    push_resident_table_metrics(&table, &mut healthy);
    assert_eq!(
        healthy,
        vec![
            ("metal_resident_buffer_count", 0),
            ("metal_resident_bytes", 0),
        ],
        "Fix: a healthy resident table must report zero counters and no error row"
    );

    // Poison the lock for real: a thread panics while holding the guard.
    let poisoner = Arc::clone(&table);
    std::thread::spawn(move || {
        let _guard = poisoner.lock().expect("Fix: a fresh Mutex must lock");
        panic!("deliberate panic to poison the resident buffer table");
    })
    .join()
    .expect_err("Fix: the poisoning thread must panic so the Mutex is poisoned");
    assert!(
        table.is_poisoned(),
        "Fix: the resident buffer table must be poisoned before the sentinel is asserted"
    );

    let mut poisoned = Vec::new();
    push_resident_table_metrics(&table, &mut poisoned);
    assert_eq!(
        poisoned,
        vec![
            ("metal_resident_buffer_count", u64::MAX),
            ("metal_resident_bytes", u64::MAX),
            ("metal_resident_buffer_error", 1),
        ],
        "Fix: a poisoned resident table must report the u64::MAX sentinel for both \
         counters and add `metal_resident_buffer_error`, never drop the rows"
    );
}
