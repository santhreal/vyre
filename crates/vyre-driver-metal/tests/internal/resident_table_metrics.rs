//! The resident-buffer metric rows, asserted against the emitter that writes them.
//!
//! `push_resident_table_metrics` is private to `runtime`, so only a module of
//! the library can name it. That is why the library includes this file directly
//! while every other module of `internal` reaches the backend through the public
//! trait and is compiled by the `all_tests` harness. No device is acquired here,
//! so the file carries no device admission; the Apple gate on its declaration is
//! what makes `runtime` exist at all.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::runtime::{push_resident_table_metrics, MetalResidentBufferTable};

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
