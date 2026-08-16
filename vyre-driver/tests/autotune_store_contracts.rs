//! Contracts for `vyre_driver::autotune_store`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::autotune_store::{AutotuneKey, AutotuneRecord, AutotuneStore};
use vyre_driver::specialization::SpecCacheKey;

use tempfile::TempDir;

fn sample_spec(spec_hash: u64) -> SpecCacheKey {
    SpecCacheKey {
        shader_hash: 0xdeadbeef,
        binding_sig: 0xfacefeed,
        workgroup_size: [128, 1, 1],
        spec_hash,
    }
}

fn sample_record(unroll: u32) -> AutotuneRecord {
    AutotuneRecord {
        workgroup_size: [128, 1, 1],
        unroll,
        tile: [16, 16, 1],
        recorded_at: "2026-05-02".to_string(),
    }
}

#[test]
fn empty_store_returns_none_for_lookup() {
    let store = AutotuneStore::default();
    let key = AutotuneKey::new(&sample_spec(1), "adapter-x");
    assert!(store.get(&key).is_none());
    assert!(store.is_empty());
    assert!(!store.is_dirty());
}

#[test]
fn put_then_get_round_trips_record() {
    let mut store = AutotuneStore::default();
    let key = AutotuneKey::new(&sample_spec(1), "adapter-x");
    store.put(key.clone(), sample_record(4));
    assert!(store.is_dirty());
    assert_eq!(store.get(&key), Some(&sample_record(4)));
    assert_eq!(store.len(), 1);
}

#[test]
fn distinct_specs_or_adapters_get_distinct_records() {
    let mut store = AutotuneStore::default();
    let key_a = AutotuneKey::new(&sample_spec(1), "adapter-x");
    let key_b = AutotuneKey::new(&sample_spec(2), "adapter-x");
    let key_c = AutotuneKey::new(&sample_spec(1), "adapter-y");
    store.put(key_a.clone(), sample_record(4));
    store.put(key_b.clone(), sample_record(8));
    store.put(key_c.clone(), sample_record(16));
    assert_eq!(store.len(), 3);
    assert_eq!(store.get(&key_a).unwrap().unroll, 4);
    assert_eq!(store.get(&key_b).unwrap().unroll, 8);
    assert_eq!(store.get(&key_c).unwrap().unroll, 16);
}

#[test]
fn save_then_load_round_trips_through_toml() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("autotune.toml");
    let mut store = AutotuneStore::default();
    let key = AutotuneKey::new(&sample_spec(7), "adapter-x");
    store.put(key.clone(), sample_record(4));
    let wrote = store.save_if_dirty(&path).unwrap();
    assert!(wrote);
    assert!(!store.is_dirty(), "save should clear the dirty flag");

    let loaded = AutotuneStore::load(&path).unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded.get(&key), Some(&sample_record(4)));
}

#[test]
fn save_takes_exclusive_lock_so_concurrent_writes_serialize() {
    // R7: two threads writing to the same autotune file must not
    // interleave. With the exclusive file lock the second writer
    // waits until the first releases, and the file is the latter
    // writer's content (not a torn mix of both).
    use std::sync::Arc;
    use std::thread;
    let dir = TempDir::new().unwrap();
    let path = Arc::new(dir.path().join("autotune.toml"));

    let path_a = Arc::clone(&path);
    let path_b = Arc::clone(&path);
    let h_a = thread::spawn(move || {
        let mut store = AutotuneStore::default();
        let key = AutotuneKey::new(&sample_spec(101), "adapter-a");
        store.put(key, sample_record(11));
        store.save_if_dirty(&path_a).unwrap();
    });
    let h_b = thread::spawn(move || {
        let mut store = AutotuneStore::default();
        let key = AutotuneKey::new(&sample_spec(202), "adapter-b");
        store.put(key, sample_record(22));
        store.save_if_dirty(&path_b).unwrap();
    });
    h_a.join().unwrap();
    h_b.join().unwrap();

    // The file must be parseable (not torn) regardless of which
    // writer won. Without the lock this race produced corrupt TOML
    // ~30% of the time on a warm 5090 box.
    let loaded = AutotuneStore::load(&path).expect("Fix: file must be valid TOML");
    assert_eq!(loaded.len(), 1, "exactly one writer's record must persist");
}

#[test]
fn save_if_dirty_no_op_when_clean() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("autotune.toml");
    let mut store = AutotuneStore::default();
    let wrote = store.save_if_dirty(&path).unwrap();
    assert!(!wrote);
    assert!(!path.exists(), "no write must not create the file");
}

#[test]
fn load_missing_file_returns_empty_store() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("does_not_exist.toml");
    let store = AutotuneStore::load(&path).unwrap();
    assert!(store.is_empty());
}

#[test]
fn forget_removes_record_and_marks_dirty() {
    let mut store = AutotuneStore::default();
    let key = AutotuneKey::new(&sample_spec(1), "adapter-x");
    store.put(key.clone(), sample_record(4));
    let dir_path = TempDir::new().unwrap();
    let path = dir_path.path().join("a.toml");
    store.save_if_dirty(&path).unwrap();
    assert!(!store.is_dirty());

    let removed = store.forget(&key);
    assert!(removed);
    assert!(store.is_dirty());
    assert!(store.is_empty());

    let removed_again = store.forget(&key);
    assert!(!removed_again);
}

#[test]
fn key_distinguishes_different_workgroup_sizes() {
    let mut a = sample_spec(1);
    let mut b = sample_spec(1);
    a.workgroup_size = [128, 1, 1];
    b.workgroup_size = [256, 1, 1];
    let ka = AutotuneKey::new(&a, "x");
    let kb = AutotuneKey::new(&b, "x");
    assert_ne!(ka, kb);
}
