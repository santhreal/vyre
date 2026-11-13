//! Pipeline cache limits, audit reporting and environment parsing.

use super::*;
use crate::backend::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

#[test]
fn parse_positive_env_rejects_unset_zero_and_invalid() {
    let name = "VYRE_TEST_PARSE_POSITIVE_ENV_UNIQUE";
    std::env::remove_var(name);
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "0");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 7);
    std::env::set_var(name, "not-a-number");
    assert_eq!(super::parse_positive_env::<usize>(name, 9), 9);
    std::env::set_var(name, "42");
    assert_eq!(super::parse_positive_env::<u32>(name, 7), 42);
    std::env::remove_var(name);
}

mod cache_audit {
    use super::super::{PipelineCacheAudit, PipelineCacheAuditReport};

    #[test]
    fn empty_audit_reports_no_data_and_no_alarm() {
        let audit = PipelineCacheAudit::new();
        let report = audit.snapshot(8000);
        assert_eq!(
            report,
            PipelineCacheAuditReport {
                hits: 0,
                misses: 0,
                unknowns: 0,
                hit_rate_bps: None,
                below_alarm_threshold: false,
            }
        );
    }

    #[test]
    fn audit_computes_hit_rate_bps_correctly() {
        let mut audit = PipelineCacheAudit::new();
        audit.observe(Some(true));
        audit.observe(Some(true));
        audit.observe(Some(true));
        audit.observe(Some(false));
        let report = audit.snapshot(0);
        assert_eq!(report.hits, 3);
        assert_eq!(report.misses, 1);
        assert_eq!(report.hit_rate_bps, Some(7500));
    }

    #[test]
    fn audit_hit_rate_uses_widened_shared_ratio_for_saturated_counters() {
        let audit = PipelineCacheAudit {
            hits: u64::MAX,
            misses: 0,
            unknowns: 0,
        };
        let report = audit.snapshot(0);

        assert_eq!(report.hit_rate_bps, Some(10_000));
    }

    #[test]
    fn audit_excludes_unknowns_from_rate_denominator() {
        let mut audit = PipelineCacheAudit::new();
        audit.observe(Some(true));
        audit.observe(None);
        audit.observe(None);
        audit.observe(Some(false));
        let report = audit.snapshot(0);
        assert_eq!(report.hits, 1);
        assert_eq!(report.misses, 1);
        assert_eq!(report.unknowns, 2);
        // 1/2 = 50%  -  unknowns must NOT dilute the rate.
        assert_eq!(report.hit_rate_bps, Some(5000));
    }

    #[test]
    fn audit_alarms_when_hit_rate_below_threshold() {
        let mut audit = PipelineCacheAudit::new();
        for _ in 0..3 {
            audit.observe(Some(true));
        }
        for _ in 0..7 {
            audit.observe(Some(false));
        }
        let report = audit.snapshot(8000);
        assert_eq!(report.hit_rate_bps, Some(3000));
        assert!(report.below_alarm_threshold);
    }

    #[test]
    fn audit_does_not_alarm_at_exactly_threshold() {
        let mut audit = PipelineCacheAudit::new();
        for _ in 0..8 {
            audit.observe(Some(true));
        }
        for _ in 0..2 {
            audit.observe(Some(false));
        }
        let report = audit.snapshot(8000);
        assert_eq!(report.hit_rate_bps, Some(8000));
        assert!(!report.below_alarm_threshold);
    }

    #[test]
    fn audit_alarm_disabled_with_zero_threshold() {
        let mut audit = PipelineCacheAudit::new();
        for _ in 0..5 {
            audit.observe(Some(false));
        }
        let report = audit.snapshot(0);
        assert_eq!(report.hit_rate_bps, Some(0));
        assert!(!report.below_alarm_threshold);
    }

    #[test]
    fn audit_no_alarm_when_no_data_even_with_threshold() {
        let mut audit = PipelineCacheAudit::new();
        audit.observe(None);
        audit.observe(None);
        let report = audit.snapshot(8000);
        assert_eq!(report.hit_rate_bps, None);
        assert!(!report.below_alarm_threshold);
    }
}

mod cache_identity {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    fn store_program(output_name: &'static str, value: u32) -> Program {
        Program::wrapped(
            vec![BufferDecl::output(output_name, 0, DataType::U32).with_count(1)],
            [64, 1, 1],
            vec![Node::store(output_name, Expr::u32(0), Expr::u32(value))],
        )
    }

    #[test]
    fn normalized_program_digest_tracks_program_structure() {
        let a = store_program("out", 7);
        let b = store_program("out", 8);
        assert_ne!(
            normalized_program_cache_digest(&a),
            normalized_program_cache_digest(&b),
            "Fix: backend shader caches must miss when program semantics change."
        );
        assert_eq!(
            normalized_program_cache_digest(&a),
            normalized_program_cache_digest(&a),
            "Fix: normalized program cache digests must be deterministic."
        );
    }

    #[test]
    fn dispatch_policy_cache_hash_tracks_codegen_policy() {
        let base = DispatchConfig {
            ulp_budget: Some(1),
            workgroup_override: Some([64, 1, 1]),
            ..Default::default()
        };
        let mut changed = base.clone();
        changed.workgroup_override = Some([128, 1, 1]);

        let mut a = blake3::Hasher::new();
        update_dispatch_policy_cache_hash(&mut a, &base);
        let mut b = blake3::Hasher::new();
        update_dispatch_policy_cache_hash(&mut b, &changed);

        assert_ne!(a.finalize(), b.finalize());
        assert_eq!(
            dispatch_policy_cache_string(&base),
            "ulp=Some(1):wg=Some([64, 1, 1]):float=contracted"
        );
    }

    #[test]
    fn shared_disk_pipeline_cache_round_trips_and_shards() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DiskPipelineCache::open(dir.path()).unwrap();
        let fp = PipelineDeviceFingerprint::from_parts(1, 2, "driver-a", "runtime-b");
        let key = [7_u8; 32];
        let path = cache.path_for(key, fp);
        let cache_key = fp.cache_key(key);
        let cache_key_hex = hex_encode(&cache_key);
        assert_eq!(
            path.parent().and_then(std::path::Path::file_name),
            Some(std::ffi::OsStr::new(&cache_key_hex[..2])),
            "Fix: cryptographic device fingerprinting must happen before shard path derivation."
        );
        assert!(cache.read(key, fp).unwrap().is_none());
        cache.write(key, fp, b"compiled bytes").unwrap();
        assert_eq!(
            cache.read(key, fp).unwrap().as_deref(),
            Some(b"compiled bytes".as_slice())
        );
    }

    #[test]
    fn subgroup_reduction_offsets_derive_from_size() {
        assert_eq!(crate::subgroup::reduction_offsets(32), vec![16, 8, 4, 2, 1]);
        assert_eq!(crate::subgroup::reduction_offsets(8), vec![4, 2, 1]);
    }
}

mod compiled_pipeline_defaults {
    use super::*;
    use crate::backend::CompiledPipeline;
    use crate::Resource;

    /// A batched timed dispatch reports one device duration per item, in
    /// submission order, and an empty batch reports nothing.
    ///
    /// WHY: a resident batch summary is computed from these entries, so an
    /// entry that lands out of order, is dropped, or survives from a
    /// previous batch attributes one item's device time to another item and
    /// silently changes every figure derived from it. This default is what
    /// a backend without a batched submission path uses, so it holds the
    /// same contract as a backend override.
    ///
    /// Does not catch whether an override enqueues the whole batch before
    /// awaiting any of it, which is only observable on a device.
    #[test]
    fn compiled_pipeline_batched_timed_default_reports_one_device_time_per_item() {
        struct TimedBatchPipeline;

        impl crate::backend::sealed::Sealed for TimedBatchPipeline {}

        impl CompiledPipeline for TimedBatchPipeline {
            fn id(&self) -> &str {
                "batched-timed-default"
            }

            fn dispatch_borrowed_timed(
                &self,
                _: &[&[u8]],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                panic!("batched timed default must route through resident-handle dispatch")
            }

            fn dispatch_persistent_handles_timed(
                &self,
                inputs: &[Resource],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                let Some(Resource::Borrowed(bytes)) = inputs.first() else {
                    return Err(BackendError::new(
                        "batched timed test item requires one borrowed resource. Fix: bind one resource per batch item.",
                    ));
                };
                let device_ns = u64::from(bytes[0]) * 1000;
                Ok(crate::TimedDispatchResult::device_timed(
                    vec![bytes.clone()],
                    7,
                    Some(device_ns),
                ))
            }
        }

        let pipeline = TimedBatchPipeline;
        let items = [
            vec![Resource::Borrowed(vec![3_u8])],
            vec![Resource::Borrowed(vec![1_u8])],
            vec![Resource::Borrowed(vec![2_u8])],
        ];
        let batches: Vec<&[Resource]> = items.iter().map(Vec::as_slice).collect();
        let mut outputs = vec![vec![Vec::with_capacity(8)]];
        let first_slot_ptr = outputs[0][0].as_ptr();
        let mut device_ns_by_item = vec![Some(999_u64)];

        pipeline
            .dispatch_persistent_handles_batched_timed(
                &batches,
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .unwrap();

        assert_eq!(
            device_ns_by_item,
            vec![Some(3000), Some(1000), Some(2000)],
            "each item reports its own device time, in submission order"
        );
        assert_eq!(
            outputs,
            vec![vec![vec![3_u8]], vec![vec![1_u8]], vec![vec![2_u8]]]
        );
        assert_eq!(
            outputs[0][0].as_ptr(),
            first_slot_ptr,
            "an existing output slot is reused rather than reallocated"
        );

        pipeline
            .dispatch_persistent_handles_batched_timed(
                &[],
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .unwrap();

        assert!(device_ns_by_item.is_empty(), "an empty batch times nothing");
        assert!(outputs.is_empty(), "an empty batch produces no output set");
    }

    /// A failing item stops the batch with its error and leaves no timing
    /// row for an item that did not run.
    #[test]
    fn compiled_pipeline_batched_timed_default_propagates_item_error() {
        struct FailingSecondItem;

        impl crate::backend::sealed::Sealed for FailingSecondItem {}

        impl CompiledPipeline for FailingSecondItem {
            fn id(&self) -> &str {
                "batched-timed-failure"
            }

            fn dispatch_borrowed_timed(
                &self,
                _: &[&[u8]],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                panic!("batched timed default must route through resident-handle dispatch")
            }

            fn dispatch_persistent_handles_timed(
                &self,
                inputs: &[Resource],
                _: &DispatchConfig,
            ) -> Result<crate::TimedDispatchResult, BackendError> {
                match inputs.first() {
                    Some(Resource::Borrowed(bytes)) if bytes[0] == 0 => Err(BackendError::new(
                        "item zero rejected. Fix: submit nonzero items.",
                    )),
                    Some(Resource::Borrowed(bytes)) => Ok(
                        crate::TimedDispatchResult::device_timed(vec![bytes.clone()], 1, Some(5)),
                    ),
                    _ => panic!("test items are borrowed"),
                }
            }
        }

        let items = [
            vec![Resource::Borrowed(vec![4_u8])],
            vec![Resource::Borrowed(vec![0_u8])],
            vec![Resource::Borrowed(vec![6_u8])],
        ];
        let batches: Vec<&[Resource]> = items.iter().map(Vec::as_slice).collect();
        let mut outputs = Vec::new();
        let mut device_ns_by_item = Vec::new();

        let error = FailingSecondItem
            .dispatch_persistent_handles_batched_timed(
                &batches,
                &DispatchConfig::default(),
                &mut outputs,
                &mut device_ns_by_item,
            )
            .expect_err("the rejected item must fail the batch");

        assert!(error.to_string().contains("item zero rejected"), "{error}");
        assert_eq!(
            device_ns_by_item,
            vec![Some(5)],
            "only the item that ran is timed"
        );
    }
}

mod on_disk {
    /// G8: content-hash on-disk pipeline cache.
    ///
    /// Keyed by `blake3(program.to_wire() || driver_version || device_gen
    /// || CURRENT_PIPELINE_CACHE_KEY_VERSION || feature_flags)`. A hit
    /// lets a backend skip target compilation and load the bytes
    /// straight into a pipeline handle  -  single-digit ms cold start
    /// after the first run.
    ///
    /// This module owns the **pure** key derivation + blob I/O. The
    /// backend supplies its native blob bytes and calls [`store`] after a successful compile;
    /// subsequent runs call [`load`] before compiling. The key
    /// versioning means a `CURRENT_PIPELINE_CACHE_KEY_VERSION` bump
    /// invalidates every existing file on disk, the same way it
    use std::fmt::Write as _;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};

    use super::{PipelineFeatureFlags, CURRENT_PIPELINE_CACHE_KEY_VERSION};
    use blake3::Hasher;

    /// Cache-file extension. Binary blob.
    pub(super) const CACHE_EXTENSION: &str = "bin";

    /// Compute the 32-byte blake3 cache key for `program` on the
    /// named backend.
    ///
    /// `driver_version` is the backend's own build identifier;
    /// `device_gen` is a caller-chosen generation bucket for the
    /// target device family. Mixing them makes a pipeline compiled
    /// for one generation miss on another, even though the Program
    /// bytes match.
    #[must_use]
    pub(super) fn compute_cache_key(
        program_wire: &[u8],
        backend_id: &str,
        driver_version: &str,
        device_gen: &str,
        feature_flags: PipelineFeatureFlags,
    ) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(&CURRENT_PIPELINE_CACHE_KEY_VERSION.to_le_bytes());
        hasher.update(&(backend_id.len() as u32).to_le_bytes());
        hasher.update(backend_id.as_bytes());
        hasher.update(&(driver_version.len() as u32).to_le_bytes());
        hasher.update(driver_version.as_bytes());
        hasher.update(&(device_gen.len() as u32).to_le_bytes());
        hasher.update(device_gen.as_bytes());
        hasher.update(&feature_flags.0.to_le_bytes());
        hasher.update(&(program_wire.len() as u64).to_le_bytes());
        hasher.update(program_wire);
        let mut out = [0_u8; 32];
        out.copy_from_slice(hasher.finalize().as_bytes());
        out
    }

    /// Filename inside `cache_dir` for `key`  -  lowercase hex +
    /// `.bin` extension. Deterministic; no salt.
    #[must_use]
    pub(super) fn cache_path(cache_dir: &Path, key: &[u8; 32]) -> PathBuf {
        // Writes to a String never fail; ignore the Result per the
        // stdlib convention for `fmt::Write` on owned buffers.
        let mut name = String::with_capacity(64 + 1 + CACHE_EXTENSION.len());
        for b in key {
            let _ = write!(&mut name, "{b:02x}");
        }
        name.push('.');
        name.push_str(CACHE_EXTENSION);
        cache_dir.join(name)
    }

    /// Load a cached blob by key. Returns `Ok(None)` on a miss
    /// (file doesn't exist) and `Err` on I/O errors.
    pub(super) fn load(cache_dir: &Path, key: &[u8; 32]) -> Result<Option<Vec<u8>>, CacheError> {
        let path = cache_path(cache_dir, key);
        match fs::read(&path) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(CacheError::Io { path, source: e }),
        }
    }

    /// Write a cached blob for `key`. Creates `cache_dir` if
    /// missing. Writes via a temp file + atomic rename so a
    /// concurrent reader either sees the old blob or the new one,
    /// never a torn write.
    pub(super) fn store(cache_dir: &Path, key: &[u8; 32], bytes: &[u8]) -> Result<(), CacheError> {
        fs::create_dir_all(cache_dir).map_err(|e| CacheError::Io {
            path: cache_dir.to_path_buf(),
            source: e,
        })?;
        let final_path = cache_path(cache_dir, key);
        let tmp_path = final_path.with_extension("bin.tmp");
        fs::write(&tmp_path, bytes).map_err(|e| CacheError::Io {
            path: tmp_path.clone(),
            source: e,
        })?;
        fs::rename(&tmp_path, &final_path).map_err(|e| CacheError::Io {
            path: final_path,
            source: e,
        })
    }

    /// Cache I/O errors.
    #[derive(Debug, thiserror::Error)]
    pub(super) enum CacheError {
        /// Disk-side I/O failure while reading or writing a cache entry.
        #[error(
            "Fix: pipeline-cache I/O failed at {path:?}. \
             Ensure the cache directory is writable: {source}"
        )]
        Io {
            /// Cache directory or file the operation targeted.
            path: PathBuf,
            /// Underlying `std::io::Error` that triggered the failure.
            #[source]
            source: io::Error,
        },
    }

    // Inline: covers `cache_path`, `compute_cache_key`, `load`, `store`, which no integration
    // test can name.
    #[cfg(test)]
    mod tests {
        use super::*;

        fn key1() -> [u8; 32] {
            [1_u8; 32]
        }

        fn key2() -> [u8; 32] {
            [2_u8; 32]
        }

        #[test]
        fn compute_cache_key_is_deterministic() {
            let a = compute_cache_key(
                b"bytes",
                "backend-a",
                "v24",
                "arch-a",
                PipelineFeatureFlags::SUBGROUP_OPS,
            );
            let b = compute_cache_key(
                b"bytes",
                "backend-a",
                "v24",
                "arch-a",
                PipelineFeatureFlags::SUBGROUP_OPS,
            );
            assert_eq!(a, b);
        }

        #[test]
        fn compute_cache_key_changes_with_driver_version() {
            let a = compute_cache_key(
                b"x",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            let b = compute_cache_key(
                b"x",
                "backend-a",
                "v25",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            assert_ne!(a, b);
        }

        #[test]
        fn compute_cache_key_changes_with_device_gen() {
            let a = compute_cache_key(
                b"x",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            let b = compute_cache_key(
                b"x",
                "backend-a",
                "v24",
                "gen-b",
                PipelineFeatureFlags::empty(),
            );
            assert_ne!(a, b);
        }

        #[test]
        fn compute_cache_key_changes_with_feature_flags() {
            let a = compute_cache_key(
                b"x",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            let b = compute_cache_key(
                b"x",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::SUBGROUP_OPS,
            );
            assert_ne!(a, b);
        }

        #[test]
        fn compute_cache_key_changes_with_program_bytes() {
            let a = compute_cache_key(
                b"prog-a",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            let b = compute_cache_key(
                b"prog-b",
                "backend-a",
                "v24",
                "gen-a",
                PipelineFeatureFlags::empty(),
            );
            assert_ne!(a, b);
        }

        #[test]
        fn compute_cache_key_not_vulnerable_to_length_extension() {
            // A naive concatenation of two variable-length fields
            // without separating them would let `("ab", "cd")`
            // collide with `("abc", "d")`. Our format prefixes each
            // field with its length, so these must differ.
            let a = compute_cache_key(b"", "ab", "cd", "gen-a", PipelineFeatureFlags::empty());
            let b = compute_cache_key(b"", "abc", "d", "gen-a", PipelineFeatureFlags::empty());
            assert_ne!(a, b);
        }

        #[test]
        fn cache_path_is_hex_and_bin_extension() {
            let d = Path::new("/tmp");
            let p = cache_path(d, &[0xAB_u8; 32]);
            let fname = p.file_name().unwrap().to_string_lossy().to_string();
            assert!(fname.ends_with(".bin"));
            assert!(fname.contains("abababab"));
            assert_eq!(fname.len(), 64 + 4); // 64 hex + ".bin"
        }

        #[test]
        fn load_miss_returns_none() {
            let dir = tempfile::tempdir().unwrap();
            let r = load(dir.path(), &key1()).unwrap();
            assert!(r.is_none());
        }

        #[test]
        fn store_then_load_roundtrips() {
            let dir = tempfile::tempdir().unwrap();
            let payload = b"compiled-target-bytes".to_vec();
            store(dir.path(), &key1(), &payload).unwrap();
            let loaded = load(dir.path(), &key1()).unwrap();
            assert_eq!(loaded.as_deref(), Some(payload.as_slice()));
        }

        #[test]
        fn store_creates_missing_cache_dir() {
            let parent = tempfile::tempdir().unwrap();
            let nested = parent.path().join("a").join("b").join("c");
            assert!(!nested.exists());
            store(&nested, &key1(), b"blob").unwrap();
            let loaded = load(&nested, &key1()).unwrap();
            assert_eq!(loaded.as_deref(), Some(b"blob".as_slice()));
        }

        #[test]
        fn different_keys_do_not_overlap() {
            let dir = tempfile::tempdir().unwrap();
            store(dir.path(), &key1(), b"one").unwrap();
            store(dir.path(), &key2(), b"two").unwrap();
            assert_eq!(
                load(dir.path(), &key1()).unwrap().as_deref(),
                Some(b"one".as_slice())
            );
            assert_eq!(
                load(dir.path(), &key2()).unwrap().as_deref(),
                Some(b"two".as_slice())
            );
        }

        #[test]
        fn overwriting_same_key_preserves_atomicity() {
            let dir = tempfile::tempdir().unwrap();
            store(dir.path(), &key1(), b"first").unwrap();
            store(dir.path(), &key1(), b"second").unwrap();
            assert_eq!(
                load(dir.path(), &key1()).unwrap().as_deref(),
                Some(b"second".as_slice())
            );
        }
    }
}
