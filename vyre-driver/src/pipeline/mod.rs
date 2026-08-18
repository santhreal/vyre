//! Concrete executable cache identity, persistence, and telemetry.

/// Shared on-disk compiled-pipeline cache.
pub(crate) mod cache;
/// Stable cache hashing and device fingerprint helpers.
pub(crate) mod hashing;

pub use cache::{
    DiskPipelineCache, PipelineCacheIdentity, PipelineCacheKey, PipelineCacheMissEvidence,
    PipelineCacheMissReason, PipelineFeatureFlags,
};
pub use hashing::{
    dispatch_policy_cache_digest, dispatch_policy_cache_string, hex_encode, hex_short,
    normalized_program_cache_digest, push_lower_hex, try_normalized_program_cache_digest,
    update_dispatch_policy_cache_hash, PipelineDeviceFingerprint,
};

/// Version mixed into every persistent pipeline cache key.
pub const CURRENT_PIPELINE_CACHE_KEY_VERSION: u32 = 1;
/// Default maximum number of compiled pipeline artifacts retained in memory.
pub const DEFAULT_PIPELINE_CACHE_ENTRIES: usize = 256;
/// Default maximum bytes retained by a backend pipeline cache.
pub const DEFAULT_PIPELINE_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Baseline one-dimensional workgroup used when a caller supplies no override.
pub const DEFAULT_1D_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// Backend-reported compiled-pipeline cache counters.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PipelineCacheSnapshot {
    /// Cache lookups that found an already-compiled artifact.
    pub hits: u64,
    /// Cache lookups that required compile/load work.
    pub misses: u64,
}

/// Pipeline reuse cache hit-rate audit.
///
/// Aggregates backend cache lookup outcomes into a report that records hit,
/// miss, and unknown counts. Unknown outcomes are excluded from the hit-rate
/// denominator because backends without real counters report `None`.
#[derive(Debug, Default, Clone)]
pub struct PipelineCacheAudit {
    hits: u64,
    misses: u64,
    unknowns: u64,
}

/// Snapshot of a [`PipelineCacheAudit`].
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineCacheAuditReport {
    /// Lookups that found an already-compiled artifact.
    pub hits: u64,
    /// Lookups that performed compile/load work.
    pub misses: u64,
    /// Lookups whose backend did not report cache state.
    pub unknowns: u64,
    /// Hit rate in basis points (0..=10_000) over the
    /// `hits + misses` denominator (excluding unknowns). `None` when
    /// `hits + misses == 0` so the caller can distinguish "no data"
    /// from "0% hit rate".
    pub hit_rate_bps: Option<u32>,
    /// Whether the hit rate is below the operator-supplied alarm
    /// threshold. Always `false` when `hit_rate_bps` is `None`.
    pub below_alarm_threshold: bool,
}

impl PipelineCacheAudit {
    /// Empty audit accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one outcome from the dispatcher.
    pub fn observe(&mut self, cache_hit: Option<bool>) {
        match cache_hit {
            Some(true) => self.hits = self.hits.saturating_add(1),
            Some(false) => self.misses = self.misses.saturating_add(1),
            None => self.unknowns = self.unknowns.saturating_add(1),
        }
    }

    /// Snapshot the audit, scoring it against `alarm_threshold_bps`.
    /// `alarm_threshold_bps = 8000` flags any audit with under 80% hit
    /// rate; pass `0` to disable the alarm.
    #[must_use]
    pub fn snapshot(&self, alarm_threshold_bps: u32) -> PipelineCacheAuditReport {
        let denominator = self.hits.saturating_add(self.misses);
        let hit_rate_bps = if denominator == 0 {
            None
        } else {
            Some(crate::numeric::ratio_basis_points_u64(
                self.hits,
                denominator,
                0,
                "pipeline cache hit rate",
                "driver",
            ))
        };
        let below_alarm_threshold = match hit_rate_bps {
            Some(rate) if alarm_threshold_bps > 0 => rate < alarm_threshold_bps,
            _ => false,
        };
        PipelineCacheAuditReport {
            hits: self.hits,
            misses: self.misses,
            unknowns: self.unknowns,
            hit_rate_bps,
            below_alarm_threshold,
        }
    }
}

/// Resolve pipeline cache limits from Tier-A operational environment settings.
#[must_use]
pub fn pipeline_cache_limits_from_env() -> (u32, usize) {
    let entries = parse_positive_env(
        "VYRE_PIPELINE_CACHE_ENTRIES",
        DEFAULT_PIPELINE_CACHE_ENTRIES as u32,
    );
    let bytes = parse_positive_env("VYRE_PIPELINE_CACHE_BYTES", DEFAULT_PIPELINE_CACHE_BYTES);
    (entries, bytes)
}

/// Parse a positive Tier-A env integer. Returns `default` when the variable is
/// unset; a present-but-invalid value (unparsable, non-positive, non-unicode)
/// is a misconfiguration surfaced loudly via `tracing::warn!` before falling
/// back so it is never silently discarded.
fn parse_positive_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + Default + std::fmt::Display + Copy,
{
    let raw = match std::env::var(name) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return default,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "ignoring non-unicode {name}: expected a positive integer; using default {default}"
            );
            return default;
        }
    };
    match raw.parse::<T>() {
        Ok(value) if value > T::default() => value,
        _ => {
            tracing::warn!(
                "ignoring invalid {name}={raw:?}: expected a positive integer; using default {default}"
            );
            default
        }
    }
}

// Inline: `pipeline` is a `pub(crate)` module and the suite calls the private
// `parse_positive_env`, neither reachable from an integration test.
#[cfg(test)]
mod tests {
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
                "ulp=Some(1):wg=Some([64, 1, 1])"
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
        use crate::{OutputBuffers, Resource};

        #[test]
        fn compiled_pipeline_borrowed_batch_default_preserves_order() {
            #[derive(Default)]
            struct BatchDefaultPipeline {
                calls: std::sync::Mutex<Vec<Vec<u8>>>,
            }

            impl crate::backend::sealed::Sealed for BatchDefaultPipeline {}

            impl CompiledPipeline for BatchDefaultPipeline {
                fn id(&self) -> &str {
                    "batch-default"
                }

                fn dispatch(
                    &self,
                    _: &[Vec<u8>],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    Err(BackendError::new(
                        "batch default test should use dispatch_borrowed. Fix: keep borrowed batch default zero-copy until each single dispatch.",
                    ))
                }

                fn dispatch_borrowed(
                    &self,
                    inputs: &[&[u8]],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    let first = inputs.first().copied().unwrap_or_default().to_vec();
                    self.calls.lock().unwrap().push(first.clone());
                    Ok(vec![first])
                }
            }

            let pipeline = BatchDefaultPipeline::default();
            let a = [1_u8, 2];
            let b = [3_u8, 4];
            let batch_a: [&[u8]; 1] = [a.as_slice()];
            let batch_b: [&[u8]; 1] = [b.as_slice()];
            let batches: [&[&[u8]]; 2] = [&batch_a, &batch_b];

            let outputs = pipeline
                .dispatch_borrowed_batched(&batches, &DispatchConfig::default())
                .unwrap();

            assert_eq!(outputs, vec![vec![a.to_vec()], vec![b.to_vec()]]);
            assert_eq!(
                *pipeline.calls.lock().unwrap(),
                vec![a.to_vec(), b.to_vec()]
            );
        }

        #[test]
        fn compiled_pipeline_default_into_records_dispatch_telemetry() {
            struct TelemetryPipeline;

            impl crate::backend::sealed::Sealed for TelemetryPipeline {}

            impl CompiledPipeline for TelemetryPipeline {
                fn id(&self) -> &str {
                    "compiled-telemetry"
                }

                fn dispatch_borrowed(
                    &self,
                    inputs: &[&[u8]],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    Ok(inputs.iter().map(|row| row.to_vec()).collect())
                }
            }

            let before = crate::observability::snapshot_dispatch_telemetry();
            let pipeline = TelemetryPipeline;
            let input = [1_u8, 2, 3];
            let mut outputs = vec![Vec::with_capacity(8)];

            pipeline
                .dispatch_borrowed_into(
                    &[input.as_slice()],
                    &DispatchConfig::default(),
                    &mut outputs,
                )
                .expect("default compiled-pipeline dispatch into must succeed");

            let after = crate::observability::snapshot_dispatch_telemetry();
            assert!(after.launches > before.launches);
            assert!(after.input_bytes >= before.input_bytes + 3);
            assert!(after.output_bytes >= before.output_bytes + 3);
            assert!(after.output_slots > before.output_slots);
            assert!(after.output_slots_reused > before.output_slots_reused);
        }

        #[test]
        fn compiled_pipeline_borrowed_batch_into_reuses_output_slots() {
            #[derive(Default)]
            struct BatchDefaultPipeline {
                calls: std::sync::Mutex<Vec<Vec<u8>>>,
            }

            impl crate::backend::sealed::Sealed for BatchDefaultPipeline {}

            impl CompiledPipeline for BatchDefaultPipeline {
                fn id(&self) -> &str {
                    "batch-default-into"
                }

                fn dispatch_borrowed(
                    &self,
                    inputs: &[&[u8]],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    let first = inputs.first().copied().unwrap_or_default().to_vec();
                    self.calls.lock().unwrap().push(first.clone());
                    Ok(vec![first])
                }
            }

            let pipeline = BatchDefaultPipeline::default();
            let a = [1_u8, 2];
            let b = [3_u8, 4];
            let batch_a: [&[u8]; 1] = [a.as_slice()];
            let batch_b: [&[u8]; 1] = [b.as_slice()];
            let batches: [&[&[u8]]; 2] = [&batch_a, &batch_b];
            let mut outputs = vec![
                vec![Vec::with_capacity(8)],
                vec![Vec::with_capacity(8)],
                vec![Vec::with_capacity(8)],
            ];
            let outer_ptr = outputs.as_ptr();
            let first_inner_ptr = outputs[0].as_ptr();
            let second_inner_ptr = outputs[1].as_ptr();
            let first_slot_ptr = outputs[0][0].as_ptr();
            let second_slot_ptr = outputs[1][0].as_ptr();

            pipeline
                .dispatch_borrowed_batched_into(&batches, &DispatchConfig::default(), &mut outputs)
                .unwrap();

            assert_eq!(outputs, vec![vec![a.to_vec()], vec![b.to_vec()]]);
            assert_eq!(outputs.as_ptr(), outer_ptr);
            assert_eq!(outputs[0].as_ptr(), first_inner_ptr);
            assert_eq!(outputs[1].as_ptr(), second_inner_ptr);
            assert_eq!(outputs[0][0].as_ptr(), first_slot_ptr);
            assert_eq!(outputs[1][0].as_ptr(), second_slot_ptr);
        }

        #[test]
        fn compiled_pipeline_persistent_handle_into_default_reuses_output_slots() {
            #[derive(Default)]
            struct PersistentDefaultPipeline {
                calls: std::sync::Mutex<Vec<Vec<u8>>>,
            }

            impl crate::backend::sealed::Sealed for PersistentDefaultPipeline {}

            impl CompiledPipeline for PersistentDefaultPipeline {
                fn id(&self) -> &str {
                    "persistent-default-into"
                }

                fn dispatch_borrowed(
                    &self,
                    _: &[&[u8]],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    Err(BackendError::new(
                        "persistent into default test should use resident-handle dispatch. Fix: keep persistent batch default on the resident API.",
                    ))
                }

                fn dispatch_persistent_handles(
                    &self,
                    inputs: &[Resource],
                    _: &DispatchConfig,
                ) -> Result<OutputBuffers, BackendError> {
                    let bytes = match inputs.first() {
                        Some(Resource::Borrowed(bytes)) => bytes.clone(),
                        Some(Resource::Resident(handle)) => handle.id().to_le_bytes().to_vec(),
                        None => Vec::new(),
                    };
                    self.calls.lock().unwrap().push(bytes.clone());
                    Ok(vec![bytes])
                }
            }

            let pipeline = PersistentDefaultPipeline::default();
            let mut outputs = vec![Vec::with_capacity(8)];
            let outer_ptr = outputs.as_ptr();
            let first_slot_ptr = outputs[0].as_ptr();

            pipeline
                .dispatch_persistent_handles_into(
                    &[Resource::Borrowed(vec![9_u8, 8, 7])],
                    &DispatchConfig::default(),
                    &mut outputs,
                )
                .unwrap();

            assert_eq!(outputs, vec![vec![9_u8, 8, 7]]);
            assert_eq!(outputs.as_ptr(), outer_ptr);
            assert_eq!(outputs[0].as_ptr(), first_slot_ptr);
            assert_eq!(*pipeline.calls.lock().unwrap(), vec![vec![9_u8, 8, 7]]);
        }

        #[test]
        fn compiled_pipeline_persistent_defaults_fail_explicitly_without_host_fallback() {
            struct UnsupportedPersistentPipeline;

            impl crate::backend::sealed::Sealed for UnsupportedPersistentPipeline {}

            impl CompiledPipeline for UnsupportedPersistentPipeline {
                fn id(&self) -> &str {
                    "unsupported-persistent"
                }

                fn dispatch_borrowed(
                    &self,
                    _: &[&[u8]],
                    _: &DispatchConfig,
                ) -> Result<Vec<Vec<u8>>, BackendError> {
                    panic!("persistent defaults must not route through host-buffer dispatch")
                }
            }

            fn assert_unsupported(error: BackendError, expected_name: &str) {
                let message = error.to_string();
                match error {
                    BackendError::UnsupportedFeature { name, backend } => {
                        assert_eq!(name, expected_name);
                        assert_eq!(backend, "unspecified");
                    }
                    other => panic!("expected explicit UnsupportedFeature, got {other:?}"),
                }
                assert!(
                    message.contains("Fix:"),
                    "unsupported persistent path must remain actionable: {message}"
                );
            }

            let pipeline = UnsupportedPersistentPipeline;
            let config = DispatchConfig::default();

            assert_unsupported(
                pipeline
                    .dispatch_persistent_handles(&[], &config)
                    .expect_err("unsupported persistent handle dispatch must fail"),
                "persistent handle dispatch",
            );
            assert_unsupported(
                pipeline
                    .dispatch_persistent_handles_timed(&[], &config)
                    .expect_err("timed persistent dispatch must preserve the unsupported error"),
                "persistent handle dispatch",
            );
            assert_unsupported(
                pipeline
                    .dispatch_persistent_resource_outputs(&[], &config)
                    .expect_err("unsupported resident output dispatch must fail"),
                "persistent resident output dispatch",
            );

            let mut outputs = vec![vec![0xA5]];
            assert_unsupported(
                pipeline
                    .dispatch_persistent_handles_into(&[], &config, &mut outputs)
                    .expect_err("persistent into dispatch must preserve the unsupported error"),
                "persistent handle dispatch",
            );
            assert_eq!(
                outputs,
                vec![vec![0xA5]],
                "failing persistent dispatch must not mutate caller output storage"
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
        pub(super) fn load(
            cache_dir: &Path,
            key: &[u8; 32],
        ) -> Result<Option<Vec<u8>>, CacheError> {
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
        pub(super) fn store(
            cache_dir: &Path,
            key: &[u8; 32],
            bytes: &[u8],
        ) -> Result<(), CacheError> {
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
}
