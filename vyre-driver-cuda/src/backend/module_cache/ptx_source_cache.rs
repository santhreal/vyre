//! In-memory cache of lowered PTX text, its retention policy, and its snapshot.

use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use vyre_driver::accounting::checked_add_usize_lazy;
use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use super::cache_key::{ptx_source_cache_key_from_program_identity, PtxSourceCacheKey};
use super::capacity_accounting::{
    increment_cache_access_u32, increment_cache_counter_u64, release_cached_source_bytes,
    reserve_cached_source_bytes, select_evicted_keys,
};
use super::ptx_disk_cache::{load_ptx_from_disk, store_ptx_to_disk};
use super::CudaPtxSourceCacheSnapshot;

const PTX_SOURCE_CACHE_SOFT_CAP: usize = 512;
const PTX_SOURCE_CACHE_RETAIN_AFTER_EVICTION: usize = PTX_SOURCE_CACHE_SOFT_CAP / 2;
const PTX_SOURCE_CACHE_SOFT_BYTES: usize = 256 * 1024 * 1024;

/// Cache of lowered PTX text. This sits in front of the CUDA module cache so
/// ordinary dispatches avoid re-running descriptor validation and PTX emission
/// before discovering that the module is already warm.
#[derive(Debug)]
pub(crate) struct CudaPtxSourceCache {
    sources: DashMap<PtxSourceCacheKey, CachedPtxSource, BuildHasherDefault<FxHasher>>,
    hits: AtomicU64,
    misses: AtomicU64,
    cached_source_bytes: AtomicUsize,
}

#[derive(Debug)]
struct CachedPtxSource {
    source: Arc<str>,
    source_bytes: usize,
    access_count: AtomicU32,
}

impl CudaPtxSourceCache {
    pub(crate) fn new() -> Self {
        Self {
            sources: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            cached_source_bytes: AtomicUsize::new(0),
        }
    }

    pub(crate) fn key_for_program(
        &self,
        program: &Program,
        config: &DispatchConfig,
        ptx_target_sm: u32,
        subgroup_size: u32,
        feature_flags: vyre_driver::PipelineFeatureFlags,
    ) -> Result<PtxSourceCacheKey, BackendError> {
        ptx_source_cache_key_from_program_identity(
            program,
            config,
            ptx_target_sm,
            subgroup_size,
            feature_flags,
        )
    }

    pub(crate) fn get_or_lower(
        &self,
        key: PtxSourceCacheKey,
        lower: impl FnOnce() -> Result<String, BackendError>,
    ) -> Result<Arc<str>, BackendError> {
        if let Some(source) = self.sources.get(&key) {
            increment_cache_access_u32(&source.access_count, "CUDA PTX source access count");
            increment_cache_counter_u64(&self.hits, "CUDA PTX source cache hits");
            return Ok(Arc::clone(&source.value().source));
        }
        // Disk persistence: PTX text is large (megabytes) but compresses
        // well; reading from disk is ~10 ms vs the multi-100 ms cost of
        // re-running the vyre IR -> PTX lowering on the same program
        // shape. Cross-process and across-runs: second run of the same
        // corpus loads every lowered PTX from disk, hitting the CUDA
        // driver's cuda-jit cache for PTX -> cuBIN compilation, and
        // skipping the vyre-side lowering entirely.
        if let Some(disk_source) = load_ptx_from_disk(&key)? {
            let arc: Arc<str> = disk_source.into();
            return self.insert_disk_cached_source(key, arc);
        }
        increment_cache_counter_u64(&self.misses, "CUDA PTX source cache misses");
        if self.sources.len() >= PTX_SOURCE_CACHE_SOFT_CAP {
            self.evict_submodular();
        }
        let source = match self.sources.entry(key) {
            Entry::Occupied(existing) => {
                increment_cache_access_u32(
                    &existing.get().access_count,
                    "CUDA PTX source access count",
                );
                Arc::clone(&existing.get().source)
            }
            Entry::Vacant(entry) => {
                let source: Arc<str> = lower()?.into();
                store_ptx_to_disk(&key, source.as_ref())?;
                let source_bytes = source.len();
                if source_bytes > PTX_SOURCE_CACHE_SOFT_BYTES {
                    return Ok(source);
                }
                reserve_cached_source_bytes(&self.cached_source_bytes, source_bytes)?;
                entry.insert(CachedPtxSource {
                    source: Arc::clone(&source),
                    source_bytes,
                    access_count: AtomicU32::new(1),
                });
                source
            }
        };
        if self.cached_source_bytes.load(Ordering::Acquire) > PTX_SOURCE_CACHE_SOFT_BYTES {
            self.evict_submodular();
        }
        Ok(source)
    }

    pub(crate) fn clear(&self) {
        self.sources.clear();
        self.hits.store(0, Ordering::Release);
        self.misses.store(0, Ordering::Release);
        self.cached_source_bytes.store(0, Ordering::Release);
    }

    pub(crate) fn snapshot(&self) -> CudaPtxSourceCacheSnapshot {
        CudaPtxSourceCacheSnapshot {
            entries: self.sources.len(),
            cached_source_bytes: self.cached_source_bytes.load(Ordering::Acquire),
            hits: self.hits.load(Ordering::Acquire),
            misses: self.misses.load(Ordering::Acquire),
        }
    }

    fn insert_disk_cached_source(
        &self,
        key: PtxSourceCacheKey,
        source: Arc<str>,
    ) -> Result<Arc<str>, BackendError> {
        let source_bytes = source.len();
        if source_bytes > PTX_SOURCE_CACHE_SOFT_BYTES {
            return Ok(source);
        }
        let cached_source_bytes = self.cached_source_bytes.load(Ordering::Acquire);
        if self.sources.len() >= PTX_SOURCE_CACHE_SOFT_CAP
            || cached_source_bytes > PTX_SOURCE_CACHE_SOFT_BYTES - source_bytes
        {
            self.evict_submodular();
        }
        match self.sources.entry(key) {
            Entry::Occupied(existing) => {
                increment_cache_access_u32(
                    &existing.get().access_count,
                    "CUDA PTX source access count",
                );
                increment_cache_counter_u64(&self.hits, "CUDA PTX source cache disk hits");
                Ok(Arc::clone(&existing.get().source))
            }
            Entry::Vacant(entry) => {
                reserve_cached_source_bytes(&self.cached_source_bytes, source_bytes)?;
                entry.insert(CachedPtxSource {
                    source: Arc::clone(&source),
                    source_bytes,
                    access_count: AtomicU32::new(1),
                });
                increment_cache_counter_u64(&self.hits, "CUDA PTX source cache disk hits");
                Ok(source)
            }
        }
    }

    fn evict_submodular(&self) {
        let mut keys = SmallVec::<[PtxSourceCacheKey; PTX_SOURCE_CACHE_SOFT_CAP]>::new();
        let mut gains = SmallVec::<[u32; PTX_SOURCE_CACHE_SOFT_CAP]>::new();
        for entry in self.sources.iter() {
            keys.push(*entry.key());
            gains.push(entry.access_count.load(Ordering::Relaxed));
        }
        let Some(to_remove) = select_evicted_keys::<_, PTX_SOURCE_CACHE_SOFT_CAP>(
            &keys,
            &mut gains,
            PTX_SOURCE_CACHE_RETAIN_AFTER_EVICTION,
            "CUDA PTX source cache",
            "PTX source cache eviction removal key",
        ) else {
            self.clear_and_report_total_eviction(keys.len());
            return;
        };

        let dropped = to_remove.len();
        let total = keys.len().max(1);
        let mut dropped_bytes = 0usize;
        for key in &to_remove {
            if let Some((_, removed)) = self.sources.remove(key) {
                let Ok(next) = checked_add_usize_lazy(dropped_bytes, removed.source_bytes, || ())
                else {
                    self.clear_and_report_total_eviction(keys.len());
                    return;
                };
                dropped_bytes = next;
            }
        }
        if dropped_bytes != 0
            && release_cached_source_bytes(&self.cached_source_bytes, dropped_bytes).is_err()
        {
            self.clear_and_report_total_eviction(keys.len());
            return;
        }
        vyre_driver::cache_eviction::record_eviction_counts(dropped, total);
    }

    /// Drop every cached source and report that the whole cache went. Reached
    /// when retention selection or byte accounting cannot be trusted to leave
    /// the cache and its byte counter agreeing.
    fn clear_and_report_total_eviction(&self, candidates: usize) {
        self.sources.clear();
        self.cached_source_bytes.store(0, Ordering::Release);
        vyre_driver::cache_eviction::record_eviction_counts(candidates, candidates);
    }
}

// Inline: covers `CudaPtxSourceCache`, `snapshot`, which no integration test can name.
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use vyre_foundation::ir::Program;

    use super::super::cache_key::PtxSourceCacheKey;
    use super::super::ptx_disk_cache::ptx_disk_cache_path;
    use super::CudaPtxSourceCache;

    #[test]
    fn ptx_source_cache_snapshot_tracks_hits_misses_and_clear() {
        let cache = CudaPtxSourceCache::new();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ptx_source_cache_snapshot_tracks_hits_misses_and_clear");
        hasher.update(&std::process::id().to_le_bytes());
        hasher.update(
            &SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("Fix: system clock must be after Unix epoch")
                .as_nanos()
                .to_le_bytes(),
        );
        let key = PtxSourceCacheKey(*hasher.finalize().as_bytes());
        let disk_path = ptx_disk_cache_path(&key)
            .expect("Fix: PTX source cache path should resolve on the test host.");
        match std::fs::remove_file(&disk_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic!(
                "failed to remove pre-existing PTX cache artifact `{}` before deterministic cache-counter test: {error}",
                disk_path.display()
            ),
        }

        let first = cache
            .get_or_lower(key, || Ok("cached-ptx-source".to_string()))
            .expect("Fix: first PTX source lowering should populate cache");
        let second = cache
            .get_or_lower(key, || panic!("cache hit must not relower PTX source"))
            .expect("Fix: second PTX source lookup should hit cache");

        assert!(Arc::ptr_eq(&first, &second));
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.entries, 1);
        assert_eq!(snapshot.cached_source_bytes, "cached-ptx-source".len());
        assert_eq!(snapshot.hits, 1);
        assert_eq!(snapshot.misses, 1);

        cache.clear();
        let snapshot = cache.snapshot();
        assert_eq!(snapshot.entries, 0);
        assert_eq!(snapshot.cached_source_bytes, 0);
        assert_eq!(snapshot.hits, 0);
        assert_eq!(snapshot.misses, 0);

        let _ = std::fs::remove_file(disk_path);
    }

    #[test]
    fn ptx_source_cache_keys_use_shared_domain_identity_for_generated_configs() {
        let cache = CudaPtxSourceCache::new();
        let program = Program::wrapped(vec![], [64, 1, 1], vec![]);

        for case in 0..2048u32 {
            let mut config = vyre_driver::DispatchConfig::default();
            if case & 1 != 0 {
                config.ulp_budget = Some((case as u8).wrapping_mul(11).wrapping_add(1));
            }
            if case & 2 != 0 {
                config.workgroup_override = Some([
                    1 + (case & 127),
                    1 + ((case.rotate_left(7) >> 3) & 31),
                    1 + ((case.rotate_right(5) >> 2) & 7),
                ]);
            }
            let flags = match case & 3 {
                0 => vyre_driver::PipelineFeatureFlags::empty(),
                1 => vyre_driver::PipelineFeatureFlags::SUBGROUP_OPS,
                2 => vyre_driver::PipelineFeatureFlags::F16
                    .union(vyre_driver::PipelineFeatureFlags::BF16),
                _ => vyre_driver::PipelineFeatureFlags::TENSOR_CORES
                    .union(vyre_driver::PipelineFeatureFlags::ASYNC_COMPUTE),
            };
            let ptx_target_sm = 70 + (case % 30);
            let subgroup_size = 1 + (case.rotate_left(3) % 64);
            let key = cache
                .key_for_program(&program, &config, ptx_target_sm, subgroup_size, flags)
                .expect("Fix: generated PTX source cache key must fit shared identity envelope");
            assert_eq!(
                key,
                cache
                    .key_for_program(&program, &config, ptx_target_sm, subgroup_size, flags)
                    .expect("Fix: repeated generated PTX source cache key must fit"),
                "Fix: CUDA PTX source cache identity must be stable for generated case {case}."
            );
            assert_ne!(
                key,
                cache
                    .key_for_program(&program, &config, ptx_target_sm + 1, subgroup_size, flags)
                    .expect("Fix: generated PTX target variation cache key must fit"),
                "Fix: CUDA PTX source cache identity must include target SM for generated case {case}."
            );
            assert_ne!(
                key,
                cache
                    .key_for_program(&program, &config, ptx_target_sm, subgroup_size + 1, flags)
                    .expect("Fix: generated subgroup variation cache key must fit"),
                "Fix: CUDA PTX source cache identity must include subgroup size for generated case {case}."
            );

            let changed_flags = flags.union(vyre_driver::PipelineFeatureFlags::PERSISTENT_THREAD);
            assert_ne!(
                key,
                cache
                    .key_for_program(
                        &program,
                        &config,
                        ptx_target_sm,
                        subgroup_size,
                        changed_flags,
                    )
                    .expect("Fix: generated feature-flag variation cache key must fit"),
                "Fix: CUDA PTX source cache identity must include feature flags for generated case {case}."
            );

            let mut changed_config = config.clone();
            changed_config.ulp_budget = Some(config.ulp_budget.unwrap_or(0).wrapping_add(1));
            assert_ne!(
                key,
                cache
                    .key_for_program(
                        &program,
                        &changed_config,
                        ptx_target_sm,
                        subgroup_size,
                        flags,
                    )
                    .expect("Fix: generated dispatch-policy variation cache key must fit"),
                "Fix: CUDA PTX source cache identity must include dispatch policy for generated case {case}."
            );
        }
    }

    /// Wire a Program to a PTX source cache key with everything except the
    /// Program held fixed.
    fn ptx_key_for(program: &Program) -> PtxSourceCacheKey {
        CudaPtxSourceCache::new()
            .key_for_program(
                program,
                &vyre_driver::DispatchConfig::default(),
                86,
                32,
                vyre_driver::PipelineFeatureFlags::empty(),
            )
            .expect("Fix: fixture Program must produce a PTX source cache key")
    }

    /// Two Programs whose buffers differ only by swapped binding slots must not
    /// share a PTX source cache key.
    ///
    /// The CUDA twin of the wgpu binding test. Note what this does NOT claim: the
    /// PTX key mixes a VSA fingerprint lane beside the normalized digest, so this
    /// property may hold through either lane. That is exactly why the assertion
    /// is written against the composed KEY rather than against the digest: it is
    /// the key that admits or serves a cached PTX artifact, so the observable
    /// contract must hold no matter which lane currently carries it, including
    /// after a future refactor drops one.
    #[test]
    fn ptx_source_cache_key_separates_swapped_buffer_bindings() {
        use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

        let entry = vec![
            Node::store("a", Expr::u32(0), Expr::u32(1)),
            Node::store("b", Expr::u32(0), Expr::u32(2)),
        ];
        let straight = Program::wrapped(
            vec![
                BufferDecl::output("a", 0, DataType::U32).with_count(64),
                BufferDecl::output("b", 1, DataType::U32).with_count(64),
            ],
            [64, 1, 1],
            entry.clone(),
        );
        let swapped = Program::wrapped(
            vec![
                BufferDecl::output("a", 1, DataType::U32).with_count(64),
                BufferDecl::output("b", 0, DataType::U32).with_count(64),
            ],
            [64, 1, 1],
            entry,
        );

        assert_ne!(
            ptx_key_for(&straight),
            ptx_key_for(&swapped),
            "Fix: binding slots reach generated PTX parameter ordering, so two binding \
             layouts must not share one PTX source cache entry."
        );
    }

    /// Two Programs differing only in a shared-memory array LENGTH must not
    /// share a PTX source cache key.
    ///
    /// PTX bakes a shared array's byte length into the `.shared` declaration, so
    /// serving one length's PTX for the other gives a kernel whose shared
    /// allocation is the wrong size.
    #[test]
    fn ptx_source_cache_key_separates_shared_memory_array_lengths() {
        use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

        let build = |shared_len: u32| {
            Program::wrapped(
                vec![
                    BufferDecl::output("out", 0, DataType::U32).with_count(64),
                    BufferDecl::workgroup("tile", shared_len, DataType::U32),
                ],
                [64, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            )
        };

        assert_ne!(
            ptx_key_for(&build(64)),
            ptx_key_for(&build(128)),
            "Fix: a shared-memory array length is baked into the PTX .shared declaration, \
             so two lengths must not share one PTX source cache entry."
        );
    }

    /// Resizing a runtime storage buffer MAY change the PTX source cache key,
    /// and this test documents WHY the CUDA side is asymmetric with wgpu.
    ///
    /// The normalized digest erases runtime storage lengths, which is safe for
    /// WGSL because the naga emitter turns every non-Shared buffer into
    /// `ArraySize::Dynamic`. The PTX emitter is NOT so disciplined:
    /// `emit_binding_len_or_max` bakes any memory class's count as an immediate
    /// when an async copy is present. The VSA fingerprint lane in the PTX key is
    /// what covers that gap. This test pins the lane's presence by asserting the
    /// key still discriminates a resize even though the digest does not, so
    /// removing the lane fails here instead of silently serving PTX with a
    /// wrong baked length.
    #[test]
    fn ptx_source_cache_key_keeps_a_lane_that_sees_runtime_storage_resize() {
        use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

        let build = |count: u32| {
            Program::wrapped(
                vec![BufferDecl::output("out", 0, DataType::U32).with_count(count)],
                [64, 1, 1],
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            )
        };
        let small = build(1024);
        let large = build(1_048_576);

        let small_digest = vyre_driver::try_normalized_program_cache_digest(&small)
            .expect("Fix: fixture Program must produce a normalized cache digest");
        let large_digest = vyre_driver::try_normalized_program_cache_digest(&large)
            .expect("Fix: fixture Program must produce a normalized cache digest");
        assert_eq!(
            small_digest, large_digest,
            "Fix: the normalized digest must erase runtime storage lengths; if this fails \
             the wgpu disk cache recompiles a shader on every input resize."
        );

        assert_ne!(
            ptx_key_for(&small),
            ptx_key_for(&large),
            "Fix: the PTX emitter can bake a storage buffer count as an immediate under an \
             async copy, so the PTX source cache key must keep a lane that sees a resize \
             even though the normalized digest deliberately does not."
        );
    }
}
