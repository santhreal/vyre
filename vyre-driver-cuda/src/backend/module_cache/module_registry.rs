//! Sharded cache of loaded CUDA modules, its eviction policy, and PTX module
//! loading.

use std::cell::RefCell;
use std::ffi::CStr;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use cudarc::driver::sys::{CUfunction, CUmodule};
use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use vyre_driver::BackendError;

use super::cache_key::{
    module_cache_key_from_domain_digest, ModuleCacheKey, PtxSourceCacheKey,
    CUDA_MODULE_FROM_PTX_SOURCE_KEY_DOMAIN, CUDA_MODULE_FROM_RAW_PTX_ARTIFACT_DOMAIN,
};
use super::capacity_accounting::{
    increment_cache_access_u32, increment_cache_counter_u64, select_evicted_keys,
};
use super::driver_module::{
    get_cuda_module_function, get_cuda_module_global, load_cuda_module_data,
    unload_cuda_module_or_log, GRID_BARRIER_SYMBOL_CSTR, GRID_BARRIER_SYMBOL_NAME,
};
use super::ptx_disk_cache::write_ptx_dump;
use crate::backend::staging_reserve::reserve_vec;

const MODULE_CACHE_SOFT_CAP: usize = 2048;
const MODULE_CACHE_RETAIN_AFTER_EVICTION: usize = MODULE_CACHE_SOFT_CAP / 2;

thread_local! {
    static PTX_CSTR_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Loaded CUDA module and its `main` entry function.
#[derive(Debug)]
struct CachedModule {
    module: CUmodule,
    main: CUfunction,
    /// Device pointer + byte size of the module-scope cooperative grid-barrier
    /// counter (`_vyre_grid_barrier`). `Some` only when the PTX declares it
    /// (a grid-sync program); the host zeroes this counter before each
    /// cooperative launch. `None` for every non-grid-sync kernel.
    grid_barrier_global: Option<(u64, usize)>,
    access_count: AtomicU32,
}

// SAFETY: FFI to libcuda.so. Pointer args were validated by the matching alloc
// / store API; lifetimes are documented in the surrounding function.
// cuda_check (or matching CUresult guard) propagates non-success codes as
// BackendError.
unsafe impl Send for CachedModule {}
// SAFETY: FFI to libcuda.so. Pointer args were validated by the matching alloc
// / store API; lifetimes are documented in the surrounding function.
// cuda_check (or matching CUresult guard) propagates non-success codes as
// BackendError.
unsafe impl Sync for CachedModule {}

impl Drop for CachedModule {
    fn drop(&mut self) {
        unload_cuda_module_or_log(self.module, "CUDA module cache drop");
    }
}

/// Sharded CUDA module cache with lock-free hit counters.
#[derive(Debug)]
pub(crate) struct CudaModuleCache {
    modules: DashMap<ModuleCacheKey, CachedModule, BuildHasherDefault<FxHasher>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl CudaModuleCache {
    pub(crate) fn new() -> Self {
        Self {
            modules: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub(crate) fn key_for_ptx_source_key(
        &self,
        ptx_source_key: PtxSourceCacheKey,
        compute_capability: (u32, u32),
    ) -> Result<ModuleCacheKey, BackendError> {
        module_cache_key_from_domain_digest(
            CUDA_MODULE_FROM_PTX_SOURCE_KEY_DOMAIN,
            compute_capability,
            ptx_source_key.as_bytes(),
        )
    }

    pub(crate) fn key_for_raw_ptx_artifact(
        &self,
        raw_ptx_source: &str,
        compute_capability: (u32, u32),
    ) -> Result<ModuleCacheKey, BackendError> {
        let raw_artifact_digest = blake3::hash(raw_ptx_source.as_bytes());
        module_cache_key_from_domain_digest(
            CUDA_MODULE_FROM_RAW_PTX_ARTIFACT_DOMAIN,
            compute_capability,
            raw_artifact_digest.as_bytes(),
        )
    }

    pub(crate) fn function_for_ptx(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
        ptx_target_sm: u32,
    ) -> Result<CUfunction, BackendError> {
        if let Some(module) = self.modules.get(&key) {
            increment_cache_access_u32(&module.access_count, "CUDA module cache access count");
            increment_cache_counter_u64(&self.hits, "CUDA module cache hits");
            return Ok(module.main);
        }
        increment_cache_counter_u64(&self.misses, "CUDA module cache misses");

        if self.modules.len() >= MODULE_CACHE_SOFT_CAP {
            self.evict_submodular();
        }
        match self.modules.entry(key) {
            Entry::Occupied(existing) => {
                increment_cache_access_u32(
                    &existing.get().access_count,
                    "CUDA module cache access count",
                );
                increment_cache_counter_u64(&self.hits, "CUDA module cache hits");
                Ok(existing.get().main)
            }
            Entry::Vacant(entry) => {
                let loaded = load_module(ptx_src, ptx_target_sm)?;
                let main = loaded.main;
                entry.insert(loaded);
                Ok(main)
            }
        }
    }

    /// Device pointer + byte size of this kernel's cooperative grid-barrier
    /// counter, loading the module if it is not cached. Returns `None` for a
    /// kernel that declares no grid barrier (the caller must already know,
    /// from `contains_grid_sync`, whether one is required and treat `None` on
    /// a grid-sync program as a hard codegen error).
    pub(crate) fn grid_barrier_global_for_ptx(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
        ptx_target_sm: u32,
    ) -> Result<Option<(u64, usize)>, BackendError> {
        if let Some(module) = self.modules.get(&key) {
            increment_cache_access_u32(&module.access_count, "CUDA module cache access count");
            increment_cache_counter_u64(&self.hits, "CUDA module cache hits");
            return Ok(module.grid_barrier_global);
        }
        increment_cache_counter_u64(&self.misses, "CUDA module cache misses");

        if self.modules.len() >= MODULE_CACHE_SOFT_CAP {
            self.evict_submodular();
        }
        match self.modules.entry(key) {
            Entry::Occupied(existing) => {
                increment_cache_access_u32(
                    &existing.get().access_count,
                    "CUDA module cache access count",
                );
                increment_cache_counter_u64(&self.hits, "CUDA module cache hits");
                Ok(existing.get().grid_barrier_global)
            }
            Entry::Vacant(entry) => {
                let loaded = load_module(ptx_src, ptx_target_sm)?;
                let global = loaded.grid_barrier_global;
                entry.insert(loaded);
                Ok(global)
            }
        }
    }

    pub(crate) fn clear(&self) {
        self.modules.clear();
    }

    pub(crate) fn len(&self) -> usize {
        self.modules.len()
    }

    pub(crate) fn snapshot(&self) -> vyre_driver::PipelineCacheSnapshot {
        vyre_driver::PipelineCacheSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }

    fn evict_submodular(&self) {
        let mut keys = SmallVec::<[ModuleCacheKey; MODULE_CACHE_SOFT_CAP]>::new();
        let mut gains = SmallVec::<[u32; MODULE_CACHE_SOFT_CAP]>::new();
        for entry in self.modules.iter() {
            keys.push(*entry.key());
            gains.push(entry.access_count.load(Ordering::Relaxed));
        }
        let Some(to_remove) = select_evicted_keys::<_, MODULE_CACHE_SOFT_CAP>(
            &keys,
            &mut gains,
            MODULE_CACHE_RETAIN_AFTER_EVICTION,
            "CUDA module cache",
            "module cache eviction removal key",
        ) else {
            self.modules.clear();
            vyre_driver::cache_eviction::record_eviction_counts(keys.len(), keys.len());
            return;
        };

        let dropped = to_remove.len();
        let total = keys.len().max(1);
        for key in &to_remove {
            self.modules.remove(key);
        }
        vyre_driver::cache_eviction::record_eviction_counts(dropped, total);
    }
}

fn load_module(ptx_src: &str, ptx_target_sm: u32) -> Result<CachedModule, BackendError> {
    let mut module = std::ptr::null_mut();
    PTX_CSTR_SCRATCH.with(|scratch| {
        let mut ptx_c = scratch.borrow_mut();
        ptx_c.clear();
        let ptx_c_capacity = ptx_src
            .len()
            .checked_add(1)
            .ok_or_else(|| BackendError::new("CUDA module PTX C-string length overflowed usize. Fix: split generated PTX before module loading."))?;
        reserve_vec(
            &mut ptx_c,
            ptx_c_capacity,
            "cuda module PTX C-string scratch",
        )?;
        ptx_c.extend_from_slice(ptx_src.as_bytes());
        ptx_c.push(0);
        if let Some(dir) = std::env::var_os("VYRE_PTX_DUMP_ALL_DIR") {
            write_ptx_dump(dir, ptx_src, "VYRE_PTX_DUMP_ALL_DIR")?;
        }
        module = match load_cuda_module_data(ptx_c.as_slice()) {
            Ok(module) => module,
            Err(res) => {
                if let Some(dir) = std::env::var_os("VYRE_PTX_DUMP_DIR") {
                    let path = write_ptx_dump(dir, ptx_src, "VYRE_PTX_DUMP_DIR")?;
                    tracing::warn!("VYRE_PTX_DUMP: wrote failing PTX to {}", path.display());
                }
                return Err(BackendError::KernelCompileFailed {
                    backend: crate::CUDA_BACKEND_ID.to_string(),
                    compiler_message: format!(
                        "cuModuleLoadData failed with {res:?} for sm_{ptx_target_sm} and PTX length {} bytes. Fix: run the PTX smoke test for this Program and verify the live CUDA driver supports the emitted PTX ISA.",
                        ptx_src.len()
                    ),
                });
            }
        };
        Ok(())
    })?;
    let func_name =
        CStr::from_bytes_with_nul(b"main\0").map_err(|error| BackendError::KernelCompileFailed {
            backend: crate::CUDA_BACKEND_ID.to_string(),
            compiler_message: format!(
                "CUDA module function symbol literal was invalid: {error}. Fix: restore the static `main` CUDA entry symbol."
            ),
        })?;
    let func = match get_cuda_module_function(module, func_name) {
        Ok(func) => func,
        Err(res) => {
            unload_cuda_module_or_log(module, "CUDA module cleanup after function lookup failure");
            return Err(BackendError::KernelCompileFailed {
                backend: crate::CUDA_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "cuModuleGetFunction(main) failed with {res:?} for sm_{ptx_target_sm}. Fix: ensure CUDA PTX emission still declares `.visible .entry main`."
                ),
            });
        }
    };
    // Resolve the cooperative grid-barrier counter when this kernel declares
    // one. The emitter always names the symbol `_vyre_grid_barrier`, so the
    // PTX text is an exact, non-lossy signal of its presence (no swallowed
    // NOT_FOUND on the 99% of kernels that have no grid barrier).
    let grid_barrier_global = if ptx_src.contains(GRID_BARRIER_SYMBOL_NAME) {
        let symbol = CStr::from_bytes_with_nul(GRID_BARRIER_SYMBOL_CSTR).map_err(|error| {
            BackendError::KernelCompileFailed {
                backend: crate::CUDA_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "CUDA grid-barrier symbol literal was invalid: {error}. Fix: restore the `{GRID_BARRIER_SYMBOL_NAME}` module-scope counter declaration in the PTX emitter."
                ),
            }
        })?;
        match get_cuda_module_global(module, symbol) {
            Ok(global) => Some(global),
            Err(res) => {
                unload_cuda_module_or_log(
                    module,
                    "CUDA module cleanup after grid-barrier global lookup failure",
                );
                return Err(BackendError::KernelCompileFailed {
                    backend: crate::CUDA_BACKEND_ID.to_string(),
                    compiler_message: format!(
                        "cuModuleGetGlobal({GRID_BARRIER_SYMBOL_NAME}) failed with {res:?} for sm_{ptx_target_sm} even though the PTX text declares it. Fix: ensure the PTX emitter emits `.global .align 4 .u32 {GRID_BARRIER_SYMBOL_NAME}[1];` at module scope for grid-sync kernels."
                    ),
                });
            }
        }
    } else {
        None
    };
    Ok(CachedModule {
        module,
        main: func,
        grid_barrier_global,
        access_count: AtomicU32::new(1),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    #[test]
    fn generated_module_keys_reuse_source_digest_without_ptx_rehash_churn() {
        let cache = super::CudaModuleCache::new();
        let mut seen = HashSet::new();

        for case in 0..4096u32 {
            let mut source_digest = [0u8; 32];
            source_digest[..4].copy_from_slice(&case.to_le_bytes());
            source_digest[4..8].copy_from_slice(&case.rotate_left(13).to_le_bytes());
            source_digest[8..12].copy_from_slice(&(!case).to_le_bytes());
            source_digest[12..16].copy_from_slice(&case.wrapping_mul(0x9e37_79b9).to_le_bytes());
            let source_key = super::PtxSourceCacheKey(source_digest);

            let key = cache
                .key_for_ptx_source_key(source_key, (9, 0))
                .expect("Fix: generated source digest module key must fit");
            assert_eq!(
                key,
                cache
                    .key_for_ptx_source_key(source_key, (9, 0))
                    .expect("Fix: repeated generated source digest module key must fit"),
                "Fix: PTX source digest to CUDA module key derivation must be stable for generated case {case}."
            );
            assert_ne!(
                key,
                cache
                    .key_for_ptx_source_key(source_key, (9, 1))
                    .expect("Fix: device-scoped generated source digest module key must fit"),
                "Fix: CUDA module keys must remain device-capability scoped for generated case {case}."
            );
            assert!(
                seen.insert(key.0),
                "Fix: generated PTX source digest case {case} collided in module-cache key space."
            );
        }
    }

    #[test]
    fn module_cache_keys_use_shared_domain_separated_identity_for_generated_inputs() {
        let cache = super::CudaModuleCache::new();

        for case in 0..2048u32 {
            let mut source_digest = [0u8; 32];
            let mut state = case ^ 0xCADA_CAFE;
            for (index, byte) in source_digest.iter_mut().enumerate() {
                state = state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .rotate_left((index as u32) & 15);
                *byte = (state >> ((index & 3) * 8)) as u8;
            }
            let source_key = super::PtxSourceCacheKey(source_digest);
            let compute_capability = (7 + (case % 4), case.rotate_left(5) % 10);
            let key = cache
                .key_for_ptx_source_key(source_key, compute_capability)
                .expect("Fix: generated shared source-key module identity must fit");
            let repeated = cache
                .key_for_ptx_source_key(source_key, compute_capability)
                .expect("Fix: repeated generated shared source-key module identity must fit");
            assert_eq!(
                key, repeated,
                "Fix: shared CUDA module identity must be stable for generated source-key case {case}."
            );

            let mut changed_digest = source_digest;
            changed_digest[(case as usize) & 31] ^= 0x80 | (case as u8 & 0x7f);
            let changed_source_key = super::PtxSourceCacheKey(changed_digest);
            assert_ne!(
                key,
                cache
                    .key_for_ptx_source_key(changed_source_key, compute_capability)
                    .expect("Fix: changed generated source digest module identity must fit"),
                "Fix: one-byte PTX source digest mutations must change CUDA module keys for generated case {case}."
            );
            assert_ne!(
                key,
                cache
                    .key_for_ptx_source_key(
                        source_key,
                        (compute_capability.0, compute_capability.1.wrapping_add(1)),
                    )
                    .expect("Fix: changed generated device capability module identity must fit"),
                "Fix: CUDA module keys must include compute-capability scope for generated case {case}."
            );

            let raw_ptx = format!(
                "// generated raw ptx artifact {case}\n.version 8.0\n.target sm_{}{}\n.visible .entry main() {{ ret; }}\n",
                compute_capability.0, compute_capability.1
            );
            let raw_key = cache
                .key_for_raw_ptx_artifact(&raw_ptx, compute_capability)
                .expect("Fix: generated raw PTX artifact module identity must fit");
            let repeated_raw_key = cache
                .key_for_raw_ptx_artifact(&raw_ptx, compute_capability)
                .expect("Fix: repeated generated raw PTX artifact module identity must fit");
            assert_eq!(
                raw_key, repeated_raw_key,
                "Fix: raw PTX artifact module identity must be stable for generated case {case}."
            );
            assert_ne!(
                key, raw_key,
                "Fix: source-key and raw-artifact CUDA module cache domains must not alias for generated case {case}."
            );
        }
    }
}
