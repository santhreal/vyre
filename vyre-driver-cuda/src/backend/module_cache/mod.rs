//! CUDA module cache: PTX text to loaded `CUfunction` lookup.
//!
//! The cache is layered: [`cache_key`] derives domain-separated identity,
//! [`ptx_source_cache`] holds lowered PTX text in memory over
//! [`ptx_disk_cache`], and [`module_registry`] holds the loaded modules that
//! [`driver_module`] produces. [`capacity_accounting`] carries the byte
//! accounting and eviction sizing both caches share.

pub(crate) mod cache_key;
mod capacity_accounting;
pub(crate) mod driver_module;
mod module_registry;
mod ptx_disk_cache;
mod ptx_source_cache;
mod trap_sidecar;

pub(crate) use cache_key::{ModuleCacheKey, PtxSourceCacheKey};
pub(crate) use driver_module::{load_cuda_module_data, unload_cuda_module};
pub(crate) use module_registry::{CudaModuleCache, ModuleGlobals};
pub(crate) use ptx_source_cache::CudaPtxSourceCache;
pub(crate) use trap_sidecar::TrapSidecar;

/// Snapshot of the CUDA PTX source cache used before driver module loading.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaPtxSourceCacheSnapshot {
    /// Number of normalized PTX source entries retained in memory.
    pub entries: usize,
    /// Number of PTX source bytes retained in memory.
    pub cached_source_bytes: usize,
    /// Number of lookups served from an existing lowered PTX source.
    pub hits: u64,
    /// Number of lookups that had to lower PTX source before insertion.
    pub misses: u64,
}
