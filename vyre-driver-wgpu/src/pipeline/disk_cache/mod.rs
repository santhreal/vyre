//! Disk-backed WGSL + compiled-pipeline cache for compiled pipeline mode.

mod io;
pub(crate) mod keys;
#[cfg(test)]
mod tests;

use std::path::Path;

use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;
use vyre_foundation::serial::wire::framing::WIRE_FORMAT_VERSION;

pub(crate) use self::io::flush_disk_pipeline_cache;
use self::io::{
    persist_bytes, read_bounded_bytes, read_bounded_utf8, read_metadata,
    MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES, MAX_WGSL_CACHE_BYTES,
};
use self::keys::{
    adapter_fingerprint, blake3_hex, hex_hash, metadata_fingerprint, NAGA_VERSION,
    WGSL_LOWERING_CONTRACT,
};
pub(crate) use self::keys::{
    compiled_pipeline_cache_key, early_pipeline_cache_key, wgsl_cache_key, CompiledPipelineCacheKey,
};
#[cfg(test)]
pub(crate) use super::disk_cache_entries::set_test_disk_pipeline_cache_root;
pub(crate) use super::disk_cache_entries::{
    cache_entry_path, disk_pipeline_cache_dir, remove_impacted_entries, CompiledPipelineMetadata,
    DiskPipelineMetadata,
};

const DISK_PIPELINE_CACHE_VERSION: u32 = 5;

pub(crate) struct PipelineCacheHandle {
    pub(crate) cache: wgpu::PipelineCache,
}

pub(crate) fn load_or_compile_disk_wgsl(
    program: &Program,
    adapter_info: &wgpu::AdapterInfo,
    config: &DispatchConfig,
    enabled_features: &crate::runtime::device::EnabledFeatures,
) -> Result<String, BackendError> {
    let fingerprint = adapter_fingerprint(adapter_info);

    let norm_digest =
        vyre_driver::try_normalized_program_cache_digest(program).map_err(|error| {
            BackendError::new(format!("WGSL disk pipeline cache digest failed: {error}"))
        })?;
    let cache_key = wgsl_cache_key(&norm_digest, &fingerprint, config);
    let cache_key_hex = hex_hash(&cache_key);
    let dir = disk_pipeline_cache_dir();
    let wgsl_path = cache_entry_path(&dir, &cache_key_hex, ".wgsl");
    let meta_path = cache_entry_path(&dir, &cache_key_hex, ".wgsl.toml");
    if let Ok(wgsl) = read_bounded_utf8(&wgsl_path, MAX_WGSL_CACHE_BYTES) {
        if wgsl_metadata_matches(&meta_path, &cache_key, &wgsl, &fingerprint, config) {
            return Ok(wgsl);
        }
    }
    let start = std::time::Instant::now();
    let wgsl = lower_wgsl(program, config, enabled_features)?;
    let elapsed = start.elapsed();
    tracing::info!(
        program_fingerprint = %cache_key_hex,
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        "WGSL cache miss  -  cold cache or program shape changed"
    );
    persist_disk_wgsl(
        &dir,
        &wgsl_path,
        &meta_path,
        &cache_key,
        &wgsl,
        &fingerprint,
        config,
    )?;
    Ok(wgsl)
}

pub(crate) fn create_compiled_pipeline_cache(
    device: &wgpu::Device,
    key: &CompiledPipelineCacheKey,
) -> Result<PipelineCacheHandle, BackendError> {
    let data = load_compiled_pipeline_blob(key)?;
    let cache = {
        #[allow(unsafe_code)]
        // SAFETY: FFI to wgpu / wgpu-hal native APIs. Handles + sizes are
        // validated by the surrounding cache layer; fallback=false makes a
        // broken advertised pipeline-cache feature fail loudly instead of
        // silently substituting an uncached driver path.
        unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("vyre persistent compiled pipeline cache"),
                data: data.as_deref(),
                fallback: false,
            })
        }
    };
    Ok(PipelineCacheHandle { cache })
}

pub(crate) fn persist_compiled_pipeline_cache(
    key: &CompiledPipelineCacheKey,
    cache: &wgpu::PipelineCache,
) -> Result<(), BackendError> {
    let Some(bytes) = cache.get_data() else {
        return Ok(());
    };
    let dir = disk_pipeline_cache_dir();
    let blob_path = cache_entry_path(&dir, &key.cache_key, ".pipeline.bin");
    let meta_path = cache_entry_path(&dir, &key.cache_key, ".pipeline.toml");
    let metadata = CompiledPipelineMetadata {
        version: DISK_PIPELINE_CACHE_VERSION,
        cache_key: key.hash,
        adapter_fingerprint: metadata_fingerprint(&key.adapter_fingerprint),
        wgsl_blake3: key.wgsl_blake3.clone(),
        program_abi_version: u32::from(WIRE_FORMAT_VERSION),
        naga_version: std::borrow::Cow::Borrowed(NAGA_VERSION),
        blob_bytes: bytes.len(),
        blob_blake3: blake3_hex(&bytes),
    };
    persist_bytes(&dir, &blob_path, &meta_path, &bytes, &metadata)
}

fn persist_disk_wgsl(
    dir: &Path,
    wgsl_path: &Path,
    meta_path: &Path,
    cache_key: &[u8; 32],
    wgsl: &str,
    fingerprint: &str,
    config: &DispatchConfig,
) -> Result<(), BackendError> {
    let metadata = DiskPipelineMetadata {
        version: DISK_PIPELINE_CACHE_VERSION,
        cache_key: *cache_key,
        wgsl_bytes: wgsl.len(),
        adapter_fingerprint: metadata_fingerprint(fingerprint),
        program_abi_version: u32::from(WIRE_FORMAT_VERSION),
        naga_version: std::borrow::Cow::Borrowed(NAGA_VERSION),
        wgsl_lowering_contract: std::borrow::Cow::Borrowed(WGSL_LOWERING_CONTRACT),
        policy: vyre_driver::dispatch_policy_cache_string(config),
        wgsl_blake3: blake3_hex(wgsl.as_bytes()),
    };
    persist_bytes(dir, wgsl_path, meta_path, wgsl.as_bytes(), &metadata)
}

fn wgsl_metadata_matches(
    meta_path: &Path,
    cache_key: &[u8; 32],
    wgsl: &str,
    fingerprint: &str,
    config: &DispatchConfig,
) -> bool {
    let Ok(metadata) = read_metadata::<DiskPipelineMetadata>(meta_path) else {
        return false;
    };
    metadata.version == DISK_PIPELINE_CACHE_VERSION
        && metadata.cache_key == *cache_key
        && metadata.wgsl_bytes == wgsl.len()
        && metadata.adapter_fingerprint == metadata_fingerprint(fingerprint)
        && metadata.program_abi_version == u32::from(WIRE_FORMAT_VERSION)
        && metadata.naga_version == NAGA_VERSION
        && metadata.wgsl_lowering_contract == WGSL_LOWERING_CONTRACT
        && metadata.policy == vyre_driver::dispatch_policy_cache_string(config)
        && metadata.wgsl_blake3 == blake3_hex(wgsl.as_bytes())
}

fn load_compiled_pipeline_blob(
    key: &CompiledPipelineCacheKey,
) -> Result<Option<Vec<u8>>, BackendError> {
    let dir = disk_pipeline_cache_dir();
    let blob_path = cache_entry_path(&dir, &key.cache_key, ".pipeline.bin");
    let meta_path = cache_entry_path(&dir, &key.cache_key, ".pipeline.toml");
    let Ok(metadata) = read_metadata::<CompiledPipelineMetadata>(&meta_path) else {
        tracing::warn!(
            cache_key = %key.cache_key,
            "compiled-pipeline cache miss  -  metadata missing or unreadable"
        );
        return Ok(None);
    };
    if metadata.version != DISK_PIPELINE_CACHE_VERSION
        || metadata.cache_key != key.hash
        || metadata.adapter_fingerprint != metadata_fingerprint(&key.adapter_fingerprint)
        || metadata.wgsl_blake3 != key.wgsl_blake3
        || metadata.program_abi_version != u32::from(WIRE_FORMAT_VERSION)
        || metadata.naga_version != NAGA_VERSION
    {
        tracing::warn!(
            cache_key = %key.cache_key,
            "compiled-pipeline cache miss  -  metadata does not match current adapter or compiler contract"
        );
        return Ok(None);
    }
    let metadata_blob_bytes = u64::try_from(metadata.blob_bytes).map_err(|source| {
        BackendError::new(format!(
            "compiled pipeline blob metadata length cannot fit u64: {source}. Fix: delete the corrupt cache entry."
        ))
    })?;
    if metadata_blob_bytes > MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES {
        tracing::warn!(
            cache_key = %key.cache_key,
            blob_bytes = metadata.blob_bytes,
            max_bytes = MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES,
            "compiled-pipeline cache miss  -  blob exceeds bounded cache read budget"
        );
        return Ok(None);
    }
    let bytes = read_bounded_bytes(&blob_path, MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES).map_err(
        |error| {
            BackendError::new(format!(
                "compiled pipeline cache blob `{}` could not be read: {error}. Fix: delete the corrupt cache entry or repair filesystem permissions.",
                blob_path.display()
            ))
        },
    )?;
    if bytes.len() != metadata.blob_bytes || blake3_hex(&bytes) != metadata.blob_blake3 {
        tracing::warn!(
            cache_key = %key.cache_key,
            "compiled-pipeline cache miss  -  blob length or digest mismatch"
        );
        return Ok(None);
    }
    Ok(Some(bytes))
}

fn lower_wgsl(
    program: &Program,
    config: &DispatchConfig,
    enabled_features: &crate::runtime::device::EnabledFeatures,
) -> Result<String, BackendError> {
    crate::emit::lower_with_features(program, config, enabled_features)
        .map_err(|error| BackendError::new(error.to_string()))
}
