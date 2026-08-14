use std::borrow::Cow;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

/// Delete the on-disk cache entries `impact_mask` marks as reached.
///
/// `impact_mask` is one entry per `cache_keys` slot, as produced by
/// [`crate::pipeline::cache_impact::RuleImpactQuery::impact_mask`]. Computing it
/// is not this function's decision: it used to take the whole rule graph and run
/// the reachability walk itself, which meant the disk cache and the in-memory
/// cache each decided what "impacted" meant.
pub(crate) fn remove_impacted_entries(
    impact_mask: &[u32],
    cache_keys: &[String],
) -> std::io::Result<()> {
    let dir = disk_pipeline_cache_dir();
    for (i, &is_impacted) in impact_mask.iter().enumerate() {
        if is_impacted != 0 {
            if let Some(key) = cache_keys.get(i) {
                let wgsl_path = cache_entry_path(&dir, key, ".wgsl");
                let meta_wgsl = cache_entry_path(&dir, key, ".wgsl.toml");
                let bin_path = cache_entry_path(&dir, key, ".pipeline.bin");
                let meta_bin = cache_entry_path(&dir, key, ".pipeline.toml");

                remove_cache_entry_file(&wgsl_path)?;
                remove_cache_entry_file(&meta_wgsl)?;
                remove_cache_entry_file(&bin_path)?;
                remove_cache_entry_file(&meta_bin)?;
            }
        }
    }
    Ok(())
}

fn remove_cache_entry_file(path: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(std::io::Error::new(
            error.kind(),
            format!(
                "Fix: failed to remove invalidated WGPU pipeline cache file `{}`: {error}",
                path.display()
            ),
        )),
    }
}

pub(crate) fn cache_entry_path(dir: &Path, key: &str, suffix: &str) -> PathBuf {
    let file_name_len = key.len().saturating_add(suffix.len());
    let mut file_name = String::with_capacity(file_name_len);
    file_name.push_str(key);
    file_name.push_str(suffix);
    dir.join(file_name)
}

#[cfg(test)]
static TEST_DISK_PIPELINE_CACHE_ROOT: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

#[cfg(test)]
pub(crate) fn set_test_disk_pipeline_cache_root(path: Option<PathBuf>) -> Option<PathBuf> {
    let mut guard = TEST_DISK_PIPELINE_CACHE_ROOT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::replace(&mut *guard, path)
}

pub(crate) fn disk_pipeline_cache_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(root) = TEST_DISK_PIPELINE_CACHE_ROOT
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
    {
        return root.join("pipeline");
    }

    std::env::var_os("VYRE_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("vyre")
        })
        .join("pipeline")
}

#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct DiskPipelineMetadata {
    pub version: u32,
    pub cache_key: [u8; 32],
    pub wgsl_bytes: usize,
    pub adapter_fingerprint: [u8; 32],
    pub program_abi_version: u32,
    pub naga_version: Cow<'static, str>,
    pub wgsl_lowering_contract: Cow<'static, str>,
    pub policy: String,
    pub wgsl_blake3: String,
}

#[derive(serde::Deserialize, serde::Serialize)]
pub(crate) struct CompiledPipelineMetadata {
    pub version: u32,
    pub cache_key: [u8; 32],
    pub adapter_fingerprint: [u8; 32],
    pub wgsl_blake3: String,
    pub program_abi_version: u32,
    pub naga_version: Cow<'static, str>,
    pub blob_bytes: usize,
    pub blob_blake3: String,
}
