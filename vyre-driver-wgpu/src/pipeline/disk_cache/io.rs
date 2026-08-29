use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use fs2::FileExt;
use vyre_driver::BackendError;

use super::keys::path_fingerprint;
use crate::staging_reserve::reserve_backend_vec;

pub(super) const MAX_WGSL_CACHE_BYTES: u64 = 64 * 1024 * 1024;
pub(super) const MAX_COMPILED_PIPELINE_CACHE_BLOB_BYTES: u64 = 64 * 1024 * 1024;
pub(super) const MAX_PIPELINE_CACHE_METADATA_BYTES: u64 = 64 * 1024;
const MAX_PENDING_DURABLE_CACHE_FILES: usize = 4096;

static PENDING_DURABLE_CACHE_FILES: LazyLock<Mutex<BTreeSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(BTreeSet::new()));

pub(crate) fn flush_disk_pipeline_cache() -> Result<(), BackendError> {
    let paths = {
        let mut guard = PENDING_DURABLE_CACHE_FILES
            .lock()
            .map_err(BackendError::poisoned_lock)?;
        let mut paths = Vec::new();
        reserve_backend_vec(
            &mut paths,
            guard.len(),
            "pipeline cache pending flush path staging",
        )?;
        paths.extend(std::mem::take(&mut *guard));
        paths
    };
    if paths.is_empty() {
        return Ok(());
    }

    if let Err(error) = flush_disk_cache_paths(&paths) {
        let mut guard = PENDING_DURABLE_CACHE_FILES
            .lock()
            .map_err(BackendError::poisoned_lock)?;
        guard.extend(paths);
        return Err(error);
    }
    Ok(())
}

fn flush_disk_cache_paths(paths: &[PathBuf]) -> Result<(), BackendError> {
    sync_cache_files_bounded(
        paths,
        vyre_driver::durable_fanout::open_for_sync,
        File::sync_data,
        "pipeline cache explicit flush",
    )?;
    let parents = vyre_driver::durable_fanout::parent_directories(paths, |parents, capacity| {
        reserve_backend_vec(parents, capacity, "pipeline cache parent directory staging")
    })?;
    sync_parent_dirs_bounded(&parents)
}

pub(super) fn persist_bytes<T: serde::Serialize>(
    dir: &Path,
    data_path: &Path,
    meta_path: &Path,
    bytes: &[u8],
    metadata: &T,
) -> Result<(), BackendError> {
    fs::create_dir_all(dir).map_err(|error| {
        trace_io_err(dir, &error, "pipeline cache directory is unwritable");
        BackendError::new(format!("failed to create pipeline cache dir: {error}"))
    })?;
    write_atomic(data_path, bytes, "pipeline cache data")?;
    let encoded = toml::to_string(metadata).map_err(|error| {
        BackendError::new(format!("failed to encode pipeline cache metadata: {error}"))
    })?;
    write_atomic(meta_path, encoded.as_bytes(), "pipeline cache metadata")
}

pub(super) fn write_atomic(path: &Path, bytes: &[u8], label: &str) -> Result<(), BackendError> {
    static TMP_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let tmp_id = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp_path = path.with_extension(format!("tmp.{}_{}", std::process::id(), tmp_id));
    let mut file = File::create(&tmp_path)
        .map_err(|error| BackendError::new(format!("failed to create {label}: {error}")))?;
    FileExt::lock_exclusive(&file).map_err(|error| BackendError::new(error.to_string()))?;
    file.write_all(bytes)
        .map_err(|error| BackendError::new(error.to_string()))?;
    FileExt::unlock(&file).map_err(|error| BackendError::new(error.to_string()))?;
    fs::rename(&tmp_path, path)
        .map_err(|error| BackendError::new(format!("failed to install {label}: {error}")))?;
    register_pending_durable_cache_file(path)?;
    Ok(())
}

fn register_pending_durable_cache_file(path: &Path) -> Result<(), BackendError> {
    let should_flush = {
        let mut guard = PENDING_DURABLE_CACHE_FILES
            .lock()
            .map_err(BackendError::poisoned_lock)?;
        guard.insert(path.to_path_buf());
        guard.len() >= MAX_PENDING_DURABLE_CACHE_FILES
    };
    if should_flush {
        flush_disk_pipeline_cache()?;
    }
    Ok(())
}

/// A directory carries no write access to request, so it is opened read-only:
/// [`vyre_driver::durable_fanout::open_for_sync`] is for the file half.
#[cfg(unix)]
fn sync_parent_dirs_bounded(parents: &[PathBuf]) -> Result<(), BackendError> {
    sync_cache_files_bounded(
        parents,
        File::open,
        File::sync_all,
        "pipeline cache directory flush",
    )
}

#[cfg(not(unix))]
fn sync_parent_dirs_bounded(_parents: &[PathBuf]) -> Result<(), BackendError> {
    Ok(())
}

fn sync_cache_files_bounded(
    paths: &[PathBuf],
    open: fn(&Path) -> std::io::Result<File>,
    sync: fn(&File) -> std::io::Result<()>,
    context: &'static str,
) -> Result<(), BackendError> {
    vyre_driver::durable_fanout::for_each_bounded(
        paths,
        |path| {
            let file = open(path).map_err(|error| {
                trace_io_err(path, &error, "pipeline cache flush open failed");
                BackendError::new(format!(
                    "{context} failed to open {}: {error}. Fix: remove the corrupted cache entry and retry.",
                    path_fingerprint(path)
                ))
            })?;
            sync(&file).map_err(|error| {
                trace_io_err(path, &error, "pipeline cache flush fsync failed");
                BackendError::new(format!(
                    "{context} failed for {}: {error}. Fix: check cache storage health and retry.",
                    path_fingerprint(path)
                ))
            })
        },
        || BackendError::new(format!("{context} worker panicked. Fix: retry the flush.")),
        |requested, source| {
            BackendError::new(format!(
                "{context} could not reserve {requested} flush worker handle(s): {source}. Fix: lower pipeline cache flush fan-out."
            ))
        },
    )
}

pub(super) fn read_metadata<T: serde::de::DeserializeOwned>(meta_path: &Path) -> Result<T, ()> {
    let Ok(mut file) = File::open(meta_path) else {
        return Err(());
    };
    let Ok(metadata) = file.metadata() else {
        return Err(());
    };
    if metadata.len() > MAX_PIPELINE_CACHE_METADATA_BYTES {
        return Err(());
    }
    if FileExt::lock_shared(&file).is_err() {
        return Err(());
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| ())?;
    let bounded_read_limit = MAX_PIPELINE_CACHE_METADATA_BYTES.checked_add(1).ok_or(())?;
    let mut text = String::new();
    text.try_reserve_exact(capacity).map_err(|_| ())?;
    let res = Read::by_ref(&mut file)
        .take(bounded_read_limit)
        .read_to_string(&mut text);
    if FileExt::unlock(&file).is_err() {
        return Err(());
    }
    if res.is_err()
        || u64::try_from(text.len()).map_or(true, |len| len > MAX_PIPELINE_CACHE_METADATA_BYTES)
    {
        return Err(());
    }
    toml::from_str::<T>(&text).map_err(|_| ())
}

pub(super) fn read_bounded_utf8(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    let bytes = read_bounded_bytes(path, max_bytes)?;
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

pub(super) fn read_bounded_bytes(path: &Path, max_bytes: u64) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cache entry too large",
        ));
    }
    let capacity = usize::try_from(metadata.len()).map_err(|source| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cache entry length cannot fit usize: {source}"),
        )
    })?;
    let bounded_read_limit = max_bytes.checked_add(1).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "cache entry max_bytes cannot add sentinel byte without overflowing u64",
        )
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(capacity).map_err(|source| {
        std::io::Error::new(
            std::io::ErrorKind::OutOfMemory,
            format!("cache entry buffer could not reserve {capacity} bytes: {source}"),
        )
    })?;
    Read::by_ref(&mut file)
        .take(bounded_read_limit)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_or(true, |len| len > max_bytes) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "cache entry exceeded bounded read limit",
        ));
    }
    Ok(bytes)
}

fn trace_io_err(path: &Path, error: &std::io::Error, context: &str) {
    tracing::error!(path_id = %path_fingerprint(path), error_kind = ?error.kind(), "{context}");
}
