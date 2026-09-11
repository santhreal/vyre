//! Persistent on-disk PTX artifact store behind the in-memory source cache,
//! plus the operator-facing PTX dump path.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use vyre_driver::accounting::rebasing_atomic_next_u64;
use vyre_driver::BackendError;

use super::cache_key::PtxSourceCacheKey;

pub(super) const PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES: u64 = 1024 * 1024 * 1024;
pub(super) static PTX_CACHE_TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(super) fn ptx_disk_cache_root() -> Result<PathBuf, BackendError> {
    if let Some(p) = std::env::var_os("VYRE_PTX_SOURCE_CACHE_DIR") {
        let path = PathBuf::from(p);
        if path.as_os_str().is_empty() {
            return Err(BackendError::new(
                "VYRE_PTX_SOURCE_CACHE_DIR is empty. Fix: set it to a writable persistent directory or unset it so XDG/HOME cache discovery can run."
                    .to_string(),
            ));
        }
        return Ok(path);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(xdg).join("vyre").join("ptx-source"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(home)
            .join(".cache")
            .join("vyre")
            .join("ptx-source"));
    }
    Err(BackendError::new(
        "CUDA PTX source cache has no VYRE_PTX_SOURCE_CACHE_DIR, XDG_CACHE_HOME, or HOME. Fix: configure a writable persistent cache root; temporary fallback is forbidden for production compile performance."
            .to_string(),
    ))
}

pub(super) fn ptx_disk_cache_path(key: &PtxSourceCacheKey) -> Result<PathBuf, BackendError> {
    let mut hex = [0u8; 64];
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (i, &b) in key.0.iter().enumerate() {
        hex[i * 2] = HEX[usize::from(b >> 4)];
        hex[i * 2 + 1] = HEX[usize::from(b & 0x0f)];
    }
    let stem = std::str::from_utf8(&hex).map_err(|error| {
        BackendError::new(format!(
            "CUDA PTX source cache generated a non-UTF8 hex key from fixed lowercase ASCII digits: {error}. Fix: inspect cache key generation before publishing PTX artifacts."
        ))
    })?;
    let dir = ptx_disk_cache_root()?.join(&stem[..2]);
    Ok(dir.join(format!("{stem}.ptx")))
}

pub(super) fn load_ptx_from_disk(key: &PtxSourceCacheKey) -> Result<Option<String>, BackendError> {
    let path = ptx_disk_cache_path(key)?;
    match std::fs::metadata(&path) {
        Ok(metadata) => {
            validate_ptx_disk_cache_file_len(metadata.len(), &path)?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(BackendError::new(format!(
                "failed to stat CUDA PTX source cache `{}`: {error}. Fix: repair cache file permissions or remove the corrupt cache entry; do not silently relower around a broken production cache.",
                path.display()
            )));
        }
    }
    read_ptx_disk_cache_source_bounded(&path)
}

fn read_ptx_disk_cache_source_bounded(
    path: &std::path::Path,
) -> Result<Option<String>, BackendError> {
    let mut reader = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(BackendError::new(format!(
                "failed to read CUDA PTX source cache `{}`: {error}. Fix: repair cache file permissions or remove the corrupt cache entry; do not silently relower around a broken production cache.",
                path.display()
            )));
        }
    };
    let mut bytes = Vec::new();
    let mut total = 0u64;
    let mut chunk = [0u8; 8192];
    loop {
        let read = std::io::Read::read(&mut reader, &mut chunk).map_err(|error| {
            BackendError::new(format!(
                "failed to read CUDA PTX source cache `{}`: {error}. Fix: repair cache file permissions or remove the corrupt cache entry; do not silently relower around a broken production cache.",
                path.display()
            ))
        })?;
        if read == 0 {
            return String::from_utf8(bytes).map(Some).map_err(|error| {
                BackendError::new(format!(
                    "CUDA PTX source cache `{}` is not UTF-8: {error}. Fix: remove the corrupt cache artifact.",
                    path.display()
                ))
            });
        }
        let read = read as u64;
        // Use checked_add, matching the checked-arithmetic contract that every
        // other counter in this module follows (see `pinning_atomic_increment_u64`
        // and the source-contract test `module_cache_eviction_buffers_fit_soft_cap_inline`).
        // saturating_add would silently freeze `total` at u64::MAX on
        // pathological input, breaking the safety-cap check that follows.
        total = total.checked_add(read).ok_or_else(|| {
            BackendError::new(format!(
                "CUDA PTX source cache `{}` size accumulator overflowed u64 during bounded read. \
                 Fix: remove the corrupt cache artifact and re-prime the cache.",
                path.display()
            ))
        })?;
        if total > PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES {
            return Err(BackendError::new(format!(
                "CUDA PTX source cache `{}` exceeds the {} byte safety limit while reading. Fix: remove the corrupt cache artifact or raise the cap deliberately after reviewing compile-cache memory pressure.",
                path.display(),
                PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES
            )));
        }
        bytes.extend_from_slice(&chunk[..read as usize]);
    }
}

pub(super) fn validate_ptx_disk_cache_file_len(
    byte_len: u64,
    path: &std::path::Path,
) -> Result<(), BackendError> {
    if byte_len > PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES {
        return Err(BackendError::new(format!(
            "CUDA PTX source cache `{}` is {byte_len} bytes, above the {} byte safety limit. Fix: remove the corrupt cache artifact or raise the artifact cap deliberately after reviewing compile-cache memory pressure.",
            path.display(),
            PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES
        )));
    }
    Ok(())
}

pub(super) fn store_ptx_to_disk(key: &PtxSourceCacheKey, source: &str) -> Result<(), BackendError> {
    let source_len = u64::try_from(source.len()).map_err(|error| {
        BackendError::new(format!(
            "CUDA PTX source cache artifact length cannot fit u64: {error}. Fix: split the generated Program before attempting disk persistence."
        ))
    })?;
    if source_len > PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES {
        return Err(BackendError::new(format!(
            "refusing to write {} byte CUDA PTX source cache artifact above the {} byte safety limit. Fix: split the generated Program, reduce monomorphized PTX size, or raise the artifact cap deliberately after reviewing compile-cache memory pressure.",
            source_len,
            PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES
        )));
    }
    let path = ptx_disk_cache_path(key)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| {
            BackendError::new(format!(
                "failed to create CUDA PTX source cache directory `{}`: {error}. Fix: set VYRE_PTX_SOURCE_CACHE_DIR to a writable cache directory or repair directory permissions.",
                parent.display()
            ))
        })?;
    }
    let tmp_id = allocate_ptx_cache_tmp_id()?;
    let tmp = path.with_extension(format!("ptx.{}.{}.tmp", std::process::id(), tmp_id));
    std::fs::write(&tmp, source.as_bytes()).map_err(|error| {
        BackendError::new(format!(
            "failed to write CUDA PTX source cache temp file `{}`: {error}. Fix: set VYRE_PTX_SOURCE_CACHE_DIR to a writable cache directory or repair filesystem permissions.",
            tmp.display()
        ))
    })?;
    std::fs::rename(&tmp, &path).map_err(|error| {
        let cleanup = match std::fs::remove_file(&tmp) {
            Ok(()) => String::new(),
            Err(cleanup_error) if cleanup_error.kind() == std::io::ErrorKind::NotFound => {
                String::new()
            }
            Err(cleanup_error) => {
                format!(" Temp cleanup also failed: {cleanup_error}. Fix: repair cache directory permissions and remove stale temp files.")
            }
        };
        BackendError::new(format!(
            "failed to publish CUDA PTX source cache `{}` from temp `{}`: {error}.{cleanup} Fix: repair cache directory permissions and filesystem atomic-rename support.",
            path.display(),
            tmp.display()
        ))
    })?;
    Ok(())
}

pub(super) fn allocate_ptx_cache_tmp_id() -> Result<u64, BackendError> {
    Ok(rebasing_atomic_next_u64(
        &PTX_CACHE_TMP_COUNTER,
        1,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |_, _| {
            tracing::error!(
                "CUDA PTX source cache temp-file counter overflowed u64; rebasing sequence to keep disk cache publication alive. Fix: inspect unexpectedly high cache write churn."
            );
        },
    ))
}

pub(super) fn write_ptx_dump(
    dir: std::ffi::OsString,
    ptx_src: &str,
    env_name: &'static str,
) -> Result<std::path::PathBuf, BackendError> {
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|error| BackendError::KernelCompileFailed {
        backend: crate::CUDA_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{env_name} points at `{}` but the directory could not be created: {error}. Fix: choose a writable PTX dump directory or unset {env_name}.",
            dir.display()
        ),
    })?;
    let hash = blake3::hash(ptx_src.as_bytes());
    let path = dir.join(format!("ptx-{}.ptx", &hash.to_hex().as_str()[..16]));
    std::fs::write(&path, ptx_src).map_err(|error| BackendError::KernelCompileFailed {
        backend: crate::CUDA_BACKEND_ID.to_string(),
        compiler_message: format!(
            "{env_name} could not write PTX dump `{}`: {error}. Fix: choose a writable PTX dump directory or unset {env_name}.",
            path.display()
        ),
    })?;
    Ok(path)
}

// Inline: covers `PTX_CACHE_TMP_COUNTER`, `PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES`,
// `allocate_ptx_cache_tmp_id`, `validate_ptx_disk_cache_file_len`, which no integration test can
// name.
#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::{
        allocate_ptx_cache_tmp_id, validate_ptx_disk_cache_file_len, PTX_CACHE_TMP_COUNTER,
        PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES,
    };

    #[test]
    fn ptx_disk_cache_rejects_oversized_artifact_before_reading() {
        let path = std::path::PathBuf::from("/tmp/vyre-oversized-ptx-cache-artifact.ptx");
        let error =
            validate_ptx_disk_cache_file_len(PTX_SOURCE_CACHE_MAX_ARTIFACT_BYTES + 1, &path)
                .expect_err("oversized PTX cache artifact must be rejected before allocation");

        let message = error.to_string();
        assert!(message.contains("above the"));
        assert!(message.contains("safety limit"));
        assert!(message.contains("remove the corrupt cache artifact"));
    }

    #[test]
    fn ptx_source_cache_temp_id_rebases_after_counter_overflow() {
        PTX_CACHE_TMP_COUNTER.store(u64::MAX, Ordering::Release);

        let id = allocate_ptx_cache_tmp_id().expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - PTX temp-file id allocation must rebase instead of failing on counter overflow",
        );

        assert_eq!(id, u64::MAX);
        assert_eq!(PTX_CACHE_TMP_COUNTER.load(Ordering::Acquire), 1);
    }
}
