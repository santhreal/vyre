//! Runtime-cache compatibility for canonical AOT envelopes.
//!
//! Cache lookup and AOT packaging use the neutral artifact digest. Target
//! payloads remain authenticated attachments and do not create a second
//! semantic identity.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Errors emitted by [`emit_runtime_cache_blob`].
#[derive(Debug, thiserror::Error)]
pub enum RuntimeCacheError {
    /// The cache root directory could not be created or is not writable.
    #[error("runtime cache directory I/O failed at {path:?}: {source}")]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },

    /// Canonical envelope serialization failed before the cache write.
    #[error("runtime cache canonical artifact serialization failed: {0}")]
    CanonicalArtifact(#[from] vyre_megakernel::CompileError),
}

/// Write a canonical artifact envelope under its neutral artifact digest.
///
/// Returns the absolute path of the file written.
///
/// # Errors
///
/// Returns [`RuntimeCacheError::CanonicalArtifact`] when the envelope cannot
/// encode, or [`RuntimeCacheError::Io`] when the atomic cache write fails.
pub fn emit_runtime_cache_blob(
    envelope: &vyre_megakernel::ArtifactEnvelope,
    cache_dir: &Path,
) -> Result<PathBuf, RuntimeCacheError> {
    fs::create_dir_all(cache_dir).map_err(|source| RuntimeCacheError::Io {
        path: cache_dir.to_path_buf(),
        source,
    })?;

    let envelope_bytes = envelope.to_bytes()?;

    let fingerprint = envelope.neutral().digest().0;
    let hex = fingerprint_hex(&fingerprint);
    let final_path = cache_dir.join(format!("{hex}.bin"));
    let tmp_path = cache_dir.join(format!(".{hex}.bin.tmp"));

    let write_one_shot = || -> io::Result<()> {
        let footer = blake3::hash(&envelope_bytes);
        let mut f = File::create(&tmp_path)?;
        f.write_all(&envelope_bytes)?;
        f.write_all(footer.as_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp_path, &final_path)?;
        Ok(())
    };

    write_one_shot().map_err(|source| match fs::remove_file(&tmp_path) {
        Ok(()) => RuntimeCacheError::Io {
            path: final_path.clone(),
            source,
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => RuntimeCacheError::Io {
            path: final_path.clone(),
            source,
        },
        Err(error) => RuntimeCacheError::Io {
            path: tmp_path.clone(),
            source: error,
        },
    })?;

    Ok(final_path)
}

/// 64-char lowercase hex of a 32-byte fingerprint, matching the runtime
/// cache's path-safe encoding.
#[must_use]
pub fn fingerprint_hex(fingerprint: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &b in fingerprint {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}
