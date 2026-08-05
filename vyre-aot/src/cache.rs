//! Runtime-cache compatibility for AOT-emitted artifacts (audit P0 #26).
//!
//! `vyre-runtime`'s `DiskCache` stores compiled pipeline blobs as
//! `<payload bytes><32-byte BLAKE3 footer>` keyed by
//! `vyre_runtime::PipelineFingerprint::of(&Program)`. The same algorithm now
//! lives in `vyre_foundation::optimizer::pipeline_fingerprint_bytes`, so an
//! AOT producer can write a runtime-cache-compatible blob alongside its
//! self-contained submission bundle. A consumer that already runs through
//! the runtime cache then picks up the AOT blob without any additional
//! plumbing.

use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use vyre_foundation::ir::Program;

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

/// Write the canonical artifact envelope to `cache_dir` with a BLAKE3 footer.
///
/// `program` supplies the existing cache lookup key while `artifact` supplies
/// the sole versioned artifact representation stored as the cache payload.
///
/// Returns the absolute path of the file written.
///
/// # Errors
///
/// Returns [`RuntimeCacheError::CanonicalArtifact`] when the envelope cannot
/// encode, or [`RuntimeCacheError::Io`] when the atomic cache write fails.
pub fn emit_runtime_cache_blob(
    program: &Program,
    artifact: &crate::artifact::CompiledArtifact,
    cache_dir: &Path,
) -> Result<PathBuf, RuntimeCacheError> {
    fs::create_dir_all(cache_dir).map_err(|source| RuntimeCacheError::Io {
        path: cache_dir.to_path_buf(),
        source,
    })?;

    let envelope_bytes = artifact.envelope().to_bytes()?;

    let fingerprint = vyre_foundation::optimizer::pipeline_fingerprint_bytes(program);
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

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Node, Program};

    fn add_one_program() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::read("input", 0, DataType::U32).with_count(64),
                BufferDecl::output("out", 1, DataType::U32)
                    .with_count(64)
                    .with_output_byte_range(0..256),
            ],
            [64, 1, 1],
            vec![Node::return_()],
        )
    }

    #[test]
    fn fingerprint_hex_is_64_lowercase_chars() {
        let bytes: [u8; 32] = [0xAB; 32];
        let hex = fingerprint_hex(&bytes);
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(hex.chars().all(|c| !c.is_ascii_uppercase()));
    }

    #[test]
    fn emit_writes_canonical_envelope_with_blake3_footer() {
        let program = add_one_program();
        let artifact = crate::compile::artifact_fixture(
            &program,
            (0..1024).map(|index| (index % 251) as u8).collect(),
        );
        let envelope_bytes = artifact.envelope().to_bytes().unwrap();
        let dir = tempfile::tempdir().expect("Fix: tempdir must succeed");
        let path = emit_runtime_cache_blob(&program, &artifact, dir.path())
            .expect("Fix: emit must succeed for a valid canonical artifact");

        let blob = std::fs::read(&path).expect("Fix: blob must be readable");
        assert_eq!(blob.len(), envelope_bytes.len() + 32);
        let payload = &blob[..envelope_bytes.len()];
        let footer = &blob[envelope_bytes.len()..];
        assert_eq!(payload, envelope_bytes);
        assert_eq!(footer, blake3::hash(&envelope_bytes).as_bytes());
    }

    #[test]
    fn emit_filename_matches_runtime_fingerprint() {
        let program = add_one_program();
        let artifact =
            crate::compile::artifact_fixture(&program, b"\x00\x01\x02\x03".to_vec());
        let dir = tempfile::tempdir().expect("Fix: tempdir must succeed");
        let path = emit_runtime_cache_blob(&program, &artifact, dir.path())
            .expect("Fix: emit must succeed");

        let expected_fingerprint = vyre_foundation::optimizer::pipeline_fingerprint_bytes(&program);
        let expected_filename = format!("{}.bin", fingerprint_hex(&expected_fingerprint));
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some(expected_filename.as_str())
        );
    }

    #[test]
    fn buffer_declaration_order_does_not_change_fingerprint() {
        // Audit P0 #26 invariant: AOT and runtime must hash to the same key
        // for semantically-equal Programs whose buffer decls happen to be in
        // a different order.
        let p1 = Program::wrapped(
            vec![
                BufferDecl::read("a", 0, DataType::U32).with_count(64),
                BufferDecl::read("b", 1, DataType::U32).with_count(64),
                BufferDecl::output("out", 2, DataType::U32)
                    .with_count(64)
                    .with_output_byte_range(0..256),
            ],
            [64, 1, 1],
            vec![Node::return_()],
        );
        let p2 = Program::wrapped(
            vec![
                BufferDecl::read("b", 1, DataType::U32).with_count(64),
                BufferDecl::output("out", 2, DataType::U32)
                    .with_count(64)
                    .with_output_byte_range(0..256),
                BufferDecl::read("a", 0, DataType::U32).with_count(64),
            ],
            [64, 1, 1],
            vec![Node::return_()],
        );
        assert_eq!(
            vyre_foundation::optimizer::pipeline_fingerprint_bytes(&p1),
            vyre_foundation::optimizer::pipeline_fingerprint_bytes(&p2),
        );
    }
}
