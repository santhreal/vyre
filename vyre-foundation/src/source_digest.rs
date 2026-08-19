//! Digest of a package's own source tree.
//!
//! An artifact cache that keys on the program alone answers a compiler change
//! with an artifact the previous compiler produced: the program is unchanged, so
//! the key is unchanged, and the fix never reaches the device. A hand-edited
//! contract label covers that only while someone remembers to edit it. The
//! digest here changes whenever the package's own sources or manifest change, so
//! a key that mixes it in invalidates on every edit of the emitter that wrote
//! the artifact.
//!
//! A build script stamps the value into the crate and a test recomputes it from
//! the tree, so both read one implementation. A build script may read only its
//! own package directory, because the workspace root is absent from the
//! published tarball; the digest therefore covers `src/**` and `Cargo.toml`
//! under one package root and nothing above it.

use std::path::{Path, PathBuf};

use crate::hashing::update_length_delimited_field;

/// Largest total input the digest reads before it refuses, in bytes.
pub const MAX_SOURCE_DIGEST_BYTES: u64 = 64 * 1024 * 1024;

/// Why a source-tree digest could not be computed.
#[derive(Debug, thiserror::Error)]
pub enum SourceDigestError {
    /// A directory on the walk could not be listed.
    #[error(
        "Fix: failed to list {path}: {source}. The digest covers every file under the package \
             source directory, so restore the directory before building."
    )]
    Directory {
        /// Directory that could not be listed.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// A file on the walk could not be read.
    #[error(
        "Fix: failed to read {path}: {source}. Every walked file is digest input, so restore \
             the file before building."
    )]
    File {
        /// File that could not be read.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// A walked path is not valid UTF-8, so it cannot be framed reproducibly.
    #[error(
        "Fix: {path} is not valid UTF-8. The digest frames each path as text so the value is \
             reproducible across hosts; rename the file to UTF-8."
    )]
    NonUtf8 {
        /// Path that is not valid UTF-8.
        path: PathBuf,
    },
    /// A walked path left the package root, so the framing would be ambiguous.
    #[error(
        "Fix: {path} escaped the package root {root}. Remove the link that leaves the package; \
             a published tarball cannot carry it."
    )]
    Escaped {
        /// Path that left the root.
        path: PathBuf,
        /// Package root the walk started from.
        root: PathBuf,
    },
    /// The tree is larger than the digest reads.
    #[error(
        "Fix: the source tree exceeds the {limit} byte digest cap at {path}. Split the package \
             or raise the cap deliberately; a build script must not read an unbounded tree."
    )]
    TooLarge {
        /// File that crossed the cap.
        path: PathBuf,
        /// Cap in bytes.
        limit: u64,
    },
    /// The environment variable the digest would be stamped into is not a name
    /// cargo can pass through.
    #[error(
        "Fix: {name:?} is not a usable environment variable name. Name the digest variable in \
             ASCII uppercase, digits, and underscores so `env!` can read it back."
    )]
    EnvVarName {
        /// Name the caller supplied.
        name: String,
    },
    /// The walk found no sources, so the digest would identify nothing.
    #[error(
        "Fix: {path} holds no files, so the digest identifies nothing and every compiler \
             version shares one cache key. Point the digest at the package root that owns the \
             sources."
    )]
    Empty {
        /// Source directory that held no files.
        path: PathBuf,
    },
}

/// Digest `src/**` and `Cargo.toml` under `package_root`, separated by `domain`.
///
/// The value is the lowercase hex BLAKE3 hash of the domain, then every source
/// file in ascending relative-path order, then the manifest. Paths are framed as
/// `/`-separated text relative to `package_root`, so the value does not depend on
/// the host's directory order or path separator.
///
/// # Errors
///
/// Returns [`SourceDigestError`] when the walk cannot read the tree, when a path
/// cannot be framed reproducibly, when the input crosses
/// [`MAX_SOURCE_DIGEST_BYTES`], or when the source directory is empty.
pub fn source_tree_digest(package_root: &Path, domain: &[u8]) -> Result<String, SourceDigestError> {
    let source_dir = package_root.join("src");
    let mut files = Vec::new();
    collect_files(package_root, &source_dir, &mut files)?;
    if files.is_empty() {
        return Err(SourceDigestError::Empty { path: source_dir });
    }
    files.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut read = 0u64;
    let mut hasher = blake3::Hasher::new();
    update_length_delimited_field(&mut hasher, b"domain", domain);
    for (relative, path) in &files {
        let bytes = read_bounded(path, &mut read)?;
        update_length_delimited_field(&mut hasher, relative.as_bytes(), &bytes);
    }
    let manifest_path = package_root.join("Cargo.toml");
    let manifest = read_bounded(&manifest_path, &mut read)?;
    update_length_delimited_field(&mut hasher, b"Cargo.toml", &manifest);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Build the `cargo:` directives that stamp the digest of `package_root`.
///
/// A build script writes the returned text to stdout unchanged. The text asks
/// cargo to rerun the script when the package's sources or manifest change, and
/// stamps the digest into `env_var`, which the crate reads back with `env!`.
/// Building the text here keeps one owner for the whole mechanism: a caller that
/// only prints it cannot forget a rerun trigger and pin a stale digest.
///
/// # Errors
///
/// Returns [`SourceDigestError::EnvVarName`] when `env_var` is not an
/// environment variable name, and every error of [`source_tree_digest`].
pub fn cargo_directives(
    package_root: &Path,
    env_var: &str,
    domain: &[u8],
) -> Result<String, SourceDigestError> {
    if env_var.is_empty()
        || !env_var
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return Err(SourceDigestError::EnvVarName {
            name: env_var.to_owned(),
        });
    }
    let digest = source_tree_digest(package_root, domain)?;
    let source_dir = package_root.join("src");
    let manifest_path = package_root.join("Cargo.toml");
    Ok(format!(
        "cargo:rerun-if-changed={}\ncargo:rerun-if-changed={}\ncargo:rustc-env={env_var}={digest}\n",
        source_dir.display(),
        manifest_path.display()
    ))
}

fn collect_files(
    root: &Path,
    dir: &Path,
    out: &mut Vec<(String, PathBuf)>,
) -> Result<(), SourceDigestError> {
    let entries = std::fs::read_dir(dir).map_err(|source| SourceDigestError::Directory {
        path: dir.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| SourceDigestError::Directory {
            path: dir.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        let kind = entry
            .file_type()
            .map_err(|source| SourceDigestError::File {
                path: path.clone(),
                source,
            })?;
        if kind.is_dir() {
            collect_files(root, &path, out)?;
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .map_err(|_| SourceDigestError::Escaped {
                path: path.clone(),
                root: root.to_path_buf(),
            })?
            .to_str()
            .ok_or_else(|| SourceDigestError::NonUtf8 { path: path.clone() })?
            .replace('\\', "/");
        out.push((relative, path));
    }
    Ok(())
}

fn read_bounded(path: &Path, read: &mut u64) -> Result<Vec<u8>, SourceDigestError> {
    let bytes = std::fs::read(path).map_err(|source| SourceDigestError::File {
        path: path.to_path_buf(),
        source,
    })?;
    *read = read.saturating_add(bytes.len() as u64);
    if *read > MAX_SOURCE_DIGEST_BYTES {
        return Err(SourceDigestError::TooLarge {
            path: path.to_path_buf(),
            limit: MAX_SOURCE_DIGEST_BYTES,
        });
    }
    Ok(bytes)
}
