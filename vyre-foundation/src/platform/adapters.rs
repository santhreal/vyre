//! Small typed platform adapters without path-string policy in domain logic (Row 118).

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use thiserror::Error;

/// Error returned by platform adapter operations.
#[derive(Debug, Error)]
pub enum PlatformAdapterError {
    /// Filesystem input/output error.
    #[error("filesystem error on {path}: {source}")]
    Io {
        /// Affected path.
        path: PathBuf,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// File size exceeded safe reading quota.
    #[error("file size for {path} exceeds quota limit of {max_bytes} bytes")]
    QuotaExceeded {
        /// Affected path.
        path: PathBuf,
        /// Quota limit in bytes.
        max_bytes: usize,
    },
    /// Thread creation failed.
    #[error("failed to spawn thread `{name}`: {source}")]
    ThreadSpawn {
        /// Thread name.
        name: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },
}

static PROCESS_START: std::sync::LazyLock<Instant> = std::sync::LazyLock::new(Instant::now);

/// Monotonic clock adapter providing high-resolution duration and nanosecond timestamps.
pub struct ClockAdapter;

impl ClockAdapter {
    /// Return monotonic nanoseconds elapsed since process initialization.
    pub fn monotonic_now_ns() -> u64 {
        u64::try_from(PROCESS_START.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Calculate elapsed nanoseconds since a previous timestamp.
    pub fn elapsed_ns(since_ns: u64) -> u64 {
        Self::monotonic_now_ns().saturating_sub(since_ns)
    }
}

static SCRATCH_COUNTER: AtomicU64 = AtomicU64::new(1);

/// RAII temporary directory that removes itself and contents upon drop.
#[derive(Debug)]
pub struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    /// Return the path of this temporary directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// Typed filesystem adapter with atomic operations and bounded readers.
pub struct FileSystemAdapter;

impl FileSystemAdapter {
    /// Atomically write binary payload to target path using a temporary staging sibling.
    pub fn atomic_write(target: &Path, data: &[u8]) -> Result<(), PlatformAdapterError> {
        let parent = target.parent().unwrap_or_else(|| Path::new("."));
        let temp_name = format!(
            ".tmp_{}_{}",
            std::process::id(),
            SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let temp_path = parent.join(temp_name);

        let mut file = File::create(&temp_path).map_err(|e| PlatformAdapterError::Io {
            path: temp_path.clone(),
            source: e,
        })?;

        file.write_all(data).map_err(|e| PlatformAdapterError::Io {
            path: temp_path.clone(),
            source: e,
        })?;

        file.sync_all().map_err(|e| PlatformAdapterError::Io {
            path: temp_path.clone(),
            source: e,
        })?;

        drop(file);

        fs::rename(&temp_path, target).map_err(|e| {
            let _ = fs::remove_file(&temp_path);
            PlatformAdapterError::Io {
                path: target.to_path_buf(),
                source: e,
            }
        })?;

        Ok(())
    }

    /// Read file content up to a bounded quota limit.
    pub fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>, PlatformAdapterError> {
        let file = File::open(path).map_err(|e| PlatformAdapterError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let metadata = file.metadata().map_err(|e| PlatformAdapterError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        if metadata.len() > (max_bytes as u64) {
            return Err(PlatformAdapterError::QuotaExceeded {
                path: path.to_path_buf(),
                max_bytes,
            });
        }

        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        file.take(max_bytes as u64)
            .read_to_end(&mut buffer)
            .map_err(|e| PlatformAdapterError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;

        Ok(buffer)
    }

    /// Create a managed scratch directory under the system or local temp directory.
    pub fn create_scratch_dir(prefix: &str) -> Result<ScratchDir, PlatformAdapterError> {
        let id = SCRATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_base = std::env::temp_dir();
        let dir_path = temp_base.join(format!("{}_{}_{}", prefix, std::process::id(), id));

        fs::create_dir_all(&dir_path).map_err(|e| PlatformAdapterError::Io {
            path: dir_path.clone(),
            source: e,
        })?;

        Ok(ScratchDir { path: dir_path })
    }
}

/// Typed threading adapter with bounded stack configuration.
pub struct ThreadAdapter;

impl ThreadAdapter {
    /// Spawn a named worker thread with a specified stack size.
    pub fn spawn_named<F, T>(
        name: impl Into<String>,
        stack_size: usize,
        f: F,
    ) -> Result<std::thread::JoinHandle<T>, PlatformAdapterError>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let thread_name = name.into();
        std::thread::Builder::new()
            .name(thread_name.clone())
            .stack_size(stack_size)
            .spawn(f)
            .map_err(|e| PlatformAdapterError::ThreadSpawn {
                name: thread_name,
                source: e,
            })
    }

    /// Cooperatively yield execution time slice to other threads.
    #[inline]
    pub fn yield_now() {
        std::thread::yield_now();
    }
}
