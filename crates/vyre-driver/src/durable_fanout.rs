//! A cache entry is durable once its file data is synced and its containing
//! directory is synced. Both syncs are latency-bound, so a flush of several
//! hundred entries runs them concurrently across a bounded worker pool.
//! `sync_paths_and_parents` is the single implementation; each cache layer
//! supplies only the mapping from `SyncFailure` to its own error type.

use std::collections::TryReserveError;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Upper bound on concurrent durability workers.
///
/// Each worker spends its time inside one filesystem sync, so the useful
/// ceiling is storage queue depth rather than core count. Without a ceiling a
/// large flush spawns one thread per path, which costs more in scheduling than
/// the syncs recover in overlap.
const MAX_WORKERS: usize = 16;

/// Pass of a durable flush.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SyncPass {
    /// Syncing the data of each written file.
    Files,
    /// Syncing each distinct parent directory, which makes the entries durable.
    Directories,
}

impl SyncPass {
    fn noun(self) -> &'static str {
        match self {
            Self::Files => "file",
            Self::Directories => "directory",
        }
    }
}

/// Failure of one step of [`sync_paths_and_parents`].
///
/// `Display` omits the path, so a message built from it is safe to log. A
/// caller that reports paths reads the `path` field.
#[derive(Debug)]
pub enum SyncFailure<'a> {
    /// Opening `path` for the sync failed.
    Open {
        /// Pass the path belongs to.
        pass: SyncPass,
        /// Path that failed to open.
        path: &'a Path,
        /// Error from the open call.
        source: io::Error,
    },
    /// Syncing `path` failed.
    Sync {
        /// Pass the path belongs to.
        pass: SyncPass,
        /// Path that failed to sync.
        path: &'a Path,
        /// Error from the sync call.
        source: io::Error,
    },
    /// A sync worker panicked.
    Panicked {
        /// Pass the worker belonged to.
        pass: SyncPass,
    },
    /// Staging storage for the pass could not be reserved.
    Reserve {
        /// Pass whose staging failed.
        pass: SyncPass,
        /// Entries requested.
        requested: usize,
        /// Allocator error.
        source: TryReserveError,
    },
}

impl fmt::Display for SyncFailure<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { pass, source, .. } => write!(
                f,
                "durable flush could not open a {} for sync: {source}. Fix: remove the corrupted cache entry and retry.",
                pass.noun()
            ),
            Self::Sync { pass, source, .. } => write!(
                f,
                "durable flush could not sync a {}: {source}. Fix: check cache storage health and retry.",
                pass.noun()
            ),
            Self::Panicked { pass } => write!(
                f,
                "durable flush {} sync worker panicked. Fix: retry the flush.",
                pass.noun()
            ),
            Self::Reserve {
                pass,
                requested,
                source,
            } => write!(
                f,
                "durable flush could not reserve {requested} {} slot(s): {source}. Fix: flush fewer cache paths per batch.",
                pass.noun()
            ),
        }
    }
}

/// `failure` as an [`io::Error`] that keeps the open or sync error's kind.
///
/// A free function rather than a `From` impl, so it can be passed directly as
/// the `on_failure` mapper of [`sync_paths_and_parents`] for every lifetime.
#[must_use]
pub fn into_io_error(failure: SyncFailure<'_>) -> io::Error {
    let kind = match &failure {
        SyncFailure::Open { source, .. } | SyncFailure::Sync { source, .. } => source.kind(),
        SyncFailure::Panicked { .. } => io::ErrorKind::Other,
        SyncFailure::Reserve { .. } => io::ErrorKind::OutOfMemory,
    };
    io::Error::new(kind, failure.to_string())
}

/// Sync the data of every file in `paths`, then every distinct parent
/// directory of them.
///
/// Files are opened with write access, which `FlushFileBuffers` requires on
/// Windows; directories are opened read-only. The directory pass runs on Unix
/// only: Windows has no directory handle to flush, and Miri models `fsync`
/// only on regular files.
///
/// # Errors
///
/// Returns `on_failure` applied to the first failure. A failing chunk is
/// joined before returning, and no later chunk starts.
pub fn sync_paths_and_parents<E: Send>(
    paths: &[PathBuf],
    on_failure: impl Fn(SyncFailure<'_>) -> E + Sync,
) -> Result<(), E> {
    for_each_bounded(
        paths,
        SyncPass::Files,
        open_for_sync,
        File::sync_data,
        &on_failure,
    )?;
    if cfg!(all(unix, not(miri))) {
        let parents = parent_directories(paths).map_err(|(requested, source)| {
            on_failure(SyncFailure::Reserve {
                pass: SyncPass::Directories,
                requested,
                source,
            })
        })?;
        for_each_bounded(
            &parents,
            SyncPass::Directories,
            |path| File::open(path),
            File::sync_all,
            &on_failure,
        )?;
    }
    Ok(())
}

/// Concurrent durability workers this host may use, resolved once.
fn worker_count() -> usize {
    static WORKERS: LazyLock<usize> = LazyLock::new(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .clamp(1, MAX_WORKERS)
    });
    *WORKERS
}

/// Sorted, distinct parent directories of `paths`.
fn parent_directories(paths: &[PathBuf]) -> Result<Vec<PathBuf>, (usize, TryReserveError)> {
    let mut parents = Vec::new();
    parents
        .try_reserve_exact(paths.len())
        .map_err(|source| (paths.len(), source))?;
    parents.extend(
        paths
            .iter()
            .filter_map(|path| path.parent().map(Path::to_path_buf)),
    );
    parents.sort_unstable();
    parents.dedup();
    Ok(parents)
}

fn open_for_sync(path: &Path) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

/// Open and sync every path across a bounded pool of scoped threads.
///
/// Paths are taken [`worker_count`] at a time and each chunk is joined before
/// the next is spawned; that join bounds the live threads. A panicking worker
/// becomes [`SyncFailure::Panicked`] instead of resuming the unwind, because
/// the caller may hold a lock over the same path set.
fn for_each_bounded<E: Send>(
    paths: &[PathBuf],
    pass: SyncPass,
    open: impl Fn(&Path) -> io::Result<File> + Sync,
    sync: fn(&File) -> io::Result<()>,
    on_failure: &(impl Fn(SyncFailure<'_>) -> E + Sync),
) -> Result<(), E> {
    let task = |path: &PathBuf| -> Result<(), E> {
        let file =
            open(path).map_err(|source| on_failure(SyncFailure::Open { pass, path, source }))?;
        sync(&file).map_err(|source| on_failure(SyncFailure::Sync { pass, path, source }))
    };
    for chunk in paths.chunks(worker_count()) {
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            handles.try_reserve_exact(chunk.len()).map_err(|source| {
                on_failure(SyncFailure::Reserve {
                    pass,
                    requested: chunk.len(),
                    source,
                })
            })?;
            for path in chunk {
                handles.push(scope.spawn(|| task(path)));
            }
            let mut first_error = None;
            for handle in handles {
                let outcome = handle
                    .join()
                    .unwrap_or_else(|_| Err(on_failure(SyncFailure::Panicked { pass })));
                if let Err(error) = outcome {
                    first_error = first_error.or(Some(error));
                }
            }
            first_error.map_or(Ok(()), Err)
        })?;
    }
    Ok(())
}

// Inline: covers `parent_directories`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parent_directories_are_sorted_and_distinct() {
        let paths = [
            PathBuf::from("b/2"),
            PathBuf::from("a/1"),
            PathBuf::from("b/1"),
            PathBuf::from("a/2"),
        ];
        assert_eq!(
            parent_directories(&paths).expect("four parent slots"),
            [PathBuf::from("a"), PathBuf::from("b")]
        );
    }

    /// WHY: a file that cannot be opened must fail the flush in the file pass,
    /// report that exact path, and keep the open error's kind.
    #[test]
    fn missing_file_fails_the_file_pass_with_its_path() {
        let dir = tempfile::tempdir().expect("isolated directory");
        let written = dir.path().join("written");
        std::fs::write(&written, b"entry").expect("write entry");
        let missing = dir.path().join("missing");
        let paths = [written, missing.clone()];
        let reported = sync_paths_and_parents(&paths, |failure| match failure {
            SyncFailure::Open { pass, path, source } => (pass, path.to_path_buf(), source.kind()),
            other => panic!("unexpected failure: {other}"),
        })
        .expect_err("missing file must fail the flush");
        assert_eq!(
            reported,
            (SyncPass::Files, missing, io::ErrorKind::NotFound)
        );
    }

    #[test]
    fn written_files_and_their_directories_sync() {
        let dir = tempfile::tempdir().expect("isolated directory");
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).expect("nested directory");
        let paths = [dir.path().join("top"), nested.join("inner")];
        for path in &paths {
            std::fs::write(path, b"entry").expect("write entry");
        }
        sync_paths_and_parents(&paths, into_io_error).expect("flush succeeds");
    }

    #[test]
    fn io_conversion_keeps_the_source_kind() {
        let error = into_io_error(SyncFailure::Sync {
            pass: SyncPass::Directories,
            path: Path::new("secret/path"),
            source: io::Error::new(io::ErrorKind::PermissionDenied, "denied"),
        });
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert!(
            !error.to_string().contains("secret"),
            "durable flush messages must not include the path: {error}"
        );
    }
}
