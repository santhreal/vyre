//! Backend-neutral bounded fan-out for durability work on a path set.
//!
//! Making a cache entry durable is one filesystem call per file plus one per
//! containing directory, and both are latency-bound rather than CPU-bound: a
//! flush of several hundred entries wants concurrency, and concurrency without
//! a ceiling wants a thread per entry. Three cache layers each answered that
//! the same way, so the answer lives here once.
//!
//! What stays with the caller is the per-item work and its diagnostics: which
//! sync call to make, what a failure reads like, and what error type it is.
//! What lives here is the ceiling, the chunk-and-join shape that enforces it,
//! and the parent-directory set the second pass runs over.

use std::collections::TryReserveError;
use std::path::PathBuf;
use std::sync::LazyLock;

/// Upper bound on concurrent durability workers.
///
/// Each worker spends its time inside one filesystem sync, so the useful
/// ceiling is storage queue depth rather than core count. Without a ceiling a
/// large flush spawns one thread per path, which costs more in scheduling than
/// the syncs recover in overlap.
const MAX_WORKERS: usize = 16;

/// Concurrent durability workers this host may use.
///
/// Resolved once: `available_parallelism` is a syscall on every platform that
/// implements it, and a flush loop asking it per chunk pays for an answer that
/// cannot change usefully within one process.
#[must_use]
pub fn worker_count() -> usize {
    static WORKERS: LazyLock<usize> = LazyLock::new(|| {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .clamp(1, MAX_WORKERS)
    });
    *WORKERS
}

/// Sorted, distinct parent directories of `paths`.
///
/// A directory entry is durable only after the directory itself is synced, and
/// a cache batch normally writes many files into few directories, so the second
/// pass runs over the deduplicated set rather than once per file.
///
/// # Errors
///
/// Returns whatever `reserve` reports when the parent vector cannot be sized
/// for `paths.len()` entries.
pub fn parent_directories<E>(
    paths: &[PathBuf],
    reserve: impl FnOnce(&mut Vec<PathBuf>, usize) -> Result<(), E>,
) -> Result<Vec<PathBuf>, E> {
    let mut parents = Vec::new();
    reserve(&mut parents, paths.len())?;
    for path in paths {
        if let Some(parent) = path.parent() {
            parents.push(parent.to_path_buf());
        }
    }
    parents.sort_unstable();
    parents.dedup();
    Ok(parents)
}

/// Run `task` over every item across a bounded pool of scoped threads.
///
/// Items are taken [`worker_count`] at a time and each chunk is joined before
/// the next is spawned. That join is what bounds live threads: spawning the
/// whole set into one scope would bound nothing, since the scope only joins at
/// its end.
///
/// The first failure in a chunk propagates and no later chunk is started, so a
/// storage failure stops the fan-out instead of repeating itself once per
/// remaining item. Every thread of the failing chunk is still joined first,
/// because the scope cannot return while one of its threads holds a borrow.
///
/// A worker that panics becomes `on_panic()`. The panic payload is dropped
/// rather than resumed: a durability fan-out reports that the batch is not
/// durable, and resuming here would unwind through the scope of a caller that
/// is holding a lock over the same path set.
///
/// # Errors
///
/// Returns the first error `task` reports, `on_panic()` when a worker panicked,
/// or `on_reserve_failure` when the per-chunk handle vector cannot be sized.
pub fn for_each_bounded<T, E>(
    items: &[T],
    task: impl Fn(&T) -> Result<(), E> + Sync,
    on_panic: impl Fn() -> E,
    on_reserve_failure: impl Fn(usize, TryReserveError) -> E,
) -> Result<(), E>
where
    T: Sync,
    E: Send,
{
    if items.is_empty() {
        return Ok(());
    }
    for chunk in items.chunks(worker_count()) {
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            handles
                .try_reserve_exact(chunk.len())
                .map_err(|source| on_reserve_failure(chunk.len(), source))?;
            for item in chunk {
                handles.push(scope.spawn(|| task(item)));
            }
            let mut first_error = None;
            for handle in handles {
                let outcome = handle.join().unwrap_or_else(|_| Err(on_panic()));
                if let Err(error) = outcome {
                    first_error = first_error.or(Some(error));
                }
            }
            match first_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        })?;
    }
    Ok(())
}
