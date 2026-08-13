//! Include resolution contract for the GPU preprocessor driver.

use std::sync::Arc;

/// Result of resolving one include request.
pub(super) type IncludeLoadResult = Result<Option<(std::path::PathBuf, Arc<[u8]>)>, String>;

/// Include resolver used by the orchestration layer after GPU directive
/// extraction emits an include request.
pub trait IncludeLoader {
    /// Resolve and load `#include <path>` (system) or `#include "path"`
    /// (local). `is_next` is true for GNU `#include_next`, where search
    /// resumes after the include directory that supplied `from`. `from`
    /// is the canonical path of the file currently being preprocessed;
    /// the impl uses it as the search base for local includes.
    ///
    /// Returns `(canonical_path, file_bytes)`. Returns `Err` for missing
    /// includes and fatal I/O errors; production callers must not silently
    /// skip a requested C header.
    fn load(
        &self,
        path: &[u8],
        is_system: bool,
        is_next: bool,
        from: &std::path::Path,
    ) -> IncludeLoadResult;
}

/// Maximum recursive `#include` depth before the driver bails out.
/// Matches the resident frontend include-depth contract.
pub const MAX_INCLUDE_DEPTH: u32 = 64;
