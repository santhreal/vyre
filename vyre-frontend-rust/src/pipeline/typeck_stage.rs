//! Type-checking stage over the frontend-owned semantic substrate.

use crate::parse::Module;
use crate::sema::{self, Resolution};

use crate::RustFrontendError;

/// Type-check a resolved module via the reusable sema substrate.
pub fn typeck(
    module: &Module,
    source: &[u8],
    resolution: &Resolution,
) -> Result<(), RustFrontendError> {
    sema::typeck(module, source, resolution).map_err(|e| RustFrontendError::Typeck(e.to_string()))
}
