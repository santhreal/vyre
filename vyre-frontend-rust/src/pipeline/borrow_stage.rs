//! Borrow-checking stage over the frontend-owned semantic substrate.

use crate::parse::Module;
use crate::sema::{self, Resolution};

use crate::RustFrontendError;

/// Borrow-check a resolved module via the reusable sema substrate.
pub fn borrow_check(module: &Module, resolution: &Resolution) -> Result<(), RustFrontendError> {
    sema::borrow_check(module, resolution).map_err(|e| RustFrontendError::Borrow(e.to_string()))
}
