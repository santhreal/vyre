//! Lowering stage over the frontend-owned typed-IR substrate.

use vyre_foundation::ir::Program;
use crate::lower as rust_lower;
use crate::parse::Module;
use crate::sema::Resolution;

use crate::RustFrontendError;

/// Lower a resolved module to Vyre IR via the reusable lowering substrate.
pub fn lower(
    module: &Module,
    resolution: &Resolution,
    lane_count: Option<u32>,
) -> Result<Program, RustFrontendError> {
    let result = match lane_count {
        Some(lanes) => rust_lower::lower_batched(module, resolution, lanes),
        None => rust_lower::lower(module, resolution),
    };
    result.map_err(|e| RustFrontendError::Lower(e.to_string()))
}
