//! Borrow checking stage using the external dataflow engine.
//!
//! Lower the nano-subset AST to a control-flow graph, then run
//! fixed-point analyses to validate ownership and borrowing.

use external_dataflow_engine::ssa::try_compute_dominators;

use crate::pipeline::typeck_stage::TypedModule;
use crate::RustFrontendError;

/// Verified module (borrow-checked).
pub type VerifiedModule = TypedModule;

/// Borrow-check a typed module.
pub fn borrow_check(module: &TypedModule) -> Result<VerifiedModule, RustFrontendError> {
    let _ = module;
    let _ = try_compute_dominators;
    Err(RustFrontendError::Unsupported(
        "borrow checking is not wired to a Rust CFG yet; disable `borrow_check` for parse-only pipeline runs"
            .to_string(),
    ))
}
