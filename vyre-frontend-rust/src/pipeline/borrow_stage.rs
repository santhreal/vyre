//! Borrow checking stage using Weir dataflow analysis.
//!
//! Lower the nano-subset AST to a control-flow graph, then run
//! Weir fixed-point analyses to validate ownership and borrowing.

use external_dataflow_engine::ssa::try_compute_dominators;

use crate::pipeline::typeck_stage::TypedModule;
use crate::RustFrontendError;

/// Verified module (borrow-checked).
pub type VerifiedModule = TypedModule;

/// Borrow-check a typed module.
///
/// TODO(v0.1.0): build CFG from AST, run Weir liveness + reaching
/// defs + points-to, emit borrow errors.
pub fn borrow_check(module: &TypedModule) -> Result<VerifiedModule, RustFrontendError> {
    // Placeholder: dominator computation is imported to prove Weir
    // dependency is wired.  Real borrow check is future work.
    let _ = try_compute_dominators;
    Ok(module.clone())
}
