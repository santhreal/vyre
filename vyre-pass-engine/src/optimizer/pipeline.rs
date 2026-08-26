//! Fixed semantic execution order for the optimizer pipeline.

use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionPolicy, SemanticExecutor};

use super::canonicalize_via_encoded::{gpu_canonicalize, CanonicalizeError};
use super::const_fold_via_encoded::{gpu_const_fold, ConstFoldError};
use super::dce_via_encoded::{gpu_dce, DceError};
use super::pattern_match_via_encoded::{gpu_algebraic_identities, PatternMatchError};

/// Errors surfaced by [`gpu_optimize`], classified by semantic stage.
#[derive(Debug)]
pub enum GpuOptimizeError {
    /// Canonicalization failed.
    Canonicalize(CanonicalizeError),
    /// Constant folding failed.
    ConstFold(ConstFoldError),
    /// Dead-code elimination failed.
    Dce(DceError),
    /// Algebraic identity rewriting failed.
    AlgebraicIdentities(PatternMatchError),
}

impl std::fmt::Display for GpuOptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canonicalize(error) => write!(f, "gpu_optimize canonicalize: {error}"),
            Self::ConstFold(error) => write!(f, "gpu_optimize const-fold: {error}"),
            Self::Dce(error) => write!(f, "gpu_optimize dce: {error}"),
            Self::AlgebraicIdentities(error) => {
                write!(f, "gpu_optimize algebraic-identities: {error}")
            }
        }
    }
}

impl std::error::Error for GpuOptimizeError {}

/// Run the complete optimizer in its canonical semantic stage order.
///
/// The same executor and immutable policy apply to every analysis kernel.
/// Schedule search, not this wrapper, selects static, persistent, fused, or
/// topology-aware execution for each validated semantic graph.
pub fn gpu_optimize(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<Program, GpuOptimizeError> {
    let program =
        gpu_canonicalize(program, executor, policy).map_err(GpuOptimizeError::Canonicalize)?;
    let program = gpu_const_fold(program, executor, policy).map_err(GpuOptimizeError::ConstFold)?;
    let program = gpu_dce(program, executor, policy).map_err(GpuOptimizeError::Dce)?;
    gpu_algebraic_identities(program, executor, policy)
        .map_err(GpuOptimizeError::AlgebraicIdentities)
}
