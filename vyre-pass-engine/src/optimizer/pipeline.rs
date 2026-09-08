//! Fixed semantic execution order for the optimizer pipeline.

use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionPolicy, SemanticExecutor};

use super::canonicalize_via_encoded::{gpu_canonicalize, CanonicalizeError};
use super::const_fold_via_encoded::{gpu_const_fold, ConstFoldError};
use super::cse_via_encoded::{
    apply_cross_scope_cse, apply_cse_let_dedupe, gpu_cse_canonicals, CseError,
};
use super::dce_via_encoded::{gpu_dce, DceError};
use super::pattern_match_via_encoded::{
    gpu_algebraic_identities_with_canonicals, PatternMatchError,
};

/// Errors surfaced by [`gpu_optimize`], classified by semantic stage.
#[derive(Debug)]
pub enum GpuOptimizeError {
    /// Canonicalization failed.
    Canonicalize(CanonicalizeError),
    /// Constant folding failed.
    ConstFold(ConstFoldError),
    /// Common-subexpression analysis failed.
    Cse(CseError),
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
            Self::Cse(error) => write!(f, "gpu_optimize cse: {error}"),
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
///
/// # Stage order
///
/// Canonicalization puts a commutative literal on the right so one rule shape
/// answers both operand orders. Constant folding then collapses the literal
/// subtrees, which is what lets the identity rules see a literal operand at all.
///
/// One canonical-id analysis serves the three consumers of structural equality:
/// the identity rules whose premise is that two operands are one value, the
/// let-level dedupe, and the cross-scope hoist. They share one arena and one
/// table, so all three index the same Expr ids. Only expression shape changes
/// between them, and the table is keyed on node-level structure, which those
/// rewrites preserve.
///
/// Dead-code elimination runs on either side of the host rewrites, and both runs
/// are load-bearing. Before them, retiring a dead binding is what empties a loop
/// body or a branch so the structural rewrites can drop the wrapper. After them,
/// propagation has replaced the only read of a binding with its literal, and the
/// binding it left behind is dead.
pub fn gpu_optimize(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<Program, GpuOptimizeError> {
    let program =
        gpu_canonicalize(program, executor, policy).map_err(GpuOptimizeError::Canonicalize)?;
    let program = gpu_const_fold(program, executor, policy).map_err(GpuOptimizeError::ConstFold)?;

    let (arena, canonical) =
        gpu_cse_canonicals(&program, executor, policy).map_err(GpuOptimizeError::Cse)?;
    let program = if arena.expr_count == 0 {
        program
    } else {
        let program = gpu_algebraic_identities_with_canonicals(
            program,
            &arena,
            &canonical,
            executor,
            policy,
        )
        .map_err(GpuOptimizeError::AlgebraicIdentities)?;
        let program = apply_cse_let_dedupe(&program, &arena, &canonical);
        apply_cross_scope_cse(&program, &arena, &canonical)
    };

    let mut program = gpu_dce(program, executor, policy).map_err(GpuOptimizeError::Dce)?;
    for rewrite in vyre_foundation::transform::HOST_REWRITES {
        program = (rewrite.apply)(&program);
    }
    gpu_dce(program, executor, policy).map_err(GpuOptimizeError::Dce)
}
