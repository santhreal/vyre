//! The contract an evaluator implements to execute IR against an environment.

use crate::error::IrResult as Result;

/// Anything that can be executed against a runtime environment.
///
/// The reference interpreter and each backend implement this trait. Two
/// `Evaluatable` implementations that produce the same output for the
/// same input + environment are certifiably equivalent under the
/// conform contract.
pub trait Evaluatable<Env: ?Sized> {
    /// The value type the evaluator produces (typically `Value` for the
    /// reference interpreter, a typed handle for GPU backends).
    type Value;

    /// Evaluate this IR structure against the environment.
    ///
    /// # Errors
    ///
    /// Returns the evaluator's structured error when the environment cannot
    /// execute this IR structure.
    fn evaluate(&self, env: &mut Env) -> Result<Self::Value>;
}
