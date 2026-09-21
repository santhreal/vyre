//! The contract a backend implements to lower IR into its own representation.

use crate::error::IrResult as Result;

/// Anything that can be lowered to a target representation.
///
/// Backends implement this trait for their target. The IR does not know
/// what targets exist  -  it only knows that calling `.lower(&mut ctx)`
/// walks the structure through the visitor contract.
///
/// # Errors
///
/// Backends report structured errors through their own context type.
pub trait Lowerable<Ctx: ?Sized> {
    /// Visit this IR structure and emit into the backend-specific context.
    ///
    /// # Errors
    ///
    /// Returns the backend context's structured error when lowering cannot
    /// represent this IR structure.
    fn lower(&self, ctx: &mut Ctx) -> Result<()>;
}
