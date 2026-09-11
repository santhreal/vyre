# Lowerable historical reference

**Status: Superseded.** Use `Lowerable.txt` and
`vyre-driver/src/backend/lowering.rs` for the frozen contract.

pub trait Lowerable<Ctx: ?Sized> {
/// Visit this IR structure and emit into the backend-specific context.
fn lower(&self, ctx: &mut Ctx) -> Result<(), crate::error::Error>;
}
