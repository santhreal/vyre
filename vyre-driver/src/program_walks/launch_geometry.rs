//! Launch-geometry dependency walks.

use vyre_foundation::ir::{Expr, Program};
use vyre_foundation::visit::any_expr_in;

/// True when the program reads launch geometry that makes workgroup shape
/// semantically visible to the kernel body.
///
/// Node nesting, operand positions, and sub-expressions all come from the
/// owning enumerations in `vyre-foundation`, so a new `Node` or `Expr` variant
/// reaches this scan without an edit here. The three hand-written matches this
/// replaces each ended in a catch-all arm, which answered "no launch-geometry
/// dependency" for any variant they had not been told about; the caller then
/// coerced a dispatch grid the kernel could observe.
#[must_use]
pub(crate) fn program_uses_launch_geometry_ids(program: &Program) -> bool {
    any_expr_in(program.entry(), &mut |expr| {
        matches!(expr, Expr::LocalId { .. } | Expr::WorkgroupId { .. })
    })
}
