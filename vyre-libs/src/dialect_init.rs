//! Making this crate's ops resolvable in the current process.

/// Install the driver's dialect registry as the process-wide op lookup.
///
/// Some builders in this crate return a program containing `Expr::Call` nodes
/// that name other ops this crate registers. Resolving those names, whether to
/// inline them before lowering or to validate the program, goes through the
/// process-wide dialect lookup, and nothing installs that lookup until someone
/// asks the driver for its registry.
///
/// Call this from any builder that emits a call, so the program it hands back
/// works for a caller who never touches `vyre-driver` directly. The install is
/// idempotent and cheap after the first time.
#[inline]
pub fn ensure_ops_resolvable() {
    let _ = ::vyre_driver::registry::DialectRegistry::global();
}
