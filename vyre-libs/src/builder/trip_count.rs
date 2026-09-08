//! The single clamp every data-derived loop bound in this crate passes through.

use vyre_foundation::ir::Expr;

/// Clamp a loop bound read from a buffer to the extents of the buffers the loop
/// body indexes.
///
/// A `Node::loop_for` whose bound is a load runs for as many iterations as that
/// value states, so one out-of-contract `u32` asks for four billion: hours in
/// the reference interpreter, a watchdog reset on a device. An out-of-bounds
/// store check inside a body does not bound the loop, because it discards the
/// result of an iteration that already ran.
///
/// Every producer contract in this crate keeps such a bound at or below the
/// extent it is clamped against, so the clamp never fires on in-contract input
/// and the result is unchanged. Where the loop variable indexes an extent
/// directly, a clamped iteration could additionally only have read or written a
/// slot that does not exist.
///
/// `indexed` is the first buffer and `also_indexed` the rest, so an empty
/// extent list is a type error rather than a clamp that is not one.
#[must_use]
pub(crate) fn clamped_by_extents<const N: usize>(
    requested: Expr,
    indexed: &str,
    also_indexed: [&str; N],
) -> Expr {
    let bound = also_indexed
        .into_iter()
        .fold(Expr::buf_len(indexed), |acc, name| {
            Expr::min(acc, Expr::buf_len(name))
        });
    Expr::min(requested, bound)
}
