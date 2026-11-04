//! Locating the lowered loop op a lowering test wants to assert about.
//!
//! Three lowering tests each carried their own copy of this descent, so
//! "where does lowering put the loop" was answered three times and each copy
//! could disagree about which body owns the op it returns.

#[cfg(test)]
use crate::descriptor::{KernelBody, KernelOp, KernelOpKind};

/// The first `StructuredForLoop` op in `body` or any descendant, with the body
/// that owns it.
///
/// The body comes back with the op because the loop's child index is an index
/// into that body's `child_bodies`, not into the root's.
#[cfg(test)]
pub(crate) fn find_loop(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
    for op in &body.ops {
        if matches!(op.kind, KernelOpKind::StructuredForLoop { .. }) {
            return Some((body, op));
        }
    }
    body.child_bodies.iter().find_map(find_loop)
}

// Inline: covers the crate-private `find_loop`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::find_loop;
    use crate::descriptor_builder::{body, for_loop, if_then, lit};

    /// The op comes back paired with the body that owns it, so its child index
    /// resolves against the right `child_bodies`.
    ///
    /// WHY: the three copies of this descent all returned a body, and a copy
    /// that returned the root instead would still find the op and then index
    /// the wrong vector, which reads as a lowering bug rather than a test bug.
    #[test]
    fn a_nested_loop_comes_back_with_the_body_that_owns_it() {
        let root = body()
            .op(lit(0, 0))
            .op(if_then(0, 0))
            .child(
                body()
                    .op(lit(0, 1))
                    .op(lit(0, 2))
                    .op(for_loop("i", 1, 2, 0))
                    .child(body().op(lit(0, 9))),
            )
            .build();

        let (owner, op) = find_loop(&root).expect("Fix: find_loop must descend into child bodies");
        let loop_body = &owner.child_bodies[op.operands[2] as usize];
        assert_eq!(loop_body.ops[0].result, Some(9));
    }

    #[test]
    fn a_body_with_no_loop_finds_nothing() {
        assert!(find_loop(&body().op(lit(0, 0)).build()).is_none());
    }
}
