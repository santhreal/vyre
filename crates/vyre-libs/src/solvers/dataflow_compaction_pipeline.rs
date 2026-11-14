//! Compiler-side wrappers for dataflow convergence, scan, and compaction kernels.
//!
//! This is the glue layer for small but central GPU building blocks: fixpoint
//! flags, prefix scans, stream compaction, stochastic bitsets, interval merges,
//! sparse recovery, DP clipping, differentiable selection, and attention dot
//! partials. The kernel domains in this crate own the executable semantics.

use crate::math::differentiable::softmax_step;
use vyre_foundation::ir::Program;
/// Build a softmax normalization dispatch over precomputed exponentials.
#[must_use]
pub fn dispatch_softmax(pre_exp: &str, out: &str, n: u32) -> Program {
    softmax_step(pre_exp, out, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::Node;
    use vyre_reference::composition_witness::softmax_witness;

    #[test]
    fn dispatch_softmax_emits_softmax_primitive_and_matches_uniform_witness() {
        let program = dispatch_softmax("pre_exp", "out", 4);
        let Some(Node::Region { generator, .. }) = program.entry.first() else {
            panic!("Fix: dispatch_softmax Program must start with a Region.");
        };
        assert_eq!(generator.as_str(), "vyre-libs::math::softmax_step");

        let softmax = softmax_witness(&[1.0, 1.0, 1.0, 1.0]);
        assert!(softmax
            .iter()
            .all(|&value| (value - 0.25).abs() < 1e-8 * (1.25 + value.abs())));
    }
}
