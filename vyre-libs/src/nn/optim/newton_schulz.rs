//! Newton-Schulz 5-step orthogonalization (F32).
//!
//! `X_{k+1} = a*X_k + b*X_k@X_k^T@X_k + c*X_k@X_k^T@X_k@X_k^T@X_k`
//! Coefficients: a=3.4445, b=-4.7750, c=2.0315.
//!
//! Used by Muon optimizer. This is a multi-pass matmul composition.

use crate::math::preconditioner::newton_schulz_poly5_f32;
use vyre_foundation::composition::tag_program;
use vyre_foundation::ir::Program;

const OP_ID: &str = "vyre-libs::optim::newton_schulz_5step";

/// Newton-Schulz orthogonalization polynomial applied for five iterations.
#[must_use]
pub fn newton_schulz_5step(mat: &str, output: &str, rows: u32, cols: u32) -> Program {
    tag_program(OP_ID, newton_schulz_poly5_f32(mat, output, rows, cols))
}

const EXPECTED_NEWTON_SCHULZ_OUTPUT_BYTES: [u8; 16] = [
    0xC6, 0xF3, 0x43, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC6, 0xF3, 0x43, 0x3F,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || newton_schulz_5step("mat", "output", 2, 2),
        Some(|| {
            let to_f32 = |w: &[f32]| vyre_primitives::wire::pack_f32_slice(w);
            vec![vec![
                to_f32(&[0.5, 0.0, 0.0, 0.5]),
            ]]
        }),
        Some(|| vec![vec![EXPECTED_NEWTON_SCHULZ_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("nn")
    .with_numeric(vyre_foundation::numeric::NumericContract::ieee_f32(64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::visit::walk_exprs;

    /// Expression-node ceiling for the five-iteration composition.
    ///
    /// Set at the order of magnitude that separates shared let-bound SSA from a
    /// polynomial tree cloned once per iteration, not at the measurement, so
    /// retuning the op does not move it and a cloned tree cannot fit under it.
    const MAX_EXPR_NODES: usize = 128;

    /// Statement-node ceiling: this is a fixed-size Category-A composition.
    const MAX_NODES: usize = 32;

    /// The count comes from `walk_exprs` because `Expr` and `Node` are
    /// `#[non_exhaustive]` outside `vyre-foundation`: a traversal written here
    /// needs a catch-all arm, and that arm reads a variant added tomorrow as a
    /// leaf. A tree that grew through the new variant would then count as small
    /// and this ceiling would pass on it. The walker's match is exhaustive in
    /// the crate that declares the enums, so it cannot stop descending.
    #[test]
    fn emitted_expression_tree_stays_linear_in_iterations() {
        let program = newton_schulz_5step("mat", "output", 2, 2);

        let mut expr_nodes = 0usize;
        walk_exprs(&program, |_| expr_nodes += 1);

        assert!(
            expr_nodes <= MAX_EXPR_NODES,
            "Fix: newton_schulz_5step must emit shared let-bound SSA expressions, not \
             recursively clone the polynomial tree; expr_nodes={expr_nodes} exceeds \
             {MAX_EXPR_NODES}"
        );
        assert!(
            program.stats().node_count <= MAX_NODES,
            "Fix: newton_schulz_5step should remain a small fixed-size Cat-A composition; \
             nodes={} exceeds {MAX_NODES}",
            program.stats().node_count
        );
    }
}
