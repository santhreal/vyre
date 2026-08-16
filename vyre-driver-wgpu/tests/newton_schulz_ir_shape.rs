//! The Newton-Schulz op's IR shape stays small enough for the wgpu emitter.
//!
//! # Why this lives in the driver crate
//!
//! `newton_schulz_5step` composes five iterations of a degree-5 polynomial. Emit
//! it by cloning the polynomial tree per iteration and the expression count grows
//! multiplicatively, which the naga lowering pays for in compile time rather than
//! in a wrong answer: nothing fails, the shader just takes minutes to build. So
//! the pin is on the shape the emitter is handed, and the proof is that the
//! program it is handed still lowers and validates.
//!
//! # Why it counts with the foundation walker
//!
//! The count comes from `vyre_foundation::visit::walk_exprs`. A
//! hand-rolled counter here would be a second traversal of a `#[non_exhaustive]`
//! enum written outside the crate that declares it: its catch-all arm makes a
//! variant added tomorrow read as a leaf, so a tree that grew through the new
//! variant would count as small and this pin would pass on it. Inside
//! `vyre-foundation` the match is exhaustive, which is the only place a traversal
//! of this IR cannot silently stop descending.

mod harness;

use vyre_foundation::visit::walk_exprs;

/// Expression-node ceiling for the five-iteration composition.
///
/// The measured shape is well under this. The ceiling is set at the order of
/// magnitude that separates shared let-bound SSA from a cloned tree, not at the
/// measurement, so an ordinary retuning of the op does not move it and a
/// recursive clone cannot fit under it.
const MAX_EXPR_NODES: usize = 128;

/// Statement-node ceiling: this is a fixed-size Category-A composition.
const MAX_NODES: usize = 32;

#[test]
fn newton_schulz_ir_shape_stays_linear() {
    let program = vyre_libs::nn::optim::newton_schulz_5step("mat", "output", 2, 2);

    let mut expr_nodes = 0usize;
    walk_exprs(&program, |_| expr_nodes += 1);

    assert!(
        expr_nodes <= MAX_EXPR_NODES,
        "Fix: newton_schulz_5step must emit shared let-bound SSA expressions, not recursively \
         clone the polynomial tree; expr_nodes={expr_nodes} exceeds {MAX_EXPR_NODES}"
    );
    assert!(
        program.stats().node_count <= MAX_NODES,
        "Fix: newton_schulz_5step should remain a small fixed-size Cat-A composition; nodes={} \
         exceeds {MAX_NODES}",
        program.stats().node_count
    );
}

/// The pinned shape is the shape the wgpu emitter is actually handed.
///
/// A shape pin on its own proves nothing about lowering: the op could stay small
/// and still fail to emit. Lowering it here is what makes the ceiling above a
/// statement about this backend's compile cost rather than about the IR alone.
#[test]
fn newton_schulz_lowers_through_the_wgpu_emitter() {
    let program = vyre_libs::nn::optim::newton_schulz_5step("mat", "output", 2, 2);
    let wgsl = harness::emit_validated_wgsl(&program);
    assert!(
        wgsl.contains("fn main"),
        "Fix: the emitted module must carry a compute entry point; got {} bytes of WGSL",
        wgsl.len()
    );
}
