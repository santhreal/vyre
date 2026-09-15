//! The resident pipeline's entry into loop-invariant code motion.
//!
//! The rule itself lives in [`crate::optimizer::passes::loops::loop_licm`],
//! which is the one owner of what may leave a loop body. This module exists
//! because the resident pipeline drives rewrites as `fn(&Program) -> Program`
//! while the pass engine drives them as `Program -> PassResult`.
//!
//! A second implementation stood here and answered a narrower question with a
//! different set of rules: it hoisted a binding whose name was also bound by a
//! sibling loop, which flat-splices two bindings of that name into one scope
//! and the validator rejects the result (V032). Keeping both meant a program
//! could be rewritten one way by the resident pipeline and another by the pass
//! engine.

use crate::ir::Program;
use crate::optimizer::passes::loops::loop_licm::LoopLicm;

/// Hoist every loop-invariant binding out of its loop.
///
/// Returns the program unchanged when it holds no loop, which is what the
/// resident pipeline reads as a declined rewrite.
#[must_use]
pub fn apply_licm(program: &Program) -> Program {
    if !program.stats().has_node_loop() {
        return program.clone();
    }
    LoopLicm::transform(program.clone()).program
}
