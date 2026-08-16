//! The seam a Tier 3 dialect crosses to reach a sibling dialect.
//!
//! Each directory under `src/` is one dialect. A dialect owns its own surface
//! and depends downward on `vyre-primitives`; it does not reach sideways into
//! another dialect's module tree. Three compositions genuinely need a sibling's
//! work: a linear layer is a bias-matmul, a tiled linear layer is a tiled
//! bias-matmul, and rule-graph change impact is a reachability closure.
//!
//! Those edges are re-exported here, so the coupling between dialects is one
//! list at the crate root rather than a path reaching three levels into another
//! dialect from a file nothing collects. `lego-audit` check 4 reports a
//! `use crate::<other_dialect>::...` inside a dialect as a finding; an edge that
//! belongs is added to this module and imported from here.
//!
//! This is not a glob-import convenience. Nothing belongs here except an item
//! one dialect composes from another.

#[cfg(feature = "analysis")]
pub use crate::analysis::dataflow_fixpoint::reachability_closure_via_into;
#[cfg(feature = "math-linalg")]
pub use crate::math::linalg::{MatmulBias, MatmulBiasTiled};
