//! The gates that judge the tree against the live operation registry.
//!
//! Each of these needs a registration to exist at run time: the registered
//! operation set, the primitive catalog those operations compose from, or the
//! rewrite proofs the optimizer submits. A gate that answers its question from
//! source text alone belongs in `xtask::gates`, which links none of this.

pub mod abstraction_gate;
pub mod gate1;
pub mod heuristic_audit;
pub mod lego_audit;
pub mod lego_quick;
pub mod verify_rewrite_proofs;
pub mod whats_similar;
