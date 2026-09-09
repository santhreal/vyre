//! Contraction candidate selection and lowering analysis.
//!
//! Logical contraction IR lowers into measured scalar, SIMD/SIMT tiled, and
//! target matrix-instruction candidates without domain ownership.

pub(crate) mod analysis;
pub(crate) mod plan;

pub use analysis::analyze;
pub use plan::{ContractionCandidate, ContractionPlan, ContractionStrategy};
