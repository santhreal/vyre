//! Release checks that compare a manifest against the live registry.
//!
//! Each of these reads a release manifest and then asks the registry whether
//! what the manifest claims is actually registered. The manifest-only halves of
//! the same checks live in `xtask::release`.

pub mod conformance_matrix;
pub mod optimization_corpus;
pub mod optimization_matrix;
