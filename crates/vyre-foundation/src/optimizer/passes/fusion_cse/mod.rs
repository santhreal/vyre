//! Fusion + CSE / DCE catalog.
//!
//! Single-Program kernel fusion and classical CSE/DCE.

/// Common-subexpression elimination  -  engine + ProgramPass registration colocated.
pub mod cse;
/// Dead-code elimination  -  engine + ProgramPass registration colocated.
pub mod dce;
/// Kernel fusion by eliminating pure single-use scalar intermediates.
pub mod fusion;
