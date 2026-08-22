//! Value-range analysis (phase 1).
//!
//! For each result-id, tracks the known integer range `[min, max]`
//! (inclusive) when statically derivable. Phase 1 is the minimum
//! viable: Lit-derived singletons (a Lit(U32(7)) op has range
//! `[7, 7]`) and trivial union via Min/Max BinOps.
//!
//! Future phases (not shipped):
//! - Add narrows from comparison-guarded branches.
//! - Add narrows from Add/Sub/Mul on known-bounded operands.
//! - Add SubgroupLocalId / LocalInvocationId range from
//!   dispatch.workgroup_size.
//! - F32 ranges (with NaN handling).
//!
//! Even phase 1 is useful: enables downstream rewrites to drop
//! bounds checks (`Lt(x, n)` with known x always < n) and to choose
//! efficient strength-reduce alternatives based on operand magnitude.

mod analysis;
mod carrier_staleness;
mod report;

pub use analysis::analyze;
pub use report::{IntRange, ValueRangeReport};

// Inline: covers the crate-private `analysis` and `carrier_staleness` submodules, which no integration test can reach.
#[cfg(test)]
mod tests;
