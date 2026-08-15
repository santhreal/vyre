//! Parked composition (belongs in vyre-libs): neural-network primitives.

/// Reusable attention score / normalization passes.
pub mod attention_passes;
/// Shared attention numeric-stability guards.
pub mod attention_stability;
/// Shared F32 numeric-stability guards.
pub mod f32_stability;

/// Reusable Quest-style KV paging passes.
pub mod quest_paging_passes;
