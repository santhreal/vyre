//! Vyre pass engine: optimizer analysis Programs executed through the semantic
//! compile-and-execute boundary.
//!
//! Each encoded pass builds a schedule-free Program graph, validates it with
//! explicit external facts, maps inputs by graph value identity, and applies
//! canonical outputs back to the IR. The pass engine contains no launch route,
//! persistence, grid, or backend selection policy.
//!
//! [`optimizer::pipeline::gpu_optimize`] runs canonicalization, constant
//! folding, dead-code elimination, and algebraic identities in one fixed
//! algorithmic order with a caller-supplied
//! `vyre_megakernel::SemanticExecutionPolicy`.

#[cfg(feature = "optimizer")]
/// The encoder plus the passes that run the compiler against its own
/// primitives. Exposed at the lib root so external consumers (driver
/// parity tests, conform runners) can reach the per-pass `*_via_encoded`
/// entry points and optimizer contract metadata without descending into
/// private module paths.
pub mod optimizer;
