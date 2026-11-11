//! Layer 3 complete compute engines.
//!
//! Each engine is a self-contained GPU compute pipeline: structured input
//! in, compute passes on a real GPU backend, typed output back.
//!
//! This module owns concrete backend execution helpers. Domain algorithms live
//! in `vyre-libs`; artifact admission and persistent execution live in
//! `vyre-runtime`.

/// Per-thread scratch arenas for record/readback hot-path vectors.
pub(crate) mod dispatch_scratch;
/// GPU-resident command graph execution.
pub mod graph;
/// Mockable multi-GPU work partitioning.
pub mod multi_gpu;
/// Shared command recording and readback for vyre IR dispatch paths.
pub(crate) mod record_and_readback;
