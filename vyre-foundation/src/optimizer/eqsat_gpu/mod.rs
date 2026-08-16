//! GPU-resident e-graph substrate.
//!
//! The CPU-side `eqsat::EGraph` materialises rewrite candidates,
//! union-find merges, and cost-based extraction in a single
//! sequential walker. For wide rewrite families (algebraic
//! identities, peephole tables, pattern-match-heavy primitives)
//! the per-iteration cost grows with the e-graph size, and the
//! hash-cons table becomes the bottleneck. This module ships the
//! GPU-resident representation: a flattened, columnar mirror of
//! the EGraph that can be uploaded to a GPU buffer and walked in
//! parallel by warp-cooperative passes.
//!
//! The mirror is additive: CPU passes keep using `EGraph::saturate`,
//! while GPU-aware passes use `GpuEGraphSnapshot::from_egraph_with`
//! to materialise the columnar arrays and merge discovered equivalences
//! back through `apply_equivalences_to_egraph`.
//!
//! Soundness: the snapshot is read-only. Any equivalence the GPU
//! discovers is merged through the same `EGraph::merge` API the
//! CPU uses, so the EGraph's saturation invariants hold by
//! construction.
//!
//! ## Why the columnar layout
//!
//! Each row of the snapshot is `(eclass_id, language_op_id,
//! children_offset, children_len)`. The children indices live in
//! a separate `children: Vec<u32>` column. This layout fits a
//! GPU's coalesced-memory access pattern: a warp reading 32
//! consecutive rows touches one cache line per column (4 columns
//! × 4 bytes × 32 lanes = 512 bytes per warp).

mod apply;
mod bridge;
mod device_image;
mod error;
mod signature;
mod snapshot;

pub use apply::{apply_equivalences, apply_equivalences_to_egraph, ApplyEquivalencesReport};
pub use bridge::{bridge_equivalence_batch_with_report, GpuEGraphBridgeReport};
pub use device_image::{GpuEGraphDeviceImage, GpuEGraphDeviceLayout, GpuEGraphDeviceSpan};
pub use error::{
    GpuEGraphBridgeError, GpuEGraphDeviceImageError, GpuEGraphSnapshotError,
    GpuEGraphSnapshotIntegrityError,
};
pub use signature::gpu_egraph_row_signature;
pub use snapshot::{Equivalence, GpuEGraphSnapshot, OpIdRegistry, SnapshotRow};
