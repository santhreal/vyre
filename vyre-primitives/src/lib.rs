//! `vyre-primitives`: the operations that cannot be composed.
//!
//! An operation is admitted here only when it cannot be expressed as a
//! composition, which means it requires its own arm in a backend emitter and
//! its own arm in the reference interpreter. That is the whole rule. How many
//! callers an operation has is not an admission criterion: something reused by
//! twenty dialects and expressible as `fn(..) -> Program` over existing IR is a
//! composition, and a composition belongs in `vyre-libs` no matter who calls
//! it.
//!
//! Two kinds of item satisfy that rule.
//!
//! 1. **Marker types** (`markers`, always on, no dependencies). The unit
//!    structs a backend emitter and the reference interpreter dispatch on. They
//!    are the names those two arms agree about, so they cannot live in a crate
//!    either side composes over.
//!
//! 2. **Hardware intrinsics** (`hardware`). Subgroup collectives, memory
//!    fences, bit instructions, fused multiply-add, inverse square root: each
//!    one is a target instruction with no IR spelling.
//!
//! The remaining domains are the substrate those two rest on: the wire format,
//! the safe-IR guards, the launch-geometry helper, and the VFS DMA kernels.
//! Every composition domain that used to be parked here now lives in
//! `vyre-libs`.
//!
//! ```text
//! vyre-primitives/
//!   src/
//!     lib.rs              # this table
//!     markers.rs          # marker types, always on
//!     wire.rs             # host/device byte layout, always on
//!     dispatch_grid.rs    # lane count to dispatch grid, always on
//!     ir_safe.rs          # guarded IR construction
//!     hardware/           # feature = "hardware"
//!     vfs/                # feature = "vyre-foundation"
//! ```
//!
//! The path is the interface. A domain `mod.rs` exposes its sub-modules rather
//! than a flat namespace, so a call site reads
//! `vyre_primitives::hardware::subgroup_add(..)` and names the operation it
//! reached.

mod dispatch_grid;
#[cfg(feature = "vyre-foundation")]
pub mod ir_safe;
mod markers;
pub mod wire;

pub use dispatch_grid::lane_grid;

/// One classification of every Cargo feature this crate declares.
///
/// A domain feature that is in neither list is how a third admission category
/// returns, so the module's own tests hold the lists to the manifest. The lists
/// are crate-private: they describe which compositions are still parked here,
/// and nothing outside may depend on that.
mod organization;

pub use markers::{
    ArithAdd, ArithMul, BitwiseAnd, BitwiseOr, BitwiseXor, Clz, CombineOp, CompareEq, CompareLt,
    Gather, HashBlake3, HashFnv1a, PatternMatchDfa, PatternMatchLiteral, Popcount, Reduce,
    RegionId, Scan, Scatter, ShiftLeft, ShiftRight, Shuffle,
};


/// Derived view over canonical primitive operation registrations.
#[cfg(feature = "inventory-registry")]
pub mod operation_catalog;

/// Category C hardware intrinsics. Ops that need a dedicated backend emitter
/// arm and a dedicated reference-interpreter arm: subgroup collectives,
/// memory fences, bit instructions, fused multiply-add, inverse square root.
#[cfg(feature = "hardware")]
pub mod hardware;

/// Virtual File System DMA primitives. Uses `vyre_foundation::ir`, so it is
/// gated behind the feature that pulls vyre-foundation in as an optional dep.
#[cfg(feature = "vyre-foundation")]
pub mod vfs;
