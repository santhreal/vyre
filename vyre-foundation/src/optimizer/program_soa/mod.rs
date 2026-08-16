//! ROADMAP A2  -  columnar / SoA fact view of a `Program` that hot
//! optimizer passes can opt into.
//!
//! This is the *additive* shape of the same A1 contract. The
//! existing `Node` enum tree stays as the canonical IR; this module
//! ships a parallel `ProgramFacts` representation that walks the
//! tree once and stores per-Node payload in flat `Vec` columns. A
//! pass that needs to ask repeated "where is name `x` bound?" or
//! "every site that touches buffer `b`?" or "every Let in preorder"
//! questions builds `ProgramFacts` once (one tree walk, O(N)) and
//! then answers each question in O(1) lookup or O(K) over the
//! reply, instead of paying a fresh tree walk per query.
//!
//! ## Why columnar
//!
//! The hot optimizer queries fall into a small fixed set:
//!   - "every Let target name in this scope" (DCE, A14, A18)
//!   - "every Var read site of name `x`" (DCE liveness, CSE)
//!   - "every site that reads / writes / RMW-atomics buffer `b`"
//!     (alias-aware load elision, atomic minimization, store
//!     forwarding, dead-store elimination)
//!   - "every Node of kind `K`" (any pass that wants to skip when
//!     no candidate node is present)
//!
//! Each of these is a sequential scan over a single column when the
//! IR is laid out as struct-of-arrays. The cache footprint of one
//! column is dramatically smaller than a tree walk that touches
//! every Node enum tag, every Box pointer, every Arc indirection,
//! and every recursive child sequence  -  and the SoA columns are
//! contiguous, so a SIMD-aware scan is straightforward.
//!
//! ## What this module is NOT
//!
//! - Not a replacement for the `Node` enum. The enum stays the
//!   ground truth; `ProgramFacts` is a derived view that gets
//!   rebuilt when the program shape changes.
//! - Not a mutation API. Hot passes still rewrite the `Node` tree.
//!   They only use `ProgramFacts` for fast read-side queries.
//! - Not the GPU-resident A10 representation. The columns live in
//!   host memory; a future GPU mirror is a separate module.


mod build;
mod facts;
mod kind;

pub use facts::{ProgramFacts, RegionMeta};
pub use kind::{kind_bit, kind_mask, BufferRefKind, NodeIndex, NodeKind};
