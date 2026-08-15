//! The named-field inputs every CSR closure entry point reads.
//!
//! Iterating a masked CSR traversal to a bounded fixpoint needs the same seven
//! values wherever it happens: the four CSR arrays, the edge-kind allow mask,
//! and the iteration budget. Every entry point in the subsystem used to restate
//! that list positionally, which cost twice.
//!
//! The first cost is drift. Nine restatements of one list are nine places to
//! gain a bound, an attribute, or a fix that the other eight miss, and the
//! closure entry points had already drifted that way: one spelled the scalar
//! allow mask `edge_kind_mask`, the name its siblings give the per-edge ARRAY.
//!
//! The second cost is silent transposition. `edge_offsets`, `edge_targets`,
//! `edge_kind_mask` and the frontier are four consecutive `&[u32]` parameters,
//! so swapping any two of them compiles and then produces a wrong closure that
//! only a differential oracle can catch. Named fields turn every one of those
//! swaps into a compile error, and [`CsrGraphView`] is the one place the four
//! CSR arrays are named.
//!
//! The seed stays a separate argument. It is not part of the group: a launch
//! planner validates the graph, the mask and the budget without ever seeing a
//! frontier, so folding the seed in would force those callers to invent one.

/// The CSR arrays one graph traversal walks.
///
/// `edge_offsets`, `edge_targets` and `edge_kind_mask` are the transposition
/// hazard this type exists to remove: three same-typed slices whose order is
/// load-bearing and whose contents are indistinguishable at a call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrGraphView<'a> {
    /// Number of graph nodes. Bounds every node index and the frontier width.
    pub node_count: u32,
    /// Row starts, `node_count + 1` entries, monotonic, first entry zero.
    pub edge_offsets: &'a [u32],
    /// Destination node of each edge, indexed by the row ranges above.
    pub edge_targets: &'a [u32],
    /// Edge-kind bitmask of each edge, parallel to `edge_targets`.
    pub edge_kind_mask: &'a [u32],
}

/// A CSR closure's graph, edge filter and iteration budget.
///
/// The seed frontier travels as its own argument because the planning entry
/// points that consume this type never read one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrClosureInputs<'a> {
    /// CSR arrays the closure walks each iteration.
    pub graph: CsrGraphView<'a>,
    /// Edges whose `edge_kind_mask` intersects this mask are traversable.
    pub allow_mask: u32,
    /// Upper bound on traversal steps. Zero runs no step at all.
    pub max_iters: u32,
}
