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

impl<'a> CsrClosureInputs<'a> {
    /// Traverse `graph` with every edge kind allowed, bounded by `max_iters`.
    ///
    /// An unrestricted allow mask is what a closure caller wants whenever the
    /// property under test is not the kind filter, which is most of them: the
    /// mask was spelled `0xFFFF_FFFF` or `u32::MAX` at more than thirty call
    /// sites, each restating the whole seven-field group around it. A caller
    /// that IS testing the filter still writes the struct literal, and the
    /// difference now reads at a glance.
    #[must_use]
    pub const fn allow_all(graph: CsrGraphView<'a>, max_iters: u32) -> Self {
        Self {
            graph,
            allow_mask: u32::MAX,
            max_iters,
        }
    }
}

/// A CSR graph shape with its arrays owned for the program's lifetime.
///
/// The four arrays of a fixed test graph are `'static` data, but
/// [`CsrGraphView`] borrows them, so a caller cannot return one from a helper
/// without keeping three locals alive. That is why the same three arrays were
/// built into locals named `off`/`tgt`/`msk` at twenty-one call sites and
/// `offsets`/`targets`/`masks` at fourteen more, each followed by the same
/// restated view. A shape holds them, and [`CsrGraphShape::view`] borrows from
/// the shape rather than from the caller's stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrGraphShape {
    /// Number of graph nodes.
    pub node_count: u32,
    /// Row starts, `node_count + 1` entries.
    pub edge_offsets: &'static [u32],
    /// Destination node of each edge.
    pub edge_targets: &'static [u32],
    /// Edge-kind bitmask of each edge.
    pub edge_kind_mask: &'static [u32],
}

impl CsrGraphShape {
    /// Borrow this shape as the view every closure entry point reads.
    #[must_use]
    pub const fn view(&self) -> CsrGraphView<'_> {
        CsrGraphView {
            node_count: self.node_count,
            edge_offsets: self.edge_offsets,
            edge_targets: self.edge_targets,
            edge_kind_mask: self.edge_kind_mask,
        }
    }

    /// Frontier words this shape's node count needs.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.node_count.div_ceil(32) as usize
    }
}

/// The fixed CSR graphs the closure contracts are written against.
///
/// Every closure entry point in the subsystem is pinned on one of these two
/// four-node shapes, and each was rebuilt per test from three array literals.
/// Literals that agree by coincidence are literals that can stop agreeing: a
/// contract that means "the chain" and a contract that means "the diamond" are
/// distinguishable here and were not before.
pub mod graphs {
    use super::CsrGraphShape;

    /// `0 -> 1 -> 2 -> 3`, one edge kind. Reaching node 3 takes three steps, so
    /// this is the shape an iteration budget is observable on.
    pub const CHAIN_4: CsrGraphShape = CsrGraphShape {
        node_count: 4,
        edge_offsets: &[0, 1, 2, 3, 3],
        edge_targets: &[1, 2, 3],
        edge_kind_mask: &[1, 1, 1],
    };

    /// `0 -> 1`, `0 -> 2`, `1 -> 3`, `2 -> 3`, one edge kind. Node 3 is reached
    /// twice in one step, so this is the shape a duplicate-arrival bug shows up
    /// on and the chain does not.
    pub const DIAMOND_4: CsrGraphShape = CsrGraphShape {
        node_count: 4,
        edge_offsets: &[0, 2, 3, 4, 4],
        edge_targets: &[1, 2, 3, 3],
        edge_kind_mask: &[1, 1, 1, 1],
    };
}
