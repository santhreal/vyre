//! The expression arena's node numbering, for the passes that index it.
//!
//! `expr_arena::encode_node` allocates one `node_top_level_exprs` slot per node
//! it encodes, in DFS prefix order over the reachable prefix of every scope. A
//! pass that reads a per-expression GPU verdict has to walk the IR and land on
//! the same slot the encoder did, or it applies one node's verdict to another.
//!
//! That numbering is the encoder's, not the IR's, so it lives here rather than
//! in the shared node walk. Which positions of a node exist is the IR's and
//! comes from `vyre_foundation::transform::rewrite_walk`; this type only says
//! which arena slot the walk is standing on.

use vyre_foundation::ir::Node;
use vyre_foundation::visit::child_bodies;

/// A cursor over `ExprArenaEncoding::node_top_level_exprs`.
pub(super) struct ArenaCursor<'a> {
    node_top_level_exprs: &'a [Vec<u32>],
    node_index: usize,
}

impl<'a> ArenaCursor<'a> {
    /// Start at the first encoded node. Slot 0 is the synthetic ROOT, which
    /// carries no expressions of its own.
    pub(super) fn at_first_real_node(node_top_level_exprs: &'a [Vec<u32>]) -> Self {
        Self {
            node_top_level_exprs,
            node_index: 1,
        }
    }

    /// The slot the cursor is standing on.
    pub(super) fn position(&self) -> usize {
        self.node_index
    }

    /// Return to a slot recorded by [`Self::position`], so a second pass over
    /// the same nodes reads the same ids the first pass did.
    pub(super) fn rewind_to(&mut self, position: usize) {
        self.node_index = position;
    }

    /// Consume this node's slot and hand back its top-level arena expr ids.
    ///
    /// An id list shorter than the node's operand count means the encoder
    /// rejected an operand, and the caller leaves that position alone.
    pub(super) fn take_node(&mut self) -> Vec<u32> {
        let ids = self
            .node_top_level_exprs
            .get(self.node_index)
            .cloned()
            .unwrap_or_default();
        self.node_index += 1;
        ids
    }

    /// Advance past a whole subtree without reading it.
    ///
    /// Child bodies come from the single exhaustive owner, so a nesting variant
    /// added to `Node` cannot silently leave the cursor short by a subtree,
    /// which would misalign every verdict after it.
    pub(super) fn skip_body(&mut self, body: &[Node]) {
        for node in &body[..super::encode::reachable_prefix_len(body)] {
            self.node_index += 1;
            for nested in child_bodies(node) {
                self.skip_body(nested);
            }
        }
    }
}
