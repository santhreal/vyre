//! `dominator_tree`  -  exact immediate-dominator primitive.
//!
//! Computes the immediate dominator (`idom`) of every reachable node in a
//! control-flow graph with a single entry.  The primitive ships both a
//! Lengauer–Tarjan CPU reference oracle and a serial lane-0 GPU `Program`
//! builder that implements the Cooper–Harvey–Kennedy iterative fixpoint
//! using parent-pointer LCA on the idom tree.
//!
//! # Wire shape
//!
//! ```text
//! pg_edge_offsets : u32[node_count + 1]   // forward CSR
//! pg_edge_targets : u32[edge_count]       // forward CSR
//! pred_offsets    : u32[node_count + 1]   // predecessor CSR
//! pred_targets    : u32[pred_edge_count]  // predecessor CSR
//! idom_out        : u32[node_count]       // output idoms; NONE = unreachable
//! ```
//!
//! `idom_out[entry] == entry` for the entry block.  Unreachable nodes keep
//! the sentinel `NONE` (== `node_count`).
//!
//! # Soundness
//!
//! Exact for every reducible and irreducible single-entry CFG.  Multi-entry
//! graphs (no path from entry to some node that has predecessors) are not
//! rejected explicitly, but the resulting idom tree is undefined for the
//! disconnected component; callers should run `reachable` first if they need
//! strict guarantees.

mod depth;
mod intersect_step;
mod program;





mod registry;

#[cfg(test)]
#[path = "../../../tests/internal/graph/dominator_tree/mod.rs"]
mod tests;

pub use depth::{
    dominator_tree_depth, dominator_tree_depth_body, dominator_tree_depth_child,
    OP_ID as DOMINATOR_TREE_DEPTH_OP_ID,
};
pub use intersect_step::{
    dominator_tree_intersect_step, dominator_tree_intersect_step_body,
    dominator_tree_intersect_step_child, OP_ID as DOMINATOR_TREE_INTERSECT_STEP_OP_ID,
};
pub use program::{
    dominator_tree_program, try_dominator_tree_program, validate_dominator_tree_inputs,
    DominatorTreeError, DominatorTreeLayout, IDOM_NONE, OP_ID,
};
