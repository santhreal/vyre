//! The one lowering-node builder.
//!
//! Both lowering entry points decode the same VAST row fields into the same
//! GPU let-bindings and pack the same six ProgramGraph columns out of them.
//! The semantic lowerer adds two attribute fields and four semantic columns on
//! top; the six structural columns below are the same rows in both, and are
//! written twice by the combined pass (once to the plain PG buffer, once as
//! the head of the semantic row).

use super::*;

/// One packed VAST row addressed by its base word index.
pub(super) struct VastRow<'a> {
    /// Buffer holding packed VAST rows.
    pub vast_nodes: &'a str,
    /// Word index of this invocation's row.
    pub base: Expr,
}

impl VastRow<'_> {
    fn field(&self, index: usize) -> Expr {
        let offset = if index == 0 {
            self.base.clone()
        } else {
            Expr::add(self.base.clone(), Expr::u32(index as u32))
        };
        Expr::load(self.vast_nodes, offset)
    }

    /// The six row fields every lowered ProgramGraph node row is built from.
    pub(super) fn structural_bindings(&self) -> Vec<Node> {
        vec![
            Node::let_bind("kind", self.field(IDX_KIND)),
            Node::let_bind("parent_idx", self.field(IDX_PARENT)),
            Node::let_bind("first_child_idx", self.field(IDX_FIRST_CHILD)),
            Node::let_bind("next_sibling_idx", self.field(IDX_NEXT_SIBLING)),
            Node::let_bind("span_start", self.field(IDX_SRC_BYTE_OFF)),
            Node::let_bind("span_len", self.field(IDX_SRC_BYTE_LEN)),
        ]
    }

    /// The two attribute fields only the semantic row carries downstream.
    pub(super) fn attribute_bindings(&self) -> Vec<Node> {
        vec![
            Node::let_bind("attr_off", self.field(IDX_ATTR_OFF)),
            Node::let_bind("attr_len", self.field(IDX_ATTR_LEN)),
        ]
    }
}

/// The six packed ProgramGraph columns
/// `(kind, span_start, span_end, parent_idx, first_child_idx, next_sibling_idx)`
/// that every lowered node row starts with.
///
/// Reads the bindings [`VastRow::structural_bindings`] emits, so it must be
/// placed after them in the same scope.
pub(super) fn store_pg_node_row(out_pg_nodes: &str, row_base: &Expr) -> Vec<Node> {
    [
        Expr::var("kind"),
        Expr::var("span_start"),
        Expr::add(Expr::var("span_start"), Expr::var("span_len")),
        Expr::var("parent_idx"),
        Expr::var("first_child_idx"),
        Expr::var("next_sibling_idx"),
    ]
    .into_iter()
    .enumerate()
    .map(|(column, value)| {
        let index = if column == 0 {
            row_base.clone()
        } else {
            Expr::add(row_base.clone(), Expr::u32(column as u32))
        };
        Node::store(out_pg_nodes, index, value)
    })
    .collect()
}
