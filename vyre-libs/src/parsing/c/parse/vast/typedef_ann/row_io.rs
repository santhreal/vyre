//! The row-table plumbing every typedef-annotation builder emits: how many
//! rows a node table is declared for, the buffer declarations that carry it,
//! and the copy-through-with-overrides store loop that writes one annotated
//! row back.
//!
//! Every builder in this module tree used to open-code all three. The store
//! loop alone existed in five copies, each free to forget a field, and each
//! declaration extent repeated the `max(1)` that keeps an empty input from
//! producing a zero-length buffer and a zero dispatch axis.

use super::*;

/// Rows a node table is declared for.
///
/// A literal node count sizes the declaration exactly. A runtime count is not
/// known at build time, so the declaration falls back to one row and the
/// emitted IR carries the real bound.
///
/// Never zero. An empty input would otherwise declare a zero-length buffer,
/// and the dispatch extent derived from that declaration has a zero axis,
/// which the CUDA launcher rejects outright rather than treating as no work.
pub(super) fn declared_rows(num_nodes: &Expr) -> u32 {
    node_count(num_nodes).max(1)
}

/// Words a node table of `rows` rows occupies.
///
/// Saturating: a hostile literal row count near `u32::MAX` must clamp rather
/// than wrap to a short length that the emitted IR then indexes past.
pub(super) fn row_table_words(rows: u32) -> u32 {
    rows.saturating_mul(VAST_NODE_STRIDE_U32)
}

/// Words a declaration-context table of `rows` rows occupies.
pub(super) fn decl_context_table_words(rows: u32) -> u32 {
    rows.saturating_mul(VAST_DECL_CONTEXT_STRIDE_U32)
}

/// Read-only node table.
pub(super) fn vast_nodes_input(name: &str, binding: u32, rows: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32)
        .with_count(row_table_words(rows))
}

/// Node table the kernel both reads and writes.
pub(super) fn vast_nodes_scratch(name: &str, binding: u32, rows: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32)
        .with_count(row_table_words(rows))
}

/// Node table the kernel only writes, readable back by the host.
pub(super) fn vast_nodes_output(name: &str, binding: u32, rows: u32) -> BufferDecl {
    BufferDecl::output(name, binding, DataType::U32).with_count(row_table_words(rows))
}

/// Read-only source haystack, sized for `haystack_len` bytes in the packing
/// the caller asked for.
pub(super) fn haystack_input(
    name: &str,
    binding: u32,
    haystack_len: &Expr,
    packed_haystack: bool,
) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32)
        .with_count(haystack_word_count(haystack_len, packed_haystack))
}

/// Read-only declaration-context side table.
pub(super) fn decl_contexts_input(name: &str, binding: u32, rows: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadOnly, DataType::U32)
        .with_count(decl_context_table_words(rows))
}

/// Declaration-context table the kernel both reads and writes.
pub(super) fn decl_contexts_scratch(name: &str, binding: u32, rows: u32) -> BufferDecl {
    BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32)
        .with_count(decl_context_table_words(rows))
}

/// Copy the whole VAST row at `base` from `vast_nodes` into `out`, taking each
/// overridden field from its carrier variable instead of from the input row.
///
/// Every annotation pass writes a full row: the fields it computes plus every
/// field it merely carries forward. Emitting that loop once is what keeps a
/// pass from silently dropping a field it does not care about, which downstream
/// passes read as a zeroed parent link or a lost symbol hash.
pub(super) fn store_row_with_overrides(
    out: &str,
    vast_nodes: &str,
    base: &Expr,
    overrides: &[(u32, &str)],
) -> Vec<Node> {
    (0..VAST_NODE_STRIDE_U32)
        .map(|field| {
            let offset = Expr::add(base.clone(), Expr::u32(field));
            let value = match overrides.iter().find(|(over, _)| *over == field) {
                Some((_, carrier)) => Expr::var(*carrier),
                None => Expr::load(vast_nodes, offset.clone()),
            };
            Node::store(out, offset, value)
        })
        .collect()
}
