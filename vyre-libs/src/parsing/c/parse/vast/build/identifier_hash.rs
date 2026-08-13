use super::*;

/// Names the emitted FNV-1a identifier hash binds in the caller's scope.
pub(crate) struct IdentifierRowHashNames<'a> {
    pub start: &'a str,
    pub len: &'a str,
    pub hash: &'a str,
    pub cursor: &'a str,
    pub byte: &'a str,
}

/// The one FNV-1a hash of a VAST row's identifier text.
///
/// Four passes used to carry their own copy of this loop: the typedef
/// annotation pass, the typedef prehash pass, `emit_identifier_hash_for_row`,
/// and `emit_identifier_source_hash_for_index`. They differ only in the names
/// they bind and in the guard that decides whether to recompute, so both are
/// parameters here and the emitted IR is unchanged for every caller.
pub(crate) struct IdentifierRowHash<'a> {
    pub vast_nodes: &'a str,
    pub haystack: &'a str,
    pub haystack_len: &'a Expr,
    pub row_base: Expr,
    pub packed_haystack: bool,
    pub names: IdentifierRowHashNames<'a>,
}

impl IdentifierRowHash<'_> {
    fn field(&self, index: u32) -> Expr {
        Expr::load(
            self.vast_nodes,
            Expr::add(self.row_base.clone(), Expr::u32(index)),
        )
    }

    fn source_offset(&self) -> Expr {
        Expr::add(Expr::var(self.names.start), Expr::var(self.names.cursor))
    }

    /// Binds the row's span start, span length, and stored symbol hash.
    pub(crate) fn bindings(&self) -> Vec<Node> {
        vec![
            Node::let_bind(self.names.start, self.field(5)),
            Node::let_bind(self.names.len, self.field(6)),
            Node::let_bind(self.names.hash, self.field(VAST_TYPEDEF_SYMBOL_FIELD)),
        ]
    }

    /// `hash == 0`, the "no hash stored yet" test callers guard on.
    pub(crate) fn hash_is_unset(&self) -> Expr {
        Expr::eq(Expr::var(self.names.hash), Expr::u32(0))
    }

    /// Recomputes the hash over the row's source bytes when `guard` holds.
    pub(crate) fn update(&self, guard: Expr) -> Node {
        Node::if_then(
            guard,
            vec![
                Node::assign(self.names.hash, Expr::u32(0x811c9dc5)),
                Node::loop_for(
                    self.names.cursor,
                    Expr::u32(0),
                    Expr::var(self.names.len),
                    vec![Node::if_then(
                        Expr::lt(self.source_offset(), self.haystack_len.clone()),
                        vec![
                            Node::let_bind(
                                self.names.byte,
                                load_source_byte(
                                    self.haystack,
                                    self.source_offset(),
                                    self.packed_haystack,
                                ),
                            ),
                            Node::assign(
                                self.names.hash,
                                Expr::bitxor(Expr::var(self.names.hash), Expr::var(self.names.byte)),
                            ),
                            Node::assign(
                                self.names.hash,
                                Expr::mul(Expr::var(self.names.hash), Expr::u32(0x01000193)),
                            ),
                        ],
                    )],
                ),
            ],
        )
    }

    /// Bindings plus the guarded recompute, for callers that decode no other
    /// fields of the row.
    pub(crate) fn nodes(&self, guard: Expr) -> Vec<Node> {
        let mut nodes = self.bindings();
        nodes.push(self.update(guard));
        nodes
    }
}

pub(crate) fn emit_identifier_hash_for_row(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: &Expr,
    row_base: Expr,
    prefix: &str,
    packed_haystack: bool,
) -> Vec<Node> {
    let start = format!("{prefix}_start");
    let len = format!("{prefix}_len");
    let hash = format!("{prefix}_hash");
    let cursor = format!("{prefix}_i");
    let byte = format!("{prefix}_byte");

    let row = IdentifierRowHash {
        vast_nodes,
        haystack,
        haystack_len,
        row_base,
        packed_haystack,
        names: IdentifierRowHashNames {
            start: &start,
            len: &len,
            hash: &hash,
            cursor: &cursor,
            byte: &byte,
        },
    };
    let guard = row.hash_is_unset();
    row.nodes(guard)
}

pub(crate) fn emit_identifier_source_hash_for_index(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: &Expr,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    packed_haystack: bool,
) -> Vec<Node> {
    let base = format!("{prefix}_hash_base");
    let start = format!("{prefix}_hash_start");
    let len = format!("{prefix}_hash_len");
    let cursor = format!("{prefix}_hash_i");
    let byte = format!("{prefix}_hash_byte");

    let row = IdentifierRowHash {
        vast_nodes,
        haystack,
        haystack_len,
        row_base: Expr::var(&base),
        packed_haystack,
        names: IdentifierRowHashNames {
            start: &start,
            len: &len,
            hash: out_name,
            cursor: &cursor,
            byte: &byte,
        },
    };

    let mut nodes = vec![Node::let_bind(
        &base,
        Expr::mul(idx, Expr::u32(VAST_NODE_STRIDE_U32)),
    )];
    let guard = row.hash_is_unset();
    nodes.extend(row.nodes(guard));
    nodes
}
