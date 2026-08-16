use super::*;

/// The FNV-1a hash of a VAST row's identifier text, as an operation of its own.
pub(crate) const IDENTIFIER_ROW_HASH_OP_ID: &str = "vyre-libs::parsing::c11_identifier_row_hash";
/// [`IDENTIFIER_ROW_HASH_OP_ID`] over a haystack holding four bytes per word.
pub(crate) const IDENTIFIER_ROW_HASH_PACKED_OP_ID: &str =
    "vyre-libs::parsing::c11_identifier_row_hash_packed_haystack";

/// The operation a hash emission names as the block it composes.
pub(crate) fn identifier_row_hash_op_id(packed_haystack: bool) -> &'static str {
    if packed_haystack {
        IDENTIFIER_ROW_HASH_PACKED_OP_ID
    } else {
        IDENTIFIER_ROW_HASH_OP_ID
    }
}

/// Names the emitted FNV-1a identifier hash writes in the caller's scope.
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
/// they write and in the guard that decides whether to recompute, so both are
/// parameters here.
///
/// The three results are declared by the caller and assigned by the scan. A
/// composition region does not export the bindings made inside it, so a phase
/// that reaches its caller through a `let` cannot be named as a block; one that
/// assigns into a declared name can.
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

    /// Declares the row's span start, span length, and stored symbol hash.
    ///
    /// The caller emits these, so the names outlive the region that fills them.
    pub(crate) fn declarations(&self) -> Vec<Node> {
        vec![
            Node::let_bind(self.names.start, Expr::u32(0)),
            Node::let_bind(self.names.len, Expr::u32(0)),
            Node::let_bind(self.names.hash, Expr::u32(0)),
        ]
    }

    /// Reads the three row fields into the declared names.
    fn field_assignments(&self) -> Vec<Node> {
        vec![
            Node::assign(self.names.start, self.field(5)),
            Node::assign(self.names.len, self.field(6)),
            Node::assign(self.names.hash, self.field(VAST_TYPEDEF_SYMBOL_FIELD)),
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
                                Expr::bitxor(
                                    Expr::var(self.names.hash),
                                    Expr::var(self.names.byte),
                                ),
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

    /// The scan itself: read the row's fields, then recompute over its bytes
    /// when `guard` holds.
    fn scan(&self, guard: Expr) -> Vec<Node> {
        let mut nodes = self.field_assignments();
        nodes.push(self.update(guard));
        nodes
    }

    /// Declarations plus the scan, for the registered operation's own program.
    pub(crate) fn nodes(&self, guard: Expr) -> Vec<Node> {
        let mut nodes = self.declarations();
        nodes.extend(self.scan(guard));
        nodes
    }

    /// Declarations plus the scan as a block of `parent_op_id`.
    pub(crate) fn composed(&self, parent_op_id: &str, guard: Expr) -> Vec<Node> {
        let mut nodes = self.declarations();
        nodes.push(child_phase(
            parent_op_id,
            identifier_row_hash_op_id(self.packed_haystack),
            self.scan(guard),
        ));
        nodes
    }
}

/// The identifier hash of the row at `row_base`, as a block of `parent_op_id`.
pub(crate) fn emit_identifier_hash_for_row(
    parent_op_id: &str,
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
    row.composed(parent_op_id, guard)
}

/// The identifier hash of row `idx`, as a block of `parent_op_id`.
pub(crate) fn emit_identifier_source_hash_for_index(
    parent_op_id: &str,
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
    nodes.extend(row.composed(parent_op_id, guard));
    nodes
}

/// The registered operation: the identifier hash of one row, on its own.
fn identifier_row_hash_program(op_id: &str, packed_haystack: bool) -> Program {
    const START: &str = "phase_hash_start";
    const LEN: &str = "phase_hash_len";
    const HASH: &str = "phase_hash";
    const CURSOR: &str = "phase_hash_cursor";
    const BYTE: &str = "phase_hash_byte";

    let haystack_len = phase_haystack_len();
    let row = IdentifierRowHash {
        vast_nodes: phase_program::NODES,
        haystack: phase_program::HAYSTACK,
        haystack_len: &haystack_len,
        row_base: vast_row_base_expr(phase_row()),
        packed_haystack,
        names: IdentifierRowHashNames {
            start: START,
            len: LEN,
            hash: HASH,
            cursor: CURSOR,
            byte: BYTE,
        },
    };
    let guard = row.hash_is_unset();
    phase_program(
        op_id,
        PhaseInputs::RowWithHaystack { packed_haystack },
        HASH,
        row.nodes(guard),
    )
}

/// The identifier hash phase over a one-byte-per-word haystack.
pub(in crate::parsing::c::parse::vast) fn c11_identifier_row_hash() -> Program {
    identifier_row_hash_program(IDENTIFIER_ROW_HASH_OP_ID, false)
}

/// The identifier hash phase over a packed haystack.
pub(in crate::parsing::c::parse::vast) fn c11_identifier_row_hash_packed_haystack() -> Program {
    identifier_row_hash_program(IDENTIFIER_ROW_HASH_PACKED_OP_ID, true)
}
