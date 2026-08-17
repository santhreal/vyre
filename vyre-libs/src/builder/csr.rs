//! Canonical CSR graph traversal and frontier pipeline composer.
//!
//! Unifies neighbor iteration (`load(row_offsets, u)` .. `load(row_offsets, u+1)`),
//! edge-kind filtering, directionality (forward, backward, bidirectional),
//! frontier representations (1D bitset, 2D batched bitset, queue-driven),
//! bitset addressing primitives, and convergence tracking across `vyre-libs`.
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, MemoryOrdering, Node, Program,
};

use crate::bitset::bitset_words;

/// Default workgroup size for CSR traversal and frontier step programs.
pub const CSR_TRAVERSAL_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Canonical name for CSR edge offsets buffer.
pub const NAME_EDGE_OFFSETS: &str = "pg_edge_offsets";
/// Canonical name for CSR edge targets buffer.
pub const NAME_EDGE_TARGETS: &str = "pg_edge_targets";
/// Canonical name for CSR edge kind mask buffer.
pub const NAME_EDGE_KIND_MASK: &str = "pg_edge_kind_mask";
/// Canonical name for CSR nodes buffer.
pub const NAME_NODES: &str = "pg_nodes";
/// Canonical name for CSR node tags buffer.
pub const NAME_NODE_TAGS: &str = "pg_node_tags";

/// Binding index for CSR node array.
pub const BINDING_NODES: u32 = 0;
/// Binding index for CSR row offset array.
pub const BINDING_EDGE_OFFSETS: u32 = 1;
/// Binding index for CSR column / edge targets array.
pub const BINDING_EDGE_TARGETS: u32 = 2;
/// Binding index for CSR edge kind mask array.
pub const BINDING_EDGE_KIND_MASK: u32 = 3;
/// Binding index for CSR node tag array.
pub const BINDING_NODE_TAGS: u32 = 4;
/// Base binding index for traversal primitive buffers.
pub const BINDING_PRIMITIVE_START: u32 = 5;

/// Checked calculation of batched frontier words: `words * query_count`.
pub fn checked_batched_frontier_words(words: u32, query_count: u32) -> Result<u32, String> {
    words.checked_mul(query_count).ok_or_else(|| {
        format!(
            "Fix: batched CSR frontier words overflow u32: words={words}, query_count={query_count}."
        )
    })
}

/// Checked calculation of CSR row offset count: `node_count + 1`.
pub fn checked_csr_offset_count(node_count: u32, builder_name: &str) -> Result<u32, String> {
    node_count
        .checked_add(1)
        .ok_or_else(|| format!("Fix: {builder_name} node_count + 1 overflows u32 ({node_count})."))
}

/// The local bindings one packed-bitset access introduces.
#[derive(Clone, Copy, Debug)]
pub struct BitAccess<'a> {
    /// Binding name for the word index of the addressed bit.
    pub word: &'a str,
    /// Binding name for the single-bit mask selecting the bit within its word.
    pub mask: &'a str,
    /// Binding name for the word value the access reads.
    pub value: &'a str,
}

/// Bind the word index and single-bit mask addressing `bit` in a packed bitset.
pub fn bind_bit_address(
    bit: &Expr,
    word: &str,
    mask: &str,
    word_index: impl FnOnce(Expr) -> Expr,
) -> [Node; 2] {
    [
        Node::let_bind(word, word_index(Expr::shr(bit.clone(), Expr::u32(5)))),
        Node::let_bind(
            mask,
            Expr::shl(Expr::u32(1), Expr::bitand(bit.clone(), Expr::u32(31))),
        ),
    ]
}

/// Load the word holding the addressed bit from `buffer` into `names.value`.
pub fn bind_word(buffer: &str, names: BitAccess<'_>) -> Node {
    Node::let_bind(names.value, Expr::load(buffer, Expr::var(names.word)))
}

/// `(value & mask) != 0`: the addressed bit is set in the word already bound.
pub fn bit_is_set(names: BitAccess<'_>) -> Expr {
    Expr::ne(
        Expr::bitand(Expr::var(names.value), Expr::var(names.mask)),
        Expr::u32(0),
    )
}

/// `(value & mask) == 0`: the addressed bit is clear in the word already bound.
pub fn bit_is_clear(value: &str, mask: &str) -> Expr {
    Expr::eq(
        Expr::bitand(Expr::var(value), Expr::var(mask)),
        Expr::u32(0),
    )
}

/// Run `body` only when bit `bit` of packed bitset `buffer` is set.
pub fn when_bit_set(
    buffer: &str,
    bit: &Expr,
    word: Option<&str>,
    value: &str,
    mask: &str,
    word_index: impl FnOnce(Expr) -> Expr,
    body: Vec<Node>,
) -> Vec<Node> {
    let index = word_index(Expr::shr(bit.clone(), Expr::u32(5)));
    let bit_mask = Node::let_bind(
        mask,
        Expr::shl(Expr::u32(1), Expr::bitand(bit.clone(), Expr::u32(31))),
    );
    let guard = Node::if_then(
        Expr::ne(
            Expr::bitand(Expr::var(value), Expr::var(mask)),
            Expr::u32(0),
        ),
        body,
    );
    match word {
        Some(word) => vec![
            Node::let_bind(word, index),
            bit_mask,
            Node::let_bind(value, Expr::load(buffer, Expr::var(word))),
            guard,
        ],
        None => vec![
            Node::let_bind(value, Expr::load(buffer, index)),
            bit_mask,
            guard,
        ],
    }
}

/// Set bit `bit` of packed bitset `buffer` with an atomic OR, running
/// `on_new_bit` only when this lane flipped the bit from 0 to 1.
pub fn set_bit(
    buffer: &str,
    bit: &Expr,
    names: BitAccess<'_>,
    word_index: impl FnOnce(Expr) -> Expr,
    on_new_bit: Vec<Node>,
) -> Vec<Node> {
    let [word, mask] = bind_bit_address(bit, names.word, names.mask, word_index);
    let or = Node::let_bind(
        names.value,
        Expr::atomic_or(buffer, Expr::var(names.word), Expr::var(names.mask)),
    );
    if on_new_bit.is_empty() {
        return vec![word, mask, or];
    }
    vec![
        word,
        mask,
        or,
        Node::if_then(bit_is_clear(names.value, names.mask), on_new_bit),
    ]
}

/// Guard `active_body` on lane `source` naming an in-bounds node whose bit is set
/// in `frontier_in` and, when `excluded_sources` is given, clear in that bitset.
pub fn active_source_lane(
    node_count: u32,
    frontier_in: &str,
    excluded_sources: Option<&str>,
    source: Expr,
    active_body: Vec<Node>,
) -> Node {
    let names = BitAccess {
        word: "word_idx",
        mask: "bit_mask",
        value: "src_word",
    };
    let [word, mask] = bind_bit_address(&Expr::var("src"), names.word, names.mask, |word| word);
    let mut lane = vec![
        Node::let_bind("src", source.clone()),
        word,
        mask,
        bind_word(frontier_in, names),
    ];
    let live = match excluded_sources {
        Some(excluded_sources) => {
            lane.push(Node::let_bind(
                "excluded_word",
                Expr::load(excluded_sources, Expr::var(names.word)),
            ));
            Expr::and(bit_is_set(names), bit_is_clear("excluded_word", names.mask))
        }
        None => bit_is_set(names),
    };
    lane.push(Node::if_then(live, active_body));
    Node::if_then(Expr::lt(source, Expr::u32(node_count)), lane)
}

/// Construct read-only standard CSR graph buffer declarations.
pub fn csr_read_only_buffers(node_count: u32, edge_count: u32) -> Vec<BufferDecl> {
    let physical_edge_count = edge_count.max(1);
    let offset_count = node_count.saturating_add(1);
    vec![
        BufferDecl::storage(
            NAME_NODES,
            BINDING_NODES,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(node_count.max(1)),
        BufferDecl::storage(
            NAME_EDGE_OFFSETS,
            BINDING_EDGE_OFFSETS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(offset_count),
        BufferDecl::storage(
            NAME_EDGE_TARGETS,
            BINDING_EDGE_TARGETS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
        BufferDecl::storage(
            NAME_EDGE_KIND_MASK,
            BINDING_EDGE_KIND_MASK,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
        BufferDecl::storage(
            NAME_NODE_TAGS,
            BINDING_NODE_TAGS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(node_count.max(1)),
    ]
}

/// Fallible constructor for read-only standard CSR graph buffer declarations.
pub fn try_csr_read_only_buffers(
    node_count: u32,
    edge_count: u32,
) -> Result<Vec<BufferDecl>, String> {
    let offset_count = node_count.checked_add(1).ok_or_else(|| {
        format!("Fix: CSR offset count overflow: node_count={node_count} + 1 overflows u32.")
    })?;
    let physical_edge_count = edge_count.max(1);
    Ok(vec![
        BufferDecl::storage(
            NAME_NODES,
            BINDING_NODES,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(node_count.max(1)),
        BufferDecl::storage(
            NAME_EDGE_OFFSETS,
            BINDING_EDGE_OFFSETS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(offset_count),
        BufferDecl::storage(
            NAME_EDGE_TARGETS,
            BINDING_EDGE_TARGETS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
        BufferDecl::storage(
            NAME_EDGE_KIND_MASK,
            BINDING_EDGE_KIND_MASK,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
        BufferDecl::storage(
            NAME_NODE_TAGS,
            BINDING_NODE_TAGS,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(node_count.max(1)),
    ])
}

/// Construct a packed frontier bitset buffer declaration.
pub fn csr_frontier_buffer(
    name: &str,
    binding: u32,
    access: BufferAccess,
    node_count: u32,
) -> BufferDecl {
    let words = bitset_words(node_count);
    BufferDecl::storage(name, binding, access, DataType::U32).with_count(words.max(1))
}

/// Construct a single or multi-word u32 buffer declaration.
pub fn csr_word_buffer(
    name: &str,
    binding: u32,
    access: BufferAccess,
    word_count: u32,
) -> BufferDecl {
    BufferDecl::storage(name, binding, access, DataType::U32).with_count(word_count.max(1))
}

/// Append standard frontier output and atomic changed buffers to `buffers`.
pub fn csr_push_frontier_changed_buffers(
    buffers: &mut Vec<BufferDecl>,
    frontier_out: &str,
    changed: &str,
    node_count: u32,
) {
    buffers.push(csr_frontier_buffer(
        frontier_out,
        BINDING_PRIMITIVE_START,
        BufferAccess::ReadWrite,
        node_count,
    ));
    buffers.push(csr_word_buffer(
        changed,
        BINDING_PRIMITIVE_START + 1,
        BufferAccess::ReadWrite,
        1,
    ));
}

/// CSR buffer names referenced during traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsrBuffers<'a> {
    /// Row offsets buffer name (length `node_count + 1`).
    pub offsets: &'a str,
    /// Column / target node indices buffer name (length `edge_count`).
    pub targets: &'a str,
    /// Per-edge kind / attribute mask buffer name (length `edge_count`).
    pub edge_kind_mask: Option<&'a str>,
}

impl<'a> CsrBuffers<'a> {
    /// Create explicit CSR buffer configuration.
    #[must_use]
    pub const fn new(offsets: &'a str, targets: &'a str, edge_kind_mask: Option<&'a str>) -> Self {
        Self {
            offsets,
            targets,
            edge_kind_mask,
        }
    }
}

impl<'a> Default for CsrBuffers<'a> {
    fn default() -> Self {
        Self {
            offsets: NAME_EDGE_OFFSETS,
            targets: NAME_EDGE_TARGETS,
            edge_kind_mask: Some(NAME_EDGE_KIND_MASK),
        }
    }
}

/// Traversal direction for a CSR frontier step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsrDirection {
    /// Successor traversal: active source node `u` expands out-edges `u → v` into destination `v`.
    Forward,
    /// Predecessor traversal: candidate `u` checks in-edges for any active target `v`.
    Backward,
    /// Bidirectional expansion: forward and backward passes fused.
    Bidirectional,
}

/// Lane assignment and in-bounds rule for one queued CSR source row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CsrQueueLanes {
    /// One thread per queued source row.
    Scalar,
    /// A fixed-size team of lanes cooperates on each queued source row.
    Team {
        /// Lanes assigned to each queued row.
        lanes: u32,
    },
    /// Stride logical lane teams across a large queue wave.
    CappedGridStride {
        /// Lanes assigned to each queued row.
        lanes: u32,
        /// Maximum queued source rows covered directly by the launch grid.
        launch_lanes: Option<u32>,
    },
}

/// What a queued row does with its edges once its degree is known.
#[derive(Clone, Copy, Debug)]
pub enum CsrQueueRowPlan<'a> {
    /// Walk every edge in this pass.
    Direct,
    /// Low-degree rows are walked directly; high-degree rows are appended to `high_queue`.
    CompactHighDegree {
        /// Queue buffer receiving high-degree source node indices.
        high_queue: &'a str,
        /// Atomic u32 counter buffer for `high_queue`.
        high_len: &'a str,
        /// Maximum entries `high_queue` can accept before overflow.
        high_queue_capacity: u32,
        /// Degree at which a row is diverted into `high_queue`.
        high_degree_threshold: u32,
    },
}

/// What a queue step writes when an allowed edge reaches an in-range destination.
#[derive(Clone, Copy, Debug)]
pub enum CsrQueueEmit<'a> {
    /// Set the destination's bit in `frontier_out`.
    Frontier {
        /// Output frontier bitset buffer.
        frontier_out: &'a str,
    },
    /// Atomic-OR into an accumulator bitset; if newly set, append destination to `next_queue`.
    EnqueueDelta {
        /// Queue buffer receiving newly reached destination node indices.
        next_queue: &'a str,
        /// Atomic u32 counter buffer for `next_queue`.
        next_len: &'a str,
        /// Resident accumulator bitset marking reached nodes.
        accumulator: &'a str,
        /// Maximum entries `next_queue` can hold before overflow.
        next_queue_capacity: u32,
    },
}

/// Canonical CSR traversal composer and builder.
#[derive(Clone, Debug)]
pub struct CsrTraversalComposer<'a> {
    /// Operation ID recorded on the wrapping IR Region.
    pub op_id: &'a str,
    /// Diagnostic name used in trap/error messages.
    pub builder_name: &'a str,
    /// CSR buffer names.
    pub buffers: CsrBuffers<'a>,
    /// Traversal direction.
    pub direction: CsrDirection,
    /// Local identifier prefix for hygiene under nested inlining.
    pub prefix: &'a str,
    /// Allowed edge-kind bitmask.
    pub allow_mask: u32,
    /// Logical vertex / node count.
    pub node_count: u32,
    /// Physical / logical edge count.
    pub edge_count: u32,
    /// Execution workgroup size.
    pub workgroup_size: [u32; 3],
}

impl<'a> CsrTraversalComposer<'a> {
    /// Create a new CSR traversal composer with defaults.
    #[must_use]
    pub fn new(op_id: &'a str, builder_name: &'a str, node_count: u32) -> Self {
        Self {
            op_id,
            builder_name,
            buffers: CsrBuffers::default(),
            direction: CsrDirection::Forward,
            prefix: "",
            allow_mask: 0xFFFF_FFFF,
            node_count,
            edge_count: 0,
            workgroup_size: CSR_TRAVERSAL_WORKGROUP_SIZE,
        }
    }

    /// Convenience constructor for forward traversal.
    #[must_use]
    pub fn forward(op_id: &'a str, node_count: u32, edge_count: u32, allow_mask: u32) -> Self {
        Self::new(op_id, op_id, node_count)
            .with_direction(CsrDirection::Forward)
            .with_allow_mask(allow_mask)
            .with_edge_count(edge_count)
    }

    /// Convenience constructor for backward traversal.
    #[must_use]
    pub fn backward(op_id: &'a str, node_count: u32, edge_count: u32, allow_mask: u32) -> Self {
        Self::new(op_id, op_id, node_count)
            .with_direction(CsrDirection::Backward)
            .with_allow_mask(allow_mask)
            .with_edge_count(edge_count)
    }

    /// Set CSR buffer names.
    #[must_use]
    pub const fn with_buffers(mut self, buffers: CsrBuffers<'a>) -> Self {
        self.buffers = buffers;
        self
    }

    /// Set traversal direction.
    #[must_use]
    pub const fn with_direction(mut self, direction: CsrDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set local identifier prefix.
    #[must_use]
    pub const fn with_prefix(mut self, prefix: &'a str) -> Self {
        self.prefix = prefix;
        self
    }

    /// Set allowed edge-kind mask.
    #[must_use]
    pub const fn with_allow_mask(mut self, allow_mask: u32) -> Self {
        self.allow_mask = allow_mask;
        self
    }

    /// Set edge count.
    #[must_use]
    pub const fn with_edge_count(mut self, edge_count: u32) -> Self {
        self.edge_count = edge_count;
        self
    }

    /// Set workgroup size.
    #[must_use]
    pub const fn with_workgroup_size(mut self, workgroup_size: [u32; 3]) -> Self {
        self.workgroup_size = workgroup_size;
        self
    }

    /// Disambiguate local binding names by prefix.
    #[must_use]
    pub fn local_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}_{name}", self.prefix)
        }
    }

    /// Emit row offset loads `[edge_start = offsets[src], edge_end = offsets[src + 1]]`.
    #[must_use]
    pub fn emit_row_offsets(&self, src: Expr, start_var: &str, end_var: &str) -> [Node; 2] {
        [
            Node::let_bind(start_var, Expr::load(self.buffers.offsets, src.clone())),
            Node::let_bind(
                end_var,
                Expr::load(self.buffers.offsets, Expr::add(src, Expr::u32(1))),
            ),
        ]
    }

    /// Emit a bounded loop over the CSR edges of source node `src`.
    #[must_use]
    pub fn emit_row_bounds_and_loop(
        &self,
        src: Expr,
        edge_var: &str,
        loop_body: Vec<Node>,
    ) -> Vec<Node> {
        let edge_start = self.local_name("edge_start");
        let edge_end = self.local_name("edge_end");
        let [lo, hi] = self.emit_row_offsets(src, edge_start.as_str(), edge_end.as_str());
        vec![
            lo,
            hi,
            Node::loop_for(
                edge_var,
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                loop_body,
            ),
        ]
    }

    /// Emit degree computation for source node `src`: `degree = offsets[src + 1] - offsets[src]`.
    #[must_use]
    pub fn emit_row_degree(
        &self,
        src: Expr,
        lo_var: &str,
        hi_var: &str,
        degree_var: &str,
    ) -> [Node; 3] {
        let [lo, hi] = self.emit_row_offsets(src, lo_var, hi_var);
        [
            lo,
            hi,
            Node::let_bind(degree_var, Expr::sub(Expr::var(hi_var), Expr::var(lo_var))),
        ]
    }

    /// Emit edge walk over source node `src` with edge-kind mask filtering and in-bounds destination check.
    #[must_use]
    pub fn emit_neighbor_walk<F>(
        &self,
        src: Expr,
        edge_var: Option<&str>,
        on_neighbor: F,
    ) -> Vec<Node>
    where
        F: Fn(Expr, Expr) -> Vec<Node>,
    {
        let edge_iter = edge_var
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.local_name("e"));
        let kind_mask_var = self.local_name("kind_mask");
        let dst_var = self.local_name("dst");

        let inner = on_neighbor(Expr::var(dst_var.as_str()), Expr::var(edge_iter.as_str()));
        let dst_guard = if self.node_count > 0 {
            vec![
                Node::let_bind(
                    dst_var.as_str(),
                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                ),
                Node::if_then(
                    Expr::lt(Expr::var(dst_var.as_str()), Expr::u32(self.node_count)),
                    inner,
                ),
            ]
        } else {
            vec![
                Node::let_bind(
                    dst_var.as_str(),
                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                ),
                Node::block(inner),
            ]
        };

        let loop_body = match self.buffers.edge_kind_mask {
            Some(mask_buf) => {
                vec![
                    Node::let_bind(
                        kind_mask_var.as_str(),
                        Expr::load(mask_buf, Expr::var(edge_iter.as_str())),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(
                                Expr::var(kind_mask_var.as_str()),
                                Expr::u32(self.allow_mask),
                            ),
                            Expr::u32(0),
                        ),
                        dst_guard,
                    ),
                ]
            }
            None => dst_guard,
        };

        self.emit_row_bounds_and_loop(src, edge_iter.as_str(), loop_body)
    }

    /// Emit ONLY the CSR edge walk for source node `src`: load edge range, filter by kind mask,
    /// atomic-OR the target bit into `frontier_out`, and invoke `on_new_bit()` when a bit flips 0→1.
    #[must_use]
    pub fn emit_edge_expand(
        &self,
        frontier_out: &str,
        src: Expr,
        frontier_index: impl Fn(Expr) -> Expr,
        on_new_bit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let edge_start = name("edge_start");
        let edge_end = name("edge_end");
        let edge_iter = name("e");
        let kind_mask = name("kind_mask");
        let dst = name("dst");
        let dst_word_idx = name("dst_word_idx");
        let dst_bit = name("dst_bit");

        let flip_body = on_new_bit();
        let pre_or_word = name(if flip_body.is_empty() { "_prev" } else { "old" });
        let on_bounded = set_bit(
            frontier_out,
            &Expr::var(dst.as_str()),
            BitAccess {
                word: dst_word_idx.as_str(),
                mask: dst_bit.as_str(),
                value: pre_or_word.as_str(),
            },
            frontier_index,
            flip_body,
        );

        let kind_buf = self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK);

        vec![
            Node::let_bind(
                edge_start.as_str(),
                Expr::load(self.buffers.offsets, src.clone()),
            ),
            Node::let_bind(
                edge_end.as_str(),
                Expr::load(self.buffers.offsets, Expr::add(src, Expr::u32(1))),
            ),
            Node::loop_for(
                edge_iter.as_str(),
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                vec![
                    Node::let_bind(
                        kind_mask.as_str(),
                        Expr::load(kind_buf, Expr::var(edge_iter.as_str())),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::var(kind_mask.as_str()), Expr::u32(self.allow_mask)),
                            Expr::u32(0),
                        ),
                        vec![
                            Node::let_bind(
                                dst.as_str(),
                                Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                            ),
                            Node::if_then(
                                Expr::lt(Expr::var(dst.as_str()), Expr::u32(self.node_count)),
                                on_bounded,
                            ),
                        ],
                    ),
                ],
            ),
        ]
    }

    /// Emit the CSR neighbor expansion for one source node `src`, reading its frontier bit INLINE
    /// and expanding out-edges only when set.
    #[must_use]
    pub fn emit_edge_scan(
        &self,
        frontier_out: &str,
        src: Expr,
        frontier_index: impl Fn(Expr) -> Expr,
        on_new_bit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let word_idx = name("word_idx");
        let bit_mask = name("bit_mask");
        let src_word = name("src_word");

        let expand = self.emit_edge_expand(frontier_out, src.clone(), &frontier_index, on_new_bit);
        when_bit_set(
            frontier_out,
            &src,
            Some(word_idx.as_str()),
            src_word.as_str(),
            bit_mask.as_str(),
            frontier_index,
            expand,
        )
    }

    /// Emit one backward CSR traversal pass for candidate node `src`.
    #[must_use]
    pub fn emit_backward_scan(
        &self,
        src: Expr,
        frontier_in: &str,
        frontier_out: &str,
        on_hit: impl Fn() -> Vec<Node>,
    ) -> Vec<Node> {
        let name = |n: &str| self.local_name(n);
        let edge_start = name("edge_start");
        let edge_end = name("edge_end");
        let edge_iter = name("e");
        let kind_mask = name("kind_mask");
        let dst = name("dst");
        let hit = name("hit");

        let kind_buf = self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK);

        let hit_actions = on_hit();
        let hit_body = if hit_actions.is_empty() {
            vec![Node::assign(hit.as_str(), Expr::u32(1))]
        } else {
            let mut b = vec![Node::assign(hit.as_str(), Expr::u32(1))];
            b.extend(hit_actions);
            b
        };

        vec![
            Node::let_bind(
                edge_start.as_str(),
                Expr::load(self.buffers.offsets, src.clone()),
            ),
            Node::let_bind(
                edge_end.as_str(),
                Expr::load(self.buffers.offsets, Expr::add(src.clone(), Expr::u32(1))),
            ),
            Node::let_bind(hit.as_str(), Expr::u32(0)),
            Node::loop_for(
                edge_iter.as_str(),
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                vec![Node::if_then(
                    Expr::eq(Expr::var(hit.as_str()), Expr::u32(0)),
                    vec![
                        Node::let_bind(
                            kind_mask.as_str(),
                            Expr::load(kind_buf, Expr::var(edge_iter.as_str())),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(
                                    Expr::var(kind_mask.as_str()),
                                    Expr::u32(self.allow_mask),
                                ),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    dst.as_str(),
                                    Expr::load(self.buffers.targets, Expr::var(edge_iter.as_str())),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var(dst.as_str()), Expr::u32(self.node_count)),
                                    when_bit_set(
                                        frontier_in,
                                        &Expr::var(dst.as_str()),
                                        None,
                                        "dst_word",
                                        "dst_bit",
                                        |word| word,
                                        hit_body,
                                    ),
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var(hit.as_str()), Expr::u32(1)),
                set_bit(
                    frontier_out,
                    &src,
                    BitAccess {
                        word: "src_word_idx",
                        mask: "src_bit",
                        value: "_prev",
                    },
                    |word| word,
                    Vec::new(),
                ),
            ),
        ]
    }

    /// Build a single-step forward CSR frontier traversal program.
    #[must_use]
    pub fn build_forward_step(&self, frontier_in: &str, frontier_out: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let active_body =
            self.emit_edge_expand(frontier_out, Expr::var("src"), |word| word, Vec::new);
        let body = vec![active_source_lane(
            self.node_count,
            frontier_in,
            None,
            t,
            active_body,
        )];

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build a single-step forward CSR traversal program excluding specified source nodes.
    #[must_use]
    pub fn build_forward_step_excluding(
        &self,
        frontier_in: &str,
        excluded_sources: &str,
        frontier_out: &str,
    ) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            excluded_sources,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 2,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let active_body =
            self.emit_edge_expand(frontier_out, Expr::var("src"), |word| word, Vec::new);
        let body = vec![active_source_lane(
            self.node_count,
            frontier_in,
            Some(excluded_sources),
            t,
            active_body,
        )];

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build a single-step backward CSR frontier traversal program.
    #[must_use]
    pub fn build_backward_step(&self, frontier_in: &str, frontier_out: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        buffers.push(csr_frontier_buffer(
            frontier_in,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadOnly,
            self.node_count,
        ));
        buffers.push(csr_frontier_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            self.node_count,
        ));

        let t = Expr::InvocationId { axis: 0 };
        let mut body = vec![
            Node::let_bind("src", t.clone()),
            Node::let_bind("hit", Expr::u32(0)),
        ];
        body.extend(vec![
            Node::let_bind(
                "edge_start",
                Expr::load(self.buffers.offsets, Expr::var("src")),
            ),
            Node::let_bind(
                "edge_end",
                Expr::load(
                    self.buffers.offsets,
                    Expr::add(Expr::var("src"), Expr::u32(1)),
                ),
            ),
            Node::loop_for(
                "e",
                Expr::var("edge_start"),
                Expr::var("edge_end"),
                vec![Node::if_then(
                    Expr::eq(Expr::var("hit"), Expr::u32(0)),
                    vec![
                        Node::let_bind(
                            "kind_mask",
                            Expr::load(
                                self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK),
                                Expr::var("e"),
                            ),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var("kind_mask"), Expr::u32(self.allow_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    "dst",
                                    Expr::load(self.buffers.targets, Expr::var("e")),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("dst"), Expr::u32(self.node_count)),
                                    when_bit_set(
                                        frontier_in,
                                        &Expr::var("dst"),
                                        None,
                                        "dst_word",
                                        "dst_bit",
                                        |word| word,
                                        vec![Node::assign("hit", Expr::u32(1))],
                                    ),
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var("hit"), Expr::u32(1)),
                set_bit(
                    frontier_out,
                    &Expr::var("src"),
                    BitAccess {
                        word: "src_word_idx",
                        mask: "src_bit",
                        value: "_prev",
                    },
                    |word| word,
                    Vec::new(),
                ),
            ),
        ]);

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(Expr::lt(t, Expr::u32(self.node_count)), body)],
            )],
        )
    }

    /// Build parallel in-place backward expansion program with atomic changed flag.
    #[must_use]
    pub fn build_parallel_backward_or_changed(&self, frontier_out: &str, changed: &str) -> Program {
        let src = Expr::InvocationId { axis: 0 };
        let body = vec![
            Node::let_bind("edge_start", Expr::load(self.buffers.offsets, src.clone())),
            Node::let_bind(
                "edge_end",
                Expr::load(self.buffers.offsets, Expr::add(src.clone(), Expr::u32(1))),
            ),
            Node::let_bind("hit", Expr::u32(0)),
            Node::loop_for(
                "e",
                Expr::var("edge_start"),
                Expr::var("edge_end"),
                vec![Node::if_then(
                    Expr::eq(Expr::var("hit"), Expr::u32(0)),
                    vec![
                        Node::let_bind(
                            "kind_mask",
                            Expr::load(
                                self.buffers.edge_kind_mask.unwrap_or(NAME_EDGE_KIND_MASK),
                                Expr::var("e"),
                            ),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var("kind_mask"), Expr::u32(self.allow_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    "dst",
                                    Expr::load(self.buffers.targets, Expr::var("e")),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("dst"), Expr::u32(self.node_count)),
                                    when_bit_set(
                                        frontier_out,
                                        &Expr::var("dst"),
                                        None,
                                        "dst_word",
                                        "dst_bit",
                                        |word| word,
                                        vec![Node::assign("hit", Expr::u32(1))],
                                    ),
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var("hit"), Expr::u32(1)),
                set_bit(
                    frontier_out,
                    &src,
                    BitAccess {
                        word: "src_word_idx",
                        mask: "src_bit",
                        value: "old",
                    },
                    |word| word,
                    vec![Node::let_bind(
                        "_changed",
                        Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
                    )],
                ),
            ),
        ];

        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        csr_push_frontier_changed_buffers(&mut buffers, frontier_out, changed, self.node_count);
        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(
                    Expr::lt(src, Expr::u32(self.node_count)),
                    body,
                )],
            )],
        )
    }

    /// Build parallel in-place forward expansion program with atomic changed flag.
    #[must_use]
    pub fn build_parallel_forward_or_changed(&self, frontier_out: &str, changed: &str) -> Program {
        let mut buffers = csr_read_only_buffers(self.node_count, self.edge_count);
        csr_push_frontier_changed_buffers(&mut buffers, frontier_out, changed, self.node_count);

        let body =
            self.emit_parallel_forward_or_changed_body(frontier_out, changed, None, None, None);

        Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(self.op_id, body)],
        )
    }

    /// Build parallel in-place forward expansion body with optional snapshot barrier.
    #[must_use]
    pub fn emit_parallel_forward_or_changed_body(
        &self,
        frontier_out: &str,
        changed: &str,
        snapshot_barrier: Option<MemoryOrdering>,
        active_gate: Option<Expr>,
        extra_changed: Option<(&str, Expr)>,
    ) -> Vec<Node> {
        let local = |name: &str| -> String {
            if self.prefix.is_empty() {
                name.to_string()
            } else {
                format!("{}_{name}", self.prefix)
            }
        };
        let src = Expr::gid_x();
        let in_bounds = local("in_bounds");
        let word_idx = local("word_idx");
        let bit_mask = local("bit_mask");
        let src_word = local("src_word");
        let src_active = local("src_active");
        let changed_old = local("changed_old");
        let extra_changed_old = local("extra_changed_old");

        let mark_changed = || {
            let mut nodes = vec![Node::let_bind(
                changed_old.as_str(),
                Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
            )];
            if let Some((extra_changed_buffer, extra_changed_index)) = &extra_changed {
                nodes.push(Node::let_bind(
                    extra_changed_old.as_str(),
                    Expr::atomic_or(
                        extra_changed_buffer,
                        extra_changed_index.clone(),
                        Expr::u32(1),
                    ),
                ));
            }
            nodes
        };

        let edge_scan =
            || self.emit_edge_expand(frontier_out, src.clone(), |word| word, &mark_changed);

        if let Some(ordering) = snapshot_barrier {
            let ungated_src_active = Expr::select(
                Expr::var(in_bounds.as_str()),
                Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                Expr::u32(0),
            );
            let src_active_expr = if let Some(active_gate) = active_gate {
                Expr::select(
                    Expr::ne(active_gate, Expr::u32(0)),
                    ungated_src_active,
                    Expr::u32(0),
                )
            } else {
                ungated_src_active
            };
            let mut preamble = vec![Node::let_bind(
                in_bounds.as_str(),
                Expr::lt(src.clone(), Expr::u32(self.node_count)),
            )];
            preamble.extend(bind_bit_address(
                &src,
                word_idx.as_str(),
                bit_mask.as_str(),
                |word| Expr::select(Expr::var(in_bounds.as_str()), word, Expr::u32(0)),
            ));
            preamble.extend([
                Node::let_bind(
                    src_word.as_str(),
                    Expr::load(frontier_out, Expr::var(word_idx.as_str())),
                ),
                Node::let_bind(src_active.as_str(), src_active_expr),
                Node::barrier_with_ordering(ordering),
                Node::if_then(
                    Expr::ne(Expr::var(src_active.as_str()), Expr::u32(0)),
                    edge_scan(),
                ),
            ]);
            return preamble;
        }

        let mut body =
            bind_bit_address(&src, word_idx.as_str(), bit_mask.as_str(), |word| word).to_vec();
        body.push(Node::let_bind(
            src_word.as_str(),
            Expr::load(frontier_out, Expr::var(word_idx.as_str())),
        ));
        body.push(Node::if_then(
            bit_is_set(BitAccess {
                word: word_idx.as_str(),
                mask: bit_mask.as_str(),
                value: src_word.as_str(),
            }),
            edge_scan(),
        ));

        vec![Node::if_then(
            Expr::lt(Expr::gid_x(), Expr::u32(self.node_count)),
            body,
        )]
    }

    /// Build parallel batched forward expansion program over multiple query frontiers.
    pub fn build_parallel_batch_forward_or_changed(
        &self,
        frontier_out: &str,
        changed: &str,
        query_count: u32,
        changed_index: Expr,
        changed_slots: u32,
        mut prologue: Vec<Node>,
        extra_buffers: Vec<BufferDecl>,
    ) -> Result<Program, String> {
        if query_count == 0 {
            return Err(format!(
                "Fix: {} requires at least one query frontier.",
                self.builder_name
            ));
        }
        let src = Expr::InvocationId { axis: 0 };
        let query = Expr::InvocationId { axis: 1 };
        let words = bitset_words(self.node_count);
        let total_words = checked_batched_frontier_words(words, query_count)?;
        let query_word_base = Expr::mul(query, Expr::u32(words));

        let mut body = vec![Node::let_bind("query_word_base", query_word_base)];
        body.extend(self.emit_edge_scan(
            frontier_out,
            src.clone(),
            |word| Expr::add(Expr::var("query_word_base"), word),
            || {
                vec![Node::let_bind(
                    "_changed",
                    Expr::atomic_or(changed, changed_index.clone(), Expr::u32(1)),
                )]
            },
        ));
        prologue.append(&mut body);

        let mut buffers = try_csr_read_only_buffers(self.node_count, self.edge_count)?;
        buffers.push(csr_word_buffer(
            frontier_out,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadWrite,
            total_words.max(1),
        ));
        buffers.push(csr_word_buffer(
            changed,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            changed_slots,
        ));
        buffers.extend(extra_buffers);

        Ok(Program::wrapped(
            buffers,
            self.workgroup_size,
            vec![wrap_anonymous_region(
                self.op_id,
                vec![Node::if_then(
                    Expr::lt(src, Expr::u32(self.node_count)),
                    prologue,
                )],
            )],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_OP_ID: &str = "test::csr::traversal";

    #[test]
    fn forward_traversal_program_emits_valid_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_forward", 64)
            .with_edge_count(128)
            .with_allow_mask(0x00FF);
        let program = composer.build_forward_step("fin", "fout");
        assert_eq!(program.workgroup_size, CSR_TRAVERSAL_WORKGROUP_SIZE);
        assert_eq!(program.buffers.len(), 7); // nodes, offsets, targets, kinds, tags, fin, fout
    }

    #[test]
    fn backward_traversal_program_emits_valid_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_backward", 32)
            .with_edge_count(64)
            .with_direction(CsrDirection::Backward);
        let program = composer.build_backward_step("fin", "fout");
        assert_eq!(program.workgroup_size, CSR_TRAVERSAL_WORKGROUP_SIZE);
        assert_eq!(program.buffers.len(), 7);
    }

    #[test]
    fn excluding_forward_traversal_declares_extra_buffer() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_excluding", 100);
        let program = composer.build_forward_step_excluding("fin", "fex", "fout");
        assert_eq!(program.buffers.len(), 8); // nodes, offsets, targets, kinds, tags, fin, fex, fout
    }

    #[test]
    fn parallel_backward_or_changed_structure() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_bwd_changed", 128)
            .with_direction(CsrDirection::Backward);
        let program = composer.build_parallel_backward_or_changed("fout", "changed");
        assert_eq!(program.buffers.len(), 7);
    }

    #[test]
    fn parallel_batch_forward_validation() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_batch", 64);
        let err = composer.build_parallel_batch_forward_or_changed(
            "fout",
            "changed",
            0,
            Expr::u32(0),
            1,
            Vec::new(),
            Vec::new(),
        );
        assert!(err.is_err());

        let ok = composer.build_parallel_batch_forward_or_changed(
            "fout",
            "changed",
            4,
            Expr::InvocationId { axis: 1 },
            4,
            Vec::new(),
            Vec::new(),
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn row_degree_emission_nodes() {
        let composer = CsrTraversalComposer::new(TEST_OP_ID, "test_degree", 10);
        let [lo, hi, deg] = composer.emit_row_degree(Expr::var("src"), "lo", "hi", "deg");
        let rendered = format!("{lo:?} {hi:?} {deg:?}");
        assert!(rendered.contains("lo"));
        assert!(rendered.contains("hi"));
        assert!(rendered.contains("deg"));
    }

    #[test]
    fn prefix_hygiene() {
        let composer =
            CsrTraversalComposer::new(TEST_OP_ID, "test_prefix", 10).with_prefix("custom");
        assert_eq!(composer.local_name("e"), "custom_e");
        assert_eq!(composer.local_name("dst"), "custom_dst");
    }
}
