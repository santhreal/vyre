//! Canonical CSR graph traversal and frontier pipeline composer.
//!
//! Unifies neighbor iteration (`load(row_offsets, u)` .. `load(row_offsets, u+1)`),
//! edge-kind filtering, directionality (forward, backward, bidirectional),
//! frontier representations (1D bitset, 2D batched bitset, queue-driven),
//! bitset addressing primitives, and convergence tracking across `vyre-libs`.
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

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
    let _ = checked_csr_offset_count(node_count, "CSR")?;
    Ok(csr_read_only_buffers(node_count, edge_count))
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

impl CsrBuffers<'static> {
    /// Canonical buffer names for CSR edge traversal.
    pub const CANONICAL: Self = Self {
        offsets: NAME_EDGE_OFFSETS,
        targets: NAME_EDGE_TARGETS,
        edge_kind_mask: Some(NAME_EDGE_KIND_MASK),
    };
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

#[path = "csr_traversal.rs"]
mod csr_traversal;
pub use csr_traversal::*;
