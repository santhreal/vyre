//! Unified device-resident token/fact graph layout.
//!
//! This module is the single owner of the device-resident token/fact graph:
//! the CSR packing, the resident byte envelope, and the out-degree skew profile
//! a backend needs to size its expansion queues. A backend crate converts the
//! layout into its own scheduler types and issues its own device calls; it does
//! not restate any of the arithmetic here. Every term is backend-neutral so the
//! same layout serves every target.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Number of rank buckets carried for token/fact out-degree skew planning.
pub const TOKEN_FACT_DEGREE_PROFILE_BUCKETS: usize = 16;

/// Power-of-two ranks used by the token/fact out-degree profile.
pub const TOKEN_FACT_DEGREE_PROFILE_RANKS: [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768,
];

/// Highest rank the profile reports, and therefore how many top out-degrees the
/// profile has to retain from a graph of any size.
const TOKEN_FACT_DEGREE_PROFILE_MAX_RANK: usize = 32_768;

/// Node class stored in the unified compiler/dataflow graph.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TokenFactNodeKind {
    /// Source or macro-expanded token.
    Token,
    /// Macro expansion boundary.
    MacroExpansion,
    /// Semantic declaration, scope, or type node.
    Semantic,
    /// Dataflow fact node.
    Fact,
    /// Diagnostic/provenance node.
    Diagnostic,
}

/// Dependency edge class stored in the unified compiler/dataflow graph.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TokenFactEdgeKind {
    /// Token stream order or token-to-token provenance.
    TokenFlow,
    /// Macro expansion/provenance relation.
    MacroExpansion,
    /// Token or semantic node emits a fact.
    SemanticFact,
    /// Fact-to-fact dataflow dependency.
    FactDependency,
    /// Diagnostic depends on source token, semantic node, or fact.
    DiagnosticProvenance,
}

/// One logical node before resident CSR packing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenFactNode {
    /// Stable producer-defined id.
    pub id: u32,
    /// Node class.
    pub kind: TokenFactNodeKind,
    /// Offset into the shared resident payload slab.
    pub payload_offset: u64,
    /// Byte length inside the shared resident payload slab.
    pub payload_bytes: u64,
}

impl TokenFactNode {
    /// One node in the shared payload slab.
    #[must_use]
    pub const fn new(
        id: u32,
        kind: TokenFactNodeKind,
        payload_offset: u64,
        payload_bytes: u64,
    ) -> Self {
        Self {
            id,
            kind,
            payload_offset,
            payload_bytes,
        }
    }
}

/// One logical edge before resident CSR packing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenFactEdge {
    /// Source node id.
    pub from: u32,
    /// Destination node id.
    pub to: u32,
    /// Edge class.
    pub kind: TokenFactEdgeKind,
}

impl TokenFactEdge {
    /// One dependency edge between two producer-defined node ids.
    #[must_use]
    pub const fn new(from: u32, to: u32, kind: TokenFactEdgeKind) -> Self {
        Self { from, to, kind }
    }
}

/// CSR layout shared by parser, semantic, diagnostic, and dataflow execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceResidentTokenFactGraph {
    /// Stable node ids in resident index order.
    pub node_ids: Vec<u32>,
    /// Node classes in resident index order.
    pub node_kinds: Vec<TokenFactNodeKind>,
    /// Payload offsets in resident index order.
    pub payload_offsets: Vec<u64>,
    /// Payload byte lengths in resident index order.
    pub payload_lengths: Vec<u64>,
    /// CSR row offsets, length `node_count + 1`.
    pub row_offsets: Vec<u32>,
    /// CSR destination node indices.
    pub column_indices: Vec<u32>,
    /// Edge classes aligned with `column_indices`.
    pub edge_kinds: Vec<TokenFactEdgeKind>,
    /// Total resident payload bytes required by the shared slab.
    pub payload_bytes: u64,
    /// Number of token-class nodes.
    pub token_nodes: u32,
    /// Number of fact-class nodes.
    pub fact_nodes: u32,
}

/// Backend-neutral resident byte envelope for a packed token/fact graph.
///
/// A backend reads this to size its resident allocations and its frontier
/// expansion queues. The record widths are the caller's concrete ABI widths;
/// everything derived from them is computed once, here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceResidentTokenFactGraphLayout {
    /// Resident node count.
    pub node_count: u64,
    /// Resident CSR edge count after row deduplication.
    pub edge_count: u64,
    /// Maximum outgoing CSR row degree in the resident token/fact graph.
    pub max_out_degree: u64,
    /// Prefix sums of top out-degrees at [`TOKEN_FACT_DEGREE_PROFILE_RANKS`].
    pub top_out_degree_prefix_sums: [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
    /// Fixed bytes per resident node record.
    pub node_record_bytes: u64,
    /// Fixed bytes per resident edge record.
    pub edge_record_bytes: u64,
    /// Bytes for resident node records.
    pub node_bytes: u64,
    /// Bytes for resident edge records.
    pub edge_bytes: u64,
    /// Bytes for the shared token/fact payload slab.
    pub payload_bytes: u64,
    /// Total bytes that must remain device-resident for the layout.
    pub resident_bytes: u64,
}

impl DeviceResidentTokenFactGraphLayout {
    /// Build a layout from aggregate byte fields when CSR row offsets are not
    /// available to the caller. This preserves correctness by treating total
    /// edge count as the maximum possible row degree.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn from_aggregate_fields(
        node_count: u64,
        edge_count: u64,
        node_record_bytes: u64,
        edge_record_bytes: u64,
        node_bytes: u64,
        edge_bytes: u64,
        payload_bytes: u64,
        resident_bytes: u64,
    ) -> Self {
        Self {
            node_count,
            edge_count,
            max_out_degree: edge_count,
            top_out_degree_prefix_sums: [edge_count; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
            node_record_bytes,
            edge_record_bytes,
            node_bytes,
            edge_bytes,
            payload_bytes,
            resident_bytes,
        }
    }
}

/// One edge staged as resident indices, kept narrow so the CSR sort moves
/// twelve bytes per edge rather than a pointer-width tuple.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StagedEdge {
    from: u32,
    to: u32,
    kind: TokenFactEdgeKind,
}

/// Reusable host-side staging for resident token/fact graph CSR packing.
///
/// Packing resolves edge endpoints by binary search over the sorted node ids
/// rather than through a hash index, so the staging holds only contiguous
/// vectors and the repeat call allocates nothing.
#[derive(Debug, Default)]
pub struct DeviceResidentTokenFactGraphScratch {
    ordered_nodes: Vec<TokenFactNode>,
    staged_edges: Vec<StagedEdge>,
    row_degrees: Vec<u32>,
}

impl DeviceResidentTokenFactGraphScratch {
    /// Create empty reusable token/fact graph packing scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn clear_preserving_capacity(&mut self) {
        self.ordered_nodes.clear();
        self.staged_edges.clear();
        self.row_degrees.clear();
    }
}

/// Unified graph layout errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeviceResidentTokenFactGraphError {
    /// Duplicate logical node id.
    DuplicateNode {
        /// Duplicate id.
        id: u32,
    },
    /// Edge references an unknown node id.
    UnknownEdgeNode {
        /// Unknown id.
        id: u32,
    },
    /// Payload range arithmetic overflowed.
    PayloadOverflow {
        /// Node whose range overflowed.
        id: u32,
    },
    /// Payload range exceeds the declared resident slab.
    PayloadOutOfBounds {
        /// Node whose range is invalid.
        id: u32,
        /// Exclusive end offset.
        end: u64,
        /// Declared slab length.
        payload_bytes: u64,
    },
    /// CSR row offsets cannot fit the release ABI.
    CsrIndexOverflow,
    /// Resident record widths must be explicit, non-zero ABI values.
    ZeroRecordWidth {
        /// Field that was zero.
        field: &'static str,
    },
    /// Public CSR fields are inconsistent with each other.
    InvalidCsrShape {
        /// Invalid CSR field or relationship.
        field: &'static str,
    },
    /// Resident byte arithmetic overflowed.
    ByteCountOverflow {
        /// Field being computed.
        field: &'static str,
    },
}

impl std::fmt::Display for DeviceResidentTokenFactGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNode { id } => write!(
                f,
                "device-resident token/fact graph has duplicate node id {id}. Fix: assign one stable id before CSR packing."
            ),
            Self::UnknownEdgeNode { id } => write!(
                f,
                "device-resident token/fact graph edge references unknown node {id}. Fix: emit all parser, semantic, and fact nodes before edges."
            ),
            Self::PayloadOverflow { id } => write!(
                f,
                "device-resident token/fact graph payload range overflowed for node {id}. Fix: shard the translation unit or payload slab before device upload."
            ),
            Self::PayloadOutOfBounds {
                id,
                end,
                payload_bytes,
            } => write!(
                f,
                "device-resident token/fact graph node {id} payload ends at {end}, beyond slab length {payload_bytes}. Fix: compute payload offsets from the shared slab allocator."
            ),
            Self::CsrIndexOverflow => write!(
                f,
                "device-resident token/fact graph exceeds u32 CSR limits. Fix: shard before resident layout packing."
            ),
            Self::ZeroRecordWidth { field } => write!(
                f,
                "device-resident token/fact graph layout received zero {field}. Fix: pass the concrete resident ABI record width."
            ),
            Self::InvalidCsrShape { field } => write!(
                f,
                "device-resident token/fact graph layout received invalid CSR {field}. Fix: rebuild the token/fact graph through the canonical resident graph planner."
            ),
            Self::ByteCountOverflow { field } => write!(
                f,
                "device-resident token/fact graph layout overflowed while computing {field}. Fix: shard the token/fact graph before resident upload."
            ),
        }
    }
}

impl std::error::Error for DeviceResidentTokenFactGraphError {}

/// Pack parser, semantic, diagnostic, and dataflow nodes into one resident CSR.
pub fn plan_device_resident_token_fact_graph(
    nodes: &[TokenFactNode],
    edges: &[TokenFactEdge],
    payload_bytes: u64,
) -> Result<DeviceResidentTokenFactGraph, DeviceResidentTokenFactGraphError> {
    let mut scratch = DeviceResidentTokenFactGraphScratch::new();
    plan_device_resident_token_fact_graph_with_scratch(nodes, edges, payload_bytes, &mut scratch)
}

/// Pack a resident token/fact graph while reusing caller-owned staging scratch.
pub fn plan_device_resident_token_fact_graph_with_scratch(
    nodes: &[TokenFactNode],
    edges: &[TokenFactEdge],
    payload_bytes: u64,
    scratch: &mut DeviceResidentTokenFactGraphScratch,
) -> Result<DeviceResidentTokenFactGraph, DeviceResidentTokenFactGraphError> {
    scratch.clear_preserving_capacity();
    scratch.ordered_nodes.reserve(nodes.len());
    scratch.staged_edges.reserve(edges.len());
    u32::try_from(nodes.len()).map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
    u32::try_from(edges.len()).map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;

    for node in nodes {
        let end = node
            .payload_offset
            .checked_add(node.payload_bytes)
            .ok_or(DeviceResidentTokenFactGraphError::PayloadOverflow { id: node.id })?;
        if end > payload_bytes {
            return Err(DeviceResidentTokenFactGraphError::PayloadOutOfBounds {
                id: node.id,
                end,
                payload_bytes,
            });
        }
        scratch.ordered_nodes.push(*node);
    }
    scratch.ordered_nodes.sort_unstable_by_key(|node| node.id);

    let node_count = scratch.ordered_nodes.len();
    let mut node_ids = Vec::with_capacity(node_count);
    let mut node_kinds = Vec::with_capacity(node_count);
    let mut payload_offsets = Vec::with_capacity(node_count);
    let mut payload_lengths = Vec::with_capacity(node_count);
    let mut token_nodes = 0_u32;
    let mut fact_nodes = 0_u32;
    for node in &scratch.ordered_nodes {
        // Ids are sorted, so a duplicate is always adjacent. Reporting the
        // smallest duplicated id keeps the error independent of input order.
        if node_ids.last() == Some(&node.id) {
            return Err(DeviceResidentTokenFactGraphError::DuplicateNode { id: node.id });
        }
        node_ids.push(node.id);
        node_kinds.push(node.kind);
        payload_offsets.push(node.payload_offset);
        payload_lengths.push(node.payload_bytes);
        match node.kind {
            TokenFactNodeKind::Token => token_nodes += 1,
            TokenFactNodeKind::Fact => fact_nodes += 1,
            TokenFactNodeKind::MacroExpansion
            | TokenFactNodeKind::Semantic
            | TokenFactNodeKind::Diagnostic => {}
        }
    }

    for edge in edges {
        let from = resident_index(&node_ids, edge.from)?;
        let to = resident_index(&node_ids, edge.to)?;
        scratch.staged_edges.push(StagedEdge {
            from,
            to,
            kind: edge.kind,
        });
    }
    scratch.staged_edges.sort_unstable();

    let mut row_offsets = Vec::with_capacity(node_count + 1);
    let mut column_indices = Vec::with_capacity(scratch.staged_edges.len());
    let mut edge_kinds = Vec::with_capacity(scratch.staged_edges.len());
    row_offsets.push(0);
    let mut edge_index = 0_usize;
    for row in 0..node_count {
        let row =
            u32::try_from(row).map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
        let mut last_edge = None;
        while let Some(&staged) = scratch.staged_edges.get(edge_index) {
            if staged.from != row {
                break;
            }
            let edge_key = (staged.to, staged.kind);
            if last_edge != Some(edge_key) {
                column_indices.push(staged.to);
                edge_kinds.push(staged.kind);
                last_edge = Some(edge_key);
            }
            edge_index += 1;
        }
        let next = u32::try_from(column_indices.len())
            .map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
        row_offsets.push(next);
    }

    Ok(DeviceResidentTokenFactGraph {
        node_ids,
        node_kinds,
        payload_offsets,
        payload_lengths,
        row_offsets,
        column_indices,
        edge_kinds,
        payload_bytes,
        token_nodes,
        fact_nodes,
    })
}

/// Resolve a producer-defined node id to its resident CSR index.
///
/// `node_ids` is sorted ascending by construction, so this is a binary search
/// over one contiguous `u32` run. That beats a hash index here: the index would
/// have to be built for every node before the first edge is resolved, and the
/// search array is already needed as the packed output column.
fn resident_index(node_ids: &[u32], id: u32) -> Result<u32, DeviceResidentTokenFactGraphError> {
    let index = node_ids
        .binary_search(&id)
        .map_err(|_| DeviceResidentTokenFactGraphError::UnknownEdgeNode { id })?;
    u32::try_from(index).map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)
}

/// Compute the resident byte envelope and out-degree skew profile for a graph.
pub fn plan_device_resident_token_fact_graph_layout(
    graph: &DeviceResidentTokenFactGraph,
    node_record_bytes: u64,
    edge_record_bytes: u64,
) -> Result<DeviceResidentTokenFactGraphLayout, DeviceResidentTokenFactGraphError> {
    let mut scratch = DeviceResidentTokenFactGraphScratch::new();
    plan_device_resident_token_fact_graph_layout_with_scratch(
        graph,
        node_record_bytes,
        edge_record_bytes,
        &mut scratch,
    )
}

/// Compute the resident byte envelope reusing caller-owned profiling scratch.
pub fn plan_device_resident_token_fact_graph_layout_with_scratch(
    graph: &DeviceResidentTokenFactGraph,
    node_record_bytes: u64,
    edge_record_bytes: u64,
    scratch: &mut DeviceResidentTokenFactGraphScratch,
) -> Result<DeviceResidentTokenFactGraphLayout, DeviceResidentTokenFactGraphError> {
    if node_record_bytes == 0 {
        return Err(DeviceResidentTokenFactGraphError::ZeroRecordWidth {
            field: "node_record_bytes",
        });
    }
    if edge_record_bytes == 0 {
        return Err(DeviceResidentTokenFactGraphError::ZeroRecordWidth {
            field: "edge_record_bytes",
        });
    }
    let node_count = u64::try_from(graph.node_ids.len()).map_err(|_| {
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "node count",
        }
    })?;
    let edge_count = u64::try_from(graph.column_indices.len()).map_err(|_| {
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "edge count",
        }
    })?;
    let (max_out_degree, top_out_degree_prefix_sums) =
        csr_out_degree_profile(graph, edge_count, &mut scratch.row_degrees)?;
    let node_bytes = checked_mul(node_count, node_record_bytes, "node bytes")?;
    let edge_bytes = checked_mul(edge_count, edge_record_bytes, "edge bytes")?;
    let resident_without_payload = checked_add(node_bytes, edge_bytes, "node plus edge bytes")?;
    let resident_bytes = checked_add(
        resident_without_payload,
        graph.payload_bytes,
        "resident bytes",
    )?;

    Ok(DeviceResidentTokenFactGraphLayout {
        node_count,
        edge_count,
        max_out_degree,
        top_out_degree_prefix_sums,
        node_record_bytes,
        edge_record_bytes,
        node_bytes,
        edge_bytes,
        payload_bytes: graph.payload_bytes,
        resident_bytes,
    })
}

/// Canonical op id for device-resident token/fact graph traversal.
pub const OP_ID: &str = "vyre-libs::device::device_resident_token_fact_graph";

/// Build a Program that dispatches device-resident token/fact graph traversal.
pub fn device_resident_token_fact_graph_program(
    nodes: &[TokenFactNode],
    edges: &[TokenFactEdge],
    payload_bytes: u64,
    node_record_bytes: u64,
    edge_record_bytes: u64,
    row_offsets_buf: &str,
    column_indices_buf: &str,
    out_buf: &str,
) -> Result<Program, DeviceResidentTokenFactGraphError> {
    let graph = plan_device_resident_token_fact_graph(nodes, edges, payload_bytes)?;
    let layout =
        plan_device_resident_token_fact_graph_layout(&graph, node_record_bytes, edge_record_bytes)?;
    let node_count = u32::try_from(layout.node_count)
        .map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
    let edge_count = u32::try_from(layout.edge_count)
        .map_err(|_| DeviceResidentTokenFactGraphError::CsrIndexOverflow)?;
    let t = Expr::InvocationId { axis: 0 };
    let body = vec![Node::store(
        out_buf,
        t.clone(),
        Expr::load(column_indices_buf, t.clone()),
    )];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(row_offsets_buf, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count.saturating_add(1)),
            BufferDecl::storage(column_indices_buf, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(edge_count),
            BufferDecl::storage(out_buf, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(edge_count),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(Expr::lt(t, Expr::u32(edge_count)), body)],
        )],
    ))
}

fn checked_mul(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, DeviceResidentTokenFactGraphError> {
    left.checked_mul(right)
        .ok_or(DeviceResidentTokenFactGraphError::ByteCountOverflow { field })
}

fn checked_add(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, DeviceResidentTokenFactGraphError> {
    left.checked_add(right)
        .ok_or(DeviceResidentTokenFactGraphError::ByteCountOverflow { field })
}

/// Validate the CSR rows and report the top out-degrees the profile needs.
///
/// Degrees are collected once into a reusable `u32` buffer, and only when a
/// graph has more rows than the highest reported rank is the buffer reduced to
/// that prefix. Sorting the retained prefix is then the whole cost. A heap kept
/// per row instead paid a sift for every row of every graph.
fn csr_out_degree_profile(
    graph: &DeviceResidentTokenFactGraph,
    edge_count: u64,
    row_degrees: &mut Vec<u32>,
) -> Result<(u64, [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS]), DeviceResidentTokenFactGraphError> {
    let expected_row_offsets = graph.node_ids.len().checked_add(1).ok_or(
        DeviceResidentTokenFactGraphError::ByteCountOverflow {
            field: "row offset count",
        },
    )?;
    if graph.row_offsets.len() != expected_row_offsets {
        return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets length",
        });
    }
    let declared_edges = u64::from(*graph.row_offsets.last().ok_or(
        DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets terminator",
        },
    )?);
    if declared_edges != edge_count {
        return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
            field: "row_offsets edge count",
        });
    }

    row_degrees.clear();
    row_degrees.reserve(graph.node_ids.len());
    let mut max_out_degree = 0_u32;
    for row in graph.row_offsets.windows(2) {
        let start = row[0];
        let end = row[1];
        if end < start {
            return Err(DeviceResidentTokenFactGraphError::InvalidCsrShape {
                field: "row_offsets ordering",
            });
        }
        let degree = end - start;
        max_out_degree = max_out_degree.max(degree);
        row_degrees.push(degree);
    }
    if row_degrees.len() > TOKEN_FACT_DEGREE_PROFILE_MAX_RANK {
        row_degrees.select_nth_unstable_by(TOKEN_FACT_DEGREE_PROFILE_MAX_RANK - 1, |lhs, rhs| {
            rhs.cmp(lhs)
        });
        row_degrees.truncate(TOKEN_FACT_DEGREE_PROFILE_MAX_RANK);
    }
    row_degrees.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

    let mut prefix_sum = 0_u64;
    let mut prefix_sums = [0_u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS];
    let mut bucket = 0_usize;
    for (index, degree) in row_degrees.iter().enumerate() {
        prefix_sum = checked_add(prefix_sum, u64::from(*degree), "top out-degree prefix sum")?;
        let rank = u64::try_from(index + 1).map_err(|_| {
            DeviceResidentTokenFactGraphError::ByteCountOverflow {
                field: "out-degree profile rank",
            }
        })?;
        while bucket < TOKEN_FACT_DEGREE_PROFILE_BUCKETS
            && rank >= TOKEN_FACT_DEGREE_PROFILE_RANKS[bucket]
        {
            prefix_sums[bucket] = prefix_sum;
            bucket += 1;
        }
    }
    while bucket < TOKEN_FACT_DEGREE_PROFILE_BUCKETS {
        prefix_sums[bucket] = prefix_sum;
        bucket += 1;
    }
    Ok((u64::from(max_out_degree), prefix_sums))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_fact_graph_packs_stable_shared_csr() {
        let graph = plan_device_resident_token_fact_graph(
            &[
                TokenFactNode::new(20, TokenFactNodeKind::Fact, 12, 4),
                TokenFactNode::new(10, TokenFactNodeKind::Token, 0, 4),
                TokenFactNode::new(30, TokenFactNodeKind::Diagnostic, 20, 8),
            ],
            &[
                TokenFactEdge::new(20, 30, TokenFactEdgeKind::DiagnosticProvenance),
                TokenFactEdge::new(10, 20, TokenFactEdgeKind::SemanticFact),
            ],
            32,
        )
        .expect("Fix: valid token/fact graph should pack");

        assert_eq!(graph.node_ids, vec![10, 20, 30]);
        assert_eq!(
            graph.node_kinds,
            vec![
                TokenFactNodeKind::Token,
                TokenFactNodeKind::Fact,
                TokenFactNodeKind::Diagnostic,
            ]
        );
        assert_eq!(graph.row_offsets, vec![0, 1, 2, 2]);
        assert_eq!(graph.column_indices, vec![1, 2]);
        assert_eq!(
            graph.edge_kinds,
            vec![
                TokenFactEdgeKind::SemanticFact,
                TokenFactEdgeKind::DiagnosticProvenance,
            ]
        );
        assert_eq!(graph.token_nodes, 1);
        assert_eq!(graph.fact_nodes, 1);
    }

    #[test]
    fn token_fact_graph_deduplicates_parallel_edges_deterministically() {
        let graph = plan_device_resident_token_fact_graph(
            &[
                TokenFactNode::new(2, TokenFactNodeKind::Fact, 4, 4),
                TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 4),
            ],
            &[
                TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact),
                TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact),
            ],
            8,
        )
        .expect("Fix: duplicate edges should deduplicate inside a resident row");

        assert_eq!(graph.row_offsets, vec![0, 1, 1]);
        assert_eq!(graph.column_indices, vec![1]);
    }

    #[test]
    fn token_fact_graph_rejects_invalid_layouts() {
        assert_eq!(
            plan_device_resident_token_fact_graph(
                &[
                    TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 1),
                    TokenFactNode::new(1, TokenFactNodeKind::Fact, 1, 1),
                ],
                &[],
                2,
            )
            .expect_err("duplicate nodes should fail"),
            DeviceResidentTokenFactGraphError::DuplicateNode { id: 1 }
        );
        assert_eq!(
            plan_device_resident_token_fact_graph(
                &[TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 1)],
                &[TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact)],
                1,
            )
            .expect_err("unknown edge nodes should fail"),
            DeviceResidentTokenFactGraphError::UnknownEdgeNode { id: 2 }
        );
        assert_eq!(
            plan_device_resident_token_fact_graph(
                &[TokenFactNode::new(1, TokenFactNodeKind::Token, 8, 8)],
                &[],
                12,
            )
            .expect_err("payload overflow beyond slab should fail"),
            DeviceResidentTokenFactGraphError::PayloadOutOfBounds {
                id: 1,
                end: 16,
                payload_bytes: 12,
            }
        );
    }

    /// WHY: duplicate detection moved from a first-seen hash probe to an
    /// adjacent-pair scan over sorted ids. The reported id must therefore be the
    /// smallest duplicated one no matter how the producer ordered its nodes, or
    /// two runs over the same graph would blame different nodes.
    #[test]
    fn token_fact_graph_reports_the_smallest_duplicate_regardless_of_input_order() {
        for order in [[9_u32, 4, 9, 4], [4, 9, 4, 9], [4, 4, 9, 9]] {
            let nodes = order
                .iter()
                .map(|id| TokenFactNode::new(*id, TokenFactNodeKind::Token, 0, 0))
                .collect::<Vec<_>>();
            assert_eq!(
                plan_device_resident_token_fact_graph(&nodes, &[], 4)
                    .expect_err("duplicate nodes should fail"),
                DeviceResidentTokenFactGraphError::DuplicateNode { id: 4 },
                "input order {order:?}"
            );
        }
    }

    #[test]
    fn token_fact_graph_packs_large_unsorted_input_with_stable_indices() {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        for id in (0..1024_u32).rev() {
            nodes.push(TokenFactNode::new(
                id,
                TokenFactNodeKind::Token,
                u64::from(id),
                1,
            ));
            if id > 0 {
                edges.push(TokenFactEdge::new(id - 1, id, TokenFactEdgeKind::TokenFlow));
            }
        }

        let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 1024)
            .expect("Fix: large unsorted token/fact graph should pack deterministically");

        assert_eq!(graph.node_ids[0], 0);
        assert_eq!(graph.node_ids[1023], 1023);
        assert_eq!(graph.row_offsets[0], 0);
        assert_eq!(graph.row_offsets[1024], 1023);
        assert_eq!(graph.column_indices[0], 1);
    }

    #[test]
    fn token_fact_graph_scratch_reuses_staging_allocations() {
        let mut scratch = DeviceResidentTokenFactGraphScratch::new();
        let nodes = [
            TokenFactNode::new(3, TokenFactNodeKind::Fact, 2, 1),
            TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 1),
            TokenFactNode::new(2, TokenFactNodeKind::Semantic, 1, 1),
        ];
        let edges = [
            TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact),
            TokenFactEdge::new(2, 3, TokenFactEdgeKind::FactDependency),
        ];
        plan_device_resident_token_fact_graph_with_scratch(&nodes, &edges, 3, &mut scratch)
            .expect("Fix: first scratch-backed token/fact graph should pack");
        let ordered_capacity = scratch.ordered_nodes.capacity();
        let staged_capacity = scratch.staged_edges.capacity();

        let graph =
            plan_device_resident_token_fact_graph_with_scratch(&nodes[..2], &[], 3, &mut scratch)
                .expect("Fix: smaller scratch-backed token/fact graph should reuse staging");

        assert_eq!(scratch.ordered_nodes.capacity(), ordered_capacity);
        assert_eq!(scratch.staged_edges.capacity(), staged_capacity);
        assert_eq!(graph.node_ids, vec![1, 3]);
        assert_eq!(
            graph.row_offsets,
            vec![0, 0, 0],
            "Fix: unknown edge rows must not leak from previous scratch contents."
        );
    }

    /// WHY: the degree profile is the only per-row allocation left on the
    /// residency path. A second layout call over a smaller graph must reuse the
    /// same buffer, or the profile reintroduces a per-plan allocation.
    #[test]
    fn layout_scratch_reuses_the_out_degree_buffer() {
        let mut scratch = DeviceResidentTokenFactGraphScratch::new();
        let nodes = (0..512_u32)
            .map(|id| TokenFactNode::new(id, TokenFactNodeKind::Fact, u64::from(id) * 4, 4))
            .collect::<Vec<_>>();
        let edges = (1..512_u32)
            .map(|id| TokenFactEdge::new(0, id, TokenFactEdgeKind::FactDependency))
            .collect::<Vec<_>>();
        let graph = plan_device_resident_token_fact_graph(&nodes, &edges, 2_048)
            .expect("Fix: profiling graph should pack");
        plan_device_resident_token_fact_graph_layout_with_scratch(&graph, 32, 16, &mut scratch)
            .expect("Fix: first layout should plan");
        let degree_capacity = scratch.row_degrees.capacity();
        assert!(degree_capacity >= 512);

        let small = plan_device_resident_token_fact_graph(&nodes[..4], &[], 2_048)
            .expect("Fix: smaller profiling graph should pack");
        plan_device_resident_token_fact_graph_layout_with_scratch(&small, 32, 16, &mut scratch)
            .expect("Fix: second layout should plan");

        assert_eq!(scratch.row_degrees.capacity(), degree_capacity);
    }

    #[test]
    fn layout_accounts_for_the_resident_byte_envelope() {
        let graph = plan_device_resident_token_fact_graph(
            &[
                TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 8),
                TokenFactNode::new(2, TokenFactNodeKind::Semantic, 8, 8),
                TokenFactNode::new(3, TokenFactNodeKind::Fact, 16, 8),
            ],
            &[
                TokenFactEdge::new(1, 2, TokenFactEdgeKind::SemanticFact),
                TokenFactEdge::new(2, 3, TokenFactEdgeKind::FactDependency),
            ],
            24,
        )
        .expect("Fix: token/fact graph should pack");

        let layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
            .expect("Fix: token/fact graph should produce a resident layout");

        assert_eq!(layout.node_count, 3);
        assert_eq!(layout.edge_count, 2);
        assert_eq!(layout.max_out_degree, 1);
        assert_eq!(layout.top_out_degree_prefix_sums[0], 1);
        assert_eq!(layout.top_out_degree_prefix_sums[1], 2);
        assert_eq!(layout.top_out_degree_prefix_sums[15], 2);
        assert_eq!(layout.node_bytes, 96);
        assert_eq!(layout.edge_bytes, 32);
        assert_eq!(layout.resident_bytes, 152);
    }

    #[test]
    fn layout_exports_max_out_degree_for_hub_heavy_queue_planning() {
        let graph = plan_device_resident_token_fact_graph(
            &[
                TokenFactNode::new(1, TokenFactNodeKind::Fact, 0, 4),
                TokenFactNode::new(2, TokenFactNodeKind::Fact, 4, 4),
                TokenFactNode::new(3, TokenFactNodeKind::Fact, 8, 4),
                TokenFactNode::new(4, TokenFactNodeKind::Fact, 12, 4),
            ],
            &[
                TokenFactEdge::new(1, 2, TokenFactEdgeKind::FactDependency),
                TokenFactEdge::new(1, 3, TokenFactEdgeKind::FactDependency),
                TokenFactEdge::new(1, 4, TokenFactEdgeKind::FactDependency),
                TokenFactEdge::new(2, 3, TokenFactEdgeKind::FactDependency),
            ],
            16,
        )
        .expect("Fix: hub-heavy token/fact graph should pack");

        let layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
            .expect("Fix: hub-heavy token/fact graph should produce a resident layout");

        assert_eq!(layout.edge_count, 4);
        assert_eq!(layout.max_out_degree, 3);
        assert_eq!(layout.top_out_degree_prefix_sums[0], 3);
        assert_eq!(layout.top_out_degree_prefix_sums[1], 4);
        assert_eq!(layout.top_out_degree_prefix_sums[2], 4);
    }

    #[test]
    fn generated_layout_profiles_top_out_degree_prefixes() {
        let mut state = 0x5eec_c0de_f00d_7715_u64;
        let mut scratch = DeviceResidentTokenFactGraphScratch::new();
        for case_index in 0..4096_u64 {
            let node_count = 1 + (next_u64(&mut state) % 64) as u32;
            let nodes = (0..node_count)
                .map(|index| {
                    TokenFactNode::new(index + 1, TokenFactNodeKind::Fact, u64::from(index) * 4, 4)
                })
                .collect::<Vec<_>>();
            let mut edges = Vec::new();
            if case_index % 4 == 0 {
                for to in 2..=node_count {
                    edges.push(TokenFactEdge::new(1, to, TokenFactEdgeKind::FactDependency));
                }
            }
            let attempts = next_u64(&mut state) % (u64::from(node_count) * 5 + 1);
            for _ in 0..attempts {
                let from = 1 + (next_u64(&mut state) % u64::from(node_count)) as u32;
                let to = 1 + (next_u64(&mut state) % u64::from(node_count)) as u32;
                let kind = if next_u64(&mut state) & 1 == 0 {
                    TokenFactEdgeKind::FactDependency
                } else {
                    TokenFactEdgeKind::DiagnosticProvenance
                };
                edges.push(TokenFactEdge::new(from, to, kind));
            }
            let graph =
                plan_device_resident_token_fact_graph(&nodes, &edges, u64::from(node_count) * 4)
                    .expect("Fix: generated token/fact graph should pack");
            let layout = plan_device_resident_token_fact_graph_layout_with_scratch(
                &graph,
                32,
                16,
                &mut scratch,
            )
            .expect("Fix: generated token/fact graph should produce a resident layout");
            let mut degrees = graph
                .row_offsets
                .windows(2)
                .map(|row| u64::from(row[1] - row[0]))
                .collect::<Vec<_>>();
            degrees.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));

            assert_eq!(
                layout.max_out_degree,
                degrees.first().copied().unwrap_or(0),
                "case {case_index}"
            );
            for (bucket, rank) in TOKEN_FACT_DEGREE_PROFILE_RANKS.iter().enumerate() {
                let expected = degrees
                    .iter()
                    .take((*rank as usize).min(degrees.len()))
                    .copied()
                    .sum::<u64>();
                assert_eq!(
                    layout.top_out_degree_prefix_sums[bucket], expected,
                    "case {case_index} bucket {bucket}"
                );
            }
        }
    }

    #[test]
    fn layout_profiles_large_graph_with_bounded_top_rank_storage() {
        let node_count = 32_770_u32;
        let nodes = (0..node_count)
            .map(|index| {
                TokenFactNode::new(index + 1, TokenFactNodeKind::Fact, u64::from(index) * 4, 4)
            })
            .collect::<Vec<_>>();
        let mut edges = Vec::with_capacity(32_858);
        for to in 2..=51 {
            edges.push(TokenFactEdge::new(1, to, TokenFactEdgeKind::FactDependency));
        }
        for to in 3..=42 {
            edges.push(TokenFactEdge::new(2, to, TokenFactEdgeKind::FactDependency));
        }
        for from in 3..=node_count {
            edges.push(TokenFactEdge::new(
                from,
                1,
                TokenFactEdgeKind::FactDependency,
            ));
        }
        let graph =
            plan_device_resident_token_fact_graph(&nodes, &edges, u64::from(node_count) * 4)
                .expect("Fix: large skewed token/fact graph should pack");

        let layout = plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
            .expect("Fix: large skewed token/fact graph should produce a resident layout");

        assert_eq!(layout.node_count, u64::from(node_count));
        assert_eq!(layout.edge_count, 32_858);
        assert_eq!(layout.max_out_degree, 50);
        assert_eq!(layout.top_out_degree_prefix_sums[0], 50);
        assert_eq!(layout.top_out_degree_prefix_sums[1], 90);
        assert_eq!(layout.top_out_degree_prefix_sums[2], 92);
        assert_eq!(layout.top_out_degree_prefix_sums[15], 32_856);
    }

    #[test]
    fn aggregate_layout_constructor_preserves_the_safe_edge_bound() {
        let layout = DeviceResidentTokenFactGraphLayout::from_aggregate_fields(
            4, 9, 32, 16, 128, 144, 64, 336,
        );

        assert_eq!(layout.max_out_degree, 9);
        assert_eq!(
            layout.top_out_degree_prefix_sums,
            [9; TOKEN_FACT_DEGREE_PROFILE_BUCKETS]
        );
        assert_eq!(layout.resident_bytes, 336);
    }

    #[test]
    fn layout_rejects_missing_abi_widths() {
        let graph = plan_device_resident_token_fact_graph(&[], &[], 0)
            .expect("Fix: empty graph still has a valid resident layout");

        assert_eq!(
            plan_device_resident_token_fact_graph_layout(&graph, 0, 8)
                .expect_err("zero node record width should fail"),
            DeviceResidentTokenFactGraphError::ZeroRecordWidth {
                field: "node_record_bytes",
            }
        );
        assert_eq!(
            plan_device_resident_token_fact_graph_layout(&graph, 8, 0)
                .expect_err("zero edge record width should fail"),
            DeviceResidentTokenFactGraphError::ZeroRecordWidth {
                field: "edge_record_bytes",
            }
        );
    }

    #[test]
    fn layout_rejects_public_graphs_with_invalid_csr_rows() {
        let mut graph = plan_device_resident_token_fact_graph(
            &[TokenFactNode::new(1, TokenFactNodeKind::Fact, 0, 4)],
            &[],
            4,
        )
        .expect("Fix: token/fact graph should pack before adversarial mutation");
        graph.row_offsets[1] = 1;

        assert_eq!(
            plan_device_resident_token_fact_graph_layout(&graph, 32, 16)
                .expect_err("invalid CSR row offsets should fail before resident planning"),
            DeviceResidentTokenFactGraphError::InvalidCsrShape {
                field: "row_offsets edge count",
            }
        );
    }

    fn next_u64(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }

    #[test]
    fn device_resident_token_fact_graph_program_builds_valid_ir() {
        let nodes = [
            TokenFactNode::new(1, TokenFactNodeKind::Token, 0, 16),
            TokenFactNode::new(2, TokenFactNodeKind::Fact, 16, 16),
        ];
        let edges = [TokenFactEdge::new(1, 2, TokenFactEdgeKind::FactDependency)];
        let program = device_resident_token_fact_graph_program(
            &nodes,
            &edges,
            32,
            32,
            16,
            "row_offsets",
            "col_indices",
            "out",
        )
        .expect("Fix: valid token/fact graph program must build");
        assert_eq!(program.buffers().len(), 3);
    }
}
