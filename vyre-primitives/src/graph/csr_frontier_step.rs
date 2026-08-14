//! Shared CSR frontier-step Program builder and CPU reference.
//!
//! Forward and reverse traversals use the same ProgramGraph ABI,
//! frontier buffers, edge-kind mask filtering, and packed-NodeSet
//! output writes. The only semantic difference is whether the input
//! frontier is tested at `src` before walking outgoing edges or at
//! `dst` while scanning a source row.
//!
//! [`csr_frontier_step_cpu_ref_into`] walks the same two directions on the
//! host. It is written from the CSR arrays alone and never reads the emitted
//! `Program`, so it stays able to disagree with the program it checks; the
//! direction is its argument because the row scan and the edge-kind filter are
//! one walk, not two.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::graph::frontier_bits::{
    active_source_lane, bind_bit_address, set_bit, when_bit_set, BitAccess,
};
use crate::graph::program_graph::{
    frontier_buffer, ProgramGraphShape, BINDING_PRIMITIVE_START, NAME_EDGE_KIND_MASK,
    NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};

/// Canonical binding index for the input frontier bitset.
pub const BINDING_FRONTIER_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output frontier bitset.
pub const BINDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 1;
/// Binding index for the excluded-source mask of the excluding forward step.
pub const BINDING_EXCLUDED_SOURCES: u32 = BINDING_PRIMITIVE_START + 1;
/// Binding index for the output frontier of the excluding forward step.
///
/// The excluded-source mask takes the slot [`BINDING_FRONTIER_OUT`] holds in
/// every other frontier step, so the output frontier sits one slot further out.
pub const BINDING_EXCLUDING_FRONTIER_OUT: u32 = BINDING_PRIMITIVE_START + 2;
pub(crate) const CSR_FRONTIER_STEP_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid for one source-lane CSR frontier step.
#[must_use]
pub const fn csr_frontier_step_dispatch_grid(node_count: u32) -> [u32; 3] {
    crate::graph::lane_grid(node_count, CSR_FRONTIER_STEP_WORKGROUP_SIZE[0])
}

/// Direction for a one-step CSR frontier traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CsrFrontierStepKind {
    /// If `src` is active, emit each allowed `dst`.
    Forward,
    /// If any allowed `dst` is active, emit `src`.
    Backward,
}

/// Bit test at a node index in a caller-supplied frontier bitset.
///
/// A frontier shorter than the graph reads as inactive rather than panicking:
/// callers stage frontier words independently of `node_count`, and a short
/// frontier must not turn a layout mistake into an out-of-bounds host read.
#[cfg(any(test, feature = "cpu-parity"))]
fn frontier_bit_is_set(frontier: &[u32], node: u32) -> bool {
    frontier
        .get((node / 32) as usize)
        .is_some_and(|word| (word & (1_u32 << (node % 32))) != 0)
}

/// Set the bit for an in-range node in a node-indexed output bitset.
#[cfg(any(test, feature = "cpu-parity"))]
fn set_node_bit(out: &mut [u32], node_count: u32, node: u32) {
    if node < node_count {
        out[(node / 32) as usize] |= 1_u32 << (node % 32);
    }
}

/// Validate the CSR buffers a host frontier step reads.
///
/// The CPU oracle is used as GPU parity evidence, so malformed graph layouts
/// must fail loudly instead of producing an empty frontier that can mask
/// upstream object corruption. Returns the logical edge count.
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn validate_csr_frontier_step_cpu_inputs(
    label: &str,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> usize {
    let expected_offsets = node_count as usize + 1;
    assert_eq!(
        edge_offsets.len(),
        expected_offsets,
        "{label} CPU oracle received {} row offsets for node_count={node_count}; Fix: pass exactly node_count + 1 CSR offsets.",
        edge_offsets.len()
    );
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    assert!(
        edge_targets.len() >= edge_count && edge_kind_mask.len() >= edge_count,
        "{label} CPU oracle received edge_count={edge_count} but targets_len={} kind_mask_len={}. Fix: pass complete CSR edge buffers.",
        edge_targets.len(),
        edge_kind_mask.len()
    );
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        assert!(
            pair[0] <= pair[1],
            "{label} CPU oracle received non-monotonic CSR offsets at row {index}: {} > {}. Fix: rebuild CSR row pointers before parity comparison.",
            pair[0],
            pair[1]
        );
    }
    edge_count
}

/// CPU reference for one CSR frontier step in either edge direction.
///
/// Both directions scan every CSR row and filter edges by `allow_mask`. `kind`
/// decides which endpoint of an allowed edge is read from `frontier_in` and
/// which endpoint is written to `out`: forward reads `src` and writes `dst`,
/// backward reads `dst` and writes `src`. Forward hoists its read out of the
/// edge loop because the read endpoint is constant across a row, and backward
/// stops a row at its first active destination because the write is idempotent.
///
/// `out` is resized to the bitset width for `node_count` and fully overwritten.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub(crate) fn csr_frontier_step_cpu_ref_into(
    kind: CsrFrontierStepKind,
    label: &str,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.resize(crate::bitset::bitset_words(node_count) as usize, 0);
    validate_csr_frontier_step_cpu_inputs(
        label,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    );
    for src in 0..node_count {
        if kind == CsrFrontierStepKind::Forward && !frontier_bit_is_set(frontier_in, src) {
            continue;
        }
        let row_start = edge_offsets[src as usize] as usize;
        let row_end = edge_offsets[src as usize + 1] as usize;
        for edge in row_start..row_end {
            if (edge_kind_mask[edge] & allow_mask) == 0 {
                continue;
            }
            let dst = edge_targets[edge];
            match kind {
                CsrFrontierStepKind::Forward => set_node_bit(out, node_count, dst),
                CsrFrontierStepKind::Backward => {
                    if frontier_bit_is_set(frontier_in, dst) {
                        set_node_bit(out, node_count, src);
                        break;
                    }
                }
            }
        }
    }
}

/// Publish one op's CPU reference for a CSR frontier step.
///
/// Every op that is one masked CSR step republishes the same reference under
/// its own name: the two traversal primitives and each predicate that fixes an
/// edge-kind mask. The pair of entry points, their inputs, and the buffer-reuse
/// contract are stated here once; each op supplies its direction, the label its
/// diagnostics carry, and its own documentation.
///
/// The macro itself is always defined and always invocable: whether the entry
/// points it publishes exist is decided once, by the `cfg` on the items it
/// expands to, so an op invokes it unconditionally and expands to nothing when
/// host references are not built.
macro_rules! define_csr_frontier_step_cpu_ref {
    (
        direction: $direction:expr,
        label: $label:literal,
        $(#[$owned_meta:meta])*
        $owned_vis:vis fn $owned:ident,
        $(#[$into_meta:meta])*
        $into_vis:vis fn $into:ident,
    ) => {
        $(#[$owned_meta])*
        #[must_use]
        #[cfg(any(test, feature = "cpu-parity"))]
        $owned_vis fn $owned(
            node_count: u32,
            edge_offsets: &[u32],
            edge_targets: &[u32],
            edge_kind_mask: &[u32],
            frontier_in: &[u32],
            allow_mask: u32,
        ) -> Vec<u32> {
            let mut out = Vec::new();
            $into(
                node_count,
                edge_offsets,
                edge_targets,
                edge_kind_mask,
                frontier_in,
                allow_mask,
                &mut out,
            );
            out
        }

        $(#[$into_meta])*
        #[cfg(any(test, feature = "cpu-parity"))]
        $into_vis fn $into(
            node_count: u32,
            edge_offsets: &[u32],
            edge_targets: &[u32],
            edge_kind_mask: &[u32],
            frontier_in: &[u32],
            allow_mask: u32,
            out: &mut Vec<u32>,
        ) {
            $crate::graph::csr_frontier_step::csr_frontier_step_cpu_ref_into(
                $direction,
                $label,
                node_count,
                edge_offsets,
                edge_targets,
                edge_kind_mask,
                frontier_in,
                allow_mask,
                out,
            );
        }
    };
}

pub(crate) use define_csr_frontier_step_cpu_ref;

/// Build a one-step CSR frontier traversal under a caller-owned op id.
#[must_use]
pub(crate) fn csr_frontier_step_program(
    op_id: &'static str,
    kind: CsrFrontierStepKind,
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut buffers = shape.read_only_buffers();
    buffers.push(frontier_buffer(
        frontier_in,
        BINDING_FRONTIER_IN,
        BufferAccess::ReadOnly,
        shape.node_count,
    ));
    buffers.push(frontier_buffer(
        frontier_out,
        BINDING_FRONTIER_OUT,
        BufferAccess::ReadWrite,
        shape.node_count,
    ));

    let body = match kind {
        CsrFrontierStepKind::Forward => forward_body(
            shape.node_count,
            frontier_in,
            None,
            frontier_out,
            allow_mask,
            t,
        ),
        CsrFrontierStepKind::Backward => vec![Node::if_then(
            Expr::lt(t.clone(), Expr::u32(shape.node_count)),
            backward_body(shape.node_count, frontier_in, frontier_out, allow_mask, t),
        )],
    };

    Program::wrapped(
        buffers,
        CSR_FRONTIER_STEP_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}
/// Build a forward CSR step that excludes active source nodes selected by
/// `excluded_sources`.
#[must_use]
pub(crate) fn csr_forward_step_excluding_program(
    op_id: &'static str,
    shape: ProgramGraphShape,
    frontier_in: &str,
    excluded_sources: &str,
    frontier_out: &str,
    allow_mask: u32,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut buffers = shape.read_only_buffers();
    buffers.push(frontier_buffer(
        frontier_in,
        BINDING_FRONTIER_IN,
        BufferAccess::ReadOnly,
        shape.node_count,
    ));
    buffers.push(frontier_buffer(
        excluded_sources,
        BINDING_EXCLUDED_SOURCES,
        BufferAccess::ReadOnly,
        shape.node_count,
    ));
    buffers.push(frontier_buffer(
        frontier_out,
        BINDING_EXCLUDING_FRONTIER_OUT,
        BufferAccess::ReadWrite,
        shape.node_count,
    ));

    Program::wrapped(
        buffers,
        CSR_FRONTIER_STEP_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(forward_body(
                shape.node_count,
                frontier_in,
                Some(excluded_sources),
                frontier_out,
                allow_mask,
                t,
            )),
        }],
    )
}

fn forward_body(
    node_count: u32,
    frontier_in: &str,
    excluded_sources: Option<&str>,
    frontier_out: &str,
    allow_mask: u32,
    t: Expr,
) -> Vec<Node> {
    let active_body = crate::graph::edge_scan::csr_edge_expand_nodes(
        ProgramGraphShape::new(node_count, 0),
        frontier_out,
        Expr::var("src"),
        |word| word,
        Vec::new,
        allow_mask,
        "",
    );
    vec![active_source_lane(
        node_count,
        frontier_in,
        excluded_sources,
        t,
        active_body,
    )]
}

fn backward_body(
    node_count: u32,
    frontier_in: &str,
    frontier_out: &str,
    allow_mask: u32,
    t: Expr,
) -> Vec<Node> {
    let mut body = vec![
        Node::let_bind("src", t),
        Node::let_bind("hit", Expr::u32(0)),
    ];
    body.extend(edge_bounds_and_loop(vec![Node::if_then(
        Expr::eq(Expr::var("hit"), Expr::u32(0)),
        vec![
            Node::let_bind("kind_mask", Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e"))),
            Node::if_then(
                Expr::ne(
                    Expr::bitand(Expr::var("kind_mask"), Expr::u32(allow_mask)),
                    Expr::u32(0),
                ),
                vec![
                    Node::let_bind("dst", Expr::load(NAME_EDGE_TARGETS, Expr::var("e"))),
                    Node::if_then(
                        Expr::lt(Expr::var("dst"), Expr::u32(node_count)),
                        when_bit_set(
                            frontier_in,
                            &Expr::var("dst"),
                            BitAccess {
                                word: "dst_word_idx",
                                mask: "dst_bit",
                                value: "dst_word",
                            },
                            |word| word,
                            vec![Node::assign("hit", Expr::u32(1))],
                        ),
                    ),
                ],
            ),
        ],
    )]));
    body.push(Node::if_then(
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
    ));
    body
}

pub(crate) fn edge_scan_body(
    allow_mask: u32,
    before_kind_body: Vec<Node>,
    on_allowed_body: Vec<Node>,
) -> Vec<Node> {
    let mut loop_body = before_kind_body;
    loop_body.push(Node::let_bind(
        "kind_mask",
        Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
    ));
    loop_body.push(Node::if_then(
        Expr::ne(
            Expr::bitand(Expr::var("kind_mask"), Expr::u32(allow_mask)),
            Expr::u32(0),
        ),
        on_allowed_body,
    ));
    edge_bounds_and_loop(loop_body)
}

fn edge_bounds_and_loop(loop_body: Vec<Node>) -> Vec<Node> {
    vec![
        Node::let_bind(
            "edge_start",
            Expr::load(NAME_EDGE_OFFSETS, Expr::var("src")),
        ),
        Node::let_bind(
            "edge_end",
            Expr::load(NAME_EDGE_OFFSETS, Expr::add(Expr::var("src"), Expr::u32(1))),
        ),
        Node::loop_for(
            "e",
            Expr::var("edge_start"),
            Expr::var("edge_end"),
            loop_body,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Queue-driven CSR frontier step
// ---------------------------------------------------------------------------
//
// Once a wave has been compacted into an active queue, a frontier step reads
// its sources from that queue instead of scanning the whole bitset. Every queue
// entry point in `graph/` is one point in a three-axis space: how lanes are
// assigned to a queued row (stride), which lanes are in bounds, and what a
// reached destination writes. `csr_queue_step_program` is the ONE
// implementation of that loop; the entry points pick coordinates.

/// Lane assignment and in-bounds rule for one queued CSR source row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CsrQueueLanes {
    /// One invocation owns one queued row and walks it end to end. In bounds
    /// when the queue index is below the static capacity and the resident length.
    Scalar,
    /// A `lanes`-wide team stripes each queued row, in bounds under the same
    /// capacity-then-length rule applied to the team's queue index.
    Team { lanes: u32 },
    /// A `lanes`-wide team stripes each queued row, in bounds against an active
    /// lane count derived from `min(len, capacity)`. `launch_lanes` caps the
    /// launch: `Some(n)` covers the tail with a grid-stride loop over `n` lanes
    /// instead of launching a worst-wave-sized grid for every half-wave.
    ActiveTeam {
        lanes: u32,
        launch_lanes: Option<u32>,
    },
}

/// What a queue step writes when an allowed edge reaches an in-range destination.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CsrQueueEmit<'a> {
    /// Set the destination bit in a packed frontier bitset.
    Frontier { frontier_out: &'a str },
    /// OR the destination bit into a monotone accumulator and append only
    /// first-time discoveries to a second queue. The observed next length can
    /// exceed the capacity; the store is clamped so callers can detect overflow
    /// pressure without corrupting resident memory.
    Delta {
        accumulator: &'a str,
        next_queue: &'a str,
        next_len: &'a str,
        next_queue_capacity: u32,
    },
}

impl CsrQueueEmit<'_> {
    /// Buffer an invalid-input program reports its diagnostic through.
    const fn failure_output(&self) -> &str {
        match self {
            Self::Frontier { frontier_out } => frontier_out,
            Self::Delta { next_len, .. } => next_len,
        }
    }
}

/// What a queued row does with its edges once its degree is known.
#[derive(Clone, Copy, Debug)]
pub(crate) enum CsrQueueRowPlan<'a> {
    /// The owning lane expands every queued row.
    ExpandAll,
    /// Rows at or above `high_degree_threshold` are compacted into a second
    /// queue for a later row-strided pass. Rows that overflow that queue are
    /// expanded by the owning lane, so correctness never depends on its sizing.
    CompactHighDegree {
        high_queue: &'a str,
        high_len: &'a str,
        high_queue_capacity: u32,
        high_degree_threshold: u32,
    },
}

/// Resident CSR and queue buffers every queue step reads, in binding order 0..5.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CsrQueueInputs<'a> {
    pub active_queue: &'a str,
    pub queue_len: &'a str,
    pub edge_offsets: &'a str,
    pub edge_targets: &'a str,
    pub edge_kind_mask: &'a str,
}

/// One queue-driven CSR frontier step.
pub(crate) struct CsrQueueStepSpec<'a> {
    /// Registered operation id the emitted Region carries.
    pub op_id: &'static str,
    /// Entry-point function name, used verbatim in sizing diagnostics.
    pub builder_name: &'static str,
    /// Variable-name prefix that keeps each entry point's emitted IR distinct.
    pub prefix: &'a str,
    pub workgroup_size: [u32; 3],
    pub inputs: CsrQueueInputs<'a>,
    pub lanes: CsrQueueLanes,
    pub row_plan: CsrQueueRowPlan<'a>,
    pub emit: CsrQueueEmit<'a>,
    pub node_count: u32,
    pub edge_count: u32,
    pub queue_capacity: u32,
    pub allow_mask: u32,
}

/// Build one queue-driven CSR frontier step.
///
/// Callers validate their own zero-capacity and zero-node preconditions first,
/// so the diagnostics stay in the entry point that names them.
#[must_use]
pub(crate) fn csr_queue_step_program(spec: &CsrQueueStepSpec<'_>) -> Program {
    let edge_offset_count =
        match crate::graph::checked_csr_offset_count(spec.node_count, spec.builder_name) {
            Ok(edge_offset_count) => edge_offset_count,
            Err(error) => {
                return crate::invalid_output_program(
                    spec.op_id,
                    spec.emit.failure_output(),
                    DataType::U32,
                    error,
                );
            }
        };
    let words = crate::bitset::bitset_words(spec.node_count);
    let physical_edge_count = spec.edge_count.max(1);

    let mut buffers = vec![
        BufferDecl::storage(
            spec.inputs.active_queue,
            0,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(spec.queue_capacity),
        BufferDecl::storage(
            spec.inputs.queue_len,
            1,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(1),
        BufferDecl::storage(
            spec.inputs.edge_offsets,
            2,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(edge_offset_count),
        BufferDecl::storage(
            spec.inputs.edge_targets,
            3,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
        BufferDecl::storage(
            spec.inputs.edge_kind_mask,
            4,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(physical_edge_count),
    ];
    match spec.emit {
        CsrQueueEmit::Frontier { frontier_out } => {
            buffers.push(
                BufferDecl::storage(frontier_out, 5, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(words),
            );
        }
        CsrQueueEmit::Delta {
            accumulator,
            next_queue,
            next_len,
            next_queue_capacity,
        } => {
            buffers.push(
                BufferDecl::storage(accumulator, 5, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(words),
            );
            buffers.push(
                BufferDecl::storage(next_queue, 6, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(next_queue_capacity),
            );
            buffers.push(
                BufferDecl::storage(next_len, 7, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(1),
            );
        }
    }
    if let CsrQueueRowPlan::CompactHighDegree {
        high_queue,
        high_len,
        high_queue_capacity,
        ..
    } = spec.row_plan
    {
        buffers.push(
            BufferDecl::storage(high_queue, 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(high_queue_capacity),
        );
        buffers.push(
            BufferDecl::storage(high_len, 7, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        );
    }

    Program::wrapped(
        buffers,
        spec.workgroup_size,
        vec![Node::Region {
            generator: Ident::from(spec.op_id),
            source_region: None,
            body: Arc::new(csr_queue_step_body(spec)),
        }],
    )
}

impl CsrQueueStepSpec<'_> {
    /// One prefixed local variable name.
    fn var(&self, suffix: &str) -> String {
        let mut name = String::with_capacity(self.prefix.len() + 1 + suffix.len());
        name.push_str(self.prefix);
        name.push('_');
        name.push_str(suffix);
        name
    }
}

fn csr_queue_step_body(spec: &CsrQueueStepSpec<'_>) -> Vec<Node> {
    let lane = Expr::InvocationId { axis: 0 };
    match spec.lanes {
        CsrQueueLanes::Scalar => {
            let idx = spec.var("idx");
            let row = csr_queue_scalar_row_nodes(spec);
            vec![
                Node::let_bind(idx.as_str(), lane),
                csr_queue_slot_guard(spec, &idx, csr_queued_source_nodes(spec, &idx, row)),
            ]
        }
        CsrQueueLanes::Team { lanes } => {
            let lane_var = spec.var("lane");
            let queue_idx = spec.var("queue_idx");
            let row = csr_queue_team_row_nodes(spec, lanes);
            let mut body = vec![Node::let_bind(lane_var.as_str(), lane)];
            body.extend(csr_queue_team_lane_split(spec, lanes, &lane_var));
            body.push(csr_queue_slot_guard(
                spec,
                &queue_idx,
                csr_queued_source_nodes(spec, &queue_idx, row),
            ));
            body
        }
        CsrQueueLanes::ActiveTeam {
            lanes,
            launch_lanes,
        } => {
            let lane_var = spec.var("lane");
            let active_slots = spec.var("active_slots");
            let active_lanes = spec.var("active_lanes");
            let mut body = vec![
                Node::let_bind(lane_var.as_str(), lane),
                Node::let_bind(
                    active_slots.as_str(),
                    Expr::min(
                        Expr::load(spec.inputs.queue_len, Expr::u32(0)),
                        Expr::u32(spec.queue_capacity),
                    ),
                ),
                Node::let_bind(
                    active_lanes.as_str(),
                    Expr::mul(Expr::var(active_slots.as_str()), Expr::u32(lanes)),
                ),
            ];
            match launch_lanes {
                None => body.push(Node::if_then(
                    Expr::lt(
                        Expr::var(lane_var.as_str()),
                        Expr::var(active_lanes.as_str()),
                    ),
                    csr_queue_active_lane_nodes(spec, lanes, Expr::var(lane_var.as_str())),
                )),
                Some(launch_lanes) => {
                    let launch = spec.var("launch_lanes");
                    let remaining = spec.var("remaining_lanes");
                    let lane_iters = spec.var("lane_iters");
                    let lane_iter = spec.var("lane_iter");
                    body.push(Node::let_bind(launch.as_str(), Expr::u32(launch_lanes)));
                    body.push(Node::if_then(
                        Expr::and(
                            Expr::lt(Expr::var(lane_var.as_str()), Expr::var(launch.as_str())),
                            Expr::lt(
                                Expr::var(lane_var.as_str()),
                                Expr::var(active_lanes.as_str()),
                            ),
                        ),
                        vec![
                            Node::let_bind(
                                remaining.as_str(),
                                Expr::sub(
                                    Expr::var(active_lanes.as_str()),
                                    Expr::var(lane_var.as_str()),
                                ),
                            ),
                            Node::let_bind(
                                lane_iters.as_str(),
                                Expr::add(
                                    Expr::u32(1),
                                    Expr::div(
                                        Expr::sub(Expr::var(remaining.as_str()), Expr::u32(1)),
                                        Expr::var(launch.as_str()),
                                    ),
                                ),
                            ),
                            Node::loop_for(
                                lane_iter.as_str(),
                                Expr::u32(0),
                                Expr::var(lane_iters.as_str()),
                                csr_queue_active_lane_nodes(
                                    spec,
                                    lanes,
                                    Expr::add(
                                        Expr::var(lane_var.as_str()),
                                        Expr::mul(
                                            Expr::var(lane_iter.as_str()),
                                            Expr::var(launch.as_str()),
                                        ),
                                    ),
                                ),
                            ),
                        ],
                    ));
                }
            }
            body
        }
    }
}

/// One lane team's whole share of work, addressed by a flat logical lane index.
fn csr_queue_active_lane_nodes(
    spec: &CsrQueueStepSpec<'_>,
    lanes: u32,
    logical_lane: Expr,
) -> Vec<Node> {
    let logical = spec.var("logical_lane");
    let queue_idx = spec.var("queue_idx");
    let row = csr_queue_team_row_nodes(spec, lanes);
    let mut nodes = vec![Node::let_bind(logical.as_str(), logical_lane)];
    nodes.extend(csr_queue_team_lane_split(spec, lanes, &logical));
    nodes.extend(csr_queued_source_nodes(spec, &queue_idx, row));
    nodes
}

/// Split a flat lane index into the queue slot it serves and its lane within
/// that slot's team. `logical` names the variable holding the flat index.
fn csr_queue_team_lane_split(spec: &CsrQueueStepSpec<'_>, lanes: u32, logical: &str) -> Vec<Node> {
    vec![
        Node::let_bind(
            spec.var("queue_idx"),
            Expr::div(Expr::var(logical), Expr::u32(lanes)),
        ),
        Node::let_bind(
            spec.var("edge_lane"),
            Expr::rem(Expr::var(logical), Expr::u32(lanes)),
        ),
    ]
}

/// Guard a queue index against the static capacity and the resident length.
fn csr_queue_slot_guard(spec: &CsrQueueStepSpec<'_>, index: &str, body: Vec<Node>) -> Node {
    Node::if_then(
        Expr::lt(Expr::var(index), Expr::u32(spec.queue_capacity)),
        vec![Node::if_then(
            Expr::lt(
                Expr::var(index),
                Expr::load(spec.inputs.queue_len, Expr::u32(0)),
            ),
            body,
        )],
    )
}

/// Load the queued source node and gate the row body on it being in range.
fn csr_queued_source_nodes(
    spec: &CsrQueueStepSpec<'_>,
    queue_index: &str,
    row_body: Vec<Node>,
) -> Vec<Node> {
    let src = spec.var("src");
    vec![
        Node::let_bind(
            src.as_str(),
            Expr::load(spec.inputs.active_queue, Expr::var(queue_index)),
        ),
        Node::if_then(
            Expr::lt(Expr::var(src.as_str()), Expr::u32(spec.node_count)),
            row_body,
        ),
    ]
}

/// CSR row bounds for the queued source.
fn csr_queue_row_bounds(spec: &CsrQueueStepSpec<'_>) -> Vec<Node> {
    let src = spec.var("src");
    vec![
        Node::let_bind(
            spec.var("edge_start"),
            Expr::load(spec.inputs.edge_offsets, Expr::var(src.as_str())),
        ),
        Node::let_bind(
            spec.var("edge_end"),
            Expr::load(
                spec.inputs.edge_offsets,
                Expr::add(Expr::var(src.as_str()), Expr::u32(1)),
            ),
        ),
    ]
}

/// Walk every edge of the queued row from the owning lane.
fn csr_queue_scalar_walk(spec: &CsrQueueStepSpec<'_>) -> Vec<Node> {
    vec![Node::loop_for(
        spec.var("e"),
        Expr::var(spec.var("edge_start").as_str()),
        Expr::var(spec.var("edge_end").as_str()),
        csr_queue_edge_guard_nodes(spec),
    )]
}

/// Row body for a scalar lane: walk the row, or compact hubs into a second
/// queue and walk only what did not fit.
fn csr_queue_scalar_row_nodes(spec: &CsrQueueStepSpec<'_>) -> Vec<Node> {
    let mut nodes = csr_queue_row_bounds(spec);
    match spec.row_plan {
        CsrQueueRowPlan::ExpandAll => nodes.extend(csr_queue_scalar_walk(spec)),
        CsrQueueRowPlan::CompactHighDegree {
            high_queue,
            high_len,
            high_queue_capacity,
            high_degree_threshold,
        } => {
            let degree = spec.var("degree");
            let high_slot = spec.var("high_slot");
            nodes.push(Node::let_bind(
                degree.as_str(),
                Expr::sub(
                    Expr::var(spec.var("edge_end").as_str()),
                    Expr::var(spec.var("edge_start").as_str()),
                ),
            ));
            nodes.push(Node::if_then_else(
                Expr::ge(Expr::var(degree.as_str()), Expr::u32(high_degree_threshold)),
                vec![
                    Node::let_bind(
                        high_slot.as_str(),
                        Expr::atomic_add(high_len, Expr::u32(0), Expr::u32(1)),
                    ),
                    Node::if_then_else(
                        Expr::lt(
                            Expr::var(high_slot.as_str()),
                            Expr::u32(high_queue_capacity),
                        ),
                        vec![Node::store(
                            high_queue,
                            Expr::var(high_slot.as_str()),
                            Expr::var(spec.var("src").as_str()),
                        )],
                        csr_queue_scalar_walk(spec),
                    ),
                ],
                csr_queue_scalar_walk(spec),
            ));
        }
    }
    nodes
}

/// Row body for a lane team: stripe the row `lanes` edges at a time so one
/// high-degree hub cannot serialize behind a single invocation.
fn csr_queue_team_row_nodes(spec: &CsrQueueStepSpec<'_>, lanes: u32) -> Vec<Node> {
    let degree = spec.var("degree");
    let full_iters = spec.var("full_iters");
    let tail_iter = spec.var("tail_iter");
    let iters = spec.var("iters");
    let iter = spec.var("iter");
    let edge_offset = spec.var("edge_offset");
    let edge = spec.var("e");
    let mut nodes = csr_queue_row_bounds(spec);
    nodes.extend([
        Node::let_bind(
            degree.as_str(),
            Expr::sub(
                Expr::var(spec.var("edge_end").as_str()),
                Expr::var(spec.var("edge_start").as_str()),
            ),
        ),
        Node::let_bind(
            full_iters.as_str(),
            Expr::div(Expr::var(degree.as_str()), Expr::u32(lanes)),
        ),
        Node::let_bind(
            tail_iter.as_str(),
            Expr::select(
                Expr::ne(
                    Expr::rem(Expr::var(degree.as_str()), Expr::u32(lanes)),
                    Expr::u32(0),
                ),
                Expr::u32(1),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            iters.as_str(),
            Expr::add(
                Expr::var(full_iters.as_str()),
                Expr::var(tail_iter.as_str()),
            ),
        ),
        Node::loop_for(
            iter.as_str(),
            Expr::u32(0),
            Expr::var(iters.as_str()),
            vec![
                Node::let_bind(
                    edge_offset.as_str(),
                    Expr::add(
                        Expr::var(spec.var("edge_lane").as_str()),
                        Expr::mul(Expr::var(iter.as_str()), Expr::u32(lanes)),
                    ),
                ),
                Node::if_then(
                    Expr::lt(Expr::var(edge_offset.as_str()), Expr::var(degree.as_str())),
                    {
                        let mut body = vec![Node::let_bind(
                            edge.as_str(),
                            Expr::add(
                                Expr::var(spec.var("edge_start").as_str()),
                                Expr::var(edge_offset.as_str()),
                            ),
                        )];
                        body.extend(csr_queue_edge_guard_nodes(spec));
                        body
                    },
                ),
            ],
        ),
    ]);
    nodes
}

/// Bounds-check the edge slot, apply the edge-kind allow mask, bounds-check the
/// destination, split it into a bitset word and bit, then emit.
fn csr_queue_edge_guard_nodes(spec: &CsrQueueStepSpec<'_>) -> Vec<Node> {
    let edge = spec.var("e");
    let kind = spec.var("kind");
    let dst = spec.var("dst");
    let dst_word = spec.var("dst_word");
    let dst_bit = spec.var("dst_bit");
    vec![Node::if_then(
        Expr::lt(Expr::var(edge.as_str()), Expr::u32(spec.edge_count)),
        vec![
            Node::let_bind(
                kind.as_str(),
                Expr::load(spec.inputs.edge_kind_mask, Expr::var(edge.as_str())),
            ),
            Node::if_then(
                Expr::ne(
                    Expr::bitand(Expr::var(kind.as_str()), Expr::u32(spec.allow_mask)),
                    Expr::u32(0),
                ),
                vec![
                    Node::let_bind(
                        dst.as_str(),
                        Expr::load(spec.inputs.edge_targets, Expr::var(edge.as_str())),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var(dst.as_str()), Expr::u32(spec.node_count)),
                        {
                            let mut body = bind_bit_address(
                                &Expr::var(dst.as_str()),
                                dst_word.as_str(),
                                dst_bit.as_str(),
                                |word| word,
                            )
                            .to_vec();
                            body.extend(csr_queue_emit_nodes(spec, &dst, &dst_word, &dst_bit));
                            body
                        },
                    ),
                ],
            ),
        ],
    )]
}

fn csr_queue_emit_nodes(
    spec: &CsrQueueStepSpec<'_>,
    dst: &str,
    dst_word: &str,
    dst_bit: &str,
) -> Vec<Node> {
    match spec.emit {
        CsrQueueEmit::Frontier { frontier_out } => {
            let mut prev = String::with_capacity(spec.prefix.len() + 7);
            prev.push('_');
            prev.push_str(spec.prefix);
            prev.push_str("_prev");
            vec![Node::let_bind(
                prev,
                Expr::atomic_or(frontier_out, Expr::var(dst_word), Expr::var(dst_bit)),
            )]
        }
        CsrQueueEmit::Delta {
            accumulator,
            next_queue,
            next_len,
            next_queue_capacity,
        } => {
            let old = spec.var("old");
            let slot = spec.var("slot");
            vec![
                Node::let_bind(
                    old.as_str(),
                    Expr::atomic_or(accumulator, Expr::var(dst_word), Expr::var(dst_bit)),
                ),
                Node::if_then(
                    Expr::eq(
                        Expr::bitand(Expr::var(old.as_str()), Expr::var(dst_bit)),
                        Expr::u32(0),
                    ),
                    vec![
                        Node::let_bind(
                            slot.as_str(),
                            Expr::atomic_add(next_len, Expr::u32(0), Expr::u32(1)),
                        ),
                        Node::if_then(
                            Expr::lt(Expr::var(slot.as_str()), Expr::u32(next_queue_capacity)),
                            vec![Node::store(
                                next_queue,
                                Expr::var(slot.as_str()),
                                Expr::var(dst),
                            )],
                        ),
                    ],
                ),
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{csr_frontier_step_dispatch_grid, CSR_FRONTIER_STEP_WORKGROUP_SIZE};

    fn scalar_forward(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Vec<u32> {
        let mut out = vec![0_u32; crate::bitset::bitset_words(node_count) as usize];
        for src in 0..node_count {
            let src_word = (src / 32) as usize;
            if frontier_in
                .get(src_word)
                .copied()
                .is_none_or(|word| (word & (1_u32 << (src % 32))) == 0)
            {
                continue;
            }
            let start = edge_offsets[src as usize] as usize;
            let end = edge_offsets[src as usize + 1] as usize;
            for edge in start..end {
                if (edge_kind_mask[edge] & allow_mask) == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                if dst < node_count {
                    out[(dst / 32) as usize] |= 1_u32 << (dst % 32);
                }
            }
        }
        out
    }

    fn scalar_backward(
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        frontier_in: &[u32],
        allow_mask: u32,
    ) -> Vec<u32> {
        let mut out = vec![0_u32; crate::bitset::bitset_words(node_count) as usize];
        for src in 0..node_count {
            let start = edge_offsets[src as usize] as usize;
            let end = edge_offsets[src as usize + 1] as usize;
            let mut hit = false;
            for edge in start..end {
                if (edge_kind_mask[edge] & allow_mask) == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                if dst < node_count {
                    let word = (dst / 32) as usize;
                    let bit = 1_u32 << (dst % 32);
                    if frontier_in
                        .get(word)
                        .copied()
                        .is_some_and(|w| (w & bit) != 0)
                    {
                        hit = true;
                        break;
                    }
                }
            }
            if hit {
                out[(src / 32) as usize] |= 1_u32 << (src % 32);
            }
        }
        out
    }

    #[test]
    fn generated_csr_frontier_step_uses_block_sized_workgroup() {
        let program = crate::graph::csr_forward_traverse::csr_forward_traverse(
            crate::graph::program_graph::ProgramGraphShape::new(1024, 1536),
            "frontier_in",
            "frontier_out",
            u32::MAX,
        );

        assert_eq!(program.workgroup_size(), CSR_FRONTIER_STEP_WORKGROUP_SIZE);
        assert!(
            program.workgroup_size()[0] > 1,
            "Fix: CSR frontier traversal must not launch one CUDA block per source node."
        );
    }

    #[test]
    fn dispatch_grid_packs_source_lanes_into_workgroups() {
        assert_eq!(csr_frontier_step_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(256), [1, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(257), [2, 1, 1]);
        assert_eq!(csr_frontier_step_dispatch_grid(513), [3, 1, 1]);
    }

    #[test]
    fn generated_csr_frontier_steps_match_scalar_reference() {
        let mut state = 0xC5A1_F00D_u32;
        for case in 0..2048_u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let node_count = (state % 97) + 1;
            let mut offsets = Vec::with_capacity(node_count as usize + 1);
            let mut targets = Vec::new();
            let mut masks = Vec::new();
            offsets.push(0);
            for src in 0..node_count {
                state = state.rotate_left(5) ^ src.wrapping_mul(0x9E37_79B9);
                let degree = state % 5;
                for edge in 0..degree {
                    state = state.rotate_left(7) ^ edge.wrapping_mul(0x85EB_CA6B);
                    let target = match edge % 5 {
                        0 => state % node_count,
                        1 => node_count,
                        2 => u32::MAX,
                        _ => state % (node_count + 3),
                    };
                    targets.push(target);
                    masks.push(1_u32 << (state & 7));
                }
                offsets.push(targets.len() as u32);
            }
            let words = crate::bitset::bitset_words(node_count) as usize;
            let mut frontier = vec![0_u32; words];
            for node in 0..node_count {
                state = state.rotate_left(3) ^ node.wrapping_mul(0x27D4_EB2D);
                if (state & 3) != 0 {
                    frontier[(node / 32) as usize] |= 1_u32 << (node % 32);
                }
            }
            let allow_mask = if case % 11 == 0 {
                0
            } else {
                (1_u32 << (case & 7)) | (1_u32 << ((case + 3) & 7))
            };

            assert_eq!(
                crate::graph::csr_forward_traverse::cpu_ref(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                scalar_forward(node_count, &offsets, &targets, &masks, &frontier, allow_mask),
                "forward case {case}"
            );
            assert_eq!(
                crate::graph::csr_backward_traverse::cpu_ref(
                    node_count, &offsets, &targets, &masks, &frontier, allow_mask,
                ),
                scalar_backward(node_count, &offsets, &targets, &masks, &frontier, allow_mask),
                "backward case {case}"
            );
        }
    }
}
