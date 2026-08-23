//! The emitted program: Kahn on lane zero, one invocation.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::OP_ID;

/// Build a single-invocation Program that runs Kahn's algorithm
/// serially on lane 0.
///
/// `offsets_buf` is a CSR row-pointer array with `node_count + 1`
/// entries; `targets_buf` is the CSR column array. `indeg_scratch`
/// and `queue_scratch` are caller-provided temporary buffers of
/// length `node_count`. `order_out` receives the topological order
/// (length `node_count` on an acyclic graph; fewer on a cyclic
/// graph because the kernel does not diagnose cycles).
///
/// Workgroup size is `[1, 1, 1]`. The kernel only executes on
/// invocation 0; other lanes return immediately.
#[must_use]
pub fn toposort_program(
    node_count: u32,
    offsets_buf: &str,
    targets_buf: &str,
    indeg_scratch: &str,
    queue_scratch: &str,
    order_out: &str,
) -> Program {
    let lane0 = Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0));

    let csr =
        crate::builder::csr::CsrTraversalComposer::new(OP_ID, OP_ID, node_count).with_buffers(
            crate::builder::csr::CsrBuffers::new(offsets_buf, targets_buf, None),
        );
    let [edge_start, edge_end] = csr.emit_row_offsets(Expr::var("v"), "edge_start", "edge_end");
    let step_inner = vec![
        Node::let_bind("v", Expr::load(queue_scratch, Expr::var("read_head"))),
        Node::assign("read_head", Expr::add(Expr::var("read_head"), Expr::u32(1))),
        Node::store(order_out, Expr::var("out_idx"), Expr::var("v")),
        Node::assign("out_idx", Expr::add(Expr::var("out_idx"), Expr::u32(1))),
        edge_start,
        edge_end,
        Node::loop_for(
            "e",
            Expr::var("edge_start"),
            Expr::var("edge_end"),
            vec![
                Node::let_bind("u", Expr::load(targets_buf, Expr::var("e"))),
                Node::let_bind(
                    "new_deg",
                    Expr::sub(Expr::load(indeg_scratch, Expr::var("u")), Expr::u32(1)),
                ),
                Node::store(indeg_scratch, Expr::var("u"), Expr::var("new_deg")),
                Node::if_then(
                    Expr::eq(Expr::var("new_deg"), Expr::u32(0)),
                    vec![
                        Node::store(queue_scratch, Expr::var("write_head"), Expr::var("u")),
                        Node::assign(
                            "write_head",
                            Expr::add(Expr::var("write_head"), Expr::u32(1)),
                        ),
                    ],
                ),
            ],
        ),
    ];

    let body = vec![
        // Zero out indeg_scratch.
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(node_count),
            vec![Node::store(indeg_scratch, Expr::var("i"), Expr::u32(0))],
        ),
        // Fill indegrees from edges. Edge count = offsets_buf[node_count].
        Node::let_bind("edge_count", Expr::load(offsets_buf, Expr::u32(node_count))),
        Node::loop_for(
            "e",
            Expr::u32(0),
            Expr::var("edge_count"),
            vec![
                Node::let_bind("t", Expr::load(targets_buf, Expr::var("e"))),
                Node::let_bind("old", Expr::load(indeg_scratch, Expr::var("t"))),
                Node::store(
                    indeg_scratch,
                    Expr::var("t"),
                    Expr::add(Expr::var("old"), Expr::u32(1)),
                ),
            ],
        ),
        // Seed queue with every node whose indegree is zero.
        Node::let_bind("write_head", Expr::u32(0)),
        Node::loop_for(
            "v",
            Expr::u32(0),
            Expr::u32(node_count),
            vec![Node::if_then(
                Expr::eq(Expr::load(indeg_scratch, Expr::var("v")), Expr::u32(0)),
                vec![
                    Node::store(queue_scratch, Expr::var("write_head"), Expr::var("v")),
                    Node::assign(
                        "write_head",
                        Expr::add(Expr::var("write_head"), Expr::u32(1)),
                    ),
                ],
            )],
        ),
        Node::let_bind("read_head", Expr::u32(0)),
        Node::let_bind("out_idx", Expr::u32(0)),
        Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::u32(node_count),
            vec![Node::if_then(
                Expr::lt(Expr::var("read_head"), Expr::var("write_head")),
                step_inner,
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(offsets_buf, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count.saturating_add(1)),
            BufferDecl::storage(targets_buf, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(indeg_scratch, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
            BufferDecl::storage(queue_scratch, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
            BufferDecl::storage(order_out, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(node_count.max(1)),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(lane0, body)],
        )],
    )
}
