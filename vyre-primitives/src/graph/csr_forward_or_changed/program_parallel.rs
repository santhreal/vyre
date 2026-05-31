use std::sync::Arc;

use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::MemoryOrdering;

use super::layout::{
    CSR_FORWARD_OR_CHANGED_CHANGED_BUFFER, CSR_FORWARD_OR_CHANGED_FRONTIER_BUFFER,
    CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE, OP_ID,
};
use crate::graph::program_graph::{
    ProgramGraphShape, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS,
};

/// Parallel in-place expansion program for production fixed-point drivers.
///
/// Unlike [`csr_forward_or_changed`], this variant gives each source node its
/// own invocation instead of walking the whole CSR from one lane. The pass is
/// monotone: each dispatch may observe only the frontier bits visible at that
/// point in the dispatch, but every newly discovered destination is ORed into
/// the same resident accumulator and sets `changed[0]`. Re-dispatch until the
/// changed flag stays zero to compute the same reachability fixpoint without a
/// full frontier readback per iteration.
#[must_use]
pub fn csr_forward_or_changed_parallel(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    let words = crate::bitset::bitset_words(shape.node_count);
    let body = csr_forward_or_changed_parallel_body_prefixed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        "",
    );
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            CSR_FORWARD_OR_CHANGED_FRONTIER_BUFFER,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            CSR_FORWARD_OR_CHANGED_CHANGED_BUFFER,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );
    Program::wrapped(
        buffers,
        CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

/// Build the parallel expansion body used by production closure drivers and
/// large persistent-BFS programs.
#[must_use]
pub fn csr_forward_or_changed_parallel_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Vec<Node> {
    csr_forward_or_changed_parallel_body_prefixed_impl(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        local_prefix,
        None,
    )
}

/// Build one parallel expansion body that snapshots source-node activity
/// before any lane writes newly reached destination bits.
#[must_use]
pub fn csr_forward_or_changed_parallel_snapshot_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Vec<Node> {
    csr_forward_or_changed_parallel_body_prefixed_impl(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        local_prefix,
        Some(MemoryOrdering::GridSync),
    )
}

fn csr_forward_or_changed_parallel_body_prefixed_impl(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
    snapshot_barrier: Option<MemoryOrdering>,
) -> Vec<Node> {
    let local = |name: &str| -> String {
        if local_prefix.is_empty() {
            name.to_string()
        } else {
            format!("{local_prefix}_{name}")
        }
    };
    let src = Expr::gid_x();
    let in_bounds = local("in_bounds");
    let word_idx = local("word_idx");
    let bit_mask = local("bit_mask");
    let src_word = local("src_word");
    let src_active = local("src_active");
    let edge_start = local("edge_start");
    let edge_end = local("edge_end");
    let edge_iter = local("e");
    let kind_mask = local("kind_mask");
    let dst = local("dst");
    let dst_word_idx = local("dst_word_idx");
    let dst_bit = local("dst_bit");
    let old = local("old");
    let changed_old = local("changed_old");

    let edge_scan = || {
        vec![
            Node::let_bind(
                edge_start.as_str(),
                Expr::load(NAME_EDGE_OFFSETS, src.clone()),
            ),
            Node::let_bind(
                edge_end.as_str(),
                Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
            ),
            Node::loop_for(
                edge_iter.as_str(),
                Expr::var(edge_start.as_str()),
                Expr::var(edge_end.as_str()),
                vec![
                    Node::let_bind(
                        kind_mask.as_str(),
                        Expr::load(NAME_EDGE_KIND_MASK, Expr::var(edge_iter.as_str())),
                    ),
                    Node::if_then(
                        Expr::ne(
                            Expr::bitand(Expr::var(kind_mask.as_str()), Expr::u32(edge_kind_mask)),
                            Expr::u32(0),
                        ),
                        vec![
                            Node::let_bind(
                                dst.as_str(),
                                Expr::load(NAME_EDGE_TARGETS, Expr::var(edge_iter.as_str())),
                            ),
                            Node::if_then(
                                Expr::lt(Expr::var(dst.as_str()), Expr::u32(shape.node_count)),
                                vec![
                                    Node::let_bind(
                                        dst_word_idx.as_str(),
                                        Expr::shr(Expr::var(dst.as_str()), Expr::u32(5)),
                                    ),
                                    Node::let_bind(
                                        dst_bit.as_str(),
                                        Expr::shl(
                                            Expr::u32(1),
                                            Expr::bitand(Expr::var(dst.as_str()), Expr::u32(31)),
                                        ),
                                    ),
                                    Node::let_bind(
                                        old.as_str(),
                                        Expr::atomic_or(
                                            frontier_out,
                                            Expr::var(dst_word_idx.as_str()),
                                            Expr::var(dst_bit.as_str()),
                                        ),
                                    ),
                                    Node::if_then(
                                        Expr::eq(
                                            Expr::bitand(
                                                Expr::var(old.as_str()),
                                                Expr::var(dst_bit.as_str()),
                                            ),
                                            Expr::u32(0),
                                        ),
                                        vec![Node::let_bind(
                                            changed_old.as_str(),
                                            Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
                                        )],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ]
    };

    if let Some(ordering) = snapshot_barrier {
        return vec![
            Node::let_bind(
                in_bounds.as_str(),
                Expr::lt(src.clone(), Expr::u32(shape.node_count)),
            ),
            Node::let_bind(
                word_idx.as_str(),
                Expr::select(
                    Expr::var(in_bounds.as_str()),
                    Expr::shr(src.clone(), Expr::u32(5)),
                    Expr::u32(0),
                ),
            ),
            Node::let_bind(
                bit_mask.as_str(),
                Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
            ),
            Node::let_bind(
                src_word.as_str(),
                Expr::load(frontier_out, Expr::var(word_idx.as_str())),
            ),
            Node::let_bind(
                src_active.as_str(),
                Expr::select(
                    Expr::var(in_bounds.as_str()),
                    Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                    Expr::u32(0),
                ),
            ),
            Node::barrier_with_ordering(ordering),
            Node::if_then(
                Expr::ne(Expr::var(src_active.as_str()), Expr::u32(0)),
                edge_scan(),
            ),
        ];
    }

    let body = vec![
        Node::let_bind(word_idx.as_str(), Expr::shr(src.clone(), Expr::u32(5))),
        Node::let_bind(
            bit_mask.as_str(),
            Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
        ),
        Node::let_bind(
            src_word.as_str(),
            Expr::load(frontier_out, Expr::var(word_idx.as_str())),
        ),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var(src_word.as_str()), Expr::var(bit_mask.as_str())),
                Expr::u32(0),
            ),
            edge_scan(),
        ),
    ];

    vec![Node::if_then(
        Expr::lt(Expr::gid_x(), Expr::u32(shape.node_count)),
        body,
    )]
}

/// Wrap a parallel expansion body as a child Region of `parent_op_id`.
#[must_use]
pub fn csr_forward_or_changed_parallel_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(csr_forward_or_changed_parallel_body_prefixed(
            shape,
            frontier_out,
            changed,
            edge_kind_mask,
            local_prefix,
        )),
    }
}

/// Wrap a snapshotting parallel expansion body as a child Region.
#[must_use]
pub fn csr_forward_or_changed_parallel_snapshot_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(csr_forward_or_changed_parallel_snapshot_body_prefixed(
            shape,
            frontier_out,
            changed,
            edge_kind_mask,
            local_prefix,
        )),
    }
}
