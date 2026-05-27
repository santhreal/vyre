//! CSR frontier expansion over an in-place accumulator bitset.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::graph::program_graph::{
    ProgramGraphShape, BINDING_PRIMITIVE_START, NAME_EDGE_KIND_MASK, NAME_EDGE_OFFSETS,
    NAME_EDGE_TARGETS,
};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::csr_forward_or_changed";
/// Canonical binding index for the frontier accumulator.
pub const CSR_FORWARD_OR_CHANGED_FRONTIER_BUFFER: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the changed flag/history buffer.
pub const CSR_FORWARD_OR_CHANGED_CHANGED_BUFFER: u32 = BINDING_PRIMITIVE_START + 1;
/// Canonical one-lane workgroup for CSR forward-or-changed programs.
pub const CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE: [u32; 3] = [1, 1, 1];
/// Iteration ceiling where a changed-history buffer avoids per-iteration zeroing.
pub const CSR_FORWARD_OR_CHANGED_HISTORY_FAST_PATH_MAX_ITERS: u32 = 64;

/// Build one in-place forward expansion pass over an accumulating frontier.
#[must_use]
pub fn csr_forward_or_changed_body(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
) -> Vec<Node> {
    csr_forward_or_changed_body_prefixed(shape, frontier_out, changed_var, edge_kind_mask, "")
}

fn local(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}_{name}")
    }
}

/// Build one traversal pass with caller-provided local-name prefixing for
/// repeated inlining under validators that disallow shadowing.
#[must_use]
pub fn csr_forward_or_changed_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    let src = local(prefix, "src");
    let word_idx = local(prefix, "word_idx");
    let bit_mask = local(prefix, "bit_mask");
    let src_word = local(prefix, "src_word");
    let edge_start = local(prefix, "edge_start");
    let edge_end = local(prefix, "edge_end");
    let edge_iter = local(prefix, "e");
    let kind_mask = local(prefix, "kind_mask");
    let dst = local(prefix, "dst");
    let dst_word_idx = local(prefix, "dst_word_idx");
    let dst_bit = local(prefix, "dst_bit");
    let old = local(prefix, "old");

    let per_source = vec![
        Node::let_bind(
            word_idx.as_str(),
            Expr::shr(Expr::var(src.as_str()), Expr::u32(5)),
        ),
        Node::let_bind(
            bit_mask.as_str(),
            Expr::shl(
                Expr::u32(1),
                Expr::bitand(Expr::var(src.as_str()), Expr::u32(31)),
            ),
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
            vec![
                Node::let_bind(
                    edge_start.as_str(),
                    Expr::load(NAME_EDGE_OFFSETS, Expr::var(src.as_str())),
                ),
                Node::let_bind(
                    edge_end.as_str(),
                    Expr::load(
                        NAME_EDGE_OFFSETS,
                        Expr::add(Expr::var(src.as_str()), Expr::u32(1)),
                    ),
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
                                Expr::bitand(
                                    Expr::var(kind_mask.as_str()),
                                    Expr::u32(edge_kind_mask),
                                ),
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
                                                Expr::bitand(
                                                    Expr::var(dst.as_str()),
                                                    Expr::u32(31),
                                                ),
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
                                            vec![Node::assign(changed_var, Expr::u32(1))],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ];

    vec![Node::if_then(
        Expr::eq(Expr::local_x(), Expr::u32(0)),
        vec![Node::loop_for(
            src.as_str(),
            Expr::u32(0),
            Expr::u32(shape.node_count),
            per_source,
        )],
    )]
}

/// Wrap one traversal pass as a child Region of `parent_op_id`.
#[must_use]
pub fn csr_forward_or_changed_child(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
) -> Node {
    csr_forward_or_changed_child_prefixed(
        parent_op_id,
        shape,
        frontier_out,
        changed_var,
        edge_kind_mask,
        "",
    )
}

/// Wrap a traversal pass with a local-name prefix for repeated inlining.
#[must_use]
pub fn csr_forward_or_changed_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed_var: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(csr_forward_or_changed_body_prefixed(
            shape,
            frontier_out,
            changed_var,
            edge_kind_mask,
            local_prefix,
        )),
    }
}

/// Standalone in-place expansion program for primitive conformance.
#[must_use]
pub fn csr_forward_or_changed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    let words = crate::bitset::bitset_words(shape.node_count);
    let mut body = vec![Node::let_bind("local_changed", Expr::u32(0))];
    body.extend(csr_forward_or_changed_body(
        shape,
        frontier_out,
        "local_changed",
        edge_kind_mask,
    ));
    body.push(Node::if_then(
        Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
        vec![Node::let_bind(
            "_changed",
            Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
        )],
    ));
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
        CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

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
    let src = Expr::InvocationId { axis: 0 };
    let words = crate::bitset::bitset_words(shape.node_count);
    let body = vec![
        Node::let_bind("word_idx", Expr::shr(src.clone(), Expr::u32(5))),
        Node::let_bind(
            "bit_mask",
            Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
        ),
        Node::let_bind("src_word", Expr::load(frontier_out, Expr::var("word_idx"))),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var("src_word"), Expr::var("bit_mask")),
                Expr::u32(0),
            ),
            vec![
                Node::let_bind("edge_start", Expr::load(NAME_EDGE_OFFSETS, src.clone())),
                Node::let_bind(
                    "edge_end",
                    Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
                ),
                Node::loop_for(
                    "e",
                    Expr::var("edge_start"),
                    Expr::var("edge_end"),
                    vec![
                        Node::let_bind(
                            "kind_mask",
                            Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var("kind_mask"), Expr::u32(edge_kind_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    "dst",
                                    Expr::load(NAME_EDGE_TARGETS, Expr::var("e")),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("dst"), Expr::u32(shape.node_count)),
                                    vec![
                                        Node::let_bind(
                                            "dst_word_idx",
                                            Expr::shr(Expr::var("dst"), Expr::u32(5)),
                                        ),
                                        Node::let_bind(
                                            "dst_bit",
                                            Expr::shl(
                                                Expr::u32(1),
                                                Expr::bitand(Expr::var("dst"), Expr::u32(31)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "old",
                                            Expr::atomic_or(
                                                frontier_out,
                                                Expr::var("dst_word_idx"),
                                                Expr::var("dst_bit"),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::eq(
                                                Expr::bitand(
                                                    Expr::var("old"),
                                                    Expr::var("dst_bit"),
                                                ),
                                                Expr::u32(0),
                                            ),
                                            vec![Node::let_bind(
                                                "_changed",
                                                Expr::atomic_or(
                                                    changed,
                                                    Expr::u32(0),
                                                    Expr::u32(1),
                                                ),
                                            )],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ];
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
        CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::lt(src.clone(), Expr::u32(shape.node_count)),
                body,
            )]),
        }],
    )
}

/// Parallel in-place expansion for several frontier accumulators at once.
///
/// Invocation axis 0 is the source node and axis 1 is the query/frontier index.
/// `frontier_out` is laid out as `query_count` consecutive bitsets, each
/// containing `bitset_words(shape.node_count)` u32 words. `changed` contains
/// one u32 flag per query.
#[must_use]
pub fn csr_forward_or_changed_parallel_batch(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
) -> Program {
    match try_csr_forward_or_changed_parallel_batch(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
    ) {
        Ok(program) => program,
        Err(_) => inert_csr_forward_or_changed_batch_program(shape, frontier_out, changed, 1),
    }
}

/// Parallel in-place expansion for several frontier accumulators with checked
/// flat-frontier sizing.
pub fn try_csr_forward_or_changed_parallel_batch(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
) -> Result<Program, String> {
    if query_count == 0 {
        return Err(
            "Fix: csr_forward_or_changed_parallel_batch requires at least one query frontier."
                .to_string(),
        );
    }
    let src = Expr::InvocationId { axis: 0 };
    let query = Expr::InvocationId { axis: 1 };
    let words = crate::bitset::bitset_words(shape.node_count);
    let total_words = checked_batched_frontier_words(words, query_count)?;
    let query_word_base = Expr::mul(query.clone(), Expr::u32(words));
    let body = vec![
        Node::let_bind("query_word_base", query_word_base.clone()),
        Node::let_bind(
            "word_idx",
            Expr::add(
                Expr::var("query_word_base"),
                Expr::shr(src.clone(), Expr::u32(5)),
            ),
        ),
        Node::let_bind(
            "bit_mask",
            Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
        ),
        Node::let_bind("src_word", Expr::load(frontier_out, Expr::var("word_idx"))),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var("src_word"), Expr::var("bit_mask")),
                Expr::u32(0),
            ),
            vec![
                Node::let_bind("edge_start", Expr::load(NAME_EDGE_OFFSETS, src.clone())),
                Node::let_bind(
                    "edge_end",
                    Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
                ),
                Node::loop_for(
                    "e",
                    Expr::var("edge_start"),
                    Expr::var("edge_end"),
                    vec![
                        Node::let_bind(
                            "kind_mask",
                            Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var("kind_mask"), Expr::u32(edge_kind_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    "dst",
                                    Expr::load(NAME_EDGE_TARGETS, Expr::var("e")),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("dst"), Expr::u32(shape.node_count)),
                                    vec![
                                        Node::let_bind(
                                            "dst_word_idx",
                                            Expr::add(
                                                Expr::var("query_word_base"),
                                                Expr::shr(Expr::var("dst"), Expr::u32(5)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "dst_bit",
                                            Expr::shl(
                                                Expr::u32(1),
                                                Expr::bitand(Expr::var("dst"), Expr::u32(31)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "old",
                                            Expr::atomic_or(
                                                frontier_out,
                                                Expr::var("dst_word_idx"),
                                                Expr::var("dst_bit"),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::eq(
                                                Expr::bitand(
                                                    Expr::var("old"),
                                                    Expr::var("dst_bit"),
                                                ),
                                                Expr::u32(0),
                                            ),
                                            vec![Node::let_bind(
                                                "_changed",
                                                Expr::atomic_or(
                                                    changed,
                                                    query.clone(),
                                                    Expr::u32(1),
                                                ),
                                            )],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ];
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(total_words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(query_count),
    );
    Ok(Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::lt(src.clone(), Expr::u32(shape.node_count)),
                body,
            )]),
        }],
    ))
}

/// Batched parallel expansion with one global convergence flag.
///
/// Same frontier layout as [`csr_forward_or_changed_parallel_batch`], but every
/// newly discovered bit ORs `changed[0]` instead of `changed[query]`. This is
/// the hot-path convergence primitive for callers that only need to know
/// whether the whole query batch changed.
#[must_use]
pub fn csr_forward_or_changed_parallel_batch_global(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
) -> Program {
    csr_forward_or_changed_parallel_batch_global_slot(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        0,
        1,
    )
}

/// Batched parallel expansion with one global convergence slot.
///
/// This variant writes `changed[changed_slot]` instead of always writing
/// `changed[0]`. Resident fixed-point drivers can allocate one changed word
/// per iteration and avoid a host-to-device reset upload before every
/// dispatch. The slot must be inside `changed_slots`.
#[must_use]
pub fn csr_forward_or_changed_parallel_batch_global_slot(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_slot: u32,
    changed_slots: u32,
) -> Program {
    match try_csr_forward_or_changed_parallel_batch_global_slot(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        changed_slot,
        changed_slots,
    ) {
        Ok(program) => program,
        Err(_) => inert_csr_forward_or_changed_batch_program(
            shape,
            frontier_out,
            changed,
            changed_slots.max(1),
        ),
    }
}

/// Batched parallel expansion with one checked global convergence slot.
pub fn try_csr_forward_or_changed_parallel_batch_global_slot(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_slot: u32,
    changed_slots: u32,
) -> Result<Program, String> {
    if query_count == 0 {
        return Err(
            "Fix: csr_forward_or_changed_parallel_batch_global requires at least one query frontier."
                .to_string(),
        );
    }
    if changed_slot >= changed_slots {
        return Err(
            "Fix: changed_slot must be inside the allocated changed_slots buffer.".to_string(),
        );
    }
    csr_forward_or_changed_parallel_batch_global_indexed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        Expr::u32(changed_slot),
        changed_slots,
        Vec::new(),
        Vec::new(),
    )
}

/// Batched parallel expansion with one dynamically selected global convergence slot.
///
/// `changed_slot_input[0]` selects the convergence word to OR. The changed
/// buffer is sized for `changed_slots` and can be zeroed once before a
/// fixed-point sequence, allowing each iteration to write a fresh slot instead
/// of requiring a host zero-upload before every dispatch.
pub fn try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    changed_slot_input: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_slots: u32,
) -> Result<Program, String> {
    if changed_slots == 0 {
        return Err(
            "Fix: csr_forward_or_changed dynamic changed-slot dispatch requires at least one changed slot."
                .to_string(),
        );
    }
    csr_forward_or_changed_parallel_batch_global_indexed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        Expr::var("changed_slot"),
        changed_slots,
        vec![Node::let_bind(
            "changed_slot",
            Expr::load(changed_slot_input, Expr::u32(0)),
        )],
        vec![BufferDecl::storage(
            changed_slot_input,
            BINDING_PRIMITIVE_START + 2,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(1)],
    )
}

#[allow(clippy::too_many_arguments)]
fn csr_forward_or_changed_parallel_batch_global_indexed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_index: Expr,
    changed_slots: u32,
    mut prologue: Vec<Node>,
    extra_buffers: Vec<BufferDecl>,
) -> Result<Program, String> {
    if query_count == 0 {
        return Err(
            "Fix: csr_forward_or_changed_parallel_batch_global requires at least one query frontier."
                .to_string(),
        );
    }
    let src = Expr::InvocationId { axis: 0 };
    let query = Expr::InvocationId { axis: 1 };
    let words = crate::bitset::bitset_words(shape.node_count);
    let total_words = checked_batched_frontier_words(words, query_count)?;
    let query_word_base = Expr::mul(query.clone(), Expr::u32(words));
    let mut body = vec![
        Node::let_bind("query_word_base", query_word_base.clone()),
        Node::let_bind(
            "word_idx",
            Expr::add(
                Expr::var("query_word_base"),
                Expr::shr(src.clone(), Expr::u32(5)),
            ),
        ),
        Node::let_bind(
            "bit_mask",
            Expr::shl(Expr::u32(1), Expr::bitand(src.clone(), Expr::u32(31))),
        ),
        Node::let_bind("src_word", Expr::load(frontier_out, Expr::var("word_idx"))),
        Node::if_then(
            Expr::ne(
                Expr::bitand(Expr::var("src_word"), Expr::var("bit_mask")),
                Expr::u32(0),
            ),
            vec![
                Node::let_bind("edge_start", Expr::load(NAME_EDGE_OFFSETS, src.clone())),
                Node::let_bind(
                    "edge_end",
                    Expr::load(NAME_EDGE_OFFSETS, Expr::add(src.clone(), Expr::u32(1))),
                ),
                Node::loop_for(
                    "e",
                    Expr::var("edge_start"),
                    Expr::var("edge_end"),
                    vec![
                        Node::let_bind(
                            "kind_mask",
                            Expr::load(NAME_EDGE_KIND_MASK, Expr::var("e")),
                        ),
                        Node::if_then(
                            Expr::ne(
                                Expr::bitand(Expr::var("kind_mask"), Expr::u32(edge_kind_mask)),
                                Expr::u32(0),
                            ),
                            vec![
                                Node::let_bind(
                                    "dst",
                                    Expr::load(NAME_EDGE_TARGETS, Expr::var("e")),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("dst"), Expr::u32(shape.node_count)),
                                    vec![
                                        Node::let_bind(
                                            "dst_word_idx",
                                            Expr::add(
                                                Expr::var("query_word_base"),
                                                Expr::shr(Expr::var("dst"), Expr::u32(5)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "dst_bit",
                                            Expr::shl(
                                                Expr::u32(1),
                                                Expr::bitand(Expr::var("dst"), Expr::u32(31)),
                                            ),
                                        ),
                                        Node::let_bind(
                                            "old",
                                            Expr::atomic_or(
                                                frontier_out,
                                                Expr::var("dst_word_idx"),
                                                Expr::var("dst_bit"),
                                            ),
                                        ),
                                        Node::if_then(
                                            Expr::eq(
                                                Expr::bitand(
                                                    Expr::var("old"),
                                                    Expr::var("dst_bit"),
                                                ),
                                                Expr::u32(0),
                                            ),
                                            vec![Node::let_bind(
                                                "_changed",
                                                Expr::atomic_or(
                                                    changed,
                                                    changed_index.clone(),
                                                    Expr::u32(1),
                                                ),
                                            )],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ];
    prologue.append(&mut body);
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(total_words.max(1)),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(changed_slots),
    );
    buffers.extend(extra_buffers);
    Ok(Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::lt(src.clone(), Expr::u32(shape.node_count)),
                prologue,
            )]),
        }],
    ))
}

fn checked_batched_frontier_words(words: u32, query_count: u32) -> Result<u32, String> {
    words.checked_mul(query_count).ok_or_else(|| {
        format!(
            "Fix: batched CSR frontier words overflow u32: words={words}, query_count={query_count}."
        )
    })
}

fn inert_csr_forward_or_changed_batch_program(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    changed_slots: u32,
) -> Program {
    let mut buffers = shape.read_only_buffers();
    buffers.push(
        BufferDecl::storage(
            frontier_out,
            BINDING_PRIMITIVE_START,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
    );
    buffers.push(
        BufferDecl::storage(
            changed,
            BINDING_PRIMITIVE_START + 1,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(changed_slots.max(1)),
    );

    Program::wrapped(
        buffers,
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::return_()]),
        }],
    )
}

/// CPU reference for one in-place expansion pass.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    let mut out = Vec::new();
    let changed = cpu_ref_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
        &mut out,
    );
    (out, changed)
}

/// CPU reference writing the expanded frontier into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
    out: &mut Vec<u32>,
) -> u32 {
    let layout = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask)
        .unwrap_or_else(|err| {
            panic!("csr_forward_or_changed CPU oracle received malformed CSR. {err}")
        });
    let words = layout.frontier_words;
    out.clear();
    out.extend_from_slice(frontier);
    out.resize(words, 0);
    if edge_offsets.is_empty() {
        return 0;
    }
    let mut changed = 0u32;
    for src in 0..node_count as usize {
        let src_word = src / 32;
        let src_bit = 1u32 << (src % 32);
        if out[src_word] & src_bit == 0 {
            continue;
        }
        let start = edge_offsets[src] as usize;
        let end = edge_offsets[src + 1] as usize;
        for edge in start..end.min(edge_targets.len()).min(edge_kind_mask.len()) {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let dst = edge_targets[edge] as usize;
            if dst >= node_count as usize {
                continue;
            }
            let word = dst / 32;
            let bit = 1u32 << (dst % 32);
            let old = out[word];
            out[word] |= bit;
            if out[word] != old {
                changed = 1;
            }
        }
    }
    changed
}

/// Iterate [`cpu_ref_into`] until the change flag reaches zero or
/// `max_iters` is exhausted.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_closure(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Vec<u32> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    cpu_ref_closure_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        seed,
        allow_mask,
        max_iters,
        &mut current,
        &mut next,
    );
    current
}

/// Iterate [`cpu_ref_into`] using caller-owned frontier buffers.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_closure_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    cpu_ref_closure_into_with_step_hook(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        seed,
        allow_mask,
        max_iters,
        current,
        next,
        |_| {},
    );
}

/// Iterate [`cpu_ref_into`] with a callback after each attempted expansion.
///
/// The hook lets consumers attach observability without owning the
/// fixed-point algorithm.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn cpu_ref_closure_into_with_step_hook<F>(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    seed: &[u32],
    allow_mask: u32,
    max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
    mut on_step: F,
) where
    F: FnMut(u32),
{
    current.clear();
    current.extend_from_slice(seed);
    for iteration in 0..max_iters {
        on_step(iteration);
        let changed = cpu_ref_into(
            node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            current,
            allow_mask,
            next,
        );
        if changed == 0 {
            std::mem::swap(current, next);
            return;
        }
        std::mem::swap(current, next);
    }
}

/// Validated dispatch layout for the forward-or-changed CSR primitive.
///
/// The primitive owns these derived counts so dispatch wrappers do not fork CSR
/// offset, edge-array, frontier, or scratch sizing rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrForwardOrChangedLayout {
    /// Number of nodes accepted by the primitive.
    pub node_count: u32,
    /// Number of words required by node-indexed scratch buffers.
    pub node_words: usize,
    /// Number of words required by the edge-offset buffer.
    pub edge_offset_words: usize,
    /// Number of edge-array words supplied to the primitive.
    pub edge_storage_words: usize,
    /// Edge count used when constructing [`ProgramGraphShape`].
    pub shape_edge_count: u32,
    /// Number of frontier words used by the dispatch buffer.
    pub frontier_words: usize,
}

/// Program identity for the forward-or-changed CSR primitive.
///
/// Dispatch consumers can cache generated programs by this key without
/// re-implementing CSR validation, changed-history selection, or launch-grid
/// policy outside `vyre-primitives`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrForwardOrChangedProgramKey {
    layout: CsrForwardOrChangedLayout,
    allow_mask: u32,
    changed_slots: u32,
    uses_changed_history: bool,
}

/// Primitive-owned identity for reusable CSR forward-or-changed static inputs.
///
/// Dispatch wrappers stage edge offsets, targets, masks, and changed-history
/// buffers according to the primitive launch plan. This key keeps the content
/// identity next to that plan so wrappers do not fork graph-fingerprint rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrForwardOrChangedStaticInputKey {
    /// Program identity selected by the primitive launch planner.
    pub program_key: CsrForwardOrChangedProgramKey,
    /// Words in the staged edge-offset input.
    pub edge_offset_words: usize,
    /// Words in each staged edge-indexed input.
    pub edge_storage_words: usize,
    /// Words in the changed readback/scratch buffer.
    pub changed_words: usize,
    /// Stable fingerprint of the padded edge-offset upload.
    pub edge_offsets_hash: u64,
    /// Stable fingerprint of the padded edge-target upload.
    pub edge_targets_hash: u64,
    /// Stable fingerprint of the padded edge-kind upload.
    pub edge_kind_mask_hash: u64,
}

impl CsrForwardOrChangedProgramKey {
    /// Validated CSR/frontier layout represented by this program.
    #[must_use]
    pub const fn layout(&self) -> CsrForwardOrChangedLayout {
        self.layout
    }

    /// Edge-kind mask accepted by this program.
    #[must_use]
    pub const fn allow_mask(&self) -> u32 {
        self.allow_mask
    }

    /// Number of changed-buffer slots this program writes.
    #[must_use]
    pub const fn changed_slots(&self) -> u32 {
        self.changed_slots
    }

    /// True when this program uses the dynamic changed-history fast path.
    #[must_use]
    pub const fn uses_changed_history(&self) -> bool {
        self.uses_changed_history
    }
}

/// Lightweight primitive-owned dispatch plan without an allocated [`Program`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrForwardOrChangedLaunchPlan {
    key: CsrForwardOrChangedProgramKey,
    dispatch_grid: [u32; 3],
}

impl CsrForwardOrChangedLaunchPlan {
    /// Validated CSR/frontier layout.
    #[must_use]
    pub const fn layout(&self) -> CsrForwardOrChangedLayout {
        self.key.layout
    }

    /// Stable key for caching the generated primitive program.
    #[must_use]
    pub const fn program_key(&self) -> CsrForwardOrChangedProgramKey {
        self.key
    }

    /// Build the selected primitive program.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the selected changed-history
    /// program cannot be represented.
    pub fn program(&self) -> Result<Program, String> {
        build_csr_forward_or_changed_dispatch_program(self.key)
    }

    /// Number of u32 words in the changed readback.
    #[must_use]
    pub const fn changed_words(&self) -> usize {
        self.key.changed_slots as usize
    }

    /// True when the launch uses per-iteration changed history and a slot input.
    #[must_use]
    pub const fn uses_changed_history(&self) -> bool {
        self.key.uses_changed_history
    }

    /// Changed-slot value to upload for this iteration when the fast path is active.
    #[must_use]
    pub const fn changed_slot_value(&self, iteration: u32) -> Option<u32> {
        if self.key.uses_changed_history {
            Some(iteration)
        } else {
            None
        }
    }

    /// Index in the changed readback that carries this iteration's convergence flag.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the caller asks for an iteration
    /// outside the changed-history buffer selected by this primitive plan.
    pub fn changed_read_index(&self, iteration: u32) -> Result<usize, String> {
        if !self.key.uses_changed_history {
            return Ok(0);
        }
        let index = usize::try_from(iteration).map_err(|_| {
            format!(
                "Fix: csr_forward_or_changed iteration {iteration} cannot be represented as a changed-history readback index."
            )
        })?;
        if index >= self.changed_words() {
            return Err(format!(
                "Fix: csr_forward_or_changed iteration {iteration} is outside changed-history length {}.",
                self.changed_words()
            ));
        }
        Ok(index)
    }

    /// Dispatch grid for one expansion pass.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        self.dispatch_grid
    }

    /// Number of u32 words in the frontier accumulator.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.key.layout.frontier_words
    }

    /// Number of u32 words in node-indexed scratch buffers.
    #[must_use]
    pub const fn node_words(&self) -> usize {
        self.key.layout.node_words
    }

    /// Number of u32 words in the edge-offset buffer.
    #[must_use]
    pub const fn edge_offset_words(&self) -> usize {
        self.key.layout.edge_offset_words
    }

    /// Number of u32 words in edge-indexed target/kind buffers after padding.
    #[must_use]
    pub const fn edge_storage_words(&self) -> usize {
        self.key.layout.edge_storage_words
    }

    /// Return the primitive-owned cache identity for static CSR graph inputs.
    ///
    /// Edge arrays must match the edge count represented by the launch plan.
    /// Empty edge-offset slices are accepted for zero-edge graphs because they
    /// normalize to the same zero-padded upload as canonical `[0; n + 1]`
    /// offsets.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the supplied CSR slices no longer
    /// match the validated launch-plan shape.
    pub fn static_input_key(
        &self,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
    ) -> Result<CsrForwardOrChangedStaticInputKey, String> {
        let layout = self.layout();
        if !edge_offsets.is_empty() && edge_offsets.len() != layout.edge_offset_words {
            return Err(format!(
                "Fix: csr_forward_or_changed static key expected either empty zero-edge offsets or {} offset words, got {}.",
                layout.edge_offset_words,
                edge_offsets.len()
            ));
        }
        let expected_edges = layout.shape_edge_count as usize;
        if edge_targets.len() != expected_edges {
            return Err(format!(
                "Fix: csr_forward_or_changed static key expected {expected_edges} edge target word(s), got {}.",
                edge_targets.len()
            ));
        }
        if edge_kind_mask.len() != expected_edges {
            return Err(format!(
                "Fix: csr_forward_or_changed static key expected {expected_edges} edge kind word(s), got {}.",
                edge_kind_mask.len()
            ));
        }
        Ok(CsrForwardOrChangedStaticInputKey {
            program_key: self.program_key(),
            edge_offset_words: layout.edge_offset_words,
            edge_storage_words: layout.edge_storage_words,
            changed_words: self.changed_words(),
            edge_offsets_hash: csr_forward_or_changed_padded_slice_fingerprint(
                edge_offsets,
                layout.edge_offset_words,
            ),
            edge_targets_hash: csr_forward_or_changed_padded_slice_fingerprint(
                edge_targets,
                layout.edge_storage_words,
            ),
            edge_kind_mask_hash: csr_forward_or_changed_padded_slice_fingerprint(
                edge_kind_mask,
                layout.edge_storage_words,
            ),
        })
    }
}

fn csr_forward_or_changed_padded_slice_fingerprint(values: &[u32], padded_words: usize) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in (padded_words as u64).to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for index in 0..padded_words {
        let value = values.get(index).copied().unwrap_or(0);
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Validate and copy a seed frontier into caller-owned frontier storage.
///
/// The reservation happens before mutation, so allocator failure cannot clobber
/// a reusable frontier buffer.
///
/// # Errors
///
/// Returns the caller's error type for bad seed width or reservation failure.
pub fn copy_csr_forward_seed_frontier_into<E>(
    seed: &[u32],
    frontier_words: usize,
    frontier: &mut Vec<u32>,
    reserve: impl FnOnce(&mut Vec<u32>, usize, &'static str) -> Result<(), E>,
    map_bad_input: impl FnOnce(String) -> E,
) -> Result<(), E> {
    if seed.len() != frontier_words {
        return Err(map_bad_input(format!(
            "Fix: csr_forward_or_changed expected seed frontier length {frontier_words} word(s), got {}. Pass a bitset sized by the primitive launch plan.",
            seed.len()
        )));
    }
    reserve(
        frontier,
        frontier_words,
        "csr_forward_or_changed frontier seed",
    )?;
    frontier.clear();
    frontier.extend_from_slice(seed);
    Ok(())
}

/// Validate that a changed readback word is the primitive's 0/1 flag.
///
/// # Errors
///
/// Returns an actionable diagnostic when a backend writes a non-boolean flag.
pub fn validate_csr_forward_or_changed_flag(changed: u32) -> Result<(), String> {
    if changed <= 1 {
        return Ok(());
    }
    Err(format!(
        "Fix: csr_forward_or_changed backend returned non-boolean changed flag {changed}; expected 0 or 1."
    ))
}

/// Validate the CSR inputs used by the forward-or-changed primitive.
///
/// # Errors
///
/// Returns an actionable diagnostic when offsets are missing, non-monotonic,
/// inconsistent with edge arrays, or when targets/kind masks have mismatched
/// lengths.
pub fn validate_csr_inputs(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<CsrForwardOrChangedLayout, String> {
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!(
            "Fix: csr_forward_or_changed node_count + 1 overflows usize for node_count={node_count}."
        )
    })?;
    let frontier_words = (crate::bitset::bitset_words(node_count) as usize).max(1);
    if edge_offsets.is_empty() {
        if edge_targets.is_empty() && edge_kind_mask.is_empty() {
            return Ok(CsrForwardOrChangedLayout {
                node_count,
                node_words: (node_count as usize).max(1),
                edge_offset_words: expected_offsets,
                edge_storage_words: 1,
                shape_edge_count: 0,
                frontier_words,
            });
        }
        return Err(format!(
            "Fix: csr_forward_or_changed empty edge_offsets may only encode an empty edge set, got targets_len={} kind_mask_len={}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: csr_forward_or_changed requires edge_offsets.len() == node_count + 1, got len={}, node_count={node_count}.",
            edge_offsets.len()
        ));
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: csr_forward_or_changed requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    let shape_edge_count = u32::try_from(edge_kind_mask.len()).map_err(|_| {
        format!(
            "Fix: csr_forward_or_changed edge count {} exceeds u32 index space.",
            edge_kind_mask.len()
        )
    })?;
    for (index, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: csr_forward_or_changed offsets must be monotonic; offsets[{index}]={} > offsets[{}]={}.",
                pair[0],
                index + 1,
                pair[1]
            ));
        }
    }
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    if edge_targets.len() < edge_count {
        return Err(format!(
            "Fix: csr_forward_or_changed final offset declares edge_count={edge_count}, but targets_len={} and kind_mask_len={}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    Ok(CsrForwardOrChangedLayout {
        node_count,
        node_words: (node_count as usize).max(1),
        edge_offset_words: expected_offsets,
        edge_storage_words: edge_kind_mask.len().max(1),
        shape_edge_count,
        frontier_words,
    })
}

/// Primitive-owned CSR forward-or-changed dispatch plan.
pub struct CsrForwardOrChangedDispatchPlan {
    launch: CsrForwardOrChangedLaunchPlan,
    program: Program,
}

impl CsrForwardOrChangedDispatchPlan {
    /// Validated CSR/frontier layout.
    #[must_use]
    pub const fn layout(&self) -> CsrForwardOrChangedLayout {
        self.launch.layout()
    }

    /// Lightweight launch plan used to build this dispatch plan.
    #[must_use]
    pub const fn launch(&self) -> CsrForwardOrChangedLaunchPlan {
        self.launch
    }

    /// Stable key for caching the generated primitive program.
    #[must_use]
    pub const fn program_key(&self) -> CsrForwardOrChangedProgramKey {
        self.launch.program_key()
    }

    /// Program selected by the primitive launch planner.
    #[must_use]
    pub const fn program(&self) -> &Program {
        &self.program
    }

    /// Number of u32 words in the changed readback.
    #[must_use]
    pub const fn changed_words(&self) -> usize {
        self.launch.changed_words()
    }

    /// True when the launch uses per-iteration changed history and a slot input.
    #[must_use]
    pub const fn uses_changed_history(&self) -> bool {
        self.launch.uses_changed_history()
    }

    /// Changed-slot value to upload for this iteration when the fast path is active.
    #[must_use]
    pub const fn changed_slot_value(&self, iteration: u32) -> Option<u32> {
        self.launch.changed_slot_value(iteration)
    }

    /// Index in the changed readback that carries this iteration's convergence flag.
    ///
    /// # Errors
    ///
    /// Returns an actionable diagnostic when the caller asks for an iteration
    /// outside the changed-history buffer selected by this primitive plan.
    pub fn changed_read_index(&self, iteration: u32) -> Result<usize, String> {
        self.launch.changed_read_index(iteration)
    }

    /// Dispatch grid for one expansion pass.
    #[must_use]
    pub const fn dispatch_grid(&self) -> [u32; 3] {
        self.launch.dispatch_grid()
    }

    /// Number of u32 words in the frontier accumulator.
    #[must_use]
    pub const fn frontier_words(&self) -> usize {
        self.launch.frontier_words()
    }

    /// Number of u32 words in node-indexed scratch buffers.
    #[must_use]
    pub const fn node_words(&self) -> usize {
        self.launch.node_words()
    }

    /// Number of u32 words in the edge-offset buffer.
    #[must_use]
    pub const fn edge_offset_words(&self) -> usize {
        self.launch.edge_offset_words()
    }

    /// Number of u32 words in edge-indexed target/kind buffers after padding.
    #[must_use]
    pub const fn edge_storage_words(&self) -> usize {
        self.launch.edge_storage_words()
    }
}

/// Validate CSR inputs and select a primitive-owned launch plan without
/// allocating the generated program.
///
/// # Errors
///
/// Returns an actionable diagnostic when CSR inputs are malformed.
pub fn plan_csr_forward_or_changed_launch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<CsrForwardOrChangedLaunchPlan, String> {
    let layout = validate_csr_inputs(node_count, edge_offsets, edge_targets, edge_kind_mask)?;
    let uses_changed_history =
        max_iters > 0 && max_iters <= CSR_FORWARD_OR_CHANGED_HISTORY_FAST_PATH_MAX_ITERS;
    let changed_slots = if uses_changed_history { max_iters } else { 1 };
    Ok(CsrForwardOrChangedLaunchPlan {
        key: CsrForwardOrChangedProgramKey {
            layout,
            allow_mask,
            changed_slots,
            uses_changed_history,
        },
        dispatch_grid: [layout.node_count.max(1), 1, 1],
    })
}

/// Build the primitive program selected by a launch-plan key.
///
/// # Errors
///
/// Returns an actionable diagnostic when the changed-history program cannot be
/// represented.
pub fn build_csr_forward_or_changed_dispatch_program(
    key: CsrForwardOrChangedProgramKey,
) -> Result<Program, String> {
    let shape = ProgramGraphShape::new(key.layout.node_count, key.layout.shape_edge_count);
    if key.uses_changed_history {
        try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
            shape,
            "frontier_out",
            "changed",
            "changed_slot",
            key.allow_mask,
            1,
            key.changed_slots,
        )
    } else {
        Ok(csr_forward_or_changed_parallel(
            shape,
            "frontier_out",
            "changed",
            key.allow_mask,
        ))
    }
}

/// Validate CSR inputs and select the primitive-owned expansion launch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when CSR inputs are malformed or the
/// changed-history fast path cannot be represented by the primitive builders.
pub fn plan_csr_forward_or_changed_dispatch(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<CsrForwardOrChangedDispatchPlan, String> {
    let launch = plan_csr_forward_or_changed_launch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        allow_mask,
        max_iters,
    )?;
    let program = launch.program()?;

    Ok(CsrForwardOrChangedDispatchPlan { launch, program })
}

#[cfg(test)]
mod dispatch_contract_tests {
    use super::*;

    #[test]
    fn static_input_key_tracks_same_shape_graph_content() {
        let plan = plan_csr_forward_or_changed_launch(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            0xFFFF_FFFF,
            4,
        )
        .expect("Fix: valid CSR should produce a launch plan");
        let first = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1, 1])
            .expect("Fix: matching CSR should produce a static input key");
        let changed_targets = plan
            .static_input_key(&[0, 1, 2, 3, 3], &[2, 3, 0], &[1, 1, 1])
            .expect("Fix: same-shape graph content should still be keyable");

        assert_eq!(first.program_key, changed_targets.program_key);
        assert_eq!(first.edge_offsets_hash, changed_targets.edge_offsets_hash);
        assert_eq!(
            first.edge_kind_mask_hash,
            changed_targets.edge_kind_mask_hash
        );
        assert_ne!(first.edge_targets_hash, changed_targets.edge_targets_hash);
        assert_ne!(first, changed_targets);
    }

    #[test]
    fn static_input_key_normalizes_empty_offsets_to_zero_padded_upload() {
        let empty_offsets_plan = plan_csr_forward_or_changed_launch(4, &[], &[], &[], 1, 2)
            .expect("Fix: empty zero-edge CSR shorthand should plan");
        let canonical_offsets_plan =
            plan_csr_forward_or_changed_launch(4, &[0, 0, 0, 0, 0], &[], &[], 1, 2)
                .expect("Fix: canonical zero-edge CSR should plan");
        let empty_key = empty_offsets_plan
            .static_input_key(&[], &[], &[])
            .expect("Fix: empty zero-edge CSR shorthand should key");
        let canonical_key = canonical_offsets_plan
            .static_input_key(&[0, 0, 0, 0, 0], &[], &[])
            .expect("Fix: canonical zero-edge CSR should key");

        assert_eq!(
            empty_offsets_plan.program_key(),
            canonical_offsets_plan.program_key()
        );
        assert_eq!(empty_key, canonical_key);
    }

    #[test]
    fn static_input_key_rejects_edge_count_drift() {
        let plan = plan_csr_forward_or_changed_launch(2, &[], &[], &[], 1, 1)
            .expect("Fix: zero-edge CSR should plan");

        let err = plan
            .static_input_key(&[], &[1], &[1])
            .expect_err("Fix: stale zero-edge plan must reject edge arrays");

        assert!(err.contains("expected 0 edge target"));
    }

    #[test]
    fn seed_copy_reserves_before_mutating_reused_frontier() {
        let mut frontier = vec![0xCAFE_BABEu32];
        let err = copy_csr_forward_seed_frontier_into(
            &[0b0001],
            1,
            &mut frontier,
            |_frontier, _words, _context| Err("injected reservation failure".to_string()),
            |message| message,
        )
        .expect_err("Fix: injected reservation failure should surface");

        assert_eq!(err, "injected reservation failure");
        assert_eq!(frontier, vec![0xCAFE_BABEu32]);
    }

    #[test]
    fn seed_copy_rejects_bad_width_without_mutating_reused_frontier() {
        let mut frontier = vec![0xCAFE_BABEu32];
        let err = copy_csr_forward_seed_frontier_into(
            &[],
            1,
            &mut frontier,
            |_frontier, _words, _context| Ok::<(), String>(()),
            |message| message,
        )
        .expect_err("Fix: bad seed width should be rejected");

        assert!(err.contains("expected seed frontier length 1"));
        assert_eq!(frontier, vec![0xCAFE_BABEu32]);
    }

    #[test]
    fn changed_flag_validation_rejects_non_boolean_values() {
        validate_csr_forward_or_changed_flag(0).expect("Fix: changed=0 is valid");
        validate_csr_forward_or_changed_flag(1).expect("Fix: changed=1 is valid");
        let err =
            validate_csr_forward_or_changed_flag(2).expect_err("Fix: changed flag must be boolean");

        assert!(err.contains("non-boolean changed flag 2"));
    }
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    crate::harness::OpEntry::new(
        OP_ID,
        || csr_forward_or_changed(ProgramGraphShape::new(4, 4), "frontier", "changed", 1),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),
                to_bytes(&[0, 2, 3, 4, 4]),
                to_bytes(&[1, 2, 3, 3]),
                to_bytes(&[1, 1, 1, 1]),
                to_bytes(&[0, 0, 0, 0]),
                to_bytes(&[0b0001]),
                to_bytes(&[0]),
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0b1111]), to_bytes(&[1])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ref_expands_in_place_frontier_pass() {
        let (frontier, changed) = cpu_ref(
            4,
            &[0, 2, 3, 4, 4],
            &[1, 2, 3, 3],
            &[1, 1, 1, 1],
            &[0b0001],
            1,
        );
        assert_eq!(frontier, vec![0b1111]);
        assert_eq!(changed, 1);
    }

    #[test]
    fn cpu_ref_closure_reaches_fixpoint() {
        let closure = cpu_ref_closure(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            10,
        );
        assert_eq!(closure, vec![0b1111]);
    }

    #[test]
    fn cpu_ref_closure_into_reuses_buffers() {
        let mut current = Vec::with_capacity(8);
        let mut next = Vec::with_capacity(8);
        cpu_ref_closure_into(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0b0001],
            0xFFFF_FFFF,
            10,
            &mut current,
            &mut next,
        );
        let current_capacity = current.capacity();
        let next_capacity = next.capacity();
        assert_eq!(current, vec![0b1111]);

        cpu_ref_closure_into(
            4,
            &[0, 1, 2, 3, 3],
            &[1, 2, 3],
            &[1, 1, 1],
            &[0],
            0xFFFF_FFFF,
            10,
            &mut current,
            &mut next,
        );
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);
        assert_eq!(current, vec![0]);
    }

    #[test]
    fn validate_csr_inputs_rejects_mismatched_and_nonmonotonic_csr() {
        let err = validate_csr_inputs(2, &[0, 1, 1], &[1], &[]).unwrap_err();
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));

        let err = validate_csr_inputs(2, &[0, 2, 1], &[1, 0], &[1, 1]).unwrap_err();
        assert!(err.contains("offsets must be monotonic"));
    }

    #[test]
    fn cpu_ref_into_rejects_malformed_csr_before_touching_output_storage() {
        let mut out = vec![0xDEAD_BEEFu32, 0xABCD_EF01];
        let ptr = out.as_ptr();
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cpu_ref_into(2, &[0, 2, 1], &[1, 0], &[1, 1], &[0b01], 1, &mut out);
        }));
        std::panic::set_hook(previous_hook);

        assert!(err.is_err(), "malformed CSR must be rejected");
        assert_eq!(
            out,
            vec![0xDEAD_BEEFu32, 0xABCD_EF01],
            "Fix: malformed CSR must not clear or resize caller output before validation."
        );
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn empty_offsets_shorthand_is_empty_edge_set_only() {
        assert_eq!(
            validate_csr_inputs(64, &[], &[], &[]).expect("Fix: empty CSR shorthand is valid"),
            CsrForwardOrChangedLayout {
                node_count: 64,
                node_words: 64,
                edge_offset_words: 65,
                edge_storage_words: 1,
                shape_edge_count: 0,
                frontier_words: 2,
            }
        );

        let err = validate_csr_inputs(64, &[], &[1], &[]).unwrap_err();
        assert!(err.contains("empty edge_offsets may only encode an empty edge set"));

        let mut out = Vec::new();
        let changed = cpu_ref_into(64, &[], &[], &[], &[0b101], 0xFFFF_FFFF, &mut out);
        assert_eq!(changed, 0);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], 0b101);
        assert_eq!(out[1], 0);
    }

    #[test]
    fn dispatch_plan_selects_changed_history_and_pins_buffer_shape() {
        let edge_offsets = vec![0u32; 66];
        let plan =
            plan_csr_forward_or_changed_dispatch(65, &edge_offsets, &[], &[], 0xFFFF_FFFF, 8)
                .expect("Fix: bounded CSR forward-or-changed plan should validate");

        assert_eq!(plan.layout().node_count, 65);
        assert_eq!(plan.frontier_words(), 3);
        assert_eq!(plan.node_words(), 65);
        assert_eq!(plan.edge_storage_words(), 1);
        assert_eq!(plan.changed_words(), 8);
        assert!(plan.uses_changed_history());
        assert_eq!(plan.changed_slot_value(3), Some(3));
        assert_eq!(plan.changed_read_index(3).unwrap(), 3);
        assert!(
            plan.changed_read_index(8).is_err(),
            "Fix: changed-history readback index must reject iterations outside the buffer"
        );
        assert_eq!(plan.dispatch_grid(), [65, 1, 1]);
        assert_eq!(
            plan.program().workgroup_size,
            CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE
        );
        assert!(
            plan.program()
                .buffers()
                .iter()
                .any(|buffer| buffer.name() == "changed_slot"),
            "Fix: changed-history fast path must expose the primitive slot selector"
        );
    }

    #[test]
    fn dispatch_plan_uses_single_changed_word_for_unbounded_or_zero_iteration_cases() {
        let plan = plan_csr_forward_or_changed_dispatch(0, &[], &[], &[], 0xFFFF_FFFF, 0)
            .expect("Fix: zero-node zero-iteration plan should validate");
        assert_eq!(plan.frontier_words(), 1);
        assert_eq!(plan.changed_words(), 1);
        assert!(!plan.uses_changed_history());
        assert_eq!(plan.changed_slot_value(0), None);
        assert_eq!(plan.changed_read_index(99).unwrap(), 0);
        assert_eq!(plan.dispatch_grid(), [1, 1, 1]);

        let long_plan = plan_csr_forward_or_changed_dispatch(1, &[0, 0], &[], &[], 0xFFFF_FFFF, 65)
            .expect("Fix: long-running plan should validate without changed history");
        assert_eq!(long_plan.changed_words(), 1);
        assert!(!long_plan.uses_changed_history());
        assert!(
            !long_plan
                .program()
                .buffers()
                .iter()
                .any(|buffer| buffer.name() == "changed_slot"),
            "Fix: unbounded path must not carry the changed-history slot input"
        );
    }

    #[test]
    fn parallel_program_keeps_frontier_and_changed_resident() {
        let program = csr_forward_or_changed_parallel(
            ProgramGraphShape::new(65, 4),
            "frontier",
            "changed",
            0xFFFF_FFFF,
        );
        assert_eq!(program.workgroup_size, [1, 1, 1]);
        let names: Vec<&str> = program.buffers.iter().map(|buffer| buffer.name()).collect();
        assert!(names.contains(&"frontier"));
        assert!(names.contains(&"changed"));
        assert!(
            names.iter().any(|name| name.starts_with("pg_")),
            "parallel CSR expansion must keep ProgramGraph buffers resident"
        );
    }

    #[test]
    fn parallel_batch_program_packs_query_frontiers() {
        let program = csr_forward_or_changed_parallel_batch(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            3,
        );
        assert_eq!(program.workgroup_size, [1, 1, 1]);
        let frontier = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "frontiers")
            .expect("Fix: frontiers buffer must exist");
        let changed = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: changed buffer must exist");
        assert_eq!(frontier.count(), 9);
        assert_eq!(changed.count(), 3);
    }

    #[test]
    fn parallel_batch_global_program_uses_one_changed_flag() {
        let program = csr_forward_or_changed_parallel_batch_global(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            3,
        );
        let frontier = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "frontiers")
            .expect("Fix: frontiers buffer must exist");
        let changed = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: changed buffer must exist");
        assert_eq!(frontier.count(), 9);
        assert_eq!(changed.count(), 1);
    }

    #[test]
    fn parallel_batch_global_slot_program_uses_changed_history_buffer() {
        let program = csr_forward_or_changed_parallel_batch_global_slot(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            3,
            5,
            8,
        );
        let frontier = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "frontiers")
            .expect("Fix: frontiers buffer must exist");
        let changed = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: changed buffer must exist");
        assert_eq!(frontier.count(), 9);
        assert_eq!(changed.count(), 8);
    }

    #[test]
    fn checked_parallel_batch_rejects_zero_queries() {
        let error = try_csr_forward_or_changed_parallel_batch(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            0,
        )
        .expect_err("checked CSR batch builder must reject empty query batches");

        assert!(
            error.contains("at least one query frontier"),
            "error should describe the invalid batch shape: {error}"
        );
    }

    #[test]
    fn checked_parallel_batch_rejects_flat_frontier_overflow() {
        let error = try_csr_forward_or_changed_parallel_batch(
            ProgramGraphShape::new(u32::MAX, 0),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            33,
        )
        .expect_err("checked CSR batch builder must reject flat frontier overflow");

        assert!(
            error.contains("frontier words overflow u32"),
            "error should describe the flat frontier overflow: {error}"
        );
    }

    #[test]
    fn legacy_parallel_batch_does_not_panic_on_flat_frontier_overflow() {
        let program = csr_forward_or_changed_parallel_batch(
            ProgramGraphShape::new(u32::MAX, 0),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            33,
        );
        let frontier = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "frontiers")
            .expect("Fix: frontiers buffer must exist");
        let changed = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: changed buffer must exist");

        assert_eq!(frontier.count(), 1);
        assert_eq!(changed.count(), 1);
    }

    #[test]
    fn checked_parallel_global_slot_rejects_invalid_changed_slot() {
        let error = try_csr_forward_or_changed_parallel_batch_global_slot(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            3,
            8,
            8,
        )
        .expect_err("checked CSR global-slot builder must reject out-of-range changed slot");

        assert!(
            error.contains("changed_slot must be inside"),
            "error should describe the invalid changed slot: {error}"
        );
    }

    #[test]
    fn legacy_parallel_global_slot_does_not_panic_on_invalid_changed_slot() {
        let program = csr_forward_or_changed_parallel_batch_global_slot(
            ProgramGraphShape::new(65, 4),
            "frontiers",
            "changed",
            0xFFFF_FFFF,
            3,
            8,
            8,
        );
        let changed = program
            .buffers
            .iter()
            .find(|buffer| buffer.name() == "changed")
            .expect("Fix: changed buffer must exist");

        assert_eq!(changed.count(), 8);
    }

    #[test]
    fn csr_forward_or_changed_batch_source_has_checked_api_without_panics() {
        let source = include_str!("csr_forward_or_changed.rs");
        let batch_source = source
            .split("/// Parallel in-place expansion for several frontier accumulators at once.")
            .nth(1)
            .expect("Fix: CSR batch builder source must be present")
            .split("/// CPU reference for one in-place expansion pass.")
            .next()
            .expect("Fix: CSR batch builder source must precede CPU oracle");

        assert!(
            batch_source.contains("pub fn try_csr_forward_or_changed_parallel_batch(")
                && batch_source
                    .contains("pub fn try_csr_forward_or_changed_parallel_batch_global_slot(")
                && !batch_source.contains(concat!("panic", "!("))
                && !batch_source.contains("assert!(")
                && !batch_source.contains(".unwrap_or_else("),
            "Fix: batched CSR forward-or-changed builders must expose checked release APIs and avoid production panics."
        );
    }
}

#[cfg(test)]
mod dynamic_changed_slot_tests {
    use super::*;

    #[test]
    fn dynamic_changed_slot_program_carries_slot_input_buffer() {
        let program = try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
            ProgramGraphShape::new(8, 8),
            "frontier",
            "changed",
            "changed_slot",
            0xFF,
            1,
            4,
        )
        .expect("Fix: dynamic changed-slot program must build");

        assert!(
            program
                .buffers()
                .iter()
                .any(|buffer| buffer.name() == "changed_slot"),
            "Fix: dynamic changed-slot program must expose a read-only slot selector input."
        );
        let rendered = format!("{:?}", program.entry);
        assert!(
            rendered.contains("changed_slot"),
            "Fix: dynamic changed-slot program must load the slot and use it for the changed write."
        );
    }

    #[test]
    fn dynamic_changed_slot_rejects_zero_changed_slots() {
        let err = try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
            ProgramGraphShape::new(8, 8),
            "frontier",
            "changed",
            "changed_slot",
            0xFF,
            1,
            0,
        )
        .expect_err("zero changed slots must be rejected");
        assert!(err.contains("at least one changed slot"));
    }
}
