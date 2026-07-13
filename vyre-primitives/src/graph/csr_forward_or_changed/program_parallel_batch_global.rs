use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::batch_shared::checked_batched_frontier_words;
use super::layout::{CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE, OP_ID};
use crate::graph::program_graph::{ProgramGraphShape, BINDING_PRIMITIVE_START};

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
    // Fail fast on an invalid global convergence slot rather than silently
    // degrading to an inert empty kernel (silent recall loss). Use
    // `try_csr_forward_or_changed_parallel_batch_global_slot` for structured
    // handling.
    try_csr_forward_or_changed_parallel_batch_global_slot(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        changed_slot,
        changed_slots,
    )
    .unwrap_or_else(|error| panic!("{error}"))
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
pub(crate) fn try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
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
pub(crate) fn csr_forward_or_changed_parallel_batch_global_indexed(
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
    // The neighbor expansion is the ONE canonical CSR edge-scan; this variant
    // supplies (a) the flat per-query frontier index `query_word_base + word` and
    // (b) an `atomic_or(changed, changed_index, 1)` on each newly-set bit. The
    // `query_word_base` let-bind is emitted first so the frontier-index closure can
    // reference it. Byte-identical to the previous hand-written body.
    let mut body = vec![Node::let_bind("query_word_base", query_word_base.clone())];
    body.extend(crate::graph::edge_scan::csr_edge_scan_nodes(
        shape,
        frontier_out,
        src.clone(),
        |word| Expr::add(Expr::var("query_word_base"), word),
        || {
            vec![Node::let_bind(
                "_changed",
                Expr::atomic_or(changed, changed_index.clone(), Expr::u32(1)),
            )]
        },
        edge_kind_mask,
        "",
    ));
    prologue.append(&mut body);
    let mut buffers = shape.try_read_only_buffers()?;
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
        CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE,
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
