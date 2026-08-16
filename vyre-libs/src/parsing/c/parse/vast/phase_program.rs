//! The buffer contract every registered per-row phase operation is built on.
//!
//! A phase answers one question about one VAST row and returns a single `u32`.
//! Three modules used to carry their own copy of the buffer list, and the copy
//! in `typedef_ann::row_phases` declared a one-word haystack no fixture could
//! fit, because no caller ever asked it for a haystack. The declarations live
//! here once, so a declared extent and the payload it must hold cannot drift
//! apart.
//!
//! A callee may not read `InvocationId`, so the row index arrives as an
//! argument. The VAST node table and the source haystack arrive as buffer
//! references, which inlining retargets onto the caller's own buffers.

use super::phase_witness::{PHASE_WITNESS_ROWS, PHASE_WITNESS_SOURCE_LEN};
use super::VAST_NODE_STRIDE_U32;
use crate::parsing::c::source_bytes::source_haystack_words;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Callee-local buffer names. These never survive inlining: a buffer argument
/// retargets onto the caller's buffer and a scalar argument is substituted.
pub(in crate::parsing::c::parse::vast) const NODES: &str = "phase_vast_nodes";
pub(in crate::parsing::c::parse::vast) const HAYSTACK: &str = "phase_haystack";
pub(in crate::parsing::c::parse::vast) const ROW: &str = "phase_row";
pub(in crate::parsing::c::parse::vast) const HAYSTACK_LEN: &str = "phase_haystack_len";
pub(in crate::parsing::c::parse::vast) const NUM_NODES: &str = "phase_num_nodes";
pub(in crate::parsing::c::parse::vast) const RESULT: &str = "phase_result";

/// The row count a scan reads from its caller, bound by every phase that walks
/// the node table forward. The emitters read the name, so the callee binds it
/// from its own parameter and inlining substitutes the caller's binding.
pub(in crate::parsing::c::parse::vast) const NUM_NODES_BINDING: &str = "annot_num_nodes";

/// What a phase reads besides the node table and the row index.
pub(in crate::parsing::c::parse::vast) enum PhaseInputs {
    /// Nothing: the scan reads structure only.
    Row,
    /// The row count, for a scan that walks the node table forward.
    RowAndNumNodes,
    /// The source haystack, its length, and the row count, for a scan that
    /// reads identifier text.
    RowWithHaystack {
        /// Four source bytes per haystack word rather than one.
        packed_haystack: bool,
    },
}

/// Signature of [`PhaseInputs::Row`].
pub(in crate::parsing::c::parse::vast) const ROW_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: NODES,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: ROW,
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: RESULT,
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

/// Signature of [`PhaseInputs::RowAndNumNodes`].
pub(in crate::parsing::c::parse::vast) const ROW_AND_NUM_NODES_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: NODES,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: ROW,
            ty: "u32",
        },
        TypedParam {
            name: NUM_NODES,
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: RESULT,
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

/// Signature of [`PhaseInputs::RowWithHaystack`].
pub(in crate::parsing::c::parse::vast) const HAYSTACK_SIGNATURE: Signature = Signature {
    inputs: &[
        TypedParam {
            name: NODES,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: HAYSTACK,
            ty: "buffer<u32>",
        },
        TypedParam {
            name: ROW,
            ty: "u32",
        },
        TypedParam {
            name: HAYSTACK_LEN,
            ty: "u32",
        },
        TypedParam {
            name: NUM_NODES,
            ty: "u32",
        },
    ],
    outputs: &[TypedParam {
        name: RESULT,
        ty: "u32",
    }],
    attrs: &[],
    bytes_extraction: false,
};

/// The row index the callee works on, read from its scalar parameter.
pub(in crate::parsing::c::parse::vast) fn phase_row() -> Expr {
    Expr::load(ROW, Expr::u32(0))
}

/// The source length the callee bounds its byte reads by.
pub(in crate::parsing::c::parse::vast) fn phase_haystack_len() -> Expr {
    Expr::load(HAYSTACK_LEN, Expr::u32(0))
}

/// Assemble a phase program: `body` computes `out_name`, which becomes the
/// op's single output.
///
/// Rows a callee's own buffer declarations are sized for: the registered
/// witness, so its fixture node buffer fits the declared shape. This is still
/// only a declaration extent. A callee is never dispatched on its own, so this
/// is the shape the registry validates against, and inlining replaces every
/// access with one on the caller's buffer, which carries the real extent.
pub(in crate::parsing::c::parse::vast) fn phase_program(
    op_id: &str,
    inputs: PhaseInputs,
    out_name: &str,
    body: Vec<Node>,
) -> Program {
    let mut buffers = vec![BufferDecl::storage(NODES, 0, BufferAccess::ReadOnly, DataType::U32)
        .with_count(PHASE_WITNESS_ROWS.saturating_mul(VAST_NODE_STRIDE_U32))];
    let mut nodes = Vec::with_capacity(body.len() + 2);
    let out_binding = match inputs {
        PhaseInputs::Row => {
            buffers
                .push(BufferDecl::storage(ROW, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1));
            2
        }
        PhaseInputs::RowAndNumNodes => {
            buffers.push(
                BufferDecl::storage(ROW, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            );
            buffers.push(
                BufferDecl::storage(NUM_NODES, 2, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
            );
            nodes.push(Node::let_bind(
                NUM_NODES_BINDING,
                Expr::load(NUM_NODES, Expr::u32(0)),
            ));
            3
        }
        PhaseInputs::RowWithHaystack { packed_haystack } => {
            buffers.push(
                BufferDecl::storage(HAYSTACK, 1, BufferAccess::ReadOnly, DataType::U32).with_count(
                    source_haystack_words(PHASE_WITNESS_SOURCE_LEN, packed_haystack),
                ),
            );
            buffers
                .push(BufferDecl::storage(ROW, 2, BufferAccess::ReadOnly, DataType::U32).with_count(1));
            buffers.push(
                BufferDecl::storage(HAYSTACK_LEN, 3, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
            );
            buffers.push(
                BufferDecl::storage(NUM_NODES, 4, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(1),
            );
            nodes.push(Node::let_bind(
                NUM_NODES_BINDING,
                Expr::load(NUM_NODES, Expr::u32(0)),
            ));
            5
        }
    };
    buffers.push(BufferDecl::output(RESULT, out_binding, DataType::U32).with_count(1));

    nodes.extend(body);
    nodes.push(Node::store(RESULT, Expr::u32(0), Expr::var(out_name)));
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(op_id, nodes)],
    )
    .with_entry_op_id(op_id)
}
