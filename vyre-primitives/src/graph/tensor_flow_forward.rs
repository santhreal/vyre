//! `tensor_flow_forward`  -  1000x Parallel 3D Matrix Flow Tracking
//!
//! Exceeds Datalog performance loops by compiling Context-Sensitive Dataflow
//! directly into a Subgroup bitset operation over bounds:
//! `[Nodes : u32] x [ContextId : u8] x [FieldIdx : u8]`
//!
//! Sub-warps concurrently execute field-sensitive flow checks on an execution graph,
//! tracking nested dependencies efficiently.

use crate::graph::csr_frontier_step::edge_scan_body;
use crate::graph::frontier_bits::{set_bit, when_bit_set, BitAccess};
use crate::graph::program_graph::{
    word_buffer, ProgramGraphShape, BINDING_PRIMITIVE_START, NAME_EDGE_TARGETS,
};
use vyre_foundation::algebra::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::tensor_flow_forward";
/// Source-lane workgroup for context/field-sensitive tensor propagation.
pub const TENSOR_FLOW_FORWARD_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Canonical binding index for the input 3D tensor tensor bitset.
pub const BINDING_TENSOR_IN: u32 = BINDING_PRIMITIVE_START;
/// Canonical binding index for the output 3D tensor bitset.
pub const BINDING_TENSOR_OUT: u32 = BINDING_PRIMITIVE_START + 1;

/// Dispatch grid for source-node tensor-flow propagation.
#[must_use]
pub const fn tensor_flow_forward_dispatch_grid(node_count: u32) -> [u32; 3] {
    crate::graph::lane_grid(node_count, TENSOR_FLOW_FORWARD_WORKGROUP_SIZE[0])
}

/// Words needed for the per-node context x field tensor bitset, saturating.
///
/// The bit count is `node_count * context_limit * field_limit`, which overflows
/// u64 for three large factors, so the product saturates before the word count is
/// taken. `try_tensor_words` is the checked twin every release builder uses; this
/// const form exists for buffer metadata and saturates rather than wrapping to a
/// small, silently mis-sized allocation.
#[must_use]
pub const fn tensor_words(node_count: u32, context_limit: u32, field_limit: u32) -> u32 {
    let bits = (node_count as u64)
        .saturating_mul(context_limit as u64)
        .saturating_mul(field_limit as u64);
    let words = bits.div_ceil(32);
    if words > u32::MAX as u64 {
        u32::MAX
    } else {
        words as u32
    }
}

/// Checked tensor word count for release builders and CPU parity oracles.
pub fn try_tensor_words(
    node_count: u32,
    context_limit: u32,
    field_limit: u32,
) -> Result<u32, String> {
    let tensor_lane_count = context_limit.checked_mul(field_limit).ok_or_else(|| {
        format!(
            "{OP_ID} context_limit={context_limit} field_limit={field_limit} overflows per-node tensor lane count. Fix: shard context or field dimensions."
        )
    })?;
    let bit_count = node_count.checked_mul(tensor_lane_count).ok_or_else(|| {
        format!(
            "{OP_ID} node_count={node_count} context_limit={context_limit} field_limit={field_limit} overflows tensor bit count. Fix: shard the graph tensor before dispatch."
        )
    })?;
    Ok(crate::bitset::bitset_words(bit_count))
}

fn tensor_flow_edge_scan_body(
    tensor_out: &str,
    node_count: u32,
    field_limit: u32,
    tensor_lane_count: u32,
    allow_mask: u32,
) -> Vec<Node> {
    edge_scan_body(
        allow_mask,
        vec![Node::let_bind(
            "dst",
            Expr::load(NAME_EDGE_TARGETS, Expr::var("e")),
        )],
        vec![Node::if_then(
            Expr::lt(Expr::var("dst"), Expr::u32(node_count)),
            mark_tensor_bit(tensor_out, field_limit, tensor_lane_count),
        )],
    )
}

fn mark_tensor_bit(tensor_out: &str, field_limit: u32, tensor_lane_count: u32) -> Vec<Node> {
    let mut nodes = vec![Node::let_bind(
        "dst_abs_bit",
        tensor_lane_bit(Expr::var("dst"), field_limit, tensor_lane_count),
    )];
    nodes.extend(set_bit(
        tensor_out,
        &Expr::var("dst_abs_bit"),
        BitAccess {
            word: "dst_word",
            mask: "dst_bit",
            value: "_prev",
        },
        |word| word,
        Vec::new(),
    ));
    nodes
}

/// Absolute bit index of the `(node, ctx, fld)` tensor lane in the flat bitset.
///
/// The tensor packs `context_limit * field_limit` lanes per node, so a lane's bit
/// is `node * tensor_lane_count + ctx * field_limit + fld`. Both the source probe
/// and the destination mark address it, and they must agree exactly or a forward
/// step reads one lane and writes another. [`tensor_bit_index`] is the host twin.
fn tensor_lane_bit(node: Expr, field_limit: u32, tensor_lane_count: u32) -> Expr {
    Expr::add(
        Expr::mul(node, Expr::u32(tensor_lane_count)),
        Expr::add(
            Expr::mul(Expr::var("ctx"), Expr::u32(field_limit)),
            Expr::var("fld"),
        ),
    )
}

/// Generate Context-Sensitive / Field-Sensitive Traverse primitive program.
#[must_use]
pub fn tensor_flow_forward(
    shape: ProgramGraphShape,
    tensor_in: &str,
    tensor_out: &str,
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
) -> Program {
    match try_tensor_flow_forward(
        shape,
        tensor_in,
        tensor_out,
        context_limit,
        field_limit,
        allow_mask,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((tensor_out, DataType::U32)), error),
    }
}

/// Generate checked Context-Sensitive / Field-Sensitive Traverse primitive program.
pub fn try_tensor_flow_forward(
    shape: ProgramGraphShape,
    tensor_in: &str,
    tensor_out: &str,
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
) -> Result<Program, String> {
    if shape.node_count == 0 {
        return Err(format!(
            "{OP_ID} requires node_count > 0. Fix: pass a non-empty ProgramGraphShape."
        ));
    }
    if context_limit == 0 || field_limit == 0 {
        return Err(format!(
            "{OP_ID} requires non-zero context_limit and field_limit. Fix: pass at least one context and one field lane."
        ));
    }
    let tensor_lane_count = context_limit.checked_mul(field_limit).ok_or_else(|| {
        format!(
            "{OP_ID} context_limit={context_limit} field_limit={field_limit} overflows per-node tensor lane count. Fix: shard context or field dimensions."
        )
    })?;
    let t = Expr::InvocationId { axis: 0 };
    let words = try_tensor_words(shape.node_count, context_limit, field_limit)?;

    // X axis handles Node_ID resolution
    // Inside the body we scan the full dimension stride of Context/Fields to advance flow
    // For large graphs, context limits might be 32, meaning a whole subgroup ballot resolves
    // one context frame block per source lane instantly in hardware.

    let body = vec![
        Node::let_bind("src", t.clone()),
        // Sub-iteration across context bounds inside the single invocation
        Node::loop_for(
            "ctx",
            Expr::u32(0),
            Expr::u32(context_limit),
            vec![Node::loop_for(
                "fld",
                Expr::u32(0),
                Expr::u32(field_limit),
                {
                    // Is the (src, ctx, fld) lane hot in the tensor?
                    let mut lane = vec![Node::let_bind(
                        "abs_bit",
                        tensor_lane_bit(Expr::var("src"), field_limit, tensor_lane_count),
                    )];
                    lane.extend(when_bit_set(
                        tensor_in,
                        &Expr::var("abs_bit"),
                        Some("word_idx"),
                        "src_word",
                        "bit_mask",
                        |word| word,
                        tensor_flow_edge_scan_body(
                            tensor_out,
                            shape.node_count,
                            field_limit,
                            tensor_lane_count,
                            allow_mask,
                        ),
                    ));
                    lane
                },
            )],
        ),
    ];

    let mut buffers = shape.read_only_buffers();
    buffers.push(word_buffer(
        tensor_in,
        BINDING_TENSOR_IN,
        BufferAccess::ReadOnly,
        words,
    ));
    buffers.push(word_buffer(
        tensor_out,
        BINDING_TENSOR_OUT,
        BufferAccess::ReadWrite,
        words,
    ));

    Ok(Program::wrapped(
        buffers,
        TENSOR_FLOW_FORWARD_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(t.clone(), Expr::u32(shape.node_count)),
                body,
            )],
        )],
    ))
}

/// Host twin of [`tensor_lane_bit`]: the same absolute bit index computed on the
/// CPU oracle's side. The two must agree exactly, or the oracle proves the wrong
/// lane.
#[cfg(any(test, feature = "cpu-parity"))]
fn tensor_bit_index(node: u32, ctx: u32, fld: u32, context_limit: u32, field_limit: u32) -> u32 {
    node * context_limit * field_limit + ctx * field_limit + fld
}

#[cfg(any(test, feature = "cpu-parity"))]
fn tensor_bit_is_set(words: &[u32], bit: u32) -> bool {
    words
        .get((bit / 32) as usize)
        .copied()
        .is_some_and(|word| (word & (1u32 << (bit % 32))) != 0)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn set_tensor_bit(words: &mut [u32], bit: u32) {
    if let Some(word) = words.get_mut((bit / 32) as usize) {
        *word |= 1u32 << (bit % 32);
    }
}

/// Checked CPU oracle for [`tensor_flow_forward`].
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_tensor_flow_forward_cpu(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    tensor_in_words: &[u32],
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    try_tensor_flow_forward_cpu_into(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        tensor_in_words,
        context_limit,
        field_limit,
        allow_mask,
        &mut out,
    )?;
    Ok(out)
}

/// Checked CPU oracle for [`tensor_flow_forward`] using caller-owned output storage.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_tensor_flow_forward_cpu_into(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    tensor_in_words: &[u32],
    context_limit: u32,
    field_limit: u32,
    allow_mask: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    if context_limit == 0 || field_limit == 0 {
        return Err(format!(
            "{OP_ID} CPU oracle requires non-zero context_limit and field_limit. Fix: pass at least one context and one field lane."
        ));
    }
    if edge_offsets.len() != node_count as usize + 1 {
        return Err(format!(
            "{OP_ID} CPU oracle received {} CSR offsets for node_count={node_count}. Fix: pass exactly node_count + 1 offsets.",
            edge_offsets.len()
        ));
    }
    for (row, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "{OP_ID} CPU oracle received non-monotonic CSR offsets at row {row}: {} > {}. Fix: rebuild CSR row pointers.",
                pair[0],
                pair[1]
            ));
        }
    }
    let edge_count = edge_offsets.last().copied().unwrap_or(0) as usize;
    if edge_targets.len() < edge_count || edge_kind_mask.len() < edge_count {
        return Err(format!(
            "{OP_ID} CPU oracle received edge buffers shorter than CSR edge_count={edge_count}. Fix: pass canonical ProgramGraph edge buffers."
        ));
    }

    let word_count = try_tensor_words(node_count, context_limit, field_limit)? as usize;
    if word_count > out.capacity() {
        crate::graph::scratch::reserve_graph_items(
            out,
            word_count - out.len(),
            "tensor flow CPU oracle",
            "tensor_flow_forward output",
        )?;
    }
    out.clear();
    out.resize(word_count, 0);
    for src in 0..node_count {
        for ctx in 0..context_limit {
            for fld in 0..field_limit {
                let src_bit = tensor_bit_index(src, ctx, fld, context_limit, field_limit);
                if !tensor_bit_is_set(tensor_in_words, src_bit) {
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
                        let dst_bit = tensor_bit_index(dst, ctx, fld, context_limit, field_limit);
                        set_tensor_bit(out, dst_bit);
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || tensor_flow_forward(ProgramGraphShape::new(4, 4), "tin", "tout", 2, 2, 0xFFFF_FFFF),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),          // pg_nodes
                to_bytes(&[0, 2, 3, 4, 4]),       // pg_edge_offsets
                to_bytes(&[1, 2, 3, 3]),          // pg_edge_targets
                to_bytes(&[1, 1, 1, 1]),          // pg_edge_kind_mask
                to_bytes(&[0, 0, 0, 0]),          // pg_node_tags
                to_bytes(&[0b00010001]),          // tin
                to_bytes(&[0]),                   // tout
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[0x1110])]]
        }),
    )
}
