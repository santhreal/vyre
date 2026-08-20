//! A fused two-pass reduction publishes only the caller's buffers.
//!
//! Fusing the block pass with the combine pass concatenates both buffer
//! tables, so the intermediate the first pass writes and the second pass reads
//! appears twice. Unless the fused declaration is backend-allocated, the
//! dispatcher counts it as a third buffer the caller must supply, and a caller
//! holding only the input and the output is rejected before the kernel runs.
//!
//! The class this closes is the intermediate leaking into the dispatch
//! signature, whatever the element count or tile happens to be.

use vyre_libs::reduce::grid_stride_tree::{grid_stride_tree_sum_u32, SUM_U32_OP_ID};

fn caller_supplied(program: &vyre_foundation::ir::Program) -> Vec<String> {
    program
        .buffers()
        .iter()
        .filter(|buffer| {
            !buffer.is_backend_allocated_output()
                && !matches!(buffer.kind(), vyre_foundation::ir::MemoryKind::Shared)
        })
        .map(|buffer| buffer.name().to_string())
        .collect()
}

#[test]
fn a_multi_block_reduction_asks_the_caller_for_only_values_and_out() {
    // count > tile forces the two-pass path rather than the single-block one.
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256);
    let supplied = caller_supplied(&program);
    assert_eq!(
        supplied,
        vec!["values".to_string(), "out".to_string()],
        "the fused reduction must allocate its own partials; op {SUM_U32_OP_ID}"
    );
}

#[test]
fn the_intermediate_is_backend_allocated_at_every_block_count() {
    // Sweep tile/count pairs so a demotion that only holds for one shape fails.
    for (count, tile) in [(8192u32, 256u32), (65536, 1024), (4096, 64), (1_048_576, 1024)] {
        let program = grid_stride_tree_sum_u32("values", "out", count, tile);
        let supplied = caller_supplied(&program);
        assert_eq!(
            supplied,
            vec!["values".to_string(), "out".to_string()],
            "count={count} tile={tile} leaked an intermediate into the dispatch signature: {supplied:?}"
        );
    }
}

#[test]
fn every_storage_buffer_binding_is_unique() {
    // Two fused passes each numbered their own bindings from zero. A collision
    // silently aliases two distinct buffers onto one slot.
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256);
    let mut seen: Vec<(u32, String)> = Vec::new();
    for buffer in program.buffers() {
        if matches!(buffer.kind(), vyre_foundation::ir::MemoryKind::Shared) {
            continue;
        }
        let binding = buffer.binding();
        assert!(
            !seen.iter().any(|(b, _)| *b == binding),
            "binding {binding} claimed by both `{}` and `{}`",
            seen.iter().find(|(b, _)| *b == binding).map(|(_, n)| n.as_str()).unwrap_or("?"),
            buffer.name()
        );
        seen.push((binding, buffer.name().to_string()));
    }
}

/// The two fused passes need a grid-level fence between them.
///
/// Pass 1 writes `partials[workgroup_id]` and pass 2 reads every entry, so a
/// workgroup-scoped barrier orders only the block that wrote its own slot. A
/// `SeqCst` barrier here lets pass 2 read slots no block has written yet, which
/// surfaces as a wrong sum rather than as a dispatch error.
#[test]
fn the_fused_reduction_carries_a_grid_level_fence() {
    use vyre_foundation::ir::Node;

    fn orderings(nodes: &[Node], out: &mut Vec<String>) {
        for node in nodes {
            match node {
                Node::Barrier { ordering } => out.push(format!("{ordering:?}")),
                Node::Region { body, .. } => orderings(body, out),
                Node::Block(body) => orderings(body, out),
                _ => {}
            }
        }
    }

    let program = grid_stride_tree_sum_u32("values", "out", 1 << 20, 1024);
    let mut found = Vec::new();
    orderings(program.entry(), &mut found);
    assert!(
        found.iter().any(|o| o == "GridSync"),
        "barriers were {found:?}; pass 2 reads partials every workgroup writes"
    );
}
