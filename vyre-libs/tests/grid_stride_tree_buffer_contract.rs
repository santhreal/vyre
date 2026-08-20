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

use vyre_libs::reduce::grid_stride_tree::{
    grid_stride_tree_sum_u32, grid_stride_tree_sum_u32_blocks, SUM_U32_OP_ID,
};

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
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256, 8);
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
    for (count, tile, blocks) in [
        (8192u32, 256u32, 8u32),
        (65536, 1024, 64),
        (4096, 64, 64),
        (1_048_576, 1024, 170),
    ] {
        let program = grid_stride_tree_sum_u32("values", "out", count, tile, blocks);
        let supplied = caller_supplied(&program);
        assert_eq!(
            supplied,
            vec!["values".to_string(), "out".to_string()],
            "count={count} tile={tile} blocks={blocks} leaked an intermediate into the dispatch signature: {supplied:?}"
        );
    }
}

#[test]
fn every_storage_buffer_binding_is_unique() {
    // Two fused passes each numbered their own bindings from zero. A collision
    // silently aliases two distinct buffers onto one slot.
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256, 8);
    let mut seen: Vec<(u32, String)> = Vec::new();
    for buffer in program.buffers() {
        if matches!(buffer.kind(), vyre_foundation::ir::MemoryKind::Shared) {
            continue;
        }
        let binding = buffer.binding();
        assert!(
            !seen.iter().any(|(b, _)| *b == binding),
            "binding {binding} claimed by both `{}` and `{}`",
            seen.iter()
                .find(|(b, _)| *b == binding)
                .map(|(_, n)| n.as_str())
                .unwrap_or("?"),
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
    use vyre_foundation::visit::child_bodies;

    fn orderings(nodes: &[Node], out: &mut Vec<String>) {
        for node in nodes {
            if let Node::Barrier { ordering } = node {
                out.push(format!("{ordering:?}"));
            }
            for body in child_bodies(node) {
                orderings(body, out);
            }
        }
    }

    let program = grid_stride_tree_sum_u32("values", "out", 1 << 20, 1024, 170);
    let mut found = Vec::new();
    orderings(program.entry(), &mut found);
    assert!(
        found.iter().any(|o| o == "GridSync"),
        "barriers were {found:?}; pass 2 reads partials every workgroup writes"
    );
}

/// Pass 1 must reach every element from a grid narrower than the input.
///
/// The block count is chosen by the caller from the device's cooperative
/// residency, so it is routinely far smaller than `count / tile`. A thread that
/// reads only its own invocation index then covers `blocks * tile` elements and
/// silently drops the rest, which reads as a plausible-looking short sum. The
/// invariant is that the emitted loop's trip count times the grid stride spans
/// the whole input.
#[test]
fn pass_one_strides_far_enough_to_cover_every_element() {
    use vyre_foundation::ir::{Expr, Node};
    use vyre_foundation::visit::child_bodies;

    fn loop_bounds(nodes: &[Node], out: &mut Vec<u32>) {
        for node in nodes {
            if let Node::Loop {
                to: Expr::LitU32(bound),
                ..
            } = node
            {
                out.push(*bound);
            }
            for body in child_bodies(node) {
                loop_bounds(body, out);
            }
        }
    }

    for (count, tile, requested) in [
        (1_048_576u32, 1024u32, 170u32),
        (1_048_576, 1024, 128),
        (65536, 1024, 3),
        (8192, 256, 5),
        (1_000_001, 1024, 170),
    ] {
        let blocks = grid_stride_tree_sum_u32_blocks(count, tile, requested);
        let program = grid_stride_tree_sum_u32("values", "out", count, tile, requested);
        let mut bounds = Vec::new();
        loop_bounds(program.entry(), &mut bounds);
        let stride = u64::from(blocks) * u64::from(tile);
        let covered = bounds
            .iter()
            .map(|trips| u64::from(*trips) * stride)
            .max()
            .unwrap_or(0);
        assert!(
            covered >= u64::from(count),
            "count={count} tile={tile} blocks={blocks}: a grid stride of {stride} over trip counts {bounds:?} reaches {covered} elements"
        );
    }
}
