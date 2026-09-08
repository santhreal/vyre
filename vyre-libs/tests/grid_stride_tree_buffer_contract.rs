//! A fused two-pass reduction publishes its result and asks for its input.
//!
//! Fusing the block pass with the combine pass concatenates both buffer
//! tables, so the intermediate the first pass writes and the second pass reads
//! appears twice. Unless the fused declaration is backend-allocated, the
//! dispatcher counts it as a buffer the caller must supply, and a caller
//! holding only the input is rejected before the kernel runs.
//!
//! The result buffer is the other half of the same ABI. Nothing reads its prior
//! contents, so it is a backend-allocated output and consumes no host input
//! slot. Leaving it plain read-write storage made the oracle demand a value for
//! it, and a placeholder for an output is what the artifact ABI rejects on a
//! device.
//!
//! The class this closes is either buffer landing on the wrong side of the
//! dispatch signature, whatever the element count or tile happens to be.

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

/// The full signature at both extents, asserted together: one caller buffer,
/// one backend-allocated output, and no third entry for the intermediate.
fn assert_dispatch_signature(program: &vyre_foundation::ir::Program, case: &str) {
    let supplied = caller_supplied(program);
    assert_eq!(
        supplied,
        vec!["values".to_string()],
        "{case}: the fused reduction must allocate its own partials and its own result; op {SUM_U32_OP_ID}"
    );
    let published: Vec<&str> = program
        .buffers()
        .iter()
        .filter(|buffer| buffer.is_backend_allocated_output() && buffer.is_pipeline_live_out())
        .map(|buffer| buffer.name())
        .collect();
    assert!(
        published.contains(&"out"),
        "{case}: the result must be published, not read from the dispatch inputs: {published:?}"
    );
}

#[test]
fn a_multi_block_reduction_asks_the_caller_for_only_its_input() {
    // count > tile forces the two-pass path rather than the single-block one.
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256, 8);
    assert_dispatch_signature(&program, "count=8192 tile=256 blocks=8");
}

#[test]
fn one_signature_holds_across_both_forms() {
    // The builder switches to the single-block form at `count <= tile`, so the
    // sweep straddles that boundary: a signature that only holds for the fused
    // form leaves the caller unable to dispatch the same op at a small count.
    for (count, tile, blocks) in [
        (1u32, 16u32, 1u32),
        (16, 16, 1),
        (255, 256, 4),
        (256, 256, 1),
        (257, 256, 4),
        (8192, 256, 8),
        (65536, 1024, 64),
        (4096, 64, 64),
        (1_048_576, 1024, 170),
    ] {
        let program = grid_stride_tree_sum_u32("values", "out", count, tile, blocks);
        assert_dispatch_signature(
            &program,
            &format!("count={count} tile={tile} blocks={blocks}"),
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
/// Pass 1 writes `partials[logical_tile_id]` and pass 2 reads every entry, so a
/// workgroup-scoped barrier orders only the block that wrote its own slot. A
/// `SeqCst` barrier here lets pass 2 read slots no block has written yet, which
/// surfaces as a wrong sum rather than as a dispatch error.
#[test]
fn the_fused_reduction_carries_a_grid_level_fence() {
    use vyre_foundation::ir::Node;
    use vyre_foundation::visit::child_bodies;

    fn orderings(nodes: &[Node], out: &mut Vec<String>) {
        for node in nodes {
            if let Node::LogicalBarrier { ordering } = node {
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
        // What a backend that reports no compute-unit count asks for: no device
        // cap at all, so the shape is the only thing standing between the
        // request and the launch.
        (1_048_576, 256, u32::MAX),
        (1_048_576, 1024, u32::MAX),
    ] {
        assert!(
            count > tile,
            "Fix: this sweep reads the strided loop bounds, and a count within one tile takes the single-block form, which has no loop"
        );
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
