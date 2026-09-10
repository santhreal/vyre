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

use vyre_libs_reduce::reduce::grid_stride_tree::{
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
    let program = grid_stride_tree_sum_u32("values", "out", 8192, 256);
    assert_dispatch_signature(&program, "count=8192 tile=256");
}

#[test]
fn one_signature_holds_across_both_forms() {
    // The builder switches to the single-block form at `count <= tile`, so the
    // sweep straddles that boundary: a signature that only holds for the fused
    // form leaves the caller unable to dispatch the same op at a small count.
    for (count, tile) in [
        (1u32, 16u32),
        (16, 16),
        (255, 256),
        (256, 256),
        (257, 256),
        (8192, 256),
        (65536, 1024),
        (4096, 64),
        (1_048_576, 1024),
    ] {
        let program = grid_stride_tree_sum_u32("values", "out", count, tile);
        assert_dispatch_signature(&program, &format!("count={count} tile={tile}"));
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

    let program = grid_stride_tree_sum_u32("values", "out", 1 << 20, 1024);
    let mut found = Vec::new();
    orderings(program.entry(), &mut found);
    assert!(
        found.iter().any(|o| o == "GridSync"),
        "barriers were {found:?}; pass 2 reads partials every workgroup writes"
    );
}

/// Every workgroup a launch runs writes a partial the combine reads.
///
/// A dispatch spans the program's widest non-shared binding and runs that span
/// divided by the declared workgroup width, and a compiled artifact records
/// that launch, so no grid a caller states reaches the device. The partial
/// buffer therefore has to hold one slot per launched workgroup. When it held
/// fewer, the surplus workgroups reduced elements nothing read: at one million
/// elements the builder sized the grid from a device compute-unit count of 80
/// while the launch ran 1024 workgroups, each of the 944 surplus workgroups
/// re-read a clamped tail element thirteen times, and the reduction measured
/// 0.53x of a multithreaded CPU baseline against a release contract of 1.10x.
///
/// The launched workgroup count is recomputed from the program's own buffer
/// table because `vyre-libs-reduce` cannot depend on the driver crate that owns
/// the rule. The rule is the one `dispatch_element_count_for_program` applies
/// to a program declaring a shared buffer: the widest non-shared binding.
#[test]
fn every_launched_workgroup_writes_a_partial_the_combine_reads() {
    for (count, tile) in [
        (1_048_576u32, 1024u32),
        (65536, 1024),
        (8192, 256),
        (1_000_001, 1024),
        (1_048_576, 256),
        (4096, 64),
        // Above `tile * tile` the block count no longer fits one tile. Capping
        // it there is what handed the surplus back to pass 1 as redundant work.
        (1_048_576, 512),
        (16_777_216, 1024),
        (1_000_001, 64),
    ] {
        assert!(
            count > tile,
            "Fix: this sweep reads the partial buffer, and a count within one tile takes the single-block form, which declares none"
        );
        let program = grid_stride_tree_sum_u32("values", "out", count, tile);
        let blocks = grid_stride_tree_sum_u32_blocks(count, tile);
        assert_eq!(
            launched_workgroups(&program),
            blocks,
            "count={count} tile={tile}: the launch runs a different grid than the builder sized its partials for"
        );
        assert_eq!(
            partial_slots(&program),
            blocks,
            "count={count} tile={tile}: the partial buffer holds a different number of slots than the grid has workgroups"
        );
        assert!(
            u64::from(blocks) * u64::from(tile) >= u64::from(count),
            "count={count} tile={tile}: {blocks} tiles of {tile} lanes do not reach every element"
        );
    }
}

/// Workgroups a dispatch of `program` runs, read from the program alone.
fn launched_workgroups(program: &vyre_foundation::ir::Program) -> u32 {
    let span = program
        .buffers()
        .iter()
        .filter(|buffer| !matches!(buffer.kind(), vyre_foundation::ir::MemoryKind::Shared))
        .map(vyre_foundation::ir::BufferDecl::count)
        .max()
        .unwrap_or(1);
    span.div_ceil(program.workgroup_size()[0].max(1))
}

/// Declared slot count of the fused program's partial buffer.
fn partial_slots(program: &vyre_foundation::ir::Program) -> u32 {
    program
        .buffers()
        .iter()
        .find(|buffer| buffer.name().ends_with("_gst_partials"))
        .map(vyre_foundation::ir::BufferDecl::count)
        .unwrap_or_else(|| {
            panic!(
                "Fix: the fused two-pass reduction must declare a partial buffer; it declared {:?}",
                program
                    .buffers()
                    .iter()
                    .map(vyre_foundation::ir::BufferDecl::name)
                    .collect::<Vec<_>>()
            )
        })
}
