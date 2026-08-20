//! Multi-workgroup grid-stride tree reductions.
//!
//! Two-pass reduction:
//! 1. Level 1: Multi-block grid-stride reduction. Each workgroup loads a strided slice
//!    of the input in a coalesced loop, reduces it locally within workgroup scratch
//!    via warp shuffles / tree reduction, and writes its block total to `partials[block_id]`
//!    (independent, contention-free writes).
//! 2. Level 2: Single-block reduction summing `partials[0..num_blocks]` into `out[0]`
//!    with zero atomics.
//!
//! This distributes work across all SMs, keeps 100% coalesced DRAM accesses,
//! and eliminates atomic serialization entirely.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::workgroup_tree::{sum_u32_child, WorkgroupReductionScope};

/// Canonical op id for multi-workgroup grid-stride tree sum over u32 elements.
pub const SUM_U32_OP_ID: &str = "vyre-libs::reduce::grid_stride_tree_sum_u32";

/// Build a multi-workgroup tree reduction program for u32 sum.
#[must_use]
pub fn grid_stride_tree_sum_u32(
    values: &str,
    out: &str,
    count: u32,
    tile: u32,
) -> Program {
    let tile = tile.max(1);
    if count <= tile {
        return single_block_tree_sum_u32(values, out, count, tile);
    }

    let num_blocks = count.div_ceil(tile);
    if num_blocks == 1 {
        return single_block_tree_sum_u32(values, out, count, tile);
    }

    let partials = format!("__{out}_gst_partials");
    let pass1 = pass1_block_reduction(values, &partials, count, tile, num_blocks);
    let pass2 = pass2_combine_reduction(&partials, out, num_blocks, tile);

    match fuse_programs(&[pass1, pass2]) {
        Ok(fused) => crate::plumbing::program::outputs::demote_intermediate_outputs(fused, out),
        Err(error) => panic!("grid_stride_tree_sum_u32 fusion failed: {error}"),
    }
}

fn single_block_tree_sum_u32(
    values: &str,
    out: &str,
    count: u32,
    tile: u32,
) -> Program {
    let scratch = "__single_tree_scratch";
    let local = Expr::LocalId { axis: 0 };

    let body = vec![
        Node::let_bind("local", local.clone()),
        Node::let_bind(
            "acc",
            Expr::select(
                Expr::lt(local.clone(), Expr::u32(count)),
                Expr::load(values, local.clone()),
                Expr::u32(0),
            ),
        ),
        Node::store(scratch, local.clone(), Expr::var("acc")),
        Node::barrier(),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        Node::if_then(
            Expr::eq(local, Expr::u32(0)),
            vec![Node::store(
                out,
                Expr::u32(0),
                Expr::load(scratch, Expr::u32(0)),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

fn pass1_block_reduction(
    values: &str,
    partials: &str,
    count: u32,
    tile: u32,
    num_blocks: u32,
) -> Program {
    let scratch = "__gst_pass1_scratch";
    let local = Expr::LocalId { axis: 0 };
    let block = Expr::WorkgroupId { axis: 0 };
    let global = Expr::InvocationId { axis: 0 };

    let body = vec![
        Node::let_bind("local", local.clone()),
        Node::let_bind("block", block.clone()),
        Node::let_bind("global", global.clone()),
        Node::let_bind(
            "acc",
            Expr::select(
                Expr::lt(global.clone(), Expr::u32(count)),
                Expr::load(values, global),
                Expr::u32(0),
            ),
        ),
        Node::store(scratch, local.clone(), Expr::var("acc")),
        Node::barrier(),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        Node::if_then(
            Expr::eq(local, Expr::u32(0)),
            vec![Node::store(
                partials,
                block,
                Expr::load(scratch, Expr::u32(0)),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(partials, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

fn pass2_combine_reduction(
    partials: &str,
    out: &str,
    num_blocks: u32,
    tile: u32,
) -> Program {
    let scratch = "__gst_pass2_scratch";
    let local = Expr::LocalId { axis: 0 };

    let body = vec![
        Node::if_then(
            Expr::is_first_workgroup(),
            vec![
                Node::let_bind("local", local.clone()),
                Node::let_bind(
                    "val",
                    Expr::select(
                        Expr::lt(local.clone(), Expr::u32(num_blocks)),
                        Expr::load(partials, local.clone()),
                        Expr::u32(0),
                    ),
                ),
                Node::store(scratch, local.clone(), Expr::var("val")),
                Node::barrier(),
                sum_u32_child(
                    SUM_U32_OP_ID,
                    tile,
                    scratch,
                    WorkgroupReductionScope::FirstWorkgroup,
                ),
                Node::if_then(
                    Expr::eq(local, Expr::u32(0)),
                    vec![Node::store(
                        out,
                        Expr::u32(0),
                        Expr::load(scratch, Expr::u32(0)),
                    )],
                ),
            ],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(partials, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        SUM_U32_OP_ID,
        || grid_stride_tree_sum_u32("values", "out", 4, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[1, 2, 3, 4]), to_bytes(&[0])]]
        }),
        Some(|| vec![vec![vec![0x0a, 0x00, 0x00, 0x00]]]),
    )
    .with_laws(&["associative", "commutative", "identity"])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_block_tree_sum_u32_builds_valid_program() {
        let program = grid_stride_tree_sum_u32("values", "out", 256, 256);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name.as_ref(), "values");
        assert_eq!(program.buffers()[1].name.as_ref(), "__single_tree_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "out");
    }

    #[test]
    fn multi_block_tree_sum_u32_fuses_two_passes() {
        let program = grid_stride_tree_sum_u32("values", "out", 1048576, 1024);
        assert_eq!(program.workgroup_size(), [1024, 1, 1]);
        assert!(program.buffers().iter().any(|b| b.name.as_ref() == "values"));
        assert!(program.buffers().iter().any(|b| b.name.as_ref() == "out"));
    }
}
