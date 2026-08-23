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

/// Effective workgroup count for [`grid_stride_tree_sum_u32`].
///
/// The builder clamps a requested count to what the shape admits: pass 2
/// combines the partials with one workgroup, so the partial count cannot
/// exceed one tile, and a grid wider than the input has idle blocks. The
/// caller must launch exactly this grid, so both read the rule from here and
/// the program cannot disagree with its own dispatch.
#[must_use]
pub fn grid_stride_tree_sum_u32_blocks(count: u32, tile: u32, blocks: u32) -> u32 {
    let tile = tile.max(1);
    blocks.clamp(1, tile).min(count.div_ceil(tile).max(1))
}

/// Build a multi-workgroup tree reduction program for u32 sum.
///
/// `blocks` is the workgroup count the caller will launch. The fused program
/// carries a whole-grid fence, so it runs as one cooperative launch whose grid
/// must be fully co-resident. Only the caller knows that device limit, so the
/// caller fixes the grid and pass 1 strides over the input to cover it.
///
/// # Panics
///
/// Both passes are built here from the same shape, so fusing them fails only
/// when this module builds a pair the fuser rejects. That is a defect in the
/// builder rather than a caller error, and it panics with the fuser's reason
/// instead of returning a `Result` no caller could act on.
#[must_use]
pub fn grid_stride_tree_sum_u32(
    values: &str,
    out: &str,
    count: u32,
    tile: u32,
    blocks: u32,
) -> Program {
    let tile = tile.max(1);
    let blocks = grid_stride_tree_sum_u32_blocks(count, tile, blocks);
    // One block is not a reason to take the single-block form: that form reads
    // one tile and nothing else, so at `count > tile` it would sum a prefix and
    // report it as the total. The strided pass covers the input at any block
    // count, including one.
    if count <= tile {
        return single_block_tree_sum_u32(values, out, count, tile);
    }

    let partials = format!("__{out}_gst_partials");
    let pass1 = pass1_block_reduction(values, &partials, count, tile, blocks);
    let pass2 = pass2_combine_reduction(&partials, out, blocks, tile);

    match fuse_programs(&[pass1, pass2]) {
        Ok(fused) => crate::plumbing::program::outputs::demote_intermediate_outputs(fused, out),
        Err(error) => panic!("grid_stride_tree_sum_u32 fusion failed: {error}"),
    }
}

fn single_block_tree_sum_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    let scratch = "__single_tree_scratch";
    let local = Expr::LocalId { axis: 0 };

    let body = vec![
        Node::let_bind("local", local.clone()),
        // The lane's element is loaded under a branch, not selected after the
        // fact: `Expr::select` evaluates both arms, so a lane past `count` would
        // still read `values[local]` past the buffer end. Every lane seeds its
        // scratch slot first, so the tail lanes contribute the identity.
        Node::store(scratch, local.clone(), Expr::u32(0)),
        Node::if_then(
            Expr::lt(local.clone(), Expr::u32(count)),
            vec![Node::store(
                scratch,
                local.clone(),
                Expr::load(values, local.clone()),
            )],
        ),
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
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
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
    blocks: u32,
) -> Program {
    let scratch = "__gst_pass1_scratch";
    let local = Expr::LocalId { axis: 0 };
    let block = Expr::WorkgroupId { axis: 0 };
    // The grid is fixed by the caller, so a thread covers the input by striding
    // the whole grid instead of owning one element. Both the stride and the
    // trip count are known here, which keeps the loop bound free of any device
    // fact.
    let stride = blocks.saturating_mul(tile).max(1);
    let iterations = count.div_ceil(stride);
    let last_index = count.saturating_sub(1);

    let body = vec![
        Node::let_bind("local", local.clone()),
        Node::let_bind("block", block.clone()),
        Node::let_bind(
            "base",
            Expr::add(Expr::mul(block.clone(), Expr::u32(tile)), local.clone()),
        ),
        Node::let_bind("acc", Expr::u32(0)),
        Node::loop_for(
            "step",
            Expr::u32(0),
            Expr::u32(iterations),
            vec![
                Node::let_bind(
                    "index",
                    Expr::add(
                        Expr::var("base"),
                        Expr::mul(Expr::var("step"), Expr::u32(stride)),
                    ),
                ),
                Node::assign(
                    "acc",
                    Expr::select(
                        Expr::lt(Expr::var("index"), Expr::u32(count)),
                        Expr::add(
                            Expr::var("acc"),
                            // A select evaluates both arms, so the discarded
                            // load is clamped back inside the buffer.
                            Expr::load(
                                values,
                                Expr::min(Expr::var("index"), Expr::u32(last_index)),
                            ),
                        ),
                        Expr::var("acc"),
                    ),
                ),
            ],
        ),
        Node::store(scratch, local.clone(), Expr::var("acc")),
        Node::barrier(),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        // The store is guarded on the block index as well as the lane, so a
        // launch wider than the grid this program was built for discards the
        // extra blocks instead of writing past `partials`. The blocks inside the
        // grid still cover the input, so an over-wide launch is slower and not
        // wrong.
        Node::if_then(
            Expr::and(
                Expr::eq(local, Expr::u32(0)),
                Expr::lt(block.clone(), Expr::u32(blocks)),
            ),
            vec![Node::store(
                partials,
                block,
                Expr::load(scratch, Expr::u32(0)),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(partials, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(blocks)
                .with_pipeline_live_out(true),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

fn pass2_combine_reduction(partials: &str, out: &str, num_blocks: u32, tile: u32) -> Program {
    let scratch = "__gst_pass2_scratch";
    let local = Expr::LocalId { axis: 0 };

    let body = vec![Node::if_then(
        Expr::is_first_workgroup(),
        vec![
            Node::let_bind("local", local.clone()),
            // Branch, not select: `Expr::select` evaluates both arms, so a lane
            // past `num_blocks` would read `partials` past its end. The tile is
            // a power of two and the block count need not be, so that tail
            // exists on most shapes.
            Node::store(scratch, local.clone(), Expr::u32(0)),
            Node::if_then(
                Expr::lt(local.clone(), Expr::u32(num_blocks)),
                vec![Node::store(
                    scratch,
                    local.clone(),
                    Expr::load(partials, local.clone()),
                )],
            ),
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
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(partials, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true),
            BufferDecl::workgroup(scratch, tile, DataType::U32),
            BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        SUM_U32_OP_ID,
        || grid_stride_tree_sum_u32("values", "out", 4, 4, 1),
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
        let program = grid_stride_tree_sum_u32("values", "out", 256, 256, 1);
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert_eq!(program.buffers().len(), 3);
        assert_eq!(program.buffers()[0].name.as_ref(), "values");
        assert_eq!(program.buffers()[1].name.as_ref(), "__single_tree_scratch");
        assert_eq!(program.buffers()[2].name.as_ref(), "out");
    }

    #[test]
    fn multi_block_tree_sum_u32_fuses_two_passes() {
        let program = grid_stride_tree_sum_u32("values", "out", 1048576, 1024, 128);
        assert_eq!(program.workgroup_size(), [1024, 1, 1]);
        assert!(program
            .buffers()
            .iter()
            .any(|b| b.name.as_ref() == "values"));
        assert!(program.buffers().iter().any(|b| b.name.as_ref() == "out"));
    }
}
