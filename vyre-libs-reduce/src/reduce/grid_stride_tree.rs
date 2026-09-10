//! Multi-workgroup tree reductions.
//!
//! Two-pass reduction:
//! 1. Level 1: one workgroup per tile of the input. Each workgroup loads its
//!    tile in one coalesced pass, reduces it in workgroup scratch via subgroup
//!    shuffles / tree reduction, and writes its block total to
//!    `partials[block_id]` (independent, contention-free writes).
//! 2. Level 2: single-block reduction summing `partials[0..num_blocks]` into
//!    `out[0]` with zero atomics, striding the partials when there are more of
//!    them than one tile holds.
//!
//! This distributes work across all SMs, keeps 100% coalesced DRAM accesses,
//! and eliminates atomic serialization entirely.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs_builder::builder::strided_loop;

use super::workgroup_tree::{sum_u32_child, WorkgroupReductionScope};

/// Canonical op id for multi-workgroup grid-stride tree sum over u32 elements.
pub const SUM_U32_OP_ID: &str = "vyre-libs::reduce::grid_stride_tree_sum_u32";

/// Workgroups a launch of [`grid_stride_tree_sum_u32`] runs.
///
/// A dispatch spans the program's widest non-shared binding, which is
/// `values`, so the launch runs this many workgroups at the declared tile
/// width and no grid a caller states reaches a compiled artifact. Pass 1
/// therefore gives each launched workgroup one tile and sizes the partial
/// buffer to them, so every workgroup that runs writes a partial the combine
/// reads.
///
/// A block count taken from anywhere else leaves the surplus workgroups
/// computing a total nothing reads. A device hint of one workgroup per compute
/// unit built an 80-workgroup grid for a launch that ran 1024, and the 944
/// surplus workgroups each re-read a clamped tail element thirteen times: 0.53x
/// of a multithreaded CPU reduction at one million elements.
#[must_use]
pub fn grid_stride_tree_sum_u32_blocks(count: u32, tile: u32) -> u32 {
    count.div_ceil(tile.max(1)).max(1)
}

/// Build a multi-workgroup tree reduction program for u32 sum.
///
/// The fused program carries a whole-grid fence between the block pass and the
/// combine pass. Its grid is [`grid_stride_tree_sum_u32_blocks`], derived from
/// the same `count` and `tile` the buffer table declares, so the program and
/// the launch a dispatch infers from that table cannot disagree.
///
/// # Panics
///
/// Both passes are built here from the same shape, so fusing them fails only
/// when this module builds a pair the fuser rejects. That is a defect in the
/// builder rather than a caller error, and it panics with the fuser's reason
/// instead of returning a `Result` no caller could act on.
#[must_use]
pub fn grid_stride_tree_sum_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    let tile = tile.max(1);
    let blocks = grid_stride_tree_sum_u32_blocks(count, tile);
    // The single-block form reads one tile and nothing else, so it is correct
    // only when the whole input fits in one tile. The two-pass form covers the
    // input at every wider shape.
    if count <= tile {
        return single_block_tree_sum_u32(values, out, count, tile);
    }

    let partials = format!("__{out}_gst_partials");
    let pass1 = pass1_block_reduction(values, &partials, count, tile, blocks);
    let pass2 = pass2_combine_reduction(&partials, out, blocks, tile);

    match fuse_programs(&[pass1, pass2]) {
        Ok(fused) => {
            vyre_libs_builder::plumbing::program::outputs::demote_intermediate_outputs(fused, out)
        }
        Err(error) => panic!("grid_stride_tree_sum_u32 fusion failed: {error}"),
    }
}

fn single_block_tree_sum_u32(values: &str, out: &str, count: u32, tile: u32) -> Program {
    let scratch = "__single_tree_scratch";
    let local = Expr::LogicalWithinTileId { axis: 0 };

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
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
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

    // Both shapes of this builder present one dispatch signature: `values` in,
    // `out` published. The same owner as the fused path decides that, so the
    // single-block form cannot drift into demanding a placeholder for its own
    // result.
    vyre_libs_builder::plumbing::program::outputs::demote_intermediate_outputs(
        Program::wrapped(
            vec![
                BufferDecl::storage(values, 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(count),
                BufferDecl::workgroup(scratch, tile, DataType::U32),
                BufferDecl::storage(out, 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            ],
            [tile, 1, 1],
            vec![wrap_anonymous_region(SUM_U32_OP_ID, body)],
        ),
        out,
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
    let local = Expr::LogicalWithinTileId { axis: 0 };
    let block = Expr::LogicalTileId { axis: 0 };
    // One workgroup owns one tile. The grid is exactly `blocks` workgroups
    // wide, so the tiles partition the input and no lane reads an element
    // another lane already reduced.
    let body = vec![
        Node::let_bind("local", local.clone()),
        Node::let_bind("block", block.clone()),
        Node::let_bind(
            "index",
            Expr::add(Expr::mul(block.clone(), Expr::u32(tile)), local.clone()),
        ),
        // The lane's element is loaded under a branch, not selected after the
        // fact: `Expr::select` evaluates both arms, so a lane in the last tile
        // past `count` would still read `values[index]` past the buffer end.
        // Every lane seeds its scratch slot first, so the tail lanes contribute
        // the identity.
        Node::store(scratch, local.clone(), Expr::u32(0)),
        Node::if_then(
            Expr::lt(Expr::var("index"), Expr::u32(count)),
            vec![Node::store(
                scratch,
                local.clone(),
                Expr::load(values, Expr::var("index")),
            )],
        ),
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        // The store is guarded on the block index as well as the lane, so a
        // launch wider than the grid this program was built for discards the
        // extra blocks instead of writing past `partials`. Those blocks read
        // past `count` and contribute nothing, so an over-wide launch is slower
        // and not wrong.
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

/// Combine every partial into `out[0]` from one workgroup.
///
/// The block count is `count / tile`, so it exceeds one tile as soon as the
/// input exceeds `tile * tile`. The combine therefore strides the partials by
/// the tile width and accumulates before the tree reduction, which keeps one
/// combine pass at every input size instead of capping the block count at a
/// tile and handing the surplus back to pass 1 as redundant work.
fn pass2_combine_reduction(partials: &str, out: &str, num_blocks: u32, tile: u32) -> Program {
    let scratch = "__gst_pass2_scratch";
    let local = Expr::LogicalWithinTileId { axis: 0 };
    let steps = num_blocks.div_ceil(tile.max(1)).max(1);

    let body = vec![Node::if_then(
        Expr::is_first_logical_tile(),
        vec![
            Node::let_bind("local", local.clone()),
            Node::let_bind("acc", Expr::u32(0)),
            // The guarded body of the shared strided loop branches rather than
            // selecting, so a lane past the last partial reads nothing at all.
            strided_loop(
                tile,
                steps,
                num_blocks,
                vec![Node::assign(
                    "acc",
                    Expr::add(Expr::var("acc"), Expr::load(partials, Expr::var("idx"))),
                )],
            ),
            Node::store(scratch, local.clone(), Expr::var("acc")),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
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
        || grid_stride_tree_sum_u32("values", "out", 4, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![to_bytes(&[1, 2, 3, 4])]]
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
        assert!(program
            .buffers()
            .iter()
            .any(|b| b.name.as_ref() == "values"));
        assert!(program.buffers().iter().any(|b| b.name.as_ref() == "out"));
    }
}
