//! Multi-workgroup tree reductions.
//!
//! Two-pass reduction:
//! 1. Level 1: one workgroup per span of the input. Each workgroup loads its
//!    span in coalesced tile-strided rounds, reduces it in workgroup scratch
//!    via a tree reduction, and writes its span total to `partials[block_id]`
//!    (independent, contention-free writes). A span is
//!    `PASS1_ELEMENTS_PER_LANE` tiles wide, so a workgroup the launch runs
//!    outside the reducing grid seeds its partial with the identity and loads
//!    nothing.
//! 2. Level 2: single-block reduction summing `partials[0..num_blocks]` into
//!    `out[0]` with zero atomics, striding the partials when there are more of
//!    them than one tile holds.
//!
//! This distributes work across all SMs, keeps 100% coalesced DRAM accesses,
//! and eliminates atomic serialization entirely.

use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::execution_plan::fusion::fuse_programs;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs_builder::builder::{strided_loop, strided_loop_from};

use super::workgroup_tree::{sum_u32_child, WorkgroupReductionScope};

/// Canonical op id for multi-workgroup grid-stride tree sum over u32 elements.
pub const SUM_U32_OP_ID: &str = "vyre-libs::reduce::grid_stride_tree_sum_u32";

/// Elements one lane loads in pass 1 of the two-pass form.
///
/// A launch spans the widest non-shared binding, so pass 1 runs
/// [`grid_stride_tree_sum_u32_blocks`] workgroups whatever this builder would
/// prefer, and the only shape it owns is how much of the input one lane
/// reduces before the shared-memory tree runs. One element per lane makes that
/// tree a fixed cost per tile rather than per input: at a 1024-wide tile a
/// 1024-thread workgroup is the only one an RTX 3080 Ti SM holds resident, so
/// the memory pipe stalls through all ten tree rounds of every tile and one
/// load per lane leaves too little in flight to cover DRAM latency in between.
///
/// Loading this many elements per lane amortizes one tree over that many
/// loads and puts that many loads in flight per lane. Measured on an
/// RTX 3080 Ti with clocks pinned, one million u32 at a 1024-wide tile:
/// 27296 ns at one element per lane, 12736 ns at four, 8640 ns at sixteen,
/// 7776 ns at thirty-two, which is 154 GB/s against 539 GB/s of a 912 GB/s
/// peak. The same sweep at a 256-wide tile runs 18592 ns to 6336 ns, so the
/// factor carries the shape at every tile the device admits rather than
/// standing in for one width.
const PASS1_ELEMENTS_PER_LANE: u32 = 32;

/// Workgroups of a pass-1 launch that reduce a span of the input.
///
/// The launch runs [`grid_stride_tree_sum_u32_blocks`] workgroups and each
/// reducing one covers `tile * PASS1_ELEMENTS_PER_LANE` elements, so this many
/// cover the input and the rest run no part of the reduction. Both counts are
/// pure functions of `count` and `tile`, which is what keeps the program and
/// the launch a dispatch infers from its buffer table in agreement.
fn pass1_working_blocks(count: u32, tile: u32) -> u32 {
    count
        .div_ceil(tile.max(1).saturating_mul(PASS1_ELEMENTS_PER_LANE))
        .max(1)
}

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
    let lane = Expr::LogicalWithinTileId { axis: 0 };
    let block = Expr::LogicalTileId { axis: 0 };
    let working = pass1_working_blocks(count, tile);
    let span = tile.saturating_mul(PASS1_ELEMENTS_PER_LANE);

    // One reducing workgroup owns one span of the input, so the spans partition
    // it and no lane reads an element another lane already reduced.
    let reduce_span = vec![
        // `local` is the within-tile lane, which is what the shared workgroup
        // tree indexes its scratch by, so this workgroup's span offset reaches
        // the load loop as its base instead. Striding by the tile keeps every
        // round of loads coalesced across the tile while the whole span stays
        // inside one workgroup.
        Node::let_bind("local", lane.clone()),
        Node::let_bind("acc", Expr::u32(0)),
        // The guarded body of the shared strided loop branches rather than
        // selecting, so a lane past the last element reads nothing at all:
        // `Expr::select` evaluates both arms and would read `values` past the
        // buffer end. A lane that loads nothing keeps the identity.
        strided_loop_from(
            Expr::add(Expr::mul(block.clone(), Expr::u32(span)), lane.clone()),
            tile,
            PASS1_ELEMENTS_PER_LANE,
            count,
            vec![Node::assign(
                "acc",
                Expr::add(Expr::var("acc"), Expr::load(values, Expr::var("idx"))),
            )],
        ),
        Node::store(scratch, lane.clone(), Expr::var("acc")),
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        sum_u32_child(
            SUM_U32_OP_ID,
            tile,
            scratch,
            WorkgroupReductionScope::EveryWorkgroup,
        ),
        Node::if_then(
            Expr::eq(lane.clone(), Expr::u32(0)),
            vec![Node::store(
                partials,
                block.clone(),
                Expr::load(scratch, Expr::u32(0)),
            )],
        ),
    ];

    let body = vec![
        // The combine reads one partial per launched workgroup, so a workgroup
        // outside the reducing grid seeds its slot with the identity and runs
        // no part of the reduction: it holds the tree rounds and the loads that
        // amortize them to the workgroups that own a span. The seed is guarded
        // on the block index, so a launch wider than the grid this program was
        // built for discards the extra workgroups instead of writing past
        // `partials`.
        Node::if_then(
            Expr::and(
                Expr::eq(lane, Expr::u32(0)),
                Expr::lt(block.clone(), Expr::u32(blocks)),
            ),
            vec![Node::store(partials, block.clone(), Expr::u32(0))],
        ),
        Node::if_then(Expr::lt(block, Expr::u32(working)), reduce_span),
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

    /// Every element is inside some reducing workgroup's span, and no reducing
    /// workgroup is outside the launch.
    ///
    /// WHY: pass 1 runs `grid_stride_tree_sum_u32_blocks` workgroups and only
    /// the first `pass1_working_blocks` of them load anything, so the two
    /// counts have to bracket the input from both sides. A reducing grid too
    /// narrow drops the tail of the input and the reduction reports a partial
    /// sum; one wider than the launch places a span in a workgroup that never
    /// runs and drops it the same way. What this does not catch is whether the
    /// spans overlap, which the device sum over a non-uniform input covers.
    #[test]
    fn the_reducing_grid_covers_the_input_and_fits_the_launch() {
        for (count, tile) in [
            (1_048_576u32, 1024u32),
            (1_048_576, 256),
            (16_777_216, 1024),
            (1_000_001, 64),
            (1_000_001, 1024),
            (4096, 64),
            (2048, 1024),
            (1025, 1024),
        ] {
            let working = pass1_working_blocks(count, tile);
            let launched = grid_stride_tree_sum_u32_blocks(count, tile);
            assert!(
                u64::from(working)
                    * u64::from(tile)
                    * u64::from(PASS1_ELEMENTS_PER_LANE)
                    >= u64::from(count),
                "count={count} tile={tile}: {working} spans of {tile} lanes loading {PASS1_ELEMENTS_PER_LANE} elements each do not reach every element"
            );
            assert!(
                working <= launched,
                "count={count} tile={tile}: {working} reducing workgroups exceed the {launched} the launch runs, so a span lands in a workgroup that never runs"
            );
        }
    }

    /// The release shape amortizes one shared-memory tree over many loads.
    ///
    /// WHY: the launch is `count / tile` workgroups whatever this builder
    /// prefers, so one element per lane paid the tree's ten rounds once per
    /// 1024 loads and left one load in flight per lane. At one million u32 and
    /// a 1024-wide tile that measured 27296 ns against 7776 ns for the shape
    /// this asserts, and the release case `foundation.reduce.sum.crossover`
    /// read 0.69x of its rayon baseline against a 1.10x contract. Setting the
    /// factor back to one collapses both counts onto each other and turns this
    /// red.
    #[test]
    fn one_reducing_workgroup_covers_many_launched_tiles() {
        let count = 1_048_576;
        let tile = 1024;
        assert_eq!(
            grid_stride_tree_sum_u32_blocks(count, tile),
            1024,
            "Fix: the launch spans the widest non-shared binding, so it runs one workgroup per tile of the input"
        );
        assert_eq!(
            pass1_working_blocks(count, tile),
            32,
            "Fix: {PASS1_ELEMENTS_PER_LANE} elements per lane over a {tile}-wide tile is {} elements per reducing workgroup",
            tile * PASS1_ELEMENTS_PER_LANE
        );
    }
}
