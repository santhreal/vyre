//! Multi-workgroup tree reductions.
//!
//! Two-pass reduction:
//! 1. Level 1: one workgroup per span of the input. Each workgroup loads its
//!    span in coalesced tile-strided rounds, reduces it in workgroup scratch
//!    via a tree reduction, and writes its span total to `partials[block_id]`
//!    (independent, contention-free writes). A span is
//!    `PASS1_ELEMENTS_PER_LANE` tiles wide, so the input is covered by
//!    [`grid_stride_tree_sum_u32_blocks`] workgroups and every one of them
//!    reduces a span.
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
/// A launch spans the widest non-shared binding, capped by the domain the
/// program's own guards admit, so the only shape this builder owns is how much
/// of the input one lane reduces before the shared-memory tree runs. One
/// element per lane makes that tree a fixed cost per tile rather than per
/// input: at a 1024-wide tile a 1024-thread workgroup is the only one an
/// RTX 3080 Ti SM holds resident, so the memory pipe stalls through all ten
/// tree rounds of every tile and one load per lane leaves too little in flight
/// to cover DRAM latency in between.
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

/// Workgroups a launch of [`grid_stride_tree_sum_u32`] runs.
///
/// Each one covers `tile * PASS1_ELEMENTS_PER_LANE` elements, so this many
/// cover the input and every workgroup the launch runs reduces a span and
/// writes the partial the combine reads. The count is a pure function of
/// `count` and `tile`, which is what keeps the program and the launch a
/// dispatch infers from it in agreement: pass 1 guards its reduction on the
/// block index being below this count, and that guard is the bound the launch
/// span is narrowed to.
///
/// A block count taken from anywhere else leaves workgroups that compute a
/// total nothing reads. A device hint of one workgroup per compute unit built
/// an 80-workgroup grid for a launch that ran 1024, and the 944 surplus
/// workgroups each re-read a clamped tail element thirteen times: 0.53x of a
/// multithreaded CPU reduction at one million elements. Sizing the partial
/// buffer to the input span divided by the tile instead ran 1024 workgroups
/// where 32 reduce, and the 992 that only seeded a partial with the identity
/// cost 10 us of block scheduling in pass 1 and 13 us in the combine, which is
/// the launch narrowing this count now states.
#[must_use]
pub fn grid_stride_tree_sum_u32_blocks(count: u32, tile: u32) -> u32 {
    count
        .div_ceil(tile.max(1).saturating_mul(PASS1_ELEMENTS_PER_LANE))
        .max(1)
}

/// Build a multi-workgroup tree reduction program for u32 sum.
///
/// The fused program carries a whole-grid fence between the block pass and the
/// combine pass. Its grid is [`grid_stride_tree_sum_u32_blocks`], derived from
/// the same `count` and `tile` the buffer table declares and stated in the
/// program as the block-index guard both passes sit under, so the program and
/// the launch a dispatch infers from it cannot disagree. That grid is
/// `PASS1_ELEMENTS_PER_LANE` times narrower than the input span, which is what
/// keeps the fence inside cooperative residency and the fused program on one
/// launch instead of a host-orchestrated segment pair.
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

/// Reduce one span of the input per workgroup into `partials[block_id]`.
///
/// Every one of the `blocks` workgroups owns a span and writes its own
/// partial, so the combine reads a buffer every launched workgroup filled and
/// no slot needs an identity seed. The reduction is guarded on the block index
/// being below `blocks`, which is the bound the launch span narrows to and the
/// reason a launch wider than this grid cannot write past `partials`.
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
            Expr::eq(lane, Expr::u32(0)),
            vec![Node::store(
                partials,
                block.clone(),
                Expr::load(scratch, Expr::u32(0)),
            )],
        ),
    ];

    let body = vec![Node::if_then(
        Expr::lt(block, Expr::u32(blocks)),
        reduce_span,
    )];

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

    /// Every element is inside some reducing workgroup's span, and no workgroup
    /// is outside the reduction.
    ///
    /// WHY: the launch runs `grid_stride_tree_sum_u32_blocks` workgroups and
    /// every one of them loads a span, so the count has to reach the end of the
    /// input and stop within one span of it. A grid too narrow drops the tail
    /// and the reduction reports a partial sum; one wider leaves a workgroup
    /// that writes a partial from an empty span, which is the 992-workgroup
    /// surplus this count used to carry. What this does not catch is whether
    /// the spans overlap, which the device sum over a non-uniform input covers.
    #[test]
    fn the_reducing_grid_covers_the_input_with_no_surplus_workgroup() {
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
            let blocks = grid_stride_tree_sum_u32_blocks(count, tile);
            let span = u64::from(tile) * u64::from(PASS1_ELEMENTS_PER_LANE);
            assert!(
                u64::from(blocks) * span >= u64::from(count),
                "count={count} tile={tile}: {blocks} spans of {tile} lanes loading {PASS1_ELEMENTS_PER_LANE} elements each do not reach every element"
            );
            assert!(
                u64::from(blocks - 1) * span < u64::from(count),
                "count={count} tile={tile}: {blocks} workgroups cover the input with a whole span to spare, so one of them reduces nothing and writes a partial the combine still reads"
            );
        }
    }

    /// The release shape amortizes one shared-memory tree over many loads and
    /// narrows the launch to the workgroups that run it.
    ///
    /// WHY: the grid used to be `count / tile` workgroups, one per tile of the
    /// input, so one element per lane paid the tree's ten rounds once per 1024
    /// loads and left one load in flight per lane. At one million u32 and a
    /// 1024-wide tile that measured 27296 ns against 7776 ns for the shape this
    /// asserts, and the release case `foundation.reduce.sum.crossover` read
    /// 0.69x of its rayon baseline against a 1.10x contract. Keeping the factor
    /// but leaving the grid at one workgroup per tile then launched 1024
    /// workgroups where 32 reduce, and the 992 that reduced nothing cost 10 us
    /// of block scheduling in pass 1 plus 13 us in the combine, which read
    /// 0.87x. Setting the factor back to one returns this count to 1024 and
    /// turns this red.
    #[test]
    fn the_reducing_grid_is_the_input_span_divided_by_the_span_of_one_workgroup() {
        let count = 1_048_576;
        let tile = 1024;
        assert_eq!(
            PASS1_ELEMENTS_PER_LANE, 32,
            "Fix: the measured shape loads 32 elements per lane; a different factor needs its own sweep"
        );
        assert_eq!(
            grid_stride_tree_sum_u32_blocks(count, tile),
            32,
            "Fix: {PASS1_ELEMENTS_PER_LANE} elements per lane over a {tile}-wide tile is {} elements per reducing workgroup",
            tile * PASS1_ELEMENTS_PER_LANE
        );
    }

    /// The launch a dispatch infers for the fused program is the reducing grid.
    ///
    /// WHY: no grid a caller states reaches a compiled artifact, so the block
    /// guard both passes sit under is the only thing that can narrow the launch
    /// away from the 1,048,576-element span of `values`. Removing the guard, or
    /// widening it back to one workgroup per tile, launches 32 times the
    /// workgroups the reduction needs and puts the fused program's grid fence
    /// past cooperative residency on an 80-SM part, which splits it into two
    /// host-orchestrated segments that each launch the full span.
    #[test]
    fn the_inferred_launch_covers_only_the_reducing_workgroups() {
        let count = 1_048_576;
        let tile = 1024;
        let program = grid_stride_tree_sum_u32("values", "out", count, tile);
        let resource_span = program
            .buffers()
            .iter()
            .filter(|buffer| !matches!(buffer.kind(), vyre_foundation::ir::MemoryKind::Shared))
            .map(|buffer| buffer.count())
            .max()
            .unwrap_or(1);
        assert_eq!(
            resource_span, count,
            "Fix: the widest non-shared binding of the fused program is its input, so that is the span the launch starts from"
        );
        assert_eq!(
            vyre_foundation::admitted_logical_span(&program, resource_span),
            grid_stride_tree_sum_u32_blocks(count, tile) * tile,
            "Fix: the launch must narrow to the workgroups the block guard admits"
        );
    }
}
