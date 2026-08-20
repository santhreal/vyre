//! Multi-block parallel prefix sum  -  bridges the gap between
//! the single-workgroup scan shape (one block of lanes) and arbitrary-length
//! scans that used to fall back to a single-thread sequential loop.
//!
//! # Why
//!
//! Small scans are handled by the same guarded workgroup
//! primitive used as the recursive bottom-out. Large scans compose that
//! primitive into a three-pass multi-block chain. Real workloads (lex
//! compaction over a 3 MB C TU, histogram CDFs over millions of bins,
//! etc.) need both: arbitrary `n` AND O(log N) wall-clock.
//!
//! This module composes local guarded scans plus a Pass-C offset
//! broadcast into a 3-pass Blelloch-style chain:
//!
//! ```text
//!   Pass A: per-block local Hillis-Steele scan.
//!           writes per-element partials and per-block totals.
//!   GridSync barrier (substrate splits the dispatch here).
//!   Pass B: scan of per-block totals.
//!           recursive  -  this fn calls itself with the totals as input.
//!           Bottoms out at the guarded single-workgroup scan.
//!   GridSync barrier.
//!   Pass C: per-element offset add.
//!           thread t: out[t] = partials[t] + scanned_block_totals[block_id(t) - 1].
//! ```
//!
//! # Returns
//!
//! A single fused `Program`. The substrate (vyre-driver/src/grid_sync)
//! splits the dispatch into three kernel launches at the GridSync
//! barriers when the backend doesn't support cooperative groups.

use vyre_foundation::composition::{tag_program, trap_program, wrap_anonymous_region};

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, PORTABLE_WORKGROUP_INVOCATIONS,
};

/// Canonical op id for inclusive sum-scan over arbitrary `n`.
pub const OP_ID_INCLUSIVE_SUM: &str = "vyre-libs::reduce::multi_block_prefix_scan_inclusive_sum";

/// Canonical op id for the exclusive-sum element-difference pass that turns the
/// inclusive multi-block scan into an exclusive one.
pub const OP_ID_EXCLUSIVE_SUM: &str = "vyre-libs::reduce::multi_block_prefix_scan_exclusive_sum";

/// Return the execution geometry requirements for multi-block prefix scan.
#[must_use]
pub const fn multi_block_prefix_scan_requirements() -> vyre_foundation::GeometryRequirements {
    vyre_foundation::GeometryRequirements::cooperative(vyre_foundation::CooperativeWidth::Agnostic)
}
/// Historical direct-scan threshold retained for callers/tests that size
/// around one level of block-total recursion. The implementation recurses and
/// bottoms out at the guarded single-workgroup scan once the block count fits
/// the portable workgroup width.
pub const SOFT_MAX_N: u32 = PORTABLE_WORKGROUP_INVOCATIONS * PORTABLE_WORKGROUP_INVOCATIONS;
fn output_byte_range(words: u32, context: &str) -> Result<usize, String> {
    usize::try_from(words)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            format!(
                "{context} words={words} overflows output byte range. Fix: shard the scan before GPU dispatch."
            )
        })
}

fn total_partial_words(num_blocks: u32, block_lanes: u32, context: &str) -> Result<u32, String> {
    num_blocks.checked_mul(block_lanes).ok_or_else(|| {
        format!(
            "vyre multi_block_prefix_scan {context} num_blocks={num_blocks} overflows partial buffer count. Fix: shard the scan before GPU dispatch."
        )
    })
}

/// Build an inclusive parallel prefix-sum Program over arbitrary `n`.
///
/// Backed by the guarded single-workgroup scan for `n ≤ PORTABLE_WORKGROUP_INVOCATIONS`;
/// otherwise a 3-pass Blelloch chain (Pass A local scan + per-block
/// totals → Pass B scan of totals → Pass C broadcast offsets).
///
/// `n == 0` returns an empty Program.
#[must_use]
pub fn multi_block_prefix_scan_sum_u32(input: &str, output: &str, n: u32) -> Program {
    multi_block_prefix_scan_sum_u32_with_block_lanes(
        input,
        output,
        n,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

/// Build an inclusive parallel prefix-sum Program with explicit lowered block lanes.
#[must_use]
pub fn multi_block_prefix_scan_sum_u32_with_block_lanes(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Program {
    match try_multi_block_prefix_scan_sum_u32_with_block_lanes(input, output, n, block_lanes) {
        Ok(program) if program.entry().is_empty() => program,
        Ok(program) => tag_program(OP_ID_INCLUSIVE_SUM, program),
        Err(error) => trap_program(OP_ID_INCLUSIVE_SUM, Some((output, DataType::U32)), error),
    }
}

/// Build an inclusive parallel prefix-sum Program with lowered launch geometry.
#[must_use]
pub fn multi_block_prefix_scan_sum_u32_with_geometry(
    input: &str,
    output: &str,
    n: u32,
    geometry: &vyre_foundation::LaunchGeometry,
) -> Program {
    multi_block_prefix_scan_sum_u32_with_block_lanes(input, output, n, geometry.workgroup[0])
}

// Registration so the op id is known to `harness::all_entries()`.
// region_chain_invariant resolves the three sub-region generators below
// (`<OP_ID_INCLUSIVE_SUM>::{guarded_single_block,pass_a,pass_c}`) against this
// registered id. `n = 64` fits the portable default and keeps the build on the guarded
// single-block path (no GridSync), so the entry constructs cleanly without a
// host-split.
//
// The fixtures used to be `None`, on the reasoning that nothing walked
// vyre-primitives fixtures. The cross-backend parity matrix does, and a
// registered op with no inputs is zero execution coverage: the op was counted as
// registered while no backend ever ran it. The fixture below is an ordinary
// inclusive scan whose expected values are closed-form, so it checks real
// arithmetic rather than merely running.
const EXPECTED_PREFIX_SCAN_OUTPUT_BYTES: [u8; 256] = [
    0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x0f, 0x00, 0x00, 0x00, 0x15, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
    0x2d, 0x00, 0x00, 0x00, 0x37, 0x00, 0x00, 0x00, 0x42, 0x00, 0x00, 0x00, 0x4e, 0x00, 0x00, 0x00,
    0x5b, 0x00, 0x00, 0x00, 0x69, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x00, 0x88, 0x00, 0x00, 0x00,
    0x99, 0x00, 0x00, 0x00, 0xab, 0x00, 0x00, 0x00, 0xbe, 0x00, 0x00, 0x00, 0xd2, 0x00, 0x00, 0x00,
    0xe7, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x00, 0x00, 0x14, 0x01, 0x00, 0x00, 0x2c, 0x01, 0x00, 0x00,
    0x45, 0x01, 0x00, 0x00, 0x5f, 0x01, 0x00, 0x00, 0x7a, 0x01, 0x00, 0x00, 0x96, 0x01, 0x00, 0x00,
    0xb3, 0x01, 0x00, 0x00, 0xd1, 0x01, 0x00, 0x00, 0xf0, 0x01, 0x00, 0x00, 0x10, 0x02, 0x00, 0x00,
    0x31, 0x02, 0x00, 0x00, 0x53, 0x02, 0x00, 0x00, 0x76, 0x02, 0x00, 0x00, 0x9a, 0x02, 0x00, 0x00,
    0xbf, 0x02, 0x00, 0x00, 0xe5, 0x02, 0x00, 0x00, 0x0c, 0x03, 0x00, 0x00, 0x34, 0x03, 0x00, 0x00,
    0x5d, 0x03, 0x00, 0x00, 0x87, 0x03, 0x00, 0x00, 0xb2, 0x03, 0x00, 0x00, 0xde, 0x03, 0x00, 0x00,
    0x0b, 0x04, 0x00, 0x00, 0x39, 0x04, 0x00, 0x00, 0x68, 0x04, 0x00, 0x00, 0x98, 0x04, 0x00, 0x00,
    0xc9, 0x04, 0x00, 0x00, 0xfb, 0x04, 0x00, 0x00, 0x2e, 0x05, 0x00, 0x00, 0x62, 0x05, 0x00, 0x00,
    0x97, 0x05, 0x00, 0x00, 0xcd, 0x05, 0x00, 0x00, 0x04, 0x06, 0x00, 0x00, 0x3c, 0x06, 0x00, 0x00,
    0x75, 0x06, 0x00, 0x00, 0xaf, 0x06, 0x00, 0x00, 0xea, 0x06, 0x00, 0x00, 0x26, 0x07, 0x00, 0x00,
    0x63, 0x07, 0x00, 0x00, 0xa1, 0x07, 0x00, 0x00, 0xe0, 0x07, 0x00, 0x00, 0x20, 0x08, 0x00, 0x00,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID_INCLUSIVE_SUM,
        || multi_block_prefix_scan_sum_u32("input", "output", 64),
        Some(|| {
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            let input: Vec<u32> = (1..=SCAN_FIXTURE_LEN).collect();
            vec![vec![to_bytes(&input)]]
        }),
        Some(|| vec![vec![EXPECTED_PREFIX_SCAN_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("reduce")
    .with_geometry_requirements(multi_block_prefix_scan_requirements())
}

/// Element count of the registered inclusive-scan fixture.
///
/// At or below the portable workgroup width so the build stays on the guarded single-block
/// path, which is the shape the sub-region generators resolve against.
const SCAN_FIXTURE_LEN: u32 = 64;

fn try_multi_block_prefix_scan_sum_u32(
    input: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    try_multi_block_prefix_scan_sum_u32_with_block_lanes(
        input,
        output,
        n,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

fn try_multi_block_prefix_scan_sum_u32_with_block_lanes(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Ok(Program::empty());
    }
    let lanes = if block_lanes.is_power_of_two() && block_lanes >= 2 {
        block_lanes
    } else {
        PORTABLE_WORKGROUP_INVOCATIONS
    };
    if n <= lanes {
        return try_guarded_single_block_scan(input, output, n, lanes);
    }

    try_multi_block_prefix_scan_chain(input, output, n, lanes)
}

/// Build an **exclusive** parallel prefix-sum Program over arbitrary `n`:
/// `output[i] = sum(input[0..i])`, `output[0] = 0`.
///
/// This is the offset buffer `math::stream_compact` requires. The single-block
/// `math::prefix_scan(ScanKind::ExclusiveSum)` serves up to
/// `math::prefix_scan::MAX_SINGLE_BLOCK_SCAN` elements, but a compaction batch
/// with more live candidates than that had no on-device exclusive scan and had
/// to convert an inclusive scan to exclusive on host.
///
/// Built as `exclusive[i] = inclusive[i] - input[i]`: the tested inclusive
/// multi-block chain writes an intermediate, then a fused element-difference
/// pass subtracts the input. Reusing the inclusive chain keeps ONE scan
/// implementation; the subtract never underflows because an inclusive prefix
/// sum always includes `input[i]`.
///
/// `n == 0` returns an empty Program.
#[must_use]
pub fn multi_block_prefix_scan_sum_exclusive_u32(input: &str, output: &str, n: u32) -> Program {
    multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
        input,
        output,
        n,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

/// Build an exclusive parallel prefix-sum Program with explicit lowered block lanes.
#[must_use]
pub fn multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Program {
    match try_multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
        input,
        output,
        n,
        block_lanes,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID_EXCLUSIVE_SUM, Some((output, DataType::U32)), error),
    }
}

/// Build an exclusive parallel prefix-sum Program with lowered launch geometry.
#[must_use]
pub fn multi_block_prefix_scan_sum_exclusive_u32_with_geometry(
    input: &str,
    output: &str,
    n: u32,
    geometry: &vyre_foundation::LaunchGeometry,
) -> Program {
    multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
        input,
        output,
        n,
        geometry.workgroup[0],
    )
}

fn try_multi_block_prefix_scan_sum_exclusive_u32(
    input: &str,
    output: &str,
    n: u32,
) -> Result<Program, String> {
    try_multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
        input,
        output,
        n,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

fn try_multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Ok(Program::empty());
    }
    let lanes = if block_lanes.is_power_of_two() && block_lanes >= 2 {
        block_lanes
    } else {
        PORTABLE_WORKGROUP_INVOCATIONS
    };
    let inclusive = format!("__{output}_mbps_inclusive");
    let scan = try_multi_block_prefix_scan_sum_u32_with_block_lanes(input, &inclusive, n, lanes)?;
    let subtract = try_exclusive_difference_pass(&inclusive, input, output, n, lanes)?;

    vyre_foundation::execution_plan::fusion::fuse_programs(&[scan, subtract])
        .map(|program| crate::plumbing::program::outputs::demote_intermediate_outputs(program, output))
        .map_err(|error| {
            format!(
                "vyre multi_block_prefix_scan exclusive fusion failed for n={n}: {error}. Fix: repair program fusion for the inclusive-scan + element-difference passes; do not substitute an empty Program."
            )
        })
}

/// Element-difference pass: `output[i] = inclusive[i] - input[i]` for `i < n`.
/// A flat one-lane-per-element Region (no GridSync), so it composes after the
/// inclusive scan and executes on the reference interpreter for `n ≤ block_lanes`.
fn try_exclusive_difference_pass(
    inclusive: &str,
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    output_byte_range(n, "exclusive difference pass")?;
    let t = Expr::InvocationId { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(n)),
        vec![Node::store(
            output,
            t.clone(),
            Expr::sub(
                Expr::load(inclusive, t.clone()),
                Expr::load(input, t.clone()),
            ),
        )],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(inclusive, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(input, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::output(output, 2, DataType::U32).with_count(n),
        ],
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(OP_ID_EXCLUSIVE_SUM, body)],
    ))
}

fn try_multi_block_prefix_scan_chain(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    if n <= block_lanes {
        return try_guarded_single_block_scan(input, output, n, block_lanes);
    }

    let num_blocks = n.div_ceil(block_lanes);

    let partials = format!("__{output}_mbps_partials");
    let block_totals = format!("__{output}_mbps_block_totals");
    let block_totals_scanned = format!("__{output}_mbps_block_totals_scanned");

    let pass_a =
        try_pass_a_local_scan(input, &partials, &block_totals, n, num_blocks, block_lanes)?;
    let pass_b = try_multi_block_prefix_scan_chain(
        &block_totals,
        &block_totals_scanned,
        num_blocks,
        block_lanes,
    )?;
    let pass_c = try_pass_c_broadcast_offsets(
        &partials,
        &block_totals_scanned,
        output,
        n,
        num_blocks,
        block_lanes,
    )?;

    vyre_foundation::execution_plan::fusion::fuse_programs(&[pass_a, pass_b, pass_c])
        .map(|program| crate::plumbing::program::outputs::demote_intermediate_outputs(program, output))
        .map_err(|error| {
            format!(
                "vyre multi_block_prefix_scan fusion failed for n={n}, num_blocks={num_blocks}: {error}. Fix: repair grid-sync fusion for the three-pass GPU scan; do not substitute an empty Program."
            )
        })
}

fn try_guarded_single_block_scan(
    input: &str,
    output: &str,
    n: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    if n == 0 {
        return Ok(Program::empty());
    }

    let lane = Expr::var("lane");
    let block = Expr::var("block");
    let scratch_a = format!("__{output}_guarded_scan_a");
    let scratch_b = format!("__{output}_guarded_scan_b");

    let mut scan_body = Vec::new();
    scan_body.push(Node::let_bind("lane", Expr::LocalId { axis: 0 }));
    scan_body.push(Node::store(&scratch_a, lane.clone(), Expr::u32(0)));
    scan_body.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(n)),
        vec![Node::store(
            &scratch_a,
            lane.clone(),
            Expr::load(input, lane.clone()),
        )],
    ));
    scan_body.push(Node::Barrier {
        ordering: MemoryOrdering::SeqCst,
    });

    scan_body.extend(crate::reduce::workgroup_scan::blelloch_inclusive_sum_nodes(
        &scratch_a,
        &scratch_b,
        &lane,
        block_lanes,
    ));

    scan_body.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(n)),
        vec![Node::store(
            output,
            lane.clone(),
            Expr::load(&scratch_a, lane.clone()),
        )],
    ));

    let output_bytes = output_byte_range(
        n,
        "vyre multi_block_prefix_scan guarded single-block output",
    )?;
    let body = vec![
        Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
        Node::if_then(Expr::eq(block, Expr::u32(0)), scan_body),
    ];
    let buffers = vec![
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
        BufferDecl::output(output, 1, DataType::U32)
            .with_count(n)
            .with_output_byte_range(0..output_bytes),
        BufferDecl::workgroup(&scratch_a, block_lanes, DataType::U32),
        BufferDecl::workgroup(&scratch_b, block_lanes, DataType::U32),
    ];

    Ok(Program::wrapped(
        buffers,
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(
            "anonymous::vyre-primitives::reduce::multi_block_prefix_scan_inclusive_sum::guarded_single_block",
            body,
        )],
    ))
}
/// Pass A  -  per-block local inclusive Hillis-Steele scan.
#[must_use]
pub fn pass_a_local_scan(
    input: &str,
    partials: &str,
    block_totals: &str,
    n: u32,
    num_blocks: u32,
) -> Program {
    pass_a_local_scan_with_block_lanes(
        input,
        partials,
        block_totals,
        n,
        num_blocks,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

/// Build Pass A with explicit block lanes.
#[must_use]
pub fn pass_a_local_scan_with_block_lanes(
    input: &str,
    partials: &str,
    block_totals: &str,
    n: u32,
    num_blocks: u32,
    block_lanes: u32,
) -> Program {
    match try_pass_a_local_scan(input, partials, block_totals, n, num_blocks, block_lanes) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID_INCLUSIVE_SUM, Some((partials, DataType::U32)), error),
    }
}

fn try_pass_a_local_scan(
    input: &str,
    partials: &str,
    block_totals: &str,
    n: u32,
    num_blocks: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    let lane = "lane";
    let scratch_a = format!("__{partials}_pass_a_scratch_a");
    let scratch_b = format!("__{partials}_pass_a_scratch_b");

    let pass = crate::reduce::workgroup_scan::BlockScanPass {
        lane,
        block: "block",
        global: "global",
        scratch_a: &scratch_a,
        scratch_b: &scratch_b,
        partials,
        block_totals,
        block_lanes,
        in_range: n,
    };
    let body = pass.nodes(vec![Node::store(
        &scratch_a,
        Expr::var(lane),
        Expr::load(input, Expr::var("global")),
    )]);

    let total_partials = total_partial_words(num_blocks, block_lanes, "Pass A")?;
    let total_partial_bytes = output_byte_range(
        total_partials,
        "vyre multi_block_prefix_scan Pass A partials",
    )?;
    let block_total_bytes = output_byte_range(
        num_blocks,
        "vyre multi_block_prefix_scan Pass A block_totals",
    )?;
    let buffers = vec![
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(n),
        BufferDecl::output(partials, 1, DataType::U32)
            .with_count(total_partials)
            .with_output_byte_range(0..total_partial_bytes),
        BufferDecl::storage(block_totals, 2, BufferAccess::ReadWrite, DataType::U32)
            .with_count(num_blocks)
            .with_pipeline_live_out(true)
            .with_output_byte_range(0..block_total_bytes),
        BufferDecl::workgroup(&scratch_a, block_lanes, DataType::U32),
        BufferDecl::workgroup(&scratch_b, block_lanes, DataType::U32),
    ];

    Ok(Program::wrapped(
        buffers,
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(
            "anonymous::vyre-primitives::reduce::multi_block_prefix_scan_inclusive_sum::pass_a",
            body,
        )],
    ))
}

/// Pass C  -  broadcast scanned per-block totals back to per-element output.
///
/// `out[B*block_lanes + L] = partials[B*block_lanes + L] + offset`,
/// where `offset = scanned_block_totals[B - 1]` (or `0` for block 0).
///
/// Uses an `if_then` (not `Expr::select`) for the `offset` lookup so the
/// `block - 1` load is never evaluated when `block == 0`. `Expr::select`
/// evaluates both arms unconditionally; with no OOB-clamp on the load
/// path that would underflow to `0xFFFFFFFF` and fault on a real GPU.
/// Build Pass C for a resident or manually-scheduled multi-block inclusive scan.
///
/// Callers supply `partials` from [`pass_a_local_scan`] and a scanned
/// `block_totals` buffer, then this pass writes the final inclusive scan.
#[must_use]
pub fn pass_c_broadcast_offsets(
    partials: &str,
    block_totals_scanned: &str,
    output: &str,
    n: u32,
    num_blocks: u32,
) -> Program {
    pass_c_broadcast_offsets_with_block_lanes(
        partials,
        block_totals_scanned,
        output,
        n,
        num_blocks,
        PORTABLE_WORKGROUP_INVOCATIONS,
    )
}

/// Build Pass C with explicit block lanes.
#[must_use]
pub fn pass_c_broadcast_offsets_with_block_lanes(
    partials: &str,
    block_totals_scanned: &str,
    output: &str,
    n: u32,
    num_blocks: u32,
    block_lanes: u32,
) -> Program {
    match try_pass_c_broadcast_offsets(
        partials,
        block_totals_scanned,
        output,
        n,
        num_blocks,
        block_lanes,
    ) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID_INCLUSIVE_SUM, Some((output, DataType::U32)), error),
    }
}

fn try_pass_c_broadcast_offsets(
    partials: &str,
    block_totals_scanned: &str,
    output: &str,
    n: u32,
    num_blocks: u32,
    block_lanes: u32,
) -> Result<Program, String> {
    let lane = Expr::var("lane");
    let block = Expr::var("block");
    let global = Expr::var("global");
    let offset = Expr::var("offset");

    let body = vec![
        Node::let_bind("lane", Expr::LocalId { axis: 0 }),
        Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
        Node::let_bind(
            "global",
            Expr::add(
                Expr::mul(block.clone(), Expr::u32(block_lanes)),
                lane.clone(),
            ),
        ),
        Node::let_bind("offset", Expr::u32(0)),
        Node::if_then(
            Expr::lt(Expr::u32(0), block.clone()),
            vec![Node::assign(
                "offset",
                Expr::load(
                    block_totals_scanned,
                    // block - 1 via wrapping; only evaluated when block ≥ 1.
                    Expr::add(block.clone(), Expr::u32(0u32.wrapping_sub(1))),
                ),
            )],
        ),
        Node::if_then(
            Expr::lt(global.clone(), Expr::u32(n)),
            vec![Node::store(
                output,
                global.clone(),
                Expr::add(Expr::load(partials, global.clone()), offset),
            )],
        ),
    ];

    let total_partials = total_partial_words(num_blocks, block_lanes, "Pass C")?;
    let output_bytes = output_byte_range(n, "vyre multi_block_prefix_scan Pass C output")?;
    let buffers = vec![
        BufferDecl::storage(partials, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(total_partials),
        BufferDecl::storage(
            block_totals_scanned,
            1,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(num_blocks),
        BufferDecl::output(output, 2, DataType::U32)
            .with_count(n)
            .with_output_byte_range(0..output_bytes),
    ];

    Ok(Program::wrapped(
        buffers,
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(
            "anonymous::vyre-primitives::reduce::multi_block_prefix_scan_inclusive_sum::pass_c",
            body,
        )],
    ))
}

#[cfg(test)]
#[path = "multi_block_prefix_scan_tests.rs"]
mod tests;
