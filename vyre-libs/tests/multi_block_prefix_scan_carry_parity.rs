//! GPU-IR value parity for the MULTI-BLOCK prefix scan across a block boundary.
//!
//! The existing coverage leaves the carry path unverified at the IR level:
//!   - `sweep_reduce_multi_block_prefix_scan_volume_oracle_matrix` compares only
//!     `cpu_ref` vs an independent CPU oracle, and its inputs are `idx % 200`
//!     (n ≤ 199 < BLOCK_LANES) (the multi-block chain never runs).
//!   - `proptest_multi_block_prefix_scan` value-checks `cpu_ref` across the boundary
//!     but asserts only the PROGRAM STRUCTURE (workgroup size, buffer shapes,
//!     markers) for the GPU builder at n > BLOCK_LANES (never its output bytes).
//!   - `proptest_text_line_index` reference_evals the scan indirectly, but only for
//!     `source in 0..=256` (single-block path, no GridSync, no Pass-C carry).
//!
//! So NOTHING drives the fused Pass-A → Pass-B → Pass-C GridSync program through
//! `reference_eval` and asserts its VALUES for n > BLOCK_LANES, where Pass-C must add
//! each block's exclusive prefix-of-block-totals (the carry) to every element. This
//! pins exactly that: GPU program == cpu_ref, byte-for-byte, spanning 2-4 Pass-A
//! blocks, plus an explicit carry check at the first block boundary.
#![cfg(feature = "reduce")]
#![forbid(unsafe_code)]

use vyre_foundation::ir::{Program, PORTABLE_WORKGROUP_INVOCATIONS};
use vyre_foundation::LaunchGeometry;
use vyre_libs::reduce::multi_block_prefix_scan;

const BLOCK_LANES: u32 = PORTABLE_WORKGROUP_INVOCATIONS;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::composition_witness::{
    exclusive_prefix_sum_witness, inclusive_prefix_sum_witness,
};
use vyre_reference::value::Value;

/// Position of the buffer `name` within `reference_eval`'s returned outputs, which are
/// the writable buffers in binding order. The multi-block chain demotes its
/// intermediates (`partials`, `block_totals_scanned`) to `pipeline_live_out`: they are
/// STILL returned and precede `output` in binding order, so `outputs[0]` is the pre-carry
/// `partials`, NOT the final scan. Delegates to the interpreter's OWN output-selection
/// predicate (`vyre_reference::output_index`) so this can never drift from the real ABI.
fn output_index(program: &Program, name: &str) -> usize {
    vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: the program must declare the `{name}` buffer"))
}

/// Run `program` once and return, for every requested buffer, its first `words`
/// values. One evaluation answers every buffer a caller reads, so a chain that
/// feeds one pass from another never runs the producing pass twice.
fn eval_buffers(program: &Program, inputs: &[Value], requested: &[(&str, usize)]) -> Vec<Vec<u32>> {
    let indices: Vec<usize> = requested
        .iter()
        .map(|(name, _)| output_index(program, name))
        .collect();
    let outputs = vyre_reference::reference_eval(program, inputs)
        .expect("multi-block prefix scan must execute under reference_eval");
    requested
        .iter()
        .zip(indices)
        .map(|((_, words), index)| {
            let mut values = unpack(&outputs[index].to_bytes());
            values.truncate(*words);
            values
        })
        .collect()
}

/// Run `program` and return the first `words` values of its `name` buffer.
fn eval_buffer(program: &Program, inputs: &[Value], name: &str, words: usize) -> Vec<u32> {
    let mut buffers = eval_buffers(program, inputs, &[(name, words)]);
    buffers.pop().expect("one requested buffer answers one read")
}

/// Run the multi-block prefix-scan GPU program through the reference interpreter and
/// return the first `n` words of the FINAL `output` buffer.
fn gpu_scan(input: &[u32]) -> Vec<u32> {
    let program = multi_block_prefix_scan::multi_block_prefix_scan_sum_u32(
        "input",
        "output",
        input.len() as u32,
    );
    eval_buffer(&program, &[Value::from(pack(input))], "output", input.len())
}

#[test]
fn multi_block_intermediates_are_globally_ordered() {
    // Pin the fused chain's intermediate buffers for a 4-block scan: Pass-A must
    // write EVERY block's total, and Pass-B must produce the INCLUSIVE scan of those
    // totals. Both are `pipeline_live_out`, so reference_eval returns them. This is
    // the exact place the GridSync-simulation gap surfaced: before the interpreter
    // honored `MemoryOrdering::GridSync`, Pass-B (workgroup 0) read block_totals
    // before Pass-A workgroups 1..3 wrote them, so `scanned` came back as the scan of
    // [t0,0,0,0] = [t0,t0,t0,t0]. Asserting the true inclusive scan pins the fix.
    let n = BLOCK_LANES * 3 + 7;
    let input: Vec<u32> = (0..n).map(|i| (i % 7) + 1).collect();
    let program = multi_block_prefix_scan::multi_block_prefix_scan_sum_u32("input", "output", n);
    let bt_idx = output_index(&program, "__output_mbps_block_totals");
    let bts_idx = output_index(&program, "__output_mbps_block_totals_scanned");
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack(&input))])
        .expect("multi-block prefix scan must execute under reference_eval");
    let block_totals = unpack(&outputs[bt_idx].to_bytes());
    let scanned = unpack(&outputs[bts_idx].to_bytes());

    let expected_totals: Vec<u32> = (0..4)
        .map(|b| {
            let start = b * BLOCK_LANES as usize;
            let end = ((b + 1) * BLOCK_LANES as usize).min(n as usize);
            input[start..end].iter().copied().sum()
        })
        .collect();
    let mut expected_scanned = Vec::new();
    let mut acc = 0u32;
    for &t in &expected_totals {
        acc += t;
        expected_scanned.push(acc);
    }

    assert_eq!(
        &block_totals[..4],
        expected_totals.as_slice(),
        "Pass-A must write every block's total (all 4 workgroups), not just block 0"
    );
    assert_eq!(
        &scanned[..4],
        expected_scanned.as_slice(),
        "Pass-B must inclusive-scan the block totals AFTER Pass-A fully completes \
         (GridSync ordering); a result of [t0,t0,t0,t0] means the grid barrier was \
         not honored and Pass-B raced Pass-A"
    );
}

#[test]
fn multi_block_gpu_program_matches_cpu_ref_across_block_boundary() {
    // Small per-element values (1..=7) keep the running sum well under u32::MAX so
    // wrapping is never in play, a divergence here is a real carry defect, not an
    // overflow-semantics artifact.
    for &n in &[
        BLOCK_LANES + 1,
        BLOCK_LANES + 500,
        BLOCK_LANES * 2,
        BLOCK_LANES * 3 + 7,
    ] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) + 1).collect();
        let gpu = gpu_scan(&input);
        let cpu = inclusive_prefix_sum_witness(&input);

        assert_eq!(gpu.len(), input.len(), "output length mismatch at n={n}");
        if let Some(i) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
            let block = i / BLOCK_LANES as usize;
            let lane = i % BLOCK_LANES as usize;
            panic!(
                "GPU multi-block prefix scan diverges from cpu_ref at n={n}: first mismatch at \
                 index {i} (block {block}, lane {lane} of {num_blocks} blocks): gpu={} cpu={} \
                 delta={}; prev-ok gpu[{}]={} cpu={}",
                gpu[i],
                cpu[i],
                cpu[i] as i64 - gpu[i] as i64,
                i.saturating_sub(1),
                gpu[i.saturating_sub(1)],
                cpu[i.saturating_sub(1)],
                num_blocks = n.div_ceil(BLOCK_LANES),
            );
        }

        // Explicit carry lock: the inclusive prefix at the first Pass-A block boundary
        // must equal the full running total through that index, i.e. Pass-C added
        // block 0's complete total to block 1's leading element. A carry that dropped
        // or double-counted block 0 would break exactly here.
        let boundary = BLOCK_LANES as usize;
        let expected_boundary: u32 = input[..=boundary].iter().copied().sum();
        assert_eq!(
            gpu[boundary], expected_boundary,
            "carry wrong at block boundary index {boundary} for n={n}"
        );
    }
}

/// Run the multi-block EXCLUSIVE prefix-scan program and return the first `n` words
/// of the final `output` buffer.
fn gpu_scan_exclusive(input: &[u32]) -> Vec<u32> {
    let program = multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32(
        "input",
        "output",
        input.len() as u32,
    );
    eval_buffer(&program, &[Value::from(pack(input))], "output", input.len())
}

#[test]
fn multi_block_exclusive_scan_matches_cpu_ref_across_block_boundary() {
    // The exclusive scan fuses the inclusive multi-block chain (Pass-A/B/C with a
    // GridSync between A and B) with a per-element subtract pass that reads the
    // inclusive result written across ALL workgroups (a SECOND GridSync before the
    // subtract). Both grid barriers must be honored, so this is independent coverage
    // of the interpreter's GridSync handling on a two-grid-sync program.
    for &n in &[BLOCK_LANES + 1, BLOCK_LANES * 2, BLOCK_LANES * 3 + 7] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) + 1).collect();
        let gpu = gpu_scan_exclusive(&input);
        let cpu = exclusive_prefix_sum_witness(&input);

        assert_eq!(gpu.len(), input.len(), "exclusive output length at n={n}");
        if let Some(i) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
            panic!(
                "multi-block EXCLUSIVE scan diverges from cpu_ref at n={n}: first mismatch at \
                 index {i}: gpu={} cpu={}",
                gpu[i], cpu[i]
            );
        }
        assert_eq!(
            gpu[0], 0,
            "exclusive prefix scan output[0] must be 0 at n={n}"
        );
    }
}

/// One Pass-A or Pass-C builder at one lane count.
type PassBuilder<'a> = &'a dyn Fn(u32, u32) -> Program;

/// WHY: the family publishes six spellings of one chain, the canonical pair plus
/// an explicit-lane pair and a geometry pair, and only the canonical pair and the
/// inclusive explicit-lane form were ever executed. A spelling nobody runs is a
/// public builder whose lane plumbing can be wrong with nothing red, and the
/// geometry forms are the ones a backend calls after it has chosen a width.
///
/// Closes: every published spelling of the multi-block scan, inclusive and
/// exclusive, at a lane count other than the portable default, enumerated
/// together so a divergence names the spelling that produced it.
///
/// Does not catch: a wrong `grid` or `shared_bytes` field, which no builder in
/// this family reads.
#[test]
fn every_published_scan_spelling_matches_its_witness() {
    const LANES: u32 = 256;
    let geometry = LaunchGeometry {
        workgroup: [LANES, 1, 1],
        ..LaunchGeometry::default()
    };
    for &n in &[LANES + 1, LANES * 2, LANES * 4 + 7] {
        let input: Vec<u32> = (0..n).map(|i| (i % 7) + 1).collect();
        let inclusive = inclusive_prefix_sum_witness(&input);
        let exclusive = exclusive_prefix_sum_witness(&input);
        let cases: [(&str, Program, &[u32]); 4] = [
            (
                "sum_u32_with_block_lanes",
                multi_block_prefix_scan::multi_block_prefix_scan_sum_u32_with_block_lanes(
                    "input", "output", n, LANES,
                ),
                inclusive.as_slice(),
            ),
            (
                "sum_u32_with_geometry",
                multi_block_prefix_scan::multi_block_prefix_scan_sum_u32_with_geometry(
                    "input", "output", n, &geometry,
                ),
                inclusive.as_slice(),
            ),
            (
                "sum_exclusive_u32_with_block_lanes",
                multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
                    "input", "output", n, LANES,
                ),
                exclusive.as_slice(),
            ),
            (
                "sum_exclusive_u32_with_geometry",
                multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32_with_geometry(
                    "input", "output", n, &geometry,
                ),
                exclusive.as_slice(),
            ),
        ];
        for (spelling, program, expected) in cases {
            let actual = eval_buffer(&program, &[Value::from(pack(&input))], "output", input.len());
            assert_eq!(
                actual, expected,
                "{spelling} at n={n} over {LANES} lanes must match its witness"
            );
        }
    }
}

/// WHY: Pass A and Pass C ship as public builders so a resident scheduler can
/// drive the three-pass chain itself, and nothing in this crate executed either
/// one. The fused program hides both: it wires the passes together with buffer
/// names it chooses, so a Pass-A total written by the wrong lane or a Pass-C
/// offset read one block off can only be seen by driving the passes separately
/// and joining them by hand.
///
/// Closes: `pass_a_local_scan` and `pass_c_broadcast_offsets` in both the
/// portable and the explicit-lane spelling, composed through the host scan of the
/// block totals that Pass B computes on the device.
///
/// Does not catch: the GridSync ordering between the passes, which only the fused
/// program declares and which
/// `multi_block_intermediates_are_globally_ordered` pins.
#[test]
fn the_published_passes_compose_into_the_fused_scan() {
    let spellings: [(&str, u32, PassBuilder<'_>, PassBuilder<'_>); 2] = [
        (
            "portable",
            BLOCK_LANES,
            &|n, blocks| {
                multi_block_prefix_scan::pass_a_local_scan(
                    "input",
                    "partials",
                    "block_totals",
                    n,
                    blocks,
                )
            },
            &|n, blocks| {
                multi_block_prefix_scan::pass_c_broadcast_offsets(
                    "partials", "scanned", "output", n, blocks,
                )
            },
        ),
        (
            "256 lanes",
            256,
            &|n, blocks| {
                multi_block_prefix_scan::pass_a_local_scan_with_block_lanes(
                    "input",
                    "partials",
                    "block_totals",
                    n,
                    blocks,
                    256,
                )
            },
            &|n, blocks| {
                multi_block_prefix_scan::pass_c_broadcast_offsets_with_block_lanes(
                    "partials", "scanned", "output", n, blocks, 256,
                )
            },
        ),
    ];

    for (spelling, lanes, pass_a, pass_c) in spellings {
        let n = lanes * 3 + 5;
        let blocks = n.div_ceil(lanes);
        let input: Vec<u32> = (0..n).map(|i| (i % 5) + 1).collect();

        // Pass A stages a whole block of lanes and zeroes what lies past `n`, so
        // the witness scans the padded block and the elements past `n` stay
        // unwritten in `partials`.
        let mut expected_partials: Vec<u32> = Vec::with_capacity((blocks * lanes) as usize);
        let mut expected_totals: Vec<u32> = Vec::with_capacity(blocks as usize);
        for block in 0..blocks {
            let start = (block * lanes) as usize;
            let padded: Vec<u32> = (start..start + lanes as usize)
                .map(|index| input.get(index).copied().unwrap_or(0))
                .collect();
            let scanned = inclusive_prefix_sum_witness(&padded);
            expected_totals.push(*scanned.last().expect("a block spans its lanes"));
            for (lane, value) in scanned.iter().enumerate() {
                let global = start + lane;
                expected_partials.push(if global < input.len() { *value } else { 0 });
            }
        }

        let program_a = pass_a(n, blocks);
        let staged = [Value::from(pack(&input))];
        let words = (blocks * lanes) as usize;
        let buffers = eval_buffers(
            &program_a,
            &staged,
            &[("partials", words), ("block_totals", blocks as usize)],
        );
        let (partials, totals) = (&buffers[0], &buffers[1]);
        assert_eq!(
            partials, &expected_partials,
            "{spelling}: Pass A must write every in-range per-element partial at n={n}"
        );
        assert_eq!(
            totals, &expected_totals,
            "{spelling}: Pass A must write every per-block total at n={n}"
        );

        let scanned_totals = inclusive_prefix_sum_witness(totals);
        let program_c = pass_c(n, blocks);
        let output = eval_buffer(
            &program_c,
            &[
                Value::from(pack(partials)),
                Value::from(pack(&scanned_totals)),
            ],
            "output",
            input.len(),
        );
        assert_eq!(
            output,
            inclusive_prefix_sum_witness(&input),
            "{spelling}: Pass C must add each block's carry at n={n}"
        );
    }
}
