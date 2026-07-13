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
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]
#![forbid(unsafe_code)]

use vyre_foundation::ir::Program;
use vyre_primitives::reduce::multi_block_prefix_scan::{self, BLOCK_LANES};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Position of the buffer `name` within `reference_eval`'s returned outputs, which are
/// the writable buffers in binding order. The multi-block chain demotes its
/// intermediates (`partials`, `block_totals_scanned`) to `pipeline_live_out`: they are
/// STILL returned and precede `output` in binding order, so `outputs[0]` is the pre-carry
/// `partials`, NOT the final scan. Delegates to the interpreter's OWN output-selection
/// predicate (`vyre_reference::output_index`) so this can never drift from the real ABI.
fn output_index(program: &Program, name: &str) -> usize {
    vyre_reference::output_index(program, name)
        .expect("Fix: multi_block_prefix_scan must declare the `output` buffer")
}

/// Run the multi-block prefix-scan GPU program through the reference interpreter and
/// return the first `n` words of the FINAL `output` buffer.
fn gpu_scan(input: &[u32]) -> Vec<u32> {
    let n = input.len() as u32;
    let program = multi_block_prefix_scan::multi_block_prefix_scan_sum_u32("input", "output", n);
    let out_idx = output_index(&program, "output");
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack(input))])
        .expect("multi-block prefix scan must execute under reference_eval");
    let mut out = unpack(&outputs[out_idx].to_bytes());
    out.truncate(input.len());
    out
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
        let cpu = multi_block_prefix_scan::cpu_ref(&input);

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
    let n = input.len() as u32;
    let program =
        multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32("input", "output", n);
    let out_idx = output_index(&program, "output");
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(pack(input))])
        .expect("multi-block exclusive prefix scan must execute under reference_eval");
    let mut out = unpack(&outputs[out_idx].to_bytes());
    out.truncate(input.len());
    out
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
        let cpu = multi_block_prefix_scan::cpu_ref_exclusive(&input);

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
