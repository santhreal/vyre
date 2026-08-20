//! Tests for the multi-block prefix scan against a host oracle.

use super::*;
use vyre_foundation::visit::any_descendant;

fn reference_inclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    reference_inclusive_scan_into(input, &mut out);
    out
}

fn reference_inclusive_scan_into(input: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut acc = 0u32;
    for &x in input {
        acc = acc.wrapping_add(x);
        out.push(acc);
    }
}

fn reference_exclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut acc = 0u32;
    for &x in input {
        out.push(acc);
        acc = acc.wrapping_add(x);
    }
    out
}

fn try_reference_inclusive_scan_into(input: &[u32], out: &mut Vec<u32>) -> Result<(), String> {
    reference_inclusive_scan_into(input, out);
    Ok(())
}

#[test]
fn reference_matches_simple_inclusive_sum() {
    assert_eq!(reference_inclusive_scan(&[1, 2, 3, 4]), vec![1, 3, 6, 10]);
    assert_eq!(reference_inclusive_scan(&[]), Vec::<u32>::new());
    assert_eq!(reference_inclusive_scan(&[7]), vec![7]);
}

#[test]
fn reference_exclusive_matches_definition() {
    assert_eq!(reference_exclusive_scan(&[1, 2, 3, 4]), vec![0, 1, 3, 6]);
    assert_eq!(reference_exclusive_scan(&[]), Vec::<u32>::new());
    assert_eq!(reference_exclusive_scan(&[7]), vec![0]);
}

/// The identity the exclusive builder is constructed from:
/// `exclusive[i] == inclusive[i] - input[i]` for every element.
#[test]
fn exclusive_equals_inclusive_minus_input() {
    let mut state = 0x1234_5678_u32;
    for _ in 0..500 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let n = (state % 40) as usize;
        let input: Vec<u32> = (0..n)
            .map(|i| state.rotate_left(i as u32 % 31) % 1000)
            .collect();
        let inclusive = reference_inclusive_scan(&input);
        let exclusive = reference_exclusive_scan(&input);
        for i in 0..n {
            assert_eq!(
                exclusive[i],
                inclusive[i] - input[i],
                "exclusive[{i}] must equal inclusive[{i}] - input[{i}] for input {input:?}"
            );
        }
    }
}

/// Execute the NOVEL element-difference pass (a flat, GridSync-free Region) on
/// the reference interpreter: `output[i] = inclusive[i] - input[i]`. This is
/// the only part of the exclusive scan that is new IR; the inclusive chain it
/// composes with is separately tested (and its GPU parity lives in the driver
/// harness, same as the inclusive multi-block scan).
#[test]
fn exclusive_difference_pass_executes_and_subtracts_input() {
    use std::sync::Arc;
    use vyre_reference::reference_eval;
    use vyre_reference::value::Value;

    let input = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let inclusive = reference_inclusive_scan(&input); // [3,4,8,9,14,23,25,31]
    let n = input.len() as u32;
    let program = try_exclusive_difference_pass("inclusive", "input", "output", n, 1024)
        .expect("difference pass builds");
    let to_value =
        |data: &[u32]| Value::Bytes(Arc::from(vyre_primitives::wire::pack_u32_slice(data)));
    let inputs = vec![
        to_value(&inclusive),
        to_value(&input),
        to_value(&vec![0u32; input.len()]),
    ];
    let results = reference_eval(&program, &inputs).expect("interpreter runs difference pass");
    let out: Vec<u32> = results[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        out,
        reference_exclusive_scan(&input),
        "difference pass must yield the exclusive scan"
    );
}

/// Feed one Value per non-workgroup buffer in binding order (real input for
/// the `input`-named buffer, a zero slot for every fused scratch/output),
/// run through the reference interpreter, and return the `output` buffer.
/// The multi-block chain fuses in intermediate buffers (`__output_mbps_*`),
/// so the naive `[input, output]` feed is insufficient; this locates
/// `output` among the returned ReadWrite buffers instead of assuming index 0.
fn run_full_scan(program: &vyre_foundation::ir::Program, input: &[u32]) -> Vec<u32> {
    use vyre_foundation::ir::BufferAccess;
    use vyre_reference::value::Value;
    let mut inputs = Vec::new();
    let mut output_idx = None;
    let mut writable_seen = 0usize;
    for buf in program.buffers() {
        if buf.access() == BufferAccess::Workgroup {
            continue;
        }
        let bytes = if buf.name() == "input" {
            vyre_primitives::wire::pack_u32_slice(input)
        } else {
            vec![0u8; (buf.count() as usize).saturating_mul(4)]
        };
        inputs.push(Value::from(bytes));
        if buf.access() == BufferAccess::ReadWrite {
            if buf.name() == "output" {
                output_idx = Some(writable_seen);
            }
            writable_seen += 1;
        }
    }
    let outputs =
        vyre_reference::reference_eval(program, &inputs).expect("multi-block scan must execute");
    let idx = output_idx.expect("output buffer must be a writable result");
    outputs[idx]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn inclusive_multi_block_chain_matches_oracle_at_large_n() {
    // n exceeds the portable default and routes through the fused three-pass GridSync chain, a
    // different algorithm than the single-block scan. The other large-n
    // tests (`large_n_emits_three_pass_chain`,
    // `recursion_handles_million_elements`) assert only STRUCTURE; this
    // checks the VALUES through reference_eval across exact and off block
    // boundaries so a broken cross-block carry cannot pass as green.
    for n in [1025, 1024 + 512, 2048, 3072 + 7] {
        // under u32::MAX: this isolates carry correctness from wrap.
        let input: Vec<u32> = (0..n).map(|i| (i % 251) + 1).collect();
        let program = multi_block_prefix_scan_sum_u32("input", "output", n);
        let actual = run_full_scan(&program, &input);
        assert_eq!(
            actual,
            reference_inclusive_scan(&input),
            "n={n}: inclusive multi-block chain diverged from the scan oracle"
        );
    }
}

#[test]
fn exclusive_multi_block_chain_matches_oracle_at_large_n() {
    // The exclusive chain (inclusive chain + element-difference pass) had NO
    // full-chain value coverage at large n: `exclusive_difference_pass_executes`
    // only runs the single difference pass at n=8. This exercises the whole
    // fused exclusive scan through reference_eval past the block boundary.
    for n in [1025, 2048, 3072 + 7] {
        let input: Vec<u32> = (0..n).map(|i| (i % 251) + 1).collect();
        let program = multi_block_prefix_scan_sum_exclusive_u32("input", "output", n);
        let actual = run_full_scan(&program, &input);
        assert_eq!(
            actual,
            reference_exclusive_scan(&input),
            "n={n}: exclusive multi-block chain diverged from the exclusive scan oracle"
        );
    }
}

#[test]
fn exclusive_scan_empty_and_oversized() {
    // n == 0 -> empty program (no work, no trap).
    let empty = multi_block_prefix_scan_sum_exclusive_u32("in", "out", 0);
    assert!(
        !program_contains_trap(&empty),
        "n=0 must be an empty, non-trap program"
    );
    // Oversized -> executable trap carrying the sizing error, not a panic.
    let oversized = multi_block_prefix_scan_sum_exclusive_u32("in", "out", u32::MAX);
    assert_eq!(oversized.buffers()[0].name(), "out");
    assert!(
        program_contains_trap(&oversized),
        "oversized exclusive scan must encode an executable trap"
    );
}

#[test]
fn exclusive_scan_small_and_large_n_declare_in_and_out() {
    // Small (single-block inclusive path) and large (3-pass GridSync path)
    // both fuse into a program that reads `in` and writes `out`.
    for &n in &[1u32, 64, 1024, 2048] {
        let program = multi_block_prefix_scan_sum_exclusive_u32("in", "out", n);
        let names: Vec<&str> = program.buffers().iter().map(BufferDecl::name).collect();
        assert!(
            !program_contains_trap(&program),
            "n={n} valid exclusive scan must not trap"
        );
        assert!(
            names.contains(&"in"),
            "n={n} must declare input `in`, got {names:?}"
        );
        assert!(
            names.contains(&"out"),
            "n={n} must declare output `out`, got {names:?}"
        );
    }
}

#[test]
fn reference_into_reuses_output_and_truncates_stale_tail() {
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&[99, 98, 97, 96]);
    let capacity = out.capacity();

    reference_inclusive_scan_into(&[u32::MAX, 1, 2], &mut out);
    assert_eq!(out, vec![u32::MAX, 0, 2]);
    assert_eq!(out.capacity(), capacity);

    reference_inclusive_scan_into(&[7], &mut out);
    assert_eq!(out, vec![7]);
    assert_eq!(out.capacity(), capacity);
}

#[test]
fn try_reference_into_reuses_output_and_clears_stale_tail() {
    let mut out = Vec::with_capacity(8);
    out.extend_from_slice(&[99, 98, 97, 96]);
    let ptr = out.as_ptr();

    try_reference_inclusive_scan_into(&[u32::MAX, 1, 2], &mut out).unwrap();

    assert_eq!(out, vec![u32::MAX, 0, 2]);
    assert_eq!(out.as_ptr(), ptr);
}

#[test]
fn compatibility_wrappers_match_reference() {
    let input = &[u32::MAX, 1, 2];
    let mut compat = Vec::with_capacity(8);
    let mut fallible = Vec::with_capacity(8);

    reference_inclusive_scan_into(input, &mut compat);
    try_reference_inclusive_scan_into(input, &mut fallible)
        .expect("Fix: small multi-block prefix-scan reference must reserve");

    assert_eq!(reference_inclusive_scan(input), fallible);
    assert_eq!(compat, fallible);
}

/// True when `program` encodes an executable trap anywhere.
///
/// Descent comes from `visit::any_descendant`, the one owner of
/// which node variants nest. The hand-written match this replaces ended in
/// `_ => false`, so a fifth body-bearing variant would have reported no trap
/// in a program whose only trap is nested inside it.
fn program_contains_trap(program: &Program) -> bool {
    program
        .entry()
        .iter()
        .any(|node| any_descendant(node, &mut |n| matches!(n, Node::Trap { .. })))
}

#[test]
fn oversized_multi_block_scan_returns_trap_program_instead_of_panicking() {
    let prog = multi_block_prefix_scan_sum_u32("in_buf", "out_buf", u32::MAX);

    assert_eq!(prog.buffers()[0].name(), "out_buf");
    assert!(
        program_contains_trap(&prog),
        "oversized scan should encode an executable trap with the sizing error"
    );
}

#[test]
fn oversized_pass_builders_return_trap_programs_instead_of_panicking() {
    let pass_a = pass_a_local_scan("in_buf", "partials", "block_totals", 1, u32::MAX);
    let pass_a_lanes =
        pass_a_local_scan_with_block_lanes("in_buf", "partials", "block_totals", 1, u32::MAX, 256);
    let pass_c =
        pass_c_broadcast_offsets("partials", "block_totals_scanned", "out_buf", 1, u32::MAX);
    let pass_c_lanes = pass_c_broadcast_offsets_with_block_lanes(
        "partials",
        "block_totals_scanned",
        "out_buf",
        1,
        u32::MAX,
        256,
    );

    assert_eq!(pass_a.buffers()[0].name(), "partials");
    assert!(program_contains_trap(&pass_a));
    assert_eq!(pass_a_lanes.buffers()[0].name(), "partials");
    assert!(program_contains_trap(&pass_a_lanes));
    assert_eq!(pass_c.buffers()[0].name(), "out_buf");
    assert!(program_contains_trap(&pass_c));
    assert_eq!(pass_c_lanes.buffers()[0].name(), "out_buf");
    assert!(program_contains_trap(&pass_c_lanes));
}

#[test]
fn every_default_builder_uses_the_portable_workgroup_width() {
    let portable = [PORTABLE_WORKGROUP_INVOCATIONS, 1, 1];
    for &n in &[1u32, 2, 64, 255, 256] {
        let program = multi_block_prefix_scan_sum_u32("in_buf", "out_buf", n);
        let names: Vec<&str> = program.buffers().iter().map(BufferDecl::name).collect();
        assert_eq!(program.workgroup_size(), portable, "inclusive n={n}");
        assert!(
            names.contains(&"in_buf"),
            "n={n} must declare in_buf, got {names:?}"
        );
        assert!(
            names.contains(&"out_buf"),
            "n={n} must declare out_buf, got {names:?}"
        );
    }

    assert_eq!(
        multi_block_prefix_scan_sum_exclusive_u32("in_buf", "out_buf", 64).workgroup_size(),
        portable,
        "exclusive default"
    );
    assert_eq!(
        pass_a_local_scan("in_buf", "partials", "block_totals", 64, 1).workgroup_size(),
        portable,
        "Pass A default"
    );
    assert_eq!(
        pass_c_broadcast_offsets("partials", "block_totals_scanned", "out_buf", 64, 1,)
            .workgroup_size(),
        portable,
        "Pass C default"
    );

    for invalid in [0, 1, 3, 255] {
        assert_eq!(
            multi_block_prefix_scan_sum_u32_with_block_lanes("in_buf", "out_buf", 64, invalid,)
                .workgroup_size(),
            portable,
            "inclusive invalid width {invalid}"
        );
        assert_eq!(
            multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes(
                "in_buf", "out_buf", 64, invalid,
            )
            .workgroup_size(),
            portable,
            "exclusive invalid width {invalid}"
        );
    }
}

#[test]
fn large_n_emits_three_pass_chain() {
    // n = 2048 emits eight portable blocks; the totals fit one workgroup.
    let prog = multi_block_prefix_scan_sum_u32("in_buf", "out_buf", 2048);
    let names: Vec<&str> = prog.buffers().iter().map(BufferDecl::name).collect();
    assert!(
        names.contains(&"in_buf"),
        "input must be declared, got {names:?}"
    );
    assert!(
        names.contains(&"out_buf"),
        "output must be declared, got {names:?}"
    );
    assert_eq!(
        prog.buffers()
            .iter()
            .filter(|buffer| buffer.is_output())
            .count(),
        1,
        "fused multi-block scan must expose only the final output buffer"
    );
}

#[test]
fn empty_input_returns_empty_program() {
    let prog = multi_block_prefix_scan_sum_u32("in_buf", "out_buf", 0);
    assert!(prog.buffers().is_empty());
}

#[test]
fn recursion_bottoms_out_at_one_workgroup_for_the_soft_maximum() {
    // SOFT_MAX_N emits one portable workgroup of Pass-A blocks, so Pass B
    // falls through to the single-workgroup `prefix_scan`. Verify build.
    let prog = multi_block_prefix_scan_sum_u32("in_buf", "out_buf", SOFT_MAX_N);
    let names: Vec<&str> = prog.buffers().iter().map(BufferDecl::name).collect();
    assert!(names.contains(&"in_buf"));
    assert!(names.contains(&"out_buf"));
}

#[test]
fn multi_block_scan_runs_at_both_256_and_1024_admitted_widths() {
    for &lanes in &[256, 1024] {
        let n = lanes * 2 + 17;
        let input: Vec<u32> = (0..n).map(|i| (i % 251) + 1).collect();
        let prog = multi_block_prefix_scan_sum_u32_with_block_lanes("input", "output", n, lanes);
        assert_eq!(prog.workgroup_size(), [lanes, 1, 1]);
        let actual = run_full_scan(&prog, &input);
        assert_eq!(actual, reference_inclusive_scan(&input));

        let geom = vyre_foundation::LaunchGeometry {
            workgroup: [lanes, 1, 1],
            ..Default::default()
        };
        let prog_geom = multi_block_prefix_scan_sum_u32_with_geometry("input", "output", n, &geom);
        assert_eq!(prog_geom.workgroup_size(), [lanes, 1, 1]);
        let actual_geom = run_full_scan(&prog_geom, &input);
        assert_eq!(actual_geom, reference_inclusive_scan(&input));

        let prog_excl =
            multi_block_prefix_scan_sum_exclusive_u32_with_block_lanes("input", "output", n, lanes);
        assert_eq!(prog_excl.workgroup_size(), [lanes, 1, 1]);
        let actual_excl = run_full_scan(&prog_excl, &input);
        assert_eq!(actual_excl, reference_exclusive_scan(&input));

        let prog_excl_geom =
            multi_block_prefix_scan_sum_exclusive_u32_with_geometry("input", "output", n, &geom);
        assert_eq!(prog_excl_geom.workgroup_size(), [lanes, 1, 1]);
        let actual_excl_geom = run_full_scan(&prog_excl_geom, &input);
        assert_eq!(actual_excl_geom, reference_exclusive_scan(&input));
    }
}
