//! Value parity for `math::tensor_scc::tensor_scc_fixpoint`: the IR PROGRAM, run through
//! `reference_eval`, must match its own `cpu_ref` bitset closure, including for seeds that
//! carry bits OUTSIDE `group_mask`.
//!
//! WHY: tensor_scc_fixpoint is unregistered and its only tests exercise `cpu_ref`; the actual
//! IR was never run through reference_eval (no validity OR value check), the same gap the
//! union_find find-walk bug fell through. This differential drives the IR and compares against
//! the CPU reference over many random bit-matrices, deliberately including out-of-group seed
//! bits (the exact input class where the IR, which seeds `active` UNMASKED, diverges from
//! cpu_ref, which masks the seed to the group first).
#![cfg(all(feature = "all-lego", feature = "cpu-parity"))]

use vyre_primitives::math::tensor_scc::{cpu_ref, tensor_scc_fixpoint};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Run the IR program and return `out_mask[0]`.
fn run_ir(matrix_rows: &[u32], seed: u32, group: u32, iteration_limit: u32) -> u32 {
    let row_count = matrix_rows.len() as u32;
    let program = tensor_scc_fixpoint("rows", "seed", "group", "out", row_count, iteration_limit);
    // Input order = buffer declaration order: rows(0), seed(1), group(2), out(3).
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(matrix_rows)),
            Value::from(pack(&[seed])),
            Value::from(pack(&[group])),
            Value::from(pack(&[0u32])),
        ],
    )
    .expect("tensor_scc_fixpoint reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "out")
        .expect("Fix: tensor_scc_fixpoint must declare output `out`");
    unpack(&outputs[index].to_bytes())[0]
}

#[test]
fn tensor_scc_ir_matches_cpu_ref_including_out_of_group_seeds() {
    let mut state = 0x9E37_79B1u32;
    for case in 0..600u32 {
        let row_count = 2 + (next_u32(&mut state) % 10); // 2..=11 rows/bits
        let bit_mask = if row_count >= 32 {
            u32::MAX
        } else {
            (1u32 << row_count) - 1
        };
        let matrix_rows: Vec<u32> = (0..row_count)
            .map(|_| next_u32(&mut state) & bit_mask)
            .collect();
        // group is a random non-empty subset of the valid bits.
        let mut group = next_u32(&mut state) & bit_mask;
        if group == 0 {
            group = 1;
        }
        // seed spans ALL valid bits (so it routinely includes bits OUTSIDE the group, the
        // divergence class). iteration_limit exceeds row_count so both sides reach the fixpoint.
        let seed = next_u32(&mut state) & bit_mask;
        let iteration_limit = row_count + 2;

        let ir = run_ir(&matrix_rows, seed, group, iteration_limit);
        let cpu = cpu_ref(&matrix_rows, seed, group, iteration_limit);
        assert_eq!(
            ir, cpu,
            "case {case}: tensor_scc IR closure {ir:#b} != cpu_ref {cpu:#b} \
             (rows={matrix_rows:?}, seed={seed:#b}, group={group:#b}, iters={iteration_limit})"
        );
    }
}

#[test]
fn tensor_scc_out_of_group_seed_bit_is_masked_before_expansion() {
    // Hand-crafted minimal divergence: node 3 (bit3) is OUTSIDE group 0b0111 but its row points
    // into the group (row3 -> bit0). A seed of only bit3 must yield the EMPTY closure, an
    // out-of-group seed node contributes nothing to a group-local closure. If the IR seeds
    // `active` unmasked it would (wrongly) follow 3->0 and return 0b0001.
    let rows = [0u32, 0, 0, 0b0001];
    let seed = 0b1000; // bit3, out of group
    let group = 0b0111;
    let ir = run_ir(&rows, seed, group, 8);
    let cpu = cpu_ref(&rows, seed, group, 8);
    assert_eq!(
        cpu, 0,
        "sanity: cpu_ref masks the out-of-group seed to empty"
    );
    assert_eq!(
        ir, cpu,
        "tensor_scc IR must mask the out-of-group seed bit before expansion (got {ir:#b})"
    );
}

#[test]
fn tensor_scc_in_group_cycle_closes() {
    // Regression on the documented behavior: a cycle wholly inside the group closes to the
    // full group. rows: 0->1, 1->2, 2->0, 3->3; group 0b0111, seed bit0.
    let rows = [0b0010u32, 0b0100, 0b0001, 0b1000];
    let seed = 0b0001;
    let group = 0b0111;
    let ir = run_ir(&rows, seed, group, 8);
    assert_eq!(ir, 0b0111, "in-group cycle must close to the whole group");
    assert_eq!(ir, cpu_ref(&rows, seed, group, 8));
}
