//! GPU-IR vs CPU-ref parity for `math::sos_certificate::sos_gram_construct` on an
//! OUT-OF-RANGE monomial-pair index.
//!
//! `gram[t] = p_coeffs[monomial_pairs[t]]`, but `monomial_pairs[t]` is DATA: nothing
//! validates the pair indices are `< coeff_count`. The CPU reference defends against
//! this with `p_coeffs.get(idx).copied().unwrap_or(0)`: an out-of-range pair index
//! yields 0. The GPU IR must gate the inner load identically. An unconditional
//! `load(p_coeffs, load(monomial_pairs, t))` would OOB-read `p_coeffs` on real GPU
//! hardware, where a global-memory read past the buffer is *undefined behavior* (it
//! can page-fault the kernel or return arbitrary memory), the same class as the
//! `bitset_test_bit` unconditional-load and `reduce::gather` skip-on-OOB parity bugs.
//!
//! WHY THIS NEEDS A STRUCTURAL LOCK, NOT JUST A BEHAVIORAL CHECK: the reference
//! interpreter (`vyre-reference`) *defines* an OOB scalar load as zero-fill
//! (`vyre-reference/src/oob.rs`), so under `reference_eval` a bare double-load
//! `load(p, load(mp, t))` and the gated `select(mp_idx < cc, load(p, mp_idx), 0)`
//! produce the SAME value (0) on the OOB lane. A behavioral `reference_eval` test
//! therefore cannot catch a regression that drops the gate, it papers over exactly
//! the UB that real hardware does not. So the behavioral tests below verify the
//! common path and document the contract, and the `structural_gate_is_present` test
//! is the real regression lock: the gate introduces a `Node::Let` (binding the pair
//! index once for the bounds check + the load) that the bare double-load has no way
//! to contain. Reverting to the unguarded load removes the `LET` node-kind and fails.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use vyre_foundation::ir::stats::NODE_KIND_LET;
use vyre_primitives::math::sos_certificate::{sos_gram_construct, sos_gram_construct_cpu};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn eval_gram(pairs: &[u32], p: &[u32], m: u32, coeff_count: u32) -> Vec<u32> {
    let program = sos_gram_construct("pairs", "p", "gram", m, coeff_count);
    let gram_init = vec![0u32; (m as usize) * (m as usize)];
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(pairs)),
            Value::from(pack(p)),
            Value::from(pack(&gram_init)),
        ],
    )
    .expect("sos_gram reference evaluation must succeed");
    // `gram` (binding 2) is the sole ReadWrite buffer; reference_eval returns only the
    // writable buffers, so it is outputs[0] (as the gather parity test uses outputs[0]
    // for its lone RW `dst`).
    unpack(&outputs[0].to_bytes())
}

#[test]
fn structural_gate_is_present() {
    // The bounds gate is emitted as a `Node::Let` binding the pair index before the
    // bounds `select`. A bare `load(p_coeffs, load(monomial_pairs, t))` contains no
    // Let. This is the regression lock (the behavioral tests below cannot catch the
    // revert because the reference interpreter zero-fills OOB loads (see module doc)).
    let program = sos_gram_construct("pairs", "p", "gram", 4, 16);
    let stats = program.stats();
    assert_ne!(
        stats.node_kinds_present & NODE_KIND_LET,
        0,
        "sos_gram_construct must bind the pair index in a Node::Let and bounds-gate the \
         p_coeffs load against coeff_count; a bare load(p_coeffs, load(monomial_pairs, t)) \
         has no Let and OOB-reads p_coeffs (undefined behavior on real GPU hardware)."
    );
}

#[test]
fn out_of_range_pair_index_matches_cpu_ref() {
    // 2x2 Gram (m=2 → 4 cells). p has 3 coefficients (coeff_count=3).
    // pairs[1] == 9 is OUT OF RANGE (>= coeff_count); the CPU ref fills it with 0.
    let pairs = [0u32, 9, 1, 2];
    let p = [10u32, 20, 30];
    let cpu = sos_gram_construct_cpu(&pairs, &p, 2);
    assert_eq!(
        cpu,
        vec![10, 0, 20, 30],
        "cpu_ref must default the OOB pair index to 0"
    );
    let gpu_ir = eval_gram(&pairs, &p, 2, 3);
    assert_eq!(
        gpu_ir, cpu,
        "sos_gram GPU-IR must match cpu_ref on an out-of-range pair index. GPU={gpu_ir:?} cpu={cpu:?}"
    );
}

#[test]
fn all_pairs_out_of_range_is_all_zero() {
    let pairs = [7u32, 8, 9, 10];
    let p = [111u32, 222];
    let cpu = sos_gram_construct_cpu(&pairs, &p, 2);
    assert_eq!(cpu, vec![0, 0, 0, 0], "all-OOB cpu_ref is all zero");
    let gpu_ir = eval_gram(&pairs, &p, 2, 2);
    assert_eq!(
        gpu_ir, cpu,
        "sos_gram GPU-IR must zero every out-of-range lane. GPU={gpu_ir:?} cpu={cpu:?}"
    );
}

#[test]
fn in_range_pairs_match_cpu_ref() {
    // Sanity: with every pair index in range, the gate must not perturb the common path.
    let m = 3u32;
    let coeff_count = 9u32;
    let pairs: Vec<u32> = (0..9u32).map(|i| (i * 7 + 3) % coeff_count).collect();
    let p: Vec<u32> = (0..9u32)
        .map(|i| i.wrapping_mul(13).wrapping_add(1))
        .collect();
    let cpu = sos_gram_construct_cpu(&pairs, &p, m);
    let gpu_ir = eval_gram(&pairs, &p, m, coeff_count);
    assert_eq!(
        gpu_ir, cpu,
        "sos_gram GPU-IR must match cpu_ref on the all-in-range common path. GPU={gpu_ir:?} cpu={cpu:?}"
    );
}
