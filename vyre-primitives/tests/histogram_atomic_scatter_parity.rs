//! GPU-IR parity for `reduce::histogram::histogram_atomic_scatter`.
//!
//! `reduce/histogram.rs` ships two builders sharing one OP_ID + cpu_ref: the
//! deterministic scan `histogram` (registered + swept) and the atomic variant
//! `histogram_atomic_scatter` (lane `t` does `atomic_add(output[input[t]], 1)`
//! gated by `bin < num_bins`). The atomic variant had NO parity test (found by
//! the registry-coverage closure gate). This pins it against the shared cpu_ref
//! via `reference_eval`: the atomic scatter must produce exactly the per-bin
//! counts, dropping out-of-range input values (matching the GPU `bin < num_bins`
//! gate and the cpu_ref's `out.get_mut(bin)` skip).
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use vyre_primitives::reduce::histogram::{cpu_ref, histogram_atomic_scatter};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn eval(input: &[u32], num_bins: u32) -> Vec<u32> {
    let program = histogram_atomic_scatter("input", "output", input.len() as u32, num_bins);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(input)),
            Value::from(pack(&vec![0u32; num_bins as usize])),
        ],
    )
    .expect("histogram_atomic_scatter reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

#[test]
fn atomic_scatter_counts_bins_matching_cpu_ref() {
    // bin5 is out of range (num_bins=3) and must be dropped.
    let input = [0u32, 2, 1, 2, 0, 5, 2];
    let num_bins = 3u32;
    let cpu = cpu_ref(&input, num_bins);
    assert_eq!(
        cpu,
        vec![2, 1, 3],
        "cpu_ref: bin0=2 (idx 0,4), bin1=1 (idx 2), bin2=3 (idx 1,3,6); bin5 dropped"
    );
    assert_eq!(
        eval(&input, num_bins),
        cpu,
        "histogram_atomic_scatter GPU-IR must equal cpu_ref"
    );
}

#[test]
fn atomic_scatter_all_same_bin() {
    let input = [1u32, 1, 1, 1];
    let num_bins = 4u32;
    let cpu = cpu_ref(&input, num_bins);
    assert_eq!(cpu, vec![0, 4, 0, 0]);
    assert_eq!(eval(&input, num_bins), cpu);
}

#[test]
fn atomic_scatter_all_out_of_range_is_all_zero() {
    let input = [7u32, 8, 9];
    let num_bins = 3u32;
    let cpu = cpu_ref(&input, num_bins);
    assert_eq!(cpu, vec![0, 0, 0], "every input >= num_bins is dropped");
    assert_eq!(eval(&input, num_bins), cpu);
}
