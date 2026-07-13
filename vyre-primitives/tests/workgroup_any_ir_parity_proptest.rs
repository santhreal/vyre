//! Tier 3 - Property: differential proptest driving the ACTUAL `reduce::workgroup_any_u32` IR (a
//! single-workgroup bitwise-OR reduction) through `reference_eval` vs `cpu_ref`. The op had
//! `reference_eval` = 0 in tests/ (its `sweep_reduce_workgroup_any_*` peer is cpu-vs-cpu).
//!
//! The kernel ORs every input word into `out[0]` within one workgroup. A missed lane, a wrong
//! identity (must be 0 for OR), or a barrier/accumulation slip diverges. The sweep runs random inputs
//! (count 1..=256, single workgroup) including all-zero (result 0), single-bit-per-lane (result is the
//! union of all bits), and full-range, asserting `out[0]` bit-exact vs `cpu_ref`.
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::reduce::workgroup_any::{cpu_ref, workgroup_any_u32};

fn run_ir(values: &[u32]) -> u32 {
    let program = workgroup_any_u32("values", "out", values.len() as u32);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs = vyre_reference::reference_eval(&program, &[pack(values), pack(&[0u32])])
        .expect("workgroup_any_u32 reference evaluation must succeed");
    let b = outputs[0].to_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn workgroup_any_ir_matches_cpu_ref(
        values in prop::collection::vec(any::<u32>(), 1..=256)
    ) {
        prop_assert_eq!(run_ir(&values), cpu_ref(&values), "len={}", values.len());
    }

    /// Single-bit-per-lane: the OR reduction must reconstruct the exact union of set bits.
    #[test]
    fn workgroup_any_ir_reconstructs_bit_union(bits in prop::collection::vec(0u32..32, 1..=256)) {
        let values: Vec<u32> = bits.iter().map(|&b| 1u32 << b).collect();
        let expected = bits.iter().fold(0u32, |a, &b| a | (1u32 << b));
        prop_assert_eq!(run_ir(&values), expected);
        prop_assert_eq!(run_ir(&values), cpu_ref(&values));
    }
}

#[test]
fn workgroup_any_ir_boundaries() {
    assert_eq!(run_ir(&[0]), 0);
    assert_eq!(run_ir(&[0, 0, 0, 0]), 0, "all-zero ORs to 0");
    assert_eq!(run_ir(&[0, 2, 4, 0]), 6, "0|2|4|0 == 6");
    assert_eq!(run_ir(&[u32::MAX]), u32::MAX);
    // 256 lanes each contributing one distinct bit-cluster: full coverage.
    let full: Vec<u32> = (0..256u32).map(|i| 1u32 << (i % 32)).collect();
    assert_eq!(
        run_ir(&full),
        u32::MAX,
        "every bit set across the workgroup"
    );
    assert_eq!(run_ir(&full), cpu_ref(&full));
}
