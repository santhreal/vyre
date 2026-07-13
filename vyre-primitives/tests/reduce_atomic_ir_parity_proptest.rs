//! Tier 3 - Property: differential proptest driving the ACTUAL atomic grid-stride reduction IR of
//! the whole `reduce::atomic_scalar` family — ALL seven variants: `reduce_max`, `reduce_min`,
//! `reduce_any`, `reduce_all`, `reduce_sum`, `reduce_count` (popcount-sum), `reduce_count_non_zero` —
//! through `reference_eval` vs each op's `cpu_ref`.
//!
//! MOTIVATION — a real coverage hole, not a duplicate. The shipped reduce proptests
//! (`proptest_reduce_min_max_laws`, `proptest_reduce_any_all`, `proptest_reduce_sum_laws`,
//! `proptest_reduce_count_non_zero`) exercise ONLY `cpu_ref` (`grep reference_eval` = 0 in every one
//! of them): they prove the CPU oracle obeys algebraic laws but NEVER run the GPU IR. The atomic
//! grid-stride kernel itself — the `SeqCst` init barrier, the `atomic_max`/`atomic_min`/`atomic_or`/
//! `atomic_and`/`atomic_add` accumulation, the `select`-for-nonzero, and the `popcount` reduction — is
//! validated ONLY by ONE hand fixture per op in the inventory registry (`count = 4`). A wrong atomic
//! opcode, an identity-init bug, a barrier omission, or a mis-seeded reduction would pass EVERY
//! existing test.
//!
//! This suite closes that hole: each op's IR is run over randomized inputs (all-zero, all-`u32::MAX`,
//! all-nonzero, sparse-nonzero) asserted bit-exact vs `cpu_ref`. Building it SURFACED AND FIXED A REAL
//! BUG — `reduce_sum` diverged at `len = 257` (IR double-counted) because `reference_eval` fires
//! `ceil(count/256)` workgroups for a `count > 256` input while the kernel is single-workgroup by
//! construction; the non-idempotent Sum/Count/CountNonZero double-counted while the idempotent
//! Max/Min/Any/All hid it. The fix (a `lane < WORKGROUP_SIZE` guard on the atomic in
//! `atomic_scalar.rs`, failing extra-workgroup lanes closed) makes the reduction correct under ANY
//! dispatch grid. This suite now SPANS the 256-lane boundary into 4+ workgroups (lengths up to 1100)
//! specifically to keep that fix regression-proof. See BACKLOG
//! FINDING-atomic-scalar-reduce-double-counts-under-multi-workgroup-dispatch.
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::reduce::{all, any, count, count_non_zero, max, min, sum};

/// Which family member to build + oracle. Covers ALL seven `AtomicReduceKind`/`AtomicBoolReduceKind`
/// variants of the shared `atomic_scalar` grid-stride kernel: Max/Min/AnyNonZero/AllNonZero plus
/// Sum/PopcountSum/CountNonZero — every distinct atomic opcode (`atomic_max/min/or/and/add`), the
/// `select`-for-nonzero, and the `popcount` reduction, driven over the same randomized inputs.
#[derive(Clone, Copy, Debug)]
enum Op {
    Max,
    Min,
    Any,
    All,
    Sum,
    Count,
    CountNonZero,
}

impl Op {
    fn build(self, count_n: u32) -> vyre_foundation::ir::Program {
        match self {
            Op::Max => max::reduce_max("values", "out", count_n),
            Op::Min => min::reduce_min("values", "out", count_n),
            Op::Any => any::reduce_any("values", "out", count_n),
            Op::All => all::reduce_all("values", "out", count_n),
            Op::Sum => sum::reduce_sum("values", "out", count_n),
            Op::Count => count::reduce_count("values", "out", count_n),
            Op::CountNonZero => count_non_zero::reduce_count_non_zero("values", "out", count_n),
        }
    }

    fn oracle(self, values: &[u32]) -> u32 {
        match self {
            Op::Max => max::cpu_ref(values),
            Op::Min => min::cpu_ref(values),
            Op::Any => any::cpu_ref(values),
            Op::All => all::cpu_ref(values),
            Op::Sum => sum::cpu_ref(values),
            Op::Count => count::cpu_ref(values),
            Op::CountNonZero => count_non_zero::cpu_ref(values),
        }
    }
}

const OPS: [Op; 7] = [
    Op::Max,
    Op::Min,
    Op::Any,
    Op::All,
    Op::Sum,
    Op::Count,
    Op::CountNonZero,
];

/// Run one op's IR and return the single scalar in `out[0]`.
fn run_ir(op: Op, values: &[u32]) -> u32 {
    let program = op.build(values.len() as u32);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(values),  // values (binding 0, RO)
            pack(&[0u32]), // out (binding 1, RW) — kernel re-inits to identity in lane 0
        ],
    )
    .expect("reduce reference evaluation must succeed");
    // Sole RW buffer is `out` (binding 1) → results[0].
    let b = outputs[0].to_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// A value generator that mixes full-range and structurally-interesting distributions, so the sweep
/// hits all-zero (kills `any`/`max`), all-`u32::MAX` (kills `min` identity), all-nonzero (kills
/// `all`), and sparse-nonzero (single live lane in a large multi-workgroup buffer).
///
/// Lengths deliberately SPAN the 256-lane WORKGROUP_SIZE boundary into 4+ notional workgroups
/// (1..=1100). `reference_eval` infers a `ceil(count/256)`-workgroup grid for `count > 256` (the
/// kernel declares `out` as ReadWrite storage, so the interpreter falls back to `max_input_elements`),
/// which is EXACTLY the multi-workgroup dispatch that used to make the non-idempotent
/// Sum/Count/CountNonZero double-count. Since the `lane < WORKGROUP_SIZE` guard landed in
/// `atomic_scalar.rs` (extra workgroups fail closed), this range now VERIFIES the fix across the
/// boundary. See BACKLOG FINDING-atomic-scalar-reduce-double-counts-under-multi-workgroup-dispatch.
fn arb_values() -> impl Strategy<Value = Vec<u32>> {
    prop_oneof![
        // Full-range, spanning the 256-lane boundary into 4+ workgroups.
        prop::collection::vec(any::<u32>(), 1..=1100),
        // Small biased values (dense collisions on min/max atomics).
        prop::collection::vec(0u32..=8, 1..=600),
        // 0/1 masks (exercise any/all + the count-style select across workgroups).
        prop::collection::vec(prop_oneof![Just(0u32), Just(1u32)], 1..=600),
        // Saturation extremes.
        prop::collection::vec(prop_oneof![Just(0u32), Just(u32::MAX)], 1..=600),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Every family member's real IR must match its oracle on the same random input.
    #[test]
    fn reduce_atomic_ir_matches_cpu_ref(values in arb_values()) {
        for op in OPS {
            let got = run_ir(op, &values);
            let want = op.oracle(&values);
            prop_assert_eq!(
                got, want,
                "{:?} diverged on len={}: IR={} cpu_ref={} (first16={:?})",
                op, values.len(), got, want,
                &values[..values.len().min(16)]
            );
        }
    }
}

// Deterministic boundary cases the random sweep is unlikely to hit exactly: lengths ON and AROUND the
// 256-lane workgroup boundary (255/256/257) and across several workgroups (512/513/1024), the exact
// multi-workgroup seam where the double-count bug lived and where the `lane < WORKGROUP_SIZE` fix is
// proven. The inventory `count = 4` fixture is single-workgroup and never reached here.
#[test]
fn reduce_atomic_ir_boundary_lengths() {
    let lengths = [1usize, 2, 255, 256, 257, 511, 512, 513, 1024];
    for &n in &lengths {
        // Ascending values: max at the end, min at the front — catches a lane that drops the tail or
        // mis-seeds the identity.
        let ascending: Vec<u32> = (0..n as u32).collect();
        // One live lane parked at the LAST index — a dropped final lane makes any/max miss it.
        let mut sparse = vec![0u32; n];
        sparse[n - 1] = 7;
        for values in [ascending, sparse, vec![u32::MAX; n], vec![0u32; n]] {
            for op in OPS {
                let got = run_ir(op, &values);
                let want = op.oracle(&values);
                assert_eq!(got, want, "{op:?} boundary n={n}: IR={got} cpu_ref={want}");
            }
        }
    }
}
