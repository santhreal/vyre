//! GPU-IR parity for `graph/sheaf::sheaf_diffusion_step` driven through `vyre_reference::reference_eval`
//! with SIGNED (negative) stalks and restriction maps.
//!
//! Why this test exists: the sheaf diffusion step computes `stalks_next = s − damping·r·s` via TWO nested
//! `fixed_mul_16_16_expr` calls, where `r` (restriction map) and `s` (stalk feature) are BOTH signed in a
//! heterophilic sheaf (negative coupling; signed node features). The existing self-substrate parity test
//! (`sheaf_heterophilic_via_reference_parity`) DELIBERATELY keeps `damping·r ≤ 1` and non-negative values
//! ("no underflow"), so the negative-operand path of the fixed multiply was UNCOVERED, exactly the class
//! of silent corruption fixed in `fixed_mul_16_16_expr` (see BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`). This is the THIRD kernel (after the AMG V-cycle and the
//! Clifford product) proving the signed multiply, and the first to drive negative sheaf stalks/restriction.
//!
//! BIT-EXACT (no tolerance): stalks/restriction are half-integers and damping ∈ {0.25, 0.5}, so
//! `damping·r·s` is an exact multiple of 0.0625 with small magnitude, exactly representable in 16.16, so
//! the fixed IR must reproduce the f64 formula `s − damping·r·s` to the bit.
#![cfg(feature = "graph")]

use vyre_primitives::graph::sheaf::sheaf_diffusion_step;
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

fn from_fixed(v: u32) -> f64 {
    f64::from(v as i32) / FIXED_ONE
}

/// A signed half-integer in {-3, -2.5, …, 3}.
fn signed_half(state: &mut u32) -> f64 {
    0.5 * f64::from((xorshift(state) % 13) as i32 - 6)
}

fn run_via_reference(stalks: &[u32], restriction: &[u32], damping: u32, cells: u32) -> Vec<u32> {
    let program = sheaf_diffusion_step("stalks", "restriction", "damping", "stalks_next", cells, 1);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(stalks)),
            Value::from(pack_u32(restriction)),
            Value::from(pack_u32(&[damping])),
            Value::from(pack_u32(&vec![0u32; cells as usize])),
        ],
    )
    .expect("sheaf_diffusion_step reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn sheaf_diffusion_step_bit_exact_over_signed_stalks_and_restrictions() {
    let mut state = 0x5E_AF_00_01u32;
    let mut neg_restriction = 0u32;
    let mut neg_stalk = 0u32;
    let mut neg_output = 0u32;
    for case in 0..400u32 {
        let cells = 1 + case % 5; // 1..5 stalk cells per dispatch
        let damping_real = if case & 1 == 0 { 0.5 } else { 0.25 };
        let damping = to_fixed(damping_real);

        let stalks_f: Vec<f64> = (0..cells).map(|_| signed_half(&mut state)).collect();
        let restriction_f: Vec<f64> = (0..cells).map(|_| signed_half(&mut state)).collect();
        let stalks_fx: Vec<u32> = stalks_f.iter().map(|&v| to_fixed(v)).collect();
        let restriction_fx: Vec<u32> = restriction_f.iter().map(|&v| to_fixed(v)).collect();

        let got = run_via_reference(&stalks_fx, &restriction_fx, damping, cells);
        assert_eq!(got.len(), cells as usize, "case {case}: output length");

        for i in 0..cells as usize {
            let s = stalks_f[i];
            let r = restriction_f[i];
            // The kernel: stalks_next = s − damping·r·s (two nested signed fixed multiplies).
            let want = s - damping_real * r * s;
            let want_word = to_fixed(want);
            assert_eq!(
                got[i],
                want_word,
                "case {case} cell {i}: signed sheaf diffusion must be BIT-EXACT to s−damping·r·s; \
                 got={} want={want} (s={s} r={r} damping={damping_real})",
                from_fixed(got[i])
            );
            if r < 0.0 {
                neg_restriction += 1;
            }
            if s < 0.0 {
                neg_stalk += 1;
            }
            if from_fixed(got[i]) < 0.0 {
                neg_output += 1;
            }
        }
    }
    assert!(
        neg_restriction > 200,
        "sweep must feed negative restriction maps (signed-mul regime), got {neg_restriction}"
    );
    assert!(
        neg_stalk > 200,
        "sweep must feed negative stalk features, got {neg_stalk}"
    );
    assert!(
        neg_output > 100,
        "sweep must produce negative diffused stalks, got {neg_output}"
    );
}

#[test]
fn sheaf_diffusion_step_hand_checked_negative_restriction() {
    // s=2, r=−1, damping=0.5 → stalks_next = 2 − 0.5·(−1)·2 = 2 − (−1) = 3.
    // A NEGATIVE restriction map drives damped_r negative → exercises the signed fixed multiply directly.
    let got = run_via_reference(&[to_fixed(2.0)], &[to_fixed(-1.0)], to_fixed(0.5), 1);
    assert_eq!(
        from_fixed(got[0]),
        3.0,
        "negative restriction: heterophilic coupling INCREASES the stalk (2 → 3)"
    );

    // s=−2, r=1, damping=0.5 → −2 − 0.5·1·(−2) = −2 + 1 = −1. Negative stalk AND negative output.
    let got = run_via_reference(&[to_fixed(-2.0)], &[to_fixed(1.0)], to_fixed(0.5), 1);
    assert_eq!(
        from_fixed(got[0]),
        -1.0,
        "negative stalk stays negative through the signed multiply (−2 → −1)"
    );

    // s=−1, r=−2, damping=0.5 → −1 − 0.5·(−2)·(−1) = −1 − 1 = −2. BOTH operands negative.
    let got = run_via_reference(&[to_fixed(-1.0)], &[to_fixed(-2.0)], to_fixed(0.5), 1);
    assert_eq!(
        from_fixed(got[0]),
        -2.0,
        "both-negative operands: the signed product damping·r·s = +1 subtracts to −2"
    );
}
