//! GPU-IR parity for `math/tensor_train::tt_contract_step`: one tensor-train bond contraction
//! `acc_out[b] = Σ_a acc_in[a]·core[a,b]` in 16.16 fixed point, driven through
//! `vyre_reference::reference_eval` with SIGNED cores and accumulators.
//!
//! Why this test exists: TT cores come out of a (truncated) SVD, so they are freely SIGNED, and the
//! running bond accumulator `acc_in` carries signed partial contractions. The kernel multiplies them
//! with `fixed_mul_16_16_expr`. Before the signed-multiply fix (BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`) that multiply reconstructed the product from the
//! UNSIGNED high word, so a negative core or accumulator entry (a u32 with the top bit set, read as
//! ~2^32) produced a garbage term, silently corrupting the contracted tensor value. The only
//! existing IR run of this kernel (`fusion_pressure_via`) uses UNIT (1.0) cores exclusively, so the
//! signed regime was never exercised; this test drives the primitive directly with signed data.
//!
//! BIT-EXACT: pure integer arithmetic, so the oracle replicates the kernel exactly 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`, accumulated with
//! wrapping u32 add. Any divergence is a real IR/dispatch defect, not a rounding artifact.
#![cfg(feature = "math")]

use vyre_primitives::math::tensor_train::tt_contract_step;
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

/// Bit-exact replica of the kernel's term multiply (the corrected SIGNED 16.16 multiply).
fn fixed_mul(a: u32, b: u32) -> u32 {
    ((i64::from(a as i32) * i64::from(b as i32)) >> 16) as i32 as u32
}

/// A signed 16.16 value in roughly `[-4.0, 4.0)`: an 18-bit magnitude, optionally negated (top bit
/// set on the negative half (the operand class the old unsigned multiply corrupted)).
fn signed_fixed(state: &mut u32) -> u32 {
    let magnitude = (xorshift(state) & 0x0003_FFFF) as i32; // [0.0, 4.0) in 16.16
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// Exact u32 16.16 oracle: `acc_out[b] = Σ_a acc_in[a]·core[a*r_next + b]`.
fn contract_fixed(acc_in: &[u32], core: &[u32], r_prev: usize, r_next: usize) -> Vec<u32> {
    (0..r_next)
        .map(|b| {
            let mut acc = 0u32;
            for a in 0..r_prev {
                acc = acc.wrapping_add(fixed_mul(acc_in[a], core[a * r_next + b]));
            }
            acc
        })
        .collect()
}

fn run_via_reference(acc_in: &[u32], core: &[u32], r_prev: u32, r_next: u32) -> Vec<u32> {
    let program = tt_contract_step("acc_in", "core", "acc_out", r_prev, r_next);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(acc_in)),
            Value::from(pack_u32(core)),
            Value::from(pack_u32(&vec![0u32; r_next as usize])),
        ],
    )
    .expect("tt_contract_step reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn tt_contract_signed_matches_exact_fixed_point_contraction() {
    let mut state = 0x7ED5_5D16u32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let r_prev = 1 + (case % 6) as usize; // 1..=6 incoming bond
        let r_next = 1 + ((case / 6) % 6) as usize; // 1..=6 outgoing bond
        let acc_in: Vec<u32> = (0..r_prev).map(|_| signed_fixed(&mut state)).collect();
        let core: Vec<u32> = (0..r_prev * r_next)
            .map(|_| signed_fixed(&mut state))
            .collect();

        neg_inputs += acc_in
            .iter()
            .chain(&core)
            .filter(|&&v| (v as i32) < 0)
            .count() as u32;

        let got = run_via_reference(&acc_in, &core, r_prev as u32, r_next as u32);
        let want = contract_fixed(&acc_in, &core, r_prev, r_next);
        assert_eq!(
            got, want,
            "case {case} (r_prev={r_prev} r_next={r_next}): SIGNED TT contraction _via {got:?} != \
             exact signed oracle {want:?} (acc_in={acc_in:?} core={core:?})"
        );

        if want.iter().any(|&v| v != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative core/accumulator entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed contractions must produce negative bond values, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 contractions were non-zero, the kernel is not being exercised"
    );
}

#[test]
fn tt_contract_hand_checked_signed() {
    // acc_in = [2.0, -1.0], core (r_prev=2 × r_next=2) = [[1.0, -0.5], [3.0, 2.0]]:
    //   acc_out[0] = (2.0)(1.0) + (-1.0)(3.0) = 2.0 - 3.0 = -1.0
    //   acc_out[1] = (2.0)(-0.5) + (-1.0)(2.0) = -1.0 - 2.0 = -3.0
    let acc_in = vec![to_fixed(2.0), to_fixed(-1.0)];
    let core = vec![to_fixed(1.0), to_fixed(-0.5), to_fixed(3.0), to_fixed(2.0)];
    let got = run_via_reference(&acc_in, &core, 2, 2);
    let want = contract_fixed(&acc_in, &core, 2, 2);
    assert_eq!(
        want,
        vec![to_fixed(-1.0), to_fixed(-3.0)],
        "sanity: signed TT contraction = [-1.0, -3.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched contraction must preserve sign: [-1.0, -3.0]"
    );
}
