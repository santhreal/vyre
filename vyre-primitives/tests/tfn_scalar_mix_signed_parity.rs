//! GPU-IR parity for `geom/tfn::tfn_scalar_mix`: the tensor-field-network scalar (type-0) channel
//! mixing `out[i,co] = Σ_ci weights[co,ci]·features[i,ci]` in 16.16 fixed point, driven through
//! `vyre_reference::reference_eval` with SIGNED features and weights.
//!
//! Why this test exists: `tfn_scalar_mix` is a per-node learnable linear layer built on
//! `fixed_mul_16_16_expr`. Learned WEIGHTS are freely SIGNED (initialized around zero, trained to
//! either sign) and equivariant FEATURES carry both signs. Before the signed-multiply fix (BACKLOG
//! `FIXED-amg-fixed-path-unsigned-mul-negatives`) that multiply reconstructed the product from the
//! UNSIGNED high word, so a negative weight or feature (a u32 with the top bit set, read as ~2^32)
//! produced a garbage term, silently corrupting the mixed channel. The primitive had NO IR-execution
//! parity coverage (only an f64 `*_cpu` test), so this is the first faithful run of the mixing kernel
//! and it exercises the signed regime the network actually operates in.
//!
//! BIT-EXACT: pure integer arithmetic, so the oracle replicates the kernel exactly 
//! `fixed_mul(a,b) = ((a as i32 as i64 * b as i32 as i64) >> 16) as i32 as u32`, accumulated with
//! wrapping u32 add. Any divergence is a real IR/dispatch defect, not a rounding artifact.
#![cfg(feature = "geom")]

use vyre_primitives::geom::tfn::tfn_scalar_mix;
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

/// Exact u32 16.16 oracle: `out[i*c_out + co] = Σ_ci weights[co*c_in + ci]·features[i*c_in + ci]`.
fn tfn_mix_fixed(
    features: &[u32],
    weights: &[u32],
    n_nodes: usize,
    c_in: usize,
    c_out: usize,
) -> Vec<u32> {
    let mut out = vec![0u32; n_nodes * c_out];
    for i in 0..n_nodes {
        for co in 0..c_out {
            let mut acc = 0u32;
            for ci in 0..c_in {
                acc = acc.wrapping_add(fixed_mul(weights[co * c_in + ci], features[i * c_in + ci]));
            }
            out[i * c_out + co] = acc;
        }
    }
    out
}

fn run_via_reference(
    features: &[u32],
    weights: &[u32],
    n_nodes: u32,
    c_in: u32,
    c_out: u32,
) -> Vec<u32> {
    let program = tfn_scalar_mix("features", "weights", "out", n_nodes, c_in, c_out);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(features)),
            Value::from(pack_u32(weights)),
            Value::from(pack_u32(&vec![0u32; (n_nodes * c_out) as usize])),
        ],
    )
    .expect("tfn_scalar_mix reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn tfn_scalar_mix_signed_matches_exact_fixed_point_mix() {
    let mut state = 0x1CE1_C0DEu32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut moved = 0u32;
    for case in 0..400u32 {
        let n_nodes = 1 + (case % 4) as usize;
        let c_in = 1 + ((case / 4) % 4) as usize;
        let c_out = 1 + ((case / 16) % 4) as usize;

        let features: Vec<u32> = (0..n_nodes * c_in)
            .map(|_| signed_fixed(&mut state))
            .collect();
        let weights: Vec<u32> = (0..c_out * c_in)
            .map(|_| signed_fixed(&mut state))
            .collect();

        neg_inputs += features
            .iter()
            .chain(&weights)
            .filter(|&&v| (v as i32) < 0)
            .count() as u32;

        let got = run_via_reference(
            &features,
            &weights,
            n_nodes as u32,
            c_in as u32,
            c_out as u32,
        );
        let want = tfn_mix_fixed(&features, &weights, n_nodes, c_in, c_out);
        assert_eq!(
            got, want,
            "case {case} (n_nodes={n_nodes} c_in={c_in} c_out={c_out}): SIGNED tfn mix _via {got:?} \
             != exact signed oracle {want:?} (features={features:?} weights={weights:?})"
        );

        if want.iter().any(|&v| v != 0) {
            moved += 1;
        }
        neg_outputs += want.iter().filter(|&&v| (v as i32) < 0).count() as u32;
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative feature/weight entries, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "signed channel mixing must produce negative mixed features, got {neg_outputs}"
    );
    assert!(
        moved > 380,
        "only {moved}/400 mixes were non-zero, the kernel is not being exercised"
    );
}

#[test]
fn tfn_scalar_mix_hand_checked_signed() {
    // n_nodes=1, c_in=2, c_out=2; features = [2.0, -1.0];
    // weights (c_out × c_in) = [[1.0, 3.0], [-2.0, 0.5]]:
    //   out[0] (co=0) = (1.0)(2.0) + ( 3.0)(-1.0) = 2.0 - 3.0 = -1.0
    //   out[1] (co=1) = (-2.0)(2.0) + (0.5)(-1.0) = -4.0 - 0.5 = -4.5
    let features = vec![to_fixed(2.0), to_fixed(-1.0)];
    let weights = vec![to_fixed(1.0), to_fixed(3.0), to_fixed(-2.0), to_fixed(0.5)];
    let got = run_via_reference(&features, &weights, 1, 2, 2);
    let want = tfn_mix_fixed(&features, &weights, 1, 2, 2);
    assert_eq!(
        want,
        vec![to_fixed(-1.0), to_fixed(-4.5)],
        "sanity: signed tfn channel mix = [-1.0, -4.5]"
    );
    assert_eq!(
        got, want,
        "the dispatched channel mix must preserve sign: [-1.0, -4.5]"
    );
}
