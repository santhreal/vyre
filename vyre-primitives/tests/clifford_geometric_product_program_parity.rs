//! GPU-IR parity for the Cl(2,0) geometric-product kernel `geom/clifford2_product`, driven through
//! `vyre_reference::reference_eval` with SIGNED (negative) multivector components.
//!
//! Why this test exists: `geom/clifford.rs` ships the `clifford2_product` Program builder + a
//! `clifford2_product_cpu` f64 reference, but its only inline `#[cfg(test)]` coverage exercises the CPU
//! reference, the GPU IR had ZERO parity coverage, and NONE with negative components. The geometric
//! product is sign-mixing (`out_s = …− a₁₂·b₁₂`, `out_e1 = …− a₂·b₁₂`, …), so with negative components the
//! per-term products feed `fixed_mul_16_16_expr` NEGATIVE operands, exactly the class of value that the
//! old UNSIGNED fixed multiply corrupted (see BACKLOG `FIXED-amg-fixed-path-unsigned-mul-negatives`). This
//! is the SECOND kernel (after the AMG V-cycle) proving the signed-multiply fix, and the first to lock
//! Clifford's GPU IR at all.
//!
//! BIT-EXACT (no tolerance): every component is a multiple of 0.5, so each product is an exact multiple of
//! 0.25 and every sum stays well below 2^23, exactly representable in 16.16. The fixed IR must therefore
//! reproduce the f64 reference to the BIT.
#![cfg(feature = "geom")]

use vyre_primitives::geom::clifford::clifford2_product;
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Encode a signed half-integer f64 as two's-complement 16.16.
fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

/// Decode a two's-complement 16.16 word to the signed value it encodes.
fn from_fixed(v: u32) -> f64 {
    f64::from(v as i32) / FIXED_ONE
}

/// A signed half-integer in {-3, -2.5, …, 3}.
fn signed_half(state: &mut u32) -> f64 {
    let steps = (xorshift(state) % 13) as i32 - 6; // -6..=6
    0.5 * f64::from(steps)
}

/// Inline f64 Cl(2,0) geometric product, the authoritative reference (identical to
/// `clifford2_product_cpu`, re-stated here so the test needs no `cpu-parity` feature). Layout per pair is
/// `[s, e1, e2, e12]`.
fn clifford_product_f64(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [a_s, a1, a2, a12] = a;
    let [b_s, b1, b2, b12] = b;
    [
        a_s * b_s + a1 * b1 + a2 * b2 - a12 * b12,
        a_s * b1 + a1 * b_s - a2 * b12 + a12 * b2,
        a_s * b2 + a2 * b_s + a1 * b12 - a12 * b1,
        a_s * b12 + a12 * b_s + a1 * b2 - a2 * b1,
    ]
}

fn run_via_reference(lhs: &[u32], rhs: &[u32], n_pairs: u32) -> Vec<u32> {
    let program = clifford2_product("lhs", "rhs", "out", n_pairs);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32(lhs)),
            Value::from(pack_u32(rhs)),
            Value::from(pack_u32(&vec![0u32; (n_pairs * 4) as usize])),
        ],
    )
    .expect("clifford2_product reference evaluation must succeed");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn clifford_geometric_product_bit_exact_over_signed_components() {
    let mut state = 0xC1_1F_00_01u32;
    let mut any_negative_component = 0u32;
    let mut any_negative_output = 0u32;
    for case in 0..300u32 {
        let n_pairs = 1 + case % 4; // 1..4 independent multivector products per dispatch
        let mut lhs_f = Vec::new();
        let mut rhs_f = Vec::new();
        for _ in 0..n_pairs {
            for _ in 0..4 {
                lhs_f.push(signed_half(&mut state));
            }
            for _ in 0..4 {
                rhs_f.push(signed_half(&mut state));
            }
        }
        let lhs_fx: Vec<u32> = lhs_f.iter().map(|&v| to_fixed(v)).collect();
        let rhs_fx: Vec<u32> = rhs_f.iter().map(|&v| to_fixed(v)).collect();

        let got = run_via_reference(&lhs_fx, &rhs_fx, n_pairs);
        assert_eq!(
            got.len(),
            (n_pairs * 4) as usize,
            "case {case}: output length"
        );

        for pair in 0..n_pairs as usize {
            let a = [
                lhs_f[pair * 4],
                lhs_f[pair * 4 + 1],
                lhs_f[pair * 4 + 2],
                lhs_f[pair * 4 + 3],
            ];
            let b = [
                rhs_f[pair * 4],
                rhs_f[pair * 4 + 1],
                rhs_f[pair * 4 + 2],
                rhs_f[pair * 4 + 3],
            ];
            let want = clifford_product_f64(a, b);
            for k in 0..4 {
                let got_word = got[pair * 4 + k];
                let want_word = to_fixed(want[k]);
                assert_eq!(
                    got_word, want_word,
                    "case {case} pair {pair} comp {k}: signed fixed Clifford product must be BIT-EXACT \
                     to the f64 reference; got={} want={} (a={a:?} b={b:?})",
                    from_fixed(got_word),
                    want[k]
                );
                if a[k] < 0.0 || b[k] < 0.0 {
                    any_negative_component += 1;
                }
                if from_fixed(got_word) < 0.0 {
                    any_negative_output += 1;
                }
            }
        }
    }
    assert!(
        any_negative_component > 400,
        "sweep must feed negative components (the signed-mul regime), got {any_negative_component}"
    );
    assert!(
        any_negative_output > 200,
        "sweep must produce negative product components, got {any_negative_output}"
    );
}

#[test]
fn clifford_geometric_product_hand_checked_identities() {
    // e1·e1 = 1 (scalar): a = b = [0, 1, 0, 0] → out_s = 0+1+0-0 = 1.
    let e1 = [0.0, 1.0, 0.0, 0.0];
    let got = run_via_reference(
        &e1.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        &e1.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        1,
    );
    let got_f: Vec<f64> = got.iter().map(|&v| from_fixed(v)).collect();
    assert_eq!(got_f, vec![1.0, 0.0, 0.0, 0.0], "e1·e1 = scalar 1");

    // e1·e2 = e12: [0,1,0,0]·[0,0,1,0] → out_e12 = a1·b2 = 1.
    let e2 = [0.0, 0.0, 1.0, 0.0];
    let got = run_via_reference(
        &e1.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        &e2.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        1,
    );
    let got_f: Vec<f64> = got.iter().map(|&v| from_fixed(v)).collect();
    assert_eq!(got_f, vec![0.0, 0.0, 0.0, 1.0], "e1·e2 = e12");

    // e2·e1 = −e12 (anticommute): NEGATIVE output component (the signed-mul regime).
    let got = run_via_reference(
        &e2.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        &e1.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        1,
    );
    let got_f: Vec<f64> = got.iter().map(|&v| from_fixed(v)).collect();
    assert_eq!(
        got_f,
        vec![0.0, 0.0, 0.0, -1.0],
        "e2·e1 = −e12 (a correct NEGATIVE component proves the signed fixed multiply)"
    );

    // A negative scalar times a vector: (−2)·(3 e1) = −6 e1.
    let neg_scalar = [-2.0, 0.0, 0.0, 0.0];
    let vec3e1 = [0.0, 3.0, 0.0, 0.0];
    let got = run_via_reference(
        &neg_scalar.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        &vec3e1.iter().map(|&v| to_fixed(v)).collect::<Vec<_>>(),
        1,
    );
    let got_f: Vec<f64> = got.iter().map(|&v| from_fixed(v)).collect();
    assert_eq!(got_f, vec![0.0, -6.0, 0.0, 0.0], "(−2)·(3e1) = −6 e1");
}
