//! Fixed-point-decode parity for `topology/simplicial::simplicial_triangle_message`, the 2-simplex
//! boundary-operator message `∂(triangle) = e_jk - e_ik + e_ij`: driven through
//! `vyre_reference::reference_eval` with SIGNED 16.16 edge features.
//!
//! Why this test exists (closes BACKLOG `OBSERVATION-topology-type-parity`): the primitive has a
//! representational split, the GPU IR buffers are raw `DataType::U32` (documented as 16.16
//! fixed-point) and the boundary op is U32 two's-complement `Expr::add`/`Expr::sub`, while the CPU
//! reference `simplicial_triangle_message_cpu` computes in `f64`. The existing coverage never bridges
//! the two: the `cpu_*` tests are f64-only and the one IR test (`ir_message_skips_malformed_triangle`)
//! feeds RAW integer u32 values (10/20/30) compared against a hand-computed u32, neither proves the
//! u32-16.16 IR and the f64 semantics AGREE once you decode 16.16 ↔ f64. This suite does exactly that:
//! it encodes SIGNED f64 features to 16.16, runs the real IR, decodes the u32 output back to f64, and
//! asserts BIT-FOR-BIT equality with the f64 boundary arithmetic that `simplicial_triangle_message_cpu`
//! performs (the inline `boundary_f64` oracle mirrors it exactly, including the OOB-triangle skip → 0).
//!
//! The ∂ operator is inherently SIGNED (`jk - ik + ij` turns negative for many inputs), so this is
//! also the first signed exercise of the primitive, a negative face feature is a u32 with the top bit
//! set, and the alternating-sign sum must decode to the correct negative f64.
//!
//! BIT-EXACT: all features are exact 16.16 values (`k / 65536` for a bounded integer `k`), so the u32
//! two's-complement sum `k_jk - k_ik + k_ij` (kept within i32 range) reinterpreted as i32 equals the
//! exact f64 boundary value (no tolerance).
#![cfg(feature = "topology")]

use vyre_primitives::topology::simplicial::simplicial_triangle_message;
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

const FIXED_ONE: f64 = 65536.0;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Encode an exact 16.16 f64 (a multiple of `1/65536`) to its u32 two's-complement word.
fn to_fixed(v: f64) -> u32 {
    (v * FIXED_ONE).round() as i64 as u32
}

/// Decode a 16.16 u32 word (two's-complement) back to f64.
fn from_fixed(u: u32) -> f64 {
    f64::from(u as i32) / FIXED_ONE
}

/// A signed exact-16.16 feature in roughly `[-32.0, 32.0)`: a bounded integer numerator over 65536,
/// optionally negated. The magnitude stays small enough that a 3-term boundary sum never leaves i32
/// range, so the u32 two's-complement result decodes to the exact signed f64.
fn signed_fixed_feature(state: &mut u32) -> u32 {
    // 21-bit magnitude over 65536 => up to ~32.0; three of these sum to < 2^23 numerator, well inside i32.
    let magnitude = (xorshift(state) & 0x001F_FFFF) as i32;
    if xorshift(state) & 1 == 0 {
        magnitude as u32
    } else {
        (-magnitude) as u32
    }
}

/// f64 boundary oracle mirroring `simplicial_triangle_message_cpu` EXACTLY: per triangle, decode the
/// three canonical faces and emit `e_jk - e_ik + e_ij`; a triangle referencing an out-of-range edge is
/// skipped (message left 0), matching both the CPU reference and the IR's in-range gate.
fn boundary_f64(
    edge_features_fp: &[u32],
    triangle_edges: &[u32],
    n_edges: u32,
    n_triangles: u32,
    d: u32,
) -> Vec<f64> {
    let n_edges = n_edges as usize;
    let n_triangles = n_triangles as usize;
    let d = d as usize;
    let mut out = vec![0.0f64; n_triangles * d];
    for tri in 0..n_triangles {
        let e_jk = triangle_edges[tri * 3] as usize;
        let e_ik = triangle_edges[tri * 3 + 1] as usize;
        let e_ij = triangle_edges[tri * 3 + 2] as usize;
        if e_jk >= n_edges || e_ik >= n_edges || e_ij >= n_edges {
            continue;
        }
        for k in 0..d {
            let jk = from_fixed(edge_features_fp[e_jk * d + k]);
            let ik = from_fixed(edge_features_fp[e_ik * d + k]);
            let ij = from_fixed(edge_features_fp[e_ij * d + k]);
            out[tri * d + k] = jk - ik + ij;
        }
    }
    out
}

fn run_via_reference(
    edge_features_fp: &[u32],
    triangle_edges: &[u32],
    n_edges: u32,
    n_triangles: u32,
    d: u32,
) -> Vec<f64> {
    let program = simplicial_triangle_message("e", "te", "tm", n_edges, n_triangles, d);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_u32_slice(edge_features_fp)),
            Value::from(pack_u32_slice(triangle_edges)),
            Value::from(pack_u32_slice(&vec![0u32; (n_triangles * d) as usize])),
        ],
    )
    .expect("simplicial_triangle_message reference evaluation must succeed");
    // Buffers: edge_features RO(0), triangle_edges RO(1), triangle_messages RW(2), the sole writable
    // buffer, so it is outputs[0] in binding order.
    decode_u32_le_bytes_all(&outputs[0].to_bytes())
        .into_iter()
        .map(from_fixed)
        .collect()
}

#[test]
fn simplicial_signed_16_16_ir_matches_f64_boundary_semantics() {
    let mut state = 0x51_3C_A1_07u32;
    let mut neg_inputs = 0u32;
    let mut neg_outputs = 0u32;
    let mut nonzero = 0u32;
    for case in 0..400u32 {
        let n_edges = 2 + (case % 6); // 2..7
        let n_triangles = 1 + ((case / 6) % 4); // 1..4
        let d = 1 + ((case / 24) % 3); // 1..3

        let edge_features_fp: Vec<u32> = (0..(n_edges * d))
            .map(|_| signed_fixed_feature(&mut state))
            .collect();
        // Every triangle references three IN-RANGE edges (0..n_edges), so the boundary op is exercised
        // (the malformed-skip path has its own case below).
        let triangle_edges: Vec<u32> = (0..(n_triangles * 3))
            .map(|_| xorshift(&mut state) % n_edges)
            .collect();

        neg_inputs += edge_features_fp.iter().filter(|&&w| (w as i32) < 0).count() as u32;

        let got = run_via_reference(&edge_features_fp, &triangle_edges, n_edges, n_triangles, d);
        let want = boundary_f64(&edge_features_fp, &triangle_edges, n_edges, n_triangles, d);
        assert_eq!(
            got, want,
            "case {case} (n_edges={n_edges} n_triangles={n_triangles} d={d}): decoded 16.16 IR must \
             equal the f64 boundary semantics bit-for-bit; features={edge_features_fp:?} \
             tris={triangle_edges:?}"
        );

        for &v in &want {
            if v != 0.0 {
                nonzero += 1;
            }
            if v < 0.0 {
                neg_outputs += 1;
            }
        }
    }
    assert!(
        neg_inputs > 500,
        "sweep must feed many negative 16.16 face features, got {neg_inputs}"
    );
    assert!(
        neg_outputs > 100,
        "the signed ∂ operator must produce negative messages, got {neg_outputs}"
    );
    assert!(
        nonzero > 380,
        "only {nonzero} non-zero messages, the boundary op is barely exercised"
    );
}

#[test]
fn simplicial_hand_checked_signed_boundary_and_malformed_skip() {
    // 3 edges, 2-D features, 2 triangles. Faces canonical order = (e_jk, e_ik, e_ij).
    //   edge 0 = [ 1.0, -4.0], edge 1 = [-2.0, 0.5], edge 2 = [ 3.0,  2.0]
    // tri 0 (e_jk=2, e_ik=1, e_ij=0):
    //   dim0: 3.0 - (-2.0) + 1.0 = 6.0 ; dim1: 2.0 - 0.5 + (-4.0) = -2.5
    // tri 1 malformed (e_jk=3 == n_edges, out of range) → skipped, both dims 0.0.
    let n_edges = 3u32;
    let n_triangles = 2u32;
    let d = 2u32;
    let edge_features_fp = [
        to_fixed(1.0),
        to_fixed(-4.0),
        to_fixed(-2.0),
        to_fixed(0.5),
        to_fixed(3.0),
        to_fixed(2.0),
    ];
    let triangle_edges = [2u32, 1, 0, 3, 0, 0];

    let got = run_via_reference(&edge_features_fp, &triangle_edges, n_edges, n_triangles, d);
    let want = boundary_f64(&edge_features_fp, &triangle_edges, n_edges, n_triangles, d);
    assert_eq!(
        want,
        vec![6.0, -2.5, 0.0, 0.0],
        "sanity: signed boundary tri0 = [6.0, -2.5], malformed tri1 skipped = [0.0, 0.0]"
    );
    assert_eq!(
        got, want,
        "the dispatched 16.16 IR must preserve sign and skip the malformed triangle: [6.0, -2.5, 0.0, 0.0]"
    );
}
