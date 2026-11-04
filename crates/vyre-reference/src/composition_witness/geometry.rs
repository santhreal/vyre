//! Sequential mathematical witnesses for geometric and equivariant operations.

/// Sequential mathematical witness for Cl(2, 0) geometric product over `[s, e1, e2, e12]`.
///
/// Geometric product in Cl(2, 0) signature (+, +):
/// - s   = a_s * b_s + a_1 * b_1 + a_2 * b_2 - a_12 * b_12
/// - e1  = a_s * b_1 + a_1 * b_s - a_2 * b_12 + a_12 * b_2
/// - e2  = a_s * b_2 + a_2 * b_s + a_1 * b_12 - a_12 * b_1
/// - e12 = a_s * b_12 + a_12 * b_s + a_1 * b_2 - a_2 * b_1
#[must_use]
pub fn clifford2_product_witness(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    let [a_s, a_1, a_2, a_12] = a;
    let [b_s, b_1, b_2, b_12] = b;
    [
        a_s * b_s + a_1 * b_1 + a_2 * b_2 - a_12 * b_12,
        a_s * b_1 + a_1 * b_s - a_2 * b_12 + a_12 * b_2,
        a_s * b_2 + a_2 * b_s + a_1 * b_12 - a_12 * b_1,
        a_s * b_12 + a_12 * b_s + a_1 * b_2 - a_2 * b_1,
    ]
}

/// Sequential mathematical witness for SE(3)-equivariant scalar channel mixing.
///
/// Computes `out[i, co] = \sum_{ci} weights[co, ci] * features[i, ci]`.
/// Bounds: missing/short input features or weights are zero-padded.
#[must_use]
pub fn tfn_scalar_mix_witness(
    features: &[f64],
    weights: &[f64],
    n_nodes: u32,
    c_in: u32,
    c_out: u32,
) -> Vec<f64> {
    let n_nodes = n_nodes as usize;
    let c_in = c_in as usize;
    let c_out = c_out as usize;
    let mut out = vec![0.0; n_nodes * c_out];
    for i in 0..n_nodes {
        for co in 0..c_out {
            let mut acc = 0.0;
            for ci in 0..c_in {
                let weight = weights.get(co * c_in + ci).copied().unwrap_or(0.0);
                let feature = features.get(i * c_in + ci).copied().unwrap_or(0.0);
                acc += weight * feature;
            }
            out[i * c_out + co] = acc;
        }
    }
    out
}
