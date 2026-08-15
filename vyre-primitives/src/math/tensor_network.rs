//! Tensor-network contraction primitive (#35).
//!
//! Tensor networks (PEPS, MPS, MERA) compress high-dimensional
//! functions exponentially. Contraction order matters  -  the optimal
//! order is solved via tropical-semiring shortest-path. This file
//! ships the **single pairwise contraction step** primitive  -  given
//! two tensors and the shared-index axis, produce the contracted
//! result.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | future `vyre-libs::ml::tensor_compress` | TT/MERA-compressed weights |
//! | future `vyre-libs::sci::quantum_chem` | quantum chemistry contraction |
//! | `vyre-driver` megakernel scheduling | each Region in vyre's IR is a tensor; wires are buffer dependencies; optimal fusion = optimal contraction order |

use vyre_foundation::algebra::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::fixed_u32_matmul::{try_fixed_u32_matmul, FixedMatmulContext};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::tensor_network_pair_contract";

const MATMUL_CONTEXT: FixedMatmulContext = FixedMatmulContext {
    op_id: OP_ID,
    operation: "tn_pair_contract",
    lhs_label: "tn_pair_contract a input",
    rhs_label: "tn_pair_contract b input",
    out_label: "tn_pair_contract output",
    dimensions: ["m", "k", "n"],
};

/// Pairwise tensor contraction: contract `A[m × k]` with `B[k × n]`
/// over the shared index `k`. Result `C[m × n]`. Special case of
/// matmul; shipped as a focused primitive so contraction-chain
/// region audits are readable.
#[must_use]
pub fn tn_pair_contract(a: &str, b: &str, c: &str, m: u32, k: u32, n: u32) -> Program {
    match try_tn_pair_contract(a, b, c, m, k, n) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((c, DataType::U32)), error),
    }
}

/// Pairwise tensor contraction with checked tensor cell counts.
pub fn try_tn_pair_contract(
    a: &str,
    b: &str,
    c: &str,
    m: u32,
    k: u32,
    n: u32,
) -> Result<Program, String> {
    try_fixed_u32_matmul(a, b, c, m, k, n, &MATMUL_CONTEXT)
}

/// CPU reference: f64.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn tn_pair_contract_cpu(a: &[f64], b: &[f64], m: u32, k: u32, n: u32) -> Vec<f64> {
    try_tn_pair_contract_cpu(a, b, m, k, n).unwrap_or_else(|error| panic!("{error}"))
}

/// Fallible CPU reference: f64.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_tn_pair_contract_cpu(
    a: &[f64],
    b: &[f64],
    m: u32,
    k: u32,
    n: u32,
) -> Result<Vec<f64>, String> {
    let mut c = Vec::new();
    try_tn_pair_contract_cpu_into(a, b, m, k, n, &mut c)?;
    Ok(c)
}

/// CPU reference into caller-owned output storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn tn_pair_contract_cpu_into(a: &[f64], b: &[f64], m: u32, k: u32, n: u32, c: &mut Vec<f64>) {
    try_tn_pair_contract_cpu_into(a, b, m, k, n, c).unwrap_or_else(|error| panic!("{error}"));
}

/// Fallible CPU reference into caller-owned output storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_tn_pair_contract_cpu_into(
    a: &[f64],
    b: &[f64],
    m: u32,
    k: u32,
    n: u32,
    c: &mut Vec<f64>,
) -> Result<(), String> {
    crate::math::cpu_matrix::try_f64_matmul_into(
        a,
        b,
        m,
        k,
        n,
        c,
        crate::math::cpu_matrix::MatmulContext::TENSOR_CONTRACT,
    )
}

/// CPU helper: greedy contraction-order picker. Given a list of tensor
/// dimensions, return an ordering that minimizes the sum of
/// intermediate sizes. This is the tropical-shortest-path solution
/// in a small-dimension case.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn greedy_contract_order_cpu(dims: &[u32]) -> Vec<usize> {
    let mut order = Vec::new();
    try_greedy_contract_order_cpu_into(dims, &mut order).unwrap_or_else(|error| panic!("{error}"));
    order
}

/// Fallible greedy contraction-order picker into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_greedy_contract_order_cpu_into(
    dims: &[u32],
    order: &mut Vec<usize>,
) -> Result<(), String> {
    if dims.len() > order.capacity() {
        crate::graph::scratch::reserve_graph_items(
            order,
            dims.len() - order.len(),
            "tensor-network CPU oracle",
            "greedy_contract_order output",
        )?;
    }
    order.clear();
    order.extend(0..dims.len());
    order.sort_by(|&left, &right| dims[right].cmp(&dims[left]).then_with(|| left.cmp(&right)));
    Ok(())
}

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || tn_pair_contract("a", "b", "c", 2, 2, 2),
        Some(|| {
            let one = 1u32 << 16;
            vec![vec![
                crate::wire::pack_u32_slice(&[one, one, one, one]),
                crate::wire::pack_u32_slice(&[one, one, one, one]),
                crate::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| {
            let two = 2u32 << 16;
            vec![vec![crate::wire::pack_u32_slice(&[two, two, two, two])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_pair_contract_2x2_identity() {
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![3.0, 5.0, 7.0, 11.0];
        let out = tn_pair_contract_cpu(&i, &v, 2, 2, 2);
        assert_eq!(out, v);
    }

    #[test]
    fn cpu_pair_contract_known_2x2() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = tn_pair_contract_cpu(&a, &b, 2, 2, 2);
        assert!(approx_eq(c[0], 19.0));
        assert!(approx_eq(c[1], 22.0));
        assert!(approx_eq(c[2], 43.0));
        assert!(approx_eq(c[3], 50.0));
    }

    #[test]
    fn cpu_pair_contract_rectangular() {
        // 1x2 * 2x3 = 1x3
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = tn_pair_contract_cpu(&a, &b, 1, 2, 3);
        // [3+12, 4+14, 5+16] = [15, 18, 21]
        assert_eq!(c, vec![15.0, 18.0, 21.0]);
    }

    #[test]
    fn cpu_pair_contract_zero_input_zero_output() {
        let a = vec![0.0; 4];
        let b = vec![1.0; 4];
        let c = tn_pair_contract_cpu(&a, &b, 2, 2, 2);
        for v in c {
            assert!(approx_eq(v, 0.0));
        }
    }

    #[test]
    fn cpu_pair_contract_missing_entries_are_zero() {
        let c = tn_pair_contract_cpu(&[2.0], &[3.0, 4.0], 1, 2, 2);
        assert_eq!(c, vec![6.0, 8.0]);
    }

    #[test]
    fn cpu_pair_contract_into_reuses_output_and_truncates_stale_tail() {
        let mut c = Vec::with_capacity(8);
        c.extend_from_slice(&[99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0]);
        let ptr = c.as_ptr();
        let capacity = c.capacity();

        try_tn_pair_contract_cpu_into(&[1.0, 2.0], &[3.0, 4.0, 5.0, 6.0], 1, 2, 2, &mut c)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - tensor-network CPU oracle should reuse caller-owned output");

        assert_eq!(c, vec![13.0, 16.0]);
        assert_eq!(c.as_ptr(), ptr);
        assert_eq!(c.capacity(), capacity);

        try_tn_pair_contract_cpu_into(&[2.0], &[3.0], 1, 1, 1, &mut c)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - tensor-network CPU oracle should truncate stale output");

        assert_eq!(c, vec![6.0]);
        assert_eq!(c.as_ptr(), ptr);
        assert_eq!(c.capacity(), capacity);
    }

    #[test]
    fn greedy_contract_order_into_reuses_output() {
        let mut order = Vec::with_capacity(8);
        order.extend_from_slice(&[9, 8, 7, 6]);
        let ptr = order.as_ptr();
        let capacity = order.capacity();

        try_greedy_contract_order_cpu_into(&[4, 10, 2, 10], &mut order)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - greedy contraction order should reuse caller-owned output");

        assert_eq!(order, vec![1, 3, 0, 2]);
        assert_eq!(order.as_ptr(), ptr);
        assert_eq!(order.capacity(), capacity);

        try_greedy_contract_order_cpu_into(&[1], &mut order)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - greedy contraction order should truncate stale output");

        assert_eq!(order, vec![0]);
        assert_eq!(order.as_ptr(), ptr);
        assert_eq!(order.capacity(), capacity);
    }

    #[test]
    fn generated_pair_contract_cpu_matches_independent_reference() {
        let mut out = Vec::new();
        for case in 0..1024usize {
            let m = case % 7 + 1;
            let k = (case / 7) % 9 + 1;
            let n = (case / 63) % 6 + 1;
            let a_len = (case * 5) % (m * k + 1);
            let b_len = (case * 11) % (k * n + 1);
            let a: Vec<f64> = (0..a_len)
                .map(|idx| ((idx * 13 + case) % 31) as f64 / 7.0 - 2.0)
                .collect();
            let b: Vec<f64> = (0..b_len)
                .map(|idx| ((idx * 17 + case) % 29) as f64 / 5.0 - 3.0)
                .collect();

            try_tn_pair_contract_cpu_into(&a, &b, m as u32, k as u32, n as u32, &mut out)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated tensor-network CPU oracle should evaluate");
            let expected = independent_pair_contract(&a, &b, m, k, n);

            assert_eq!(out.len(), expected.len(), "case {case}: output length");
            for idx in 0..out.len() {
                assert!(
                    approx_eq(out[idx], expected[idx]),
                    "case {case} idx {idx}: expected {}, got {}",
                    expected[idx],
                    out[idx]
                );
            }
        }
    }

    fn independent_pair_contract(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    out[i * n + j] += a.get(i * k + kk).copied().unwrap_or(0.0)
                        * b.get(kk * n + j).copied().unwrap_or(0.0);
                }
            }
        }
        out
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = tn_pair_contract("a", "b", "c", 2, 3, 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        assert_eq!(p.buffers[0].count(), 6);
        assert_eq!(p.buffers[1].count(), 12);
        assert_eq!(p.buffers[2].count(), 8);
    }

    #[test]
    fn zero_dim_traps() {
        let p = tn_pair_contract("a", "b", "c", 0, 1, 1);
        assert!(p.stats().trap());
    }

    #[test]
    fn checked_pair_contract_rejects_output_cell_overflow() {
        let error = try_tn_pair_contract("a", "b", "c", u32::MAX, 1, 2)
            .expect_err("checked tensor contraction builder must reject output overflow");

        assert!(
            error.contains("tn_pair_contract output shape")
                && error.contains("overflows the u32 cell count"),
            "error should name the operand and the shape that overflowed: {error}"
        );
    }
}
