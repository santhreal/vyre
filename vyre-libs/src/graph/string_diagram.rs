//! String diagram compilation primitive (#53).
//!
//! String diagrams (Selinger 2010, Coecke-Kissinger ZX) are the visual
//! language of monoidal categories  -  a generalized tensor network.
//! Recent work (Patterson 2022 DisCoPy) compiles them to numeric
//! tensor contractions.
//!
//! This file ships the **monoidal composition step** primitive  -
//! sequential composition `g · f` of two morphisms encoded as small
//! tensors `f: A → B` and `g: B → C`, producing `g · f: A → C`. This
//! is matrix multiplication with categorical intent carried in the
//! stable op id.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::math::fixed_u32_matmul::{try_fixed_u32_matmul, FixedMatmulContext};

/// Op id.
pub const OP_ID: &str = "vyre-libs::graph::monoidal_compose";

const MATMUL_CONTEXT: FixedMatmulContext = FixedMatmulContext {
    op_id: OP_ID,
    operation: "monoidal_compose",
    lhs_label: "monoidal_compose f input",
    rhs_label: "monoidal_compose g input",
    out_label: "monoidal_compose output",
    dimensions: ["a", "b", "c"],
};

/// Sequential composition step. Same shape as
/// `crate::math::tensor_network::tn_pair_contract`; ships under graph because
/// string diagrams are graphs of morphisms.
#[must_use]
pub fn monoidal_compose(f: &str, g: &str, out: &str, a: u32, b: u32, c: u32) -> Program {
    match try_monoidal_compose(f, g, out, a, b, c) {
        Ok(program) => program,
        Err(error) => trap_program(OP_ID, Some((out, DataType::U32)), error),
    }
}

/// Sequential composition step with checked tensor cell counts.
pub fn try_monoidal_compose(
    f: &str,
    g: &str,
    out: &str,
    a: u32,
    b: u32,
    c: u32,
) -> Result<Program, String> {
    try_fixed_u32_matmul(f, g, out, a, b, c, &MATMUL_CONTEXT)
}

const EXPECTED_STRING_DIAGRAM_OUTPUT_BYTES: [u8; 16] =
    [0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 7, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || monoidal_compose("f", "g", "out", 2, 2, 2),
        Some(|| {
            let one = 1u32 << 16;
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[one, 0, 0, one]),
                vyre_primitives::wire::pack_u32_slice(&[2 * one, 3 * one, 5 * one, 7 * one]),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_STRING_DIAGRAM_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_laws(&["associative", "identity"])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_monoidal_compose_cpu_into(
        f: &[f64],
        g: &[f64],
        a: u32,
        b: u32,
        c: u32,
        out: &mut Vec<f64>,
    ) -> Result<(), String> {
        let out_len = (a as usize).checked_mul(c as usize).ok_or_else(|| {
            "Fix: monoidal_compose CPU output cell count overflowed u32.".to_owned()
        })?;
        if out_len > 1_000_000_000 {
            return Err("Fix: monoidal_compose CPU output reserve exceeded maximum.".to_owned());
        }
        vyre_reference::composition_witness::dense_matrix_multiply_witness_into(
            f, g, a as usize, b as usize, c as usize, out,
        );
        Ok(())
    }

    fn monoidal_compose_cpu(f: &[f64], g: &[f64], a: u32, b: u32, c: u32) -> Vec<f64> {
        let mut out = Vec::new();
        try_monoidal_compose_cpu_into(f, g, a, b, c, &mut out)
            .expect("monoidal_compose_cpu failed");
        out
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_identity_compose_passthrough() {
        let f = vec![1.0, 2.0, 3.0, 4.0];
        let i = vec![1.0, 0.0, 0.0, 1.0];
        let out = monoidal_compose_cpu(&f, &i, 2, 2, 2);
        assert_eq!(out, f);
    }

    #[test]
    fn cpu_short_inputs_are_zero_padded() {
        let out = monoidal_compose_cpu(&[2.0], &[3.0, 4.0], 1, 2, 2);
        assert_eq!(out, vec![6.0, 8.0]);
    }

    #[test]
    fn checked_cpu_ref_reuses_output_and_truncates_stale_tail() {
        let mut out = Vec::with_capacity(4);
        out.extend_from_slice(&[99.0, 98.0, 97.0, 96.0]);
        let capacity = out.capacity();

        try_monoidal_compose_cpu_into(&[2.0, 3.0], &[5.0, 7.0, 11.0, 13.0], 1, 2, 2, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - checked CPU oracle should reuse caller-owned storage");

        assert_eq!(out.len(), 2);
        assert!(approx_eq(out[0], 43.0));
        assert!(approx_eq(out[1], 53.0));
        assert_eq!(out.capacity(), capacity);

        try_monoidal_compose_cpu_into(&[4.0], &[6.0], 1, 1, 1, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - checked CPU oracle should truncate stale output cells");

        assert_eq!(out, vec![24.0]);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn checked_cpu_ref_preserves_output_on_reservation_failure() {
        let mut out = vec![1.0, 2.0, 3.0];
        let err = try_monoidal_compose_cpu_into(&[], &[], u32::MAX, 1, u32::MAX, &mut out)
            .expect_err("checked CPU oracle must reject impossible output reservations");

        assert!(
            err.contains("monoidal_compose CPU output") || err.contains("reserve"),
            "error should describe output reservation failure: {err}"
        );
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn cpu_associativity_holds() {
        // (h · g) · f = h · (g · f)
        let f = vec![1.0, 2.0]; // 1x2
        let g = vec![3.0, 4.0]; // 2x1
        let h = vec![5.0]; // 1x1
        let lhs_inner = monoidal_compose_cpu(&f, &g, 1, 2, 1); // 1x1
        let lhs = monoidal_compose_cpu(&lhs_inner, &h, 1, 1, 1); // 1x1
        let rhs_inner = monoidal_compose_cpu(&g, &h, 2, 1, 1); // 2x1
        let rhs = monoidal_compose_cpu(&f, &rhs_inner, 1, 2, 1); // 1x1
        assert!(approx_eq(lhs[0], rhs[0]));
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = monoidal_compose("f", "g", "h", 2, 3, 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        assert_eq!(p.buffers[0].count(), 6);
        assert_eq!(p.buffers[1].count(), 12);
        assert_eq!(p.buffers[2].count(), 8);
    }

    #[test]
    fn zero_a_traps() {
        let p = monoidal_compose("f", "g", "h", 0, 1, 1);
        assert!(p.stats().trap());
    }

    #[test]
    fn checked_monoidal_compose_rejects_zero_dimension() {
        let error = try_monoidal_compose("f", "g", "h", 0, 1, 1)
            .expect_err("checked monoidal compose builder must reject zero dimensions");

        assert!(
            error.contains("requires a, b, c > 0"),
            "error should describe the invalid tensor shape: {error}"
        );
    }

    #[test]
    fn checked_monoidal_compose_rejects_output_cell_overflow() {
        let error = try_monoidal_compose("f", "g", "h", u32::MAX, 1, 2)
            .expect_err("checked monoidal compose builder must reject output overflow");

        assert!(
            error.contains("monoidal_compose output shape")
                && error.contains("overflows the u32 cell count"),
            "error should name the operand and the shape that overflowed: {error}"
        );
    }

    #[test]
    fn legacy_monoidal_compose_does_not_panic_on_output_cell_overflow() {
        let program = monoidal_compose("f", "g", "h", u32::MAX, 1, 2);

        assert!(program.stats().trap());
    }
}
