//! Two's-complement negation of u32 lanes, wrapping at zero.

use vyre_foundation::ir::{Expr, Program};

const OP_ID: &str = "vyre-libs::math::wrapping_neg";

/// Computes wrapping negation.
#[must_use]
pub fn wrapping_neg(a: &str, out: &str, size: u32) -> Program {
    crate::builder::elementwise::u32_elementwise_unary(OP_ID, a, out, size, |value| {
        Expr::sub(Expr::u32(0), value)
    })
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || wrapping_neg("a", "out", 4),
        Some(|| {
            let a = [0u32, 1, u32::MAX, 42];
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![to_bytes(&a)]]
        }),
        Some(|| {
            // [0, u32::MAX (from 1), 1 (from u32::MAX), 0xFFFF_FFD6 (from 42)]
            vec![vec![vec![
                0x00, 0x00, 0x00, 0x00, // 0
                0xff, 0xff, 0xff, 0xff, // u32::MAX (wrapping_neg of 1)
                0x01, 0x00, 0x00, 0x00, // 1 (wrapping_neg of u32::MAX)
                0xd6, 0xff, 0xff, 0xff, // 0xFFFF_FFD6 (wrapping_neg of 42)
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_u32;

    fn negated(input: &[u32]) -> Vec<u32> {
        let n = (input.len() as u32).max(1);
        eval_u32(
            "wrapping_neg",
            &wrapping_neg("input", "out", n),
            &[input],
            n as usize,
        )
    }

    #[test]
    fn generated_wrapping_neg_matches_rust_reference() {
        let mut state = 0x6E67_A71E_u32;
        for case in 0..2048u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (state % 65 + 1) as usize;
            let mut input = Vec::with_capacity(len);
            for lane in 0..len {
                state = state.rotate_left(5) ^ (lane as u32).wrapping_mul(0x9E37_79B9);
                input.push(match lane % 8 {
                    0 => 0,
                    1 => 1,
                    2 => u32::MAX,
                    3 => i32::MIN as u32,
                    4 => i32::MAX as u32,
                    _ => state,
                });
            }

            let expected = input
                .iter()
                .copied()
                .map(u32::wrapping_neg)
                .collect::<Vec<_>>();
            assert_eq!(
                negated(&input),
                expected,
                "generated wrapping-neg case {case}"
            );
        }
    }
}
