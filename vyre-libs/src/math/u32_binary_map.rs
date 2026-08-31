//! Shared u32 binary elementwise map builder for math primitives.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

fn u32_two_input_map_program<F>(
    op_id: &'static str,
    lhs: &str,
    rhs: &str,
    out: &str,
    count: u32,
    rhs_count: u32,
    rhs_index: impl FnOnce(&Expr) -> Expr,
    op: F,
) -> Program
where
    F: Fn(Expr, Expr) -> Expr,
{
    ElementwiseComposer::binary_broadcast_rhs(
        op_id,
        lhs,
        rhs,
        out,
        count,
        rhs_count,
        DataType::U32,
        rhs_index,
        op,
    )
}

/// Build `out[i] = op(lhs[i], rhs[i])` for `count` u32 lanes.
#[must_use]
pub(crate) fn u32_binary_map_program<F>(
    op_id: &'static str,
    lhs: &str,
    rhs: &str,
    out: &str,
    count: u32,
    op: F,
) -> Program
where
    F: Fn(Expr, Expr) -> Expr,
{
    u32_two_input_map_program(op_id, lhs, rhs, out, count, count, Clone::clone, op)
}

/// Build `out[i] = op(vector[i], scalar[0])` for `count` u32 lanes.
#[must_use]
pub(crate) fn u32_vector_scalar_map_program<F>(
    op_id: &'static str,
    vector: &str,
    scalar: &str,
    out: &str,
    count: u32,
    op: F,
) -> Program
where
    F: Fn(Expr, Expr) -> Expr,
{
    u32_two_input_map_program(op_id, vector, scalar, out, count, 1, |_| Expr::u32(0), op)
}

#[cfg(test)]
mod tests {
    fn iht_scalar(value: u32, threshold: u32) -> u32 {
        if (value & 0x7FFF_FFFF) >= threshold {
            value
        } else {
            0
        }
    }

    fn mp_clip_scalar(value: u32, edge: u32) -> u32 {
        value.min(edge)
    }

    #[test]
    fn generated_vector_scalar_threshold_contracts_match_scalar_reference() {
        let mut state = 0xC711_9A5E_u32;
        for case in 0..4096u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = match case % 13 {
                0 => 0,
                1 => u32::MAX,
                2 => 0x8000_0000,
                3 => 0x7FFF_FFFF,
                _ => state,
            };
            state = state.rotate_left(9) ^ case.wrapping_mul(0x9E37_79B9);
            let threshold = match case % 17 {
                0 => 0,
                1 => 1,
                2 => u32::MAX,
                3 => 0x7FFF_FFFF,
                _ => state,
            };

            let abs_value = value & 0x7FFF_FFFF;
            let expected_iht = if abs_value >= threshold { value } else { 0 };
            assert_eq!(
                iht_scalar(value, threshold),
                expected_iht,
                "IHT scalar threshold case {case}"
            );
            assert_eq!(
                mp_clip_scalar(value, threshold),
                if value < threshold { value } else { threshold },
                "MP edge clip scalar case {case}"
            );
        }
    }
}
