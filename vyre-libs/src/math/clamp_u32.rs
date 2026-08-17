//! Cat-A `clamp_u32`  -  element-wise `x.clamp(lo, hi)`.
//!
//! This is a pure composition of `Expr::min` and `Expr::max`; both are
//! existing `BinOp` primitives and require no dedicated target lowering.
//!
//! Signature takes three buffers + one output  -  the binary helper
//! doesn't fit, so the Program is constructed inline (still wrapped in
//! a `Node::Region` per the Region chain invariant).
//!
//! CPU reference: `u32::clamp` bit-exact.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::ir::{DataType, Expr, Program};

const OP_ID: &str = "vyre-libs::math::clamp_u32";

/// Map `out[i] = input[i].clamp(lo[i], hi[i])` over n elements.
#[must_use]
pub fn clamp_u32(input: &str, lo: &str, hi: &str, out: &str, n: u32) -> Program {
    ElementwiseComposer::ternary(
        OP_ID,
        input,
        lo,
        hi,
        DataType::U32,
        out,
        DataType::U32,
        n,
        |x, l, h| Expr::min(Expr::max(x, l), h),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || clamp_u32("input", "lo", "hi", "out", 4),
        Some(|| {
            let input = [0u32, 5, 10, u32::MAX];
            let lo = [3u32, 3, 3, 100];
            let hi = [8u32, 8, 8, 200];
            let to_bytes = vyre_primitives::wire::pack_u32_slice;
            vec![vec![to_bytes(&input), to_bytes(&lo), to_bytes(&hi)]]
        }),
        Some(|| {
            // [3, 5, 8, 200]
            vec![vec![vec![
                0x03, 0x00, 0x00, 0x00, // 3
                0x05, 0x00, 0x00, 0x00, // 5
                0x08, 0x00, 0x00, 0x00, // 8
                0xc8, 0x00, 0x00, 0x00, // 200
            ]]]
        }),
    )
    .with_category("math")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn run(input: &[u32], lo: &[u32], hi: &[u32]) -> Vec<u32> {
        let n = input.len() as u32;
        let program = clamp_u32("input", "lo", "hi", "out", n.max(1));
        let to_bytes = vyre_primitives::wire::pack_u32_slice;
        let inputs = vec![
            Value::Bytes(to_bytes(input).into()),
            Value::Bytes(to_bytes(lo).into()),
            Value::Bytes(to_bytes(hi).into()),
            Value::Bytes(vec![0u8; (n.max(1) * 4) as usize].into()),
        ];
        let outputs = vyre_reference::reference_eval(&program, &inputs)
            .expect("Fix: clamp_u32 must run; restore this invariant before continuing.");
        let raw = outputs[0].to_bytes();
        vyre_primitives::wire::decode_u32_le_bytes_all(&raw)
    }

    #[test]
    fn matches_rust_ref_small() {
        let input = [0u32, 5, 10, u32::MAX];
        let lo = [3u32, 3, 3, 100];
        let hi = [8u32, 8, 8, 200];
        let got = run(&input, &lo, &hi);
        let expected: Vec<u32> = input
            .iter()
            .zip(lo.iter())
            .zip(hi.iter())
            .map(|((&x, &l), &h)| x.clamp(l, h))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn passthrough_when_in_range() {
        let input = [5u32];
        let lo = [0u32];
        let hi = [10u32];
        assert_eq!(run(&input, &lo, &hi), vec![5]);
    }
}
