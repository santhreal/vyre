//! Elementwise logical operations (nand, nor).
//!
//! `and`, `or`, and `xor` are registered once, in `crate::bitset`.
//! Only the synthesized combinations with no single-kernel equivalent live here.

use crate::builder::elementwise::u32_elementwise_binary;
use vyre_foundation::define_dialect;
use vyre_foundation::dialect_lookup::{Signature, TypedParam};
use vyre_foundation::ir::Program;
use vyre_foundation::operation::OperationTier;

const LOGICAL_BINARY_SIG: Signature = Signature {
    inputs: &[
        TypedParam {
            name: "a",
            ty: "buffer<u32>",
        },
        TypedParam {
            name: "b",
            ty: "buffer<u32>",
        },
    ],
    outputs: &[TypedParam {
        name: "out",
        ty: "buffer<u32>",
    }],
    attrs: &[],
    bytes_extraction: false,
};

/// Build the synthesized bitwise NAND operation.
#[must_use]
pub fn nand(a: &str, b: &str, out: &str, size: u32) -> Program {
    u32_elementwise_binary(
        "vyre-libs::logical::nand",
        a,
        b,
        out,
        size,
        |left, right| {
            vyre_foundation::ir::Expr::bitnot(vyre_foundation::ir::Expr::bitand(left, right))
        },
    )
}

/// Build the synthesized bitwise NOR operation.
#[must_use]
pub fn nor(a: &str, b: &str, out: &str, size: u32) -> Program {
    u32_elementwise_binary("vyre-libs::logical::nor", a, b, out, size, |left, right| {
        vyre_foundation::ir::Expr::bitnot(vyre_foundation::ir::Expr::bitor(left, right))
    })
}

fn nand_test_inputs() -> Vec<Vec<Vec<u8>>> {
    let a = [0xFF00_FF00u32, 0x00FF_00FF, 0xFFFF_FFFF, 0x0000_0000];
    let b = [0xF0F0_F0F0u32, 0x0F0F_0F0F, 0xFFFF_FFFF, 0x0000_0000];
    let to_bytes = vyre_primitives::wire::pack_u32_slice;
    vec![vec![to_bytes(&a), to_bytes(&b)]]
}

fn nand_expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![
        0xFF, 0x0F, 0xFF, 0x0F, 0xF0, 0xFF, 0xF0, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
        0xFF,
    ]]]
}

fn nor_test_inputs() -> Vec<Vec<Vec<u8>>> {
    let a = [0xFF00_FF00u32, 0x00FF_00FF, 0xFFFF_FFFF, 0x0000_0000];
    let b = [0xF0F0_F0F0u32, 0x0F0F_0F0F, 0xFFFF_FFFF, 0x0000_0000];
    let to_bytes = vyre_primitives::wire::pack_u32_slice;
    vec![vec![to_bytes(&a), to_bytes(&b)]]
}

fn nor_expected_output() -> Vec<Vec<Vec<u8>>> {
    vec![vec![vec![
        0x0F, 0x00, 0x0F, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
        0xFF,
    ]]]
}

define_dialect! {
    /// Declarative logical operations dialect.
    dialect: "vyre-libs::logical",
    name: dialect,
    visitor: LogicalVisitor,
    version: 1,
    min_supported_version: 1,
    tier: OperationTier::Library,
    category: "logical",
    summary: "Synthesized elementwise logical operations.",

    operations: [
        {
            op: Nand,
            discriminant: 0,
            name: "nand",
            id: "vyre-libs::logical::nand",
            version: 1,
            summary: "Bitwise NAND.",
            signature: LOGICAL_BINARY_SIG,
            is_composable: true,
            build: || nand("a", "b", "out", 4),
            test_inputs: nand_test_inputs,
            expected_output: nand_expected_output,
            call_builder: call_nand,
        },
        {
            op: Nor,
            discriminant: 1,
            name: "nor",
            id: "vyre-libs::logical::nor",
            version: 1,
            summary: "Bitwise NOR.",
            signature: LOGICAL_BINARY_SIG,
            is_composable: true,
            build: || nor("a", "b", "out", 4),
            test_inputs: nor_test_inputs,
            expected_output: nor_expected_output,
            call_builder: call_nor,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_bytes::eval_bytes;

    fn eval_u32_binary(program: &vyre_foundation::ir::Program, a: &[u32], b: &[u32]) -> Vec<u32> {
        let outputs = eval_bytes(
            "logical",
            program,
            vec![
                vyre_primitives::wire::pack_u32_slice(a),
                vyre_primitives::wire::pack_u32_slice(b),
                vec![0_u8; a.len() * core::mem::size_of::<u32>()],
            ],
        );
        vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0])
    }

    #[test]
    fn generated_nand_nor_match_scalar_reference() {
        let mut state = 0x10CC_A11E_u32;
        for case in 0..1024_u32 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let len = (state as usize % 33) + 1;
            let mut a = Vec::with_capacity(len);
            let mut b = Vec::with_capacity(len);
            for index in 0..len {
                state = state.rotate_left(5) ^ (index as u32).wrapping_mul(0x9E37_79B9);
                a.push(match index % 4 {
                    0 => state,
                    1 => !state,
                    2 => 0,
                    _ => u32::MAX,
                });
                state = state.rotate_left(9) ^ (case.wrapping_mul(0x85EB_CA6B));
                b.push(match index % 5 {
                    0 => state,
                    1 => !state,
                    2 => 0xAAAA_AAAA,
                    3 => 0x5555_5555,
                    _ => u32::MAX,
                });
            }

            let nand_program = nand("a", "b", "out", len as u32);
            let nor_program = nor("a", "b", "out", len as u32);
            let expected_nand: Vec<u32> = a
                .iter()
                .zip(&b)
                .map(|(left, right)| !(left & right))
                .collect();
            let expected_nor: Vec<u32> = a
                .iter()
                .zip(&b)
                .map(|(left, right)| !(left | right))
                .collect();

            assert_eq!(
                eval_u32_binary(&nand_program, &a, &b),
                expected_nand,
                "case {case}"
            );
            assert_eq!(
                eval_u32_binary(&nor_program, &a, &b),
                expected_nor,
                "case {case}"
            );
        }
    }

    #[test]
    fn dialect_metadata_closure() {
        assert_eq!(dialect::Op::Nand.op_id(), "vyre-libs::logical::nand");
        assert_eq!(dialect::Op::Nor.op_id(), "vyre-libs::logical::nor");
        assert_eq!(dialect::ALL_OP_IDS.len(), 2);
    }
}
