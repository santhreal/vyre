//! Elementwise logical operations (nand, nor).
//!
//! `and`, `or`, and `xor` are registered once, in `crate::bitset`.
//! Only the synthesized combinations with no single-kernel equivalent live here.

macro_rules! define_synthesized_logical_binary {
    ($module:ident, $function:ident, $op_id:literal, $expr:expr, $expected:expr, $expected_bytes:expr, $doc:literal) => {
        pub(crate) mod $module {
            use super::wrap::build_logical_binary;
            use vyre_foundation::ir::Program;

            const OP_ID: &str = $op_id;
            const EXPECTED_OUTPUT_BYTES: [u8; 16] = $expected_bytes;

            /// Build the synthesized logical binary operation.
            #[must_use]
            pub fn $function(a: &str, b: &str, out: &str, size: u32) -> Program {
                build_logical_binary(OP_ID, a, b, out, size, $expr)
            }

            inventory::submit! {
                vyre_foundation::operation::OperationRegistration::library(
                    OP_ID,
                    || $function("a", "b", "out", 4),
                    Some(|| {
                        let a = [0xFF00_FF00u32, 0x00FF_00FF, 0xFFFF_FFFF, 0x0000_0000];
                        let b = [0xF0F0_F0F0u32, 0x0F0F_0F0F, 0xFFFF_FFFF, 0x0000_0000];
                        let to_bytes = vyre_primitives::wire::pack_u32_slice;
                        vec![vec![to_bytes(&a), to_bytes(&b)]]
                    }),
                    Some(|| {
                        vec![vec![EXPECTED_OUTPUT_BYTES.to_vec()]]
                    }),
                )
            }
        }

        #[doc = $doc]
        pub use $module::$function;
    };
}

define_synthesized_logical_binary!(
    nand,
    nand,
    "vyre-libs::logical::nand",
    |left, right| vyre_foundation::ir::Expr::bitnot(vyre_foundation::ir::Expr::bitand(left, right)),
    &[0x0FFF_0FFF, 0xFFF0_FFF0, 0x0000_0000, 0xFFFF_FFFF],
    [
        0xFF, 0x0F, 0xFF, 0x0F, 0xF0, 0xFF, 0xF0, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
        0xFF
    ],
    "Bitwise NAND."
);
define_synthesized_logical_binary!(
    nor,
    nor,
    "vyre-libs::logical::nor",
    |left, right| vyre_foundation::ir::Expr::bitnot(vyre_foundation::ir::Expr::bitor(left, right)),
    &[0x000F_000F, 0xF000_F000, 0x0000_0000, 0xFFFF_FFFF],
    [
        0x0F, 0x00, 0x0F, 0x00, 0x00, 0xF0, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
        0xFF
    ],
    "Bitwise NOR."
);
mod wrap;

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::value::Value;

    fn eval_u32_binary(program: &vyre_foundation::ir::Program, a: &[u32], b: &[u32]) -> Vec<u32> {
        let outputs = vyre_reference::reference_eval(
            program,
            &[
                Value::from(vyre_primitives::wire::pack_u32_slice(a)),
                Value::from(vyre_primitives::wire::pack_u32_slice(b)),
                Value::from(vec![0_u8; a.len() * core::mem::size_of::<u32>()]),
            ],
        )
        .expect("Fix: logical elementwise program must execute in the reference interpreter.");
        vyre_primitives::wire::decode_u32_le_bytes_all(&outputs[0].to_bytes())
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
}
