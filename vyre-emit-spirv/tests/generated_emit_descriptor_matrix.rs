//! Generated descriptor matrix for SPIR-V emission invariants.
//!
//! The adversarial corpus covers hostile shapes. This test covers generated
//! ordinary kernels with varied dispatch geometry and arithmetic chain depth,
//! pinning verification, word emission, and byte emission contracts.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn generated_descriptor(seed: u32) -> KernelDescriptor {
    let chain_len = 1 + (seed as usize % 12);
    let mut literals = vec![LiteralValue::U32(0)];
    let mut ops = vec![lit(0, 0)];
    let mut accumulator = 0u32;

    for idx in 0..chain_len {
        let literal_idx = literals.len() as u32;
        let literal_value = seed
            .wrapping_mul(0x9e37_79b9)
            .rotate_left((idx as u32) & 31)
            .wrapping_add(idx as u32);
        literals.push(LiteralValue::U32(literal_value));
        let literal_result = ops.len() as u32;
        ops.push(lit(literal_idx, literal_result));
        let binop_result = ops.len() as u32;
        let kind = match idx % 4 {
            0 => BinOp::Add,
            1 => BinOp::BitXor,
            2 => BinOp::BitOr,
            _ => BinOp::BitAnd,
        };
        ops.push(op(
            KernelOpKind::BinOpKind(kind),
            [accumulator, literal_result],
            binop_result,
        ));
        accumulator = binop_result;
    }

    ops.push(effect(KernelOpKind::StoreGlobal, [0, 0, accumulator]));

    descriptor(&format!("generated_spirv_{seed:08x}"))
        .slot(global_rw(0, DataType::U32, "out"))
        .dispatch(
            1 + (seed & 255),
            1 + ((seed >> 8) & 7),
            1 + ((seed >> 16) & 3),
        )
        .body(body().literals(literals).ops(ops))
        .build()
}

fn words_from_le_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("exact 4-byte chunk")))
        .collect()
}

#[test]
fn generated_descriptors_verify_before_spirv_emission() {
    for seed in 0..256u32 {
        let desc = generated_descriptor(seed.wrapping_mul(0x045d_9f3b));
        let descriptor = vyre_lower::verify_descriptor(&desc).unwrap_or_else(|err| {
            panic!("descriptor verification failed for {}: {err:?}", desc.id)
        });
        let words = vyre_emit_spirv::emit(&descriptor)
            .unwrap_or_else(|err| panic!("SPIR-V emit failed for {}: {err:?}", desc.id));

        assert_eq!(words[0], vyre_emit_spirv::SPIRV_MAGIC, "{}", desc.id);
        assert!(words.len() > 16, "{} kernel too small", desc.id);
    }
}

#[test]
fn generated_descriptors_bytes_match_word_emission() {
    for seed in 0..128u32 {
        let desc = generated_descriptor(seed ^ 0xa501_7b1d);
        let descriptor = vyre_lower::verify_descriptor(&desc).unwrap_or_else(|err| {
            panic!("descriptor verification failed for {}: {err:?}", desc.id)
        });
        let words = vyre_emit_spirv::emit(&descriptor)
            .unwrap_or_else(|err| panic!("SPIR-V emit failed for {}: {err:?}", desc.id));
        let bytes = vyre_emit_spirv::emit_bytes(&descriptor)
            .unwrap_or_else(|err| panic!("byte emit failed for {}: {err:?}", desc.id));

        assert_eq!(bytes.len(), words.len() * 4, "{}", desc.id);
        assert_eq!(words_from_le_bytes(&bytes), words, "{}", desc.id);
    }
}

#[test]
fn generated_descriptors_verify_before_byte_emission() {
    for seed in 0..128u32 {
        let desc = generated_descriptor(seed.rotate_left(7));
        let descriptor = vyre_lower::verify_descriptor(&desc).unwrap_or_else(|err| {
            panic!("descriptor verification failed for {}: {err:?}", desc.id)
        });
        let bytes = vyre_emit_spirv::emit_bytes(&descriptor)
            .unwrap_or_else(|err| panic!("byte emit failed for {}: {err:?}", desc.id));
        assert!(bytes.len() >= 4, "{}", desc.id);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().expect("SPIR-V header word")),
            vyre_emit_spirv::SPIRV_MAGIC,
            "{}",
            desc.id
        );
    }
}
