//! SPIR-V emitter contracts over the public `vyre_emit_spirv` surface:
//! module header words, byte emission and the errors a rejected
//! descriptor produces.

use vyre_emit_spirv::*;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn one_store_kernel() -> KernelDescriptor {
    descriptor("store_one")
        .slot(global_rw(0, DataType::U32, "out"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .literal(LiteralValue::U32(0))
                .literal(LiteralValue::U32(7)),
        )
        .build()
}

#[test]
fn empty_kernel_emits_valid_spirv_with_magic_header() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let words = emit(&desc).unwrap();
    assert!(!words.is_empty());
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "first word must be the SPIR-V magic number"
    );
}

#[test]
fn one_store_kernel_emits_non_trivial_spirv() {
    let words = emit(&one_store_kernel()).unwrap();
    assert!(
        words.len() > 16,
        "real kernel should produce more than the header"
    );
    assert_eq!(words[0], SPIRV_MAGIC);
}

#[test]
fn emit_bytes_matches_words_in_le() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let words = emit(&desc).unwrap();
    let bytes = emit_bytes(&desc).unwrap();
    assert_eq!(bytes.len(), words.len() * 4);
    let first_word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(first_word, SPIRV_MAGIC);
}

#[test]
fn emit_with_unsupported_op_propagates_naga_error() {
    let desc = descriptor("bad")
        .body(body().ops([op(
            KernelOpKind::SubgroupReduce {
                op: vyre_lower::SubgroupReduceOp::Add,
            },
            [0],
            0,
        )]))
        .build();
    let r = emit(&desc);
    assert!(matches!(r, Err(EmitError::NagaEmit(_))));
}

#[test]
fn binop_add_emits_valid_spirv() {
    let kernel = descriptor("add")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(
                        KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                        [0, 1],
                        2,
                    ),
                ])
                .literals([LiteralValue::U32(3), LiteralValue::U32(4)]),
        )
        .build();
    let words = emit(&kernel).unwrap();
    assert_eq!(words[0], SPIRV_MAGIC);
    assert!(words.len() > 16);
}

#[test]
fn spirv_magic_constant_matches_spec() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203);
}

#[test]
fn emit_from_naga_module_independently_consumable() {
    // Build a valid naga::Module via emit-naga, then convert.
    let module = vyre_emit_naga::emit(&descriptor("k").build()).unwrap();
    let words = emit_from_naga_module(&module).unwrap();
    assert_eq!(words[0], SPIRV_MAGIC);
}
