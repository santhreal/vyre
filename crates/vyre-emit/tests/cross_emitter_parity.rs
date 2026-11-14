//! Cross-emitter parity: every descriptor in the corpus verifies and emits
//! through Naga, PTX, and SPIR-V. A failure in any target breaks the
//! substrate-neutral `KernelDescriptor` contract for that shape.

use vyre_compiler::lower::descriptor_builder::{
    body, descriptor, effect, global_rw, lit, op, store_literal_kernel,
};
use vyre_compiler::lower::{KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};
use vyre_foundation::ir::{BinOp, DataType};

fn out_slot(ty: DataType) -> vyre_compiler::lower::BindingSlot {
    global_rw(0, ty, "out")
}

/// `LocalInvocationId` keeps an arithmetic op alive through constant folding,
/// so the instruction assertions observe real emission.
fn invocation_id(result: u32) -> KernelOp {
    op(KernelOpKind::LocalInvocationId, [0], result)
}

fn empty() -> KernelDescriptor {
    descriptor("empty").build()
}

fn single_store() -> KernelDescriptor {
    store_literal_kernel("single_store", out_slot(DataType::U32), [64, 1, 1])
}

/// Literal-only add: folded by the rewrite stack before emission.
fn literal_add() -> KernelDescriptor {
    descriptor("literal_add")
        .slot(out_slot(DataType::U32))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::U32(3),
                    LiteralValue::U32(4),
                    LiteralValue::U32(0),
                ])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(lit(2, 2))
                .op(op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 3))
                .op(effect(KernelOpKind::StoreGlobal, [0, 2, 3])),
        )
        .build()
}

/// An additive identity and an absorbing zero, both eliminated before any
/// emitter sees them.
fn identity_heavy() -> KernelDescriptor {
    descriptor("identity_heavy")
        .slot(out_slot(DataType::U32))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(99)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(op(KernelOpKind::BinOpKind(BinOp::Add), [1, 0], 2))
                .op(op(KernelOpKind::BinOpKind(BinOp::Mul), [1, 0], 3))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1])),
        )
        .build()
}

/// Store-load-store collapsed by load forwarding and dead-store elimination.
fn store_load_store() -> KernelDescriptor {
    descriptor("stl")
        .slot(global_rw(0, DataType::U32, "buf"))
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 2))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 2])),
        )
        .build()
}

fn add_store() -> KernelDescriptor {
    descriptor("add_store")
        .slot(out_slot(DataType::U32))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(7), LiteralValue::U32(0)])
                .op(invocation_id(0))
                .op(lit(0, 1))
                .op(lit(1, 2))
                .op(op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 3))
                .op(effect(KernelOpKind::StoreGlobal, [0, 2, 3])),
        )
        .build()
}

fn mul_store() -> KernelDescriptor {
    descriptor("mul_store")
        .slot(out_slot(DataType::U32))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0)])
                .op(invocation_id(0))
                .op(invocation_id(1))
                .op(lit(0, 2))
                .op(op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 3))
                .op(effect(KernelOpKind::StoreGlobal, [0, 2, 3])),
        )
        .build()
}

fn fma_store() -> KernelDescriptor {
    descriptor("fma_store")
        .slot(out_slot(DataType::F32))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([
                    LiteralValue::F32(7.0),
                    LiteralValue::U32(0),
                    LiteralValue::F32(11.0),
                ])
                .op(invocation_id(0))
                .op(op(
                    KernelOpKind::Cast {
                        target: DataType::F32,
                    },
                    [0],
                    1,
                ))
                .op(lit(0, 2))
                .op(lit(1, 3))
                .op(lit(2, 5))
                .op(op(KernelOpKind::Fma, [1, 2, 5], 4))
                .op(effect(KernelOpKind::StoreGlobal, [0, 3, 4])),
        )
        .build()
}

fn corpus() -> [KernelDescriptor; 8] {
    [
        empty(),
        single_store(),
        literal_add(),
        identity_heavy(),
        store_load_store(),
        add_store(),
        mul_store(),
        fma_store(),
    ]
}

#[test]
fn every_descriptor_emits_through_every_target() {
    for desc in corpus() {
        let id = &desc.id;
        let verified = vyre_compiler::lower::verify_descriptor(&desc)
            .unwrap_or_else(|failure| panic!("`{id}` failed verification: {failure:?}"));
        assert_eq!(&verified.id, id, "verification preserves the kernel id");

        let naga = vyre_emit::naga::emit(&verified)
            .unwrap_or_else(|error| panic!("Naga emit failed for `{id}`: {error:?}"));
        let entry = naga
            .entry_points
            .first()
            .unwrap_or_else(|| panic!("naga module for `{id}` has no entry point"));
        assert_eq!(entry.name, "main");
        assert_eq!(entry.workgroup_size, desc.dispatch.workgroup_size);

        let ptx = vyre_emit::ptx::emit(&verified)
            .unwrap_or_else(|error| panic!("PTX emit failed for `{id}`: {error:?}"));
        for directive in [".version", ".target", ".address_size", ".entry"] {
            assert!(
                ptx.contains(directive),
                "PTX for `{id}` lacks `{directive}`"
            );
        }
        assert!(
            vyre_emit::ptx::emit(&desc).is_ok(),
            "unverified PTX emit diverges from verified emit on `{id}`"
        );

        let spirv = vyre_emit::spirv::emit(&verified)
            .unwrap_or_else(|error| panic!("SPIR-V emit failed for `{id}`: {error:?}"));
        assert_eq!(spirv.first().copied(), Some(vyre_emit::spirv::SPIRV_MAGIC));
        assert!(spirv.len() > 16, "SPIR-V for `{id}` is a bare header");
    }
}

#[test]
fn ptx_contains_the_instruction_for_each_live_op() {
    for (desc, instruction) in [
        (add_store(), "add"),
        (mul_store(), "mul.lo"),
        (fma_store(), "fma.rn"),
    ] {
        let verified =
            vyre_compiler::lower::verify_descriptor(&desc).expect("descriptor verification");
        let ptx = vyre_emit::ptx::emit(&verified).expect("PTX emit");
        assert!(
            ptx.contains(instruction),
            "PTX for `{}` lacks `{instruction}`",
            desc.id
        );
    }
}
