//! Cross-emitter parity between `vyre-emit-ptx` and `vyre-emit-naga`.
//!
//! Every descriptor in the matrix must lower through both emitters.
//! Failure in either means the substrate-neutral promise of
//! `KernelDescriptor` is broken for that shape.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

/// `LocalInvocationId` keeps each arithmetic op alive through constant
/// folding, so the instruction assertions below observe real emission.
fn invocation_id(result: u32) -> vyre_lower::KernelOp {
    op(KernelOpKind::LocalInvocationId, [0], result)
}

fn add_descriptor() -> KernelDescriptor {
    descriptor("add_store")
        .slot(global_rw(0, DataType::U32, "out"))
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

fn mul_descriptor() -> KernelDescriptor {
    descriptor("mul_store")
        .slot(global_rw(0, DataType::U32, "out"))
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

fn fma_descriptor() -> KernelDescriptor {
    descriptor("fma_store")
        .slot(global_rw(0, DataType::F32, "out"))
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

fn op_corpus() -> Vec<KernelDescriptor> {
    vec![add_descriptor(), mul_descriptor(), fma_descriptor()]
}

#[test]
fn every_op_lowers_through_ptx_and_naga() {
    for desc in op_corpus() {
        let verified = vyre_lower::verify_descriptor(&desc).expect("descriptor verification");
        let ptx = vyre_emit_ptx::emit(&verified)
            .unwrap_or_else(|error| panic!("PTX emit failed for `{}`: {error:?}", desc.id));
        assert!(
            ptx.contains(".version"),
            "ptx for `{}` must include a version directive",
            desc.id
        );

        let naga = vyre_emit_naga::emit(&verified)
            .unwrap_or_else(|error| panic!("Naga emit failed for `{}`: {error:?}", desc.id));
        assert!(
            !naga.entry_points.is_empty(),
            "naga module for `{}` must expose an entry point",
            desc.id
        );
    }
}

#[test]
fn ptx_contains_expected_instruction_for_each_op() {
    let cases = vec![
        ("add_store", "add"),
        ("mul_store", "mul.lo"),
        ("fma_store", "fma.rn"),
    ];
    for (id, instr) in cases {
        let desc = op_corpus().into_iter().find(|d| d.id == id).unwrap();
        let ptx = vyre_emit_ptx::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();
        assert!(
            ptx.contains(instr),
            "PTX for `{}` missing expected instruction `{}`",
            id,
            instr
        );
    }
}

#[test]
fn naga_and_ptx_entry_points_share_workgroup_size() {
    for desc in op_corpus() {
        let naga_module = vyre_emit_naga::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();
        let ptx = vyre_emit_ptx::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();

        let entry = &naga_module.entry_points[0];
        assert_eq!(entry.workgroup_size, desc.dispatch.workgroup_size);

        assert!(
            ptx.contains(".entry"),
            "PTX for `{}` missing .entry",
            desc.id
        );
    }
}

#[test]
fn ptx_audit_carries_kernel_id() {
    use vyre_emit_ptx::ComputeCapability;
    for desc in op_corpus() {
        let report = vyre_emit_ptx::patterns::audit(&desc, ComputeCapability::SM_80);
        assert_eq!(report.kernel_id, desc.id);
    }
}

#[test]
fn verified_and_raw_ptx_succeed_together() {
    for desc in op_corpus() {
        let raw = vyre_emit_ptx::emit(&desc);
        let descriptor = vyre_lower::verify_descriptor(&desc).unwrap_or_else(|failure| {
            panic!("descriptor `{}` failed verification: {failure:?}", desc.id)
        });
        let verified = vyre_emit_ptx::emit(&descriptor);
        assert_eq!(
            raw.is_ok(),
            verified.is_ok(),
            "ptx divergence on `{}`",
            desc.id,
        );
    }
}

#[test]
fn ptx_output_contains_required_directives() {
    for desc in op_corpus() {
        let ptx = vyre_emit_ptx::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();
        assert!(
            ptx.contains(".version") && ptx.contains(".target"),
            "PTX for `{}` missing .version or .target directive",
            desc.id
        );
        assert!(
            ptx.contains(".address_size"),
            "PTX for `{}` missing .address_size",
            desc.id
        );
    }
}
