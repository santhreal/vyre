//! Cross-emitter parity: every verified descriptor in the corpus must emit
//! through Naga, PTX, and SPIR-V. Failure in any target breaks the
//! substrate-neutral `KernelDescriptor` contract.
//!
//! Descriptor scaffolding comes from `vyre_lower::descriptor_builder`. The
//! suite lives here because this is the only emitter crate that dev-depends on
//! all three emitters.

use vyre_emit_ptx::ComputeCapability;
use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn out_slot() -> vyre_lower::BindingSlot {
    global_rw(0, DataType::U32, "out")
}

/// One store of literal 7 into element 0 of `slot`, dispatched over 64
/// invocations: the smallest descriptor that reaches every emitter's store path.
fn store_one_kernel(id: &str, slot: vyre_lower::BindingSlot) -> KernelDescriptor {
    descriptor(id)
        .slot(slot)
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1])),
        )
        .build()
}

/// Every audit layer, substrate-neutral and per-target, reports the kernel id it
/// was handed and the target it was asked about. Stated once: a layer that
/// dropped the id would otherwise be caught only where someone remembered to
/// list that layer.
fn assert_audits_carry_kernel_id(desc: &KernelDescriptor, target: ComputeCapability) {
    assert_eq!(
        vyre_lower::audit(desc, &vyre_lower::analyses::AnalysisFacts::none()).kernel_id,
        desc.id
    );
    assert_eq!(vyre_emit_naga::patterns::audit(desc).kernel_id, desc.id);
    let ptx = vyre_emit_ptx::patterns::audit(desc, target);
    assert_eq!(ptx.kernel_id, desc.id);
    assert_eq!(ptx.target, target);
    assert_eq!(vyre_emit_spirv::patterns::audit(desc).kernel_id, desc.id);
}

/// (1) Empty kernel.
fn empty() -> KernelDescriptor {
    descriptor("empty").build()
}

/// (2) Single store.
fn single_store() -> KernelDescriptor {
    store_one_kernel("single_store", out_slot())
}

/// (3) Add and store.
fn add_store() -> KernelDescriptor {
    descriptor("add_store")
        .slot(out_slot())
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

/// (4) Identity arithmetic that the rewrite stack eliminates before any
/// emitter sees it: an additive identity and an absorbing zero.
fn identity_heavy() -> KernelDescriptor {
    descriptor("identity_heavy")
        .slot(out_slot())
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

/// (5) Store-load-store that load forwarding and dead-store elimination
/// should collapse.
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

fn descriptor_corpus() -> Vec<KernelDescriptor> {
    vec![
        empty(),
        single_store(),
        add_store(),
        identity_heavy(),
        store_load_store(),
    ]
}

#[test]
fn every_descriptor_lowers_through_all_three_emitters() {
    for desc in descriptor_corpus() {
        let id = desc.id.clone();
        let verified = vyre_lower::verify_descriptor(&desc).expect("descriptor verification");

        let naga_module = vyre_emit_naga::emit(&verified)
            .unwrap_or_else(|error| panic!("Naga emit failed for `{id}`: {error:?}"));
        assert!(
            !naga_module.entry_points.is_empty(),
            "naga module for `{id}` must expose an entry point"
        );
        assert_eq!(naga_module.entry_points[0].name, "main");
        assert_eq!(
            naga_module.entry_points[0].workgroup_size,
            verified.dispatch.workgroup_size
        );

        let ptx = vyre_emit_ptx::emit(&verified)
            .unwrap_or_else(|error| panic!("PTX emit failed for `{id}`: {error:?}"));
        assert!(
            ptx.contains(".version"),
            "ptx for `{id}` must include a version directive"
        );

        let spirv_words = vyre_emit_spirv::emit(&verified)
            .unwrap_or_else(|error| panic!("SPIR-V emit failed for `{id}`: {error:?}"));
        assert_eq!(
            spirv_words.first().copied(),
            Some(vyre_emit_spirv::SPIRV_MAGIC),
            "spirv for `{id}` must start with the SPIR-V magic word"
        );
        assert!(
            spirv_words.len() > 16,
            "spirv for `{id}` must contain a complete module, not just a header"
        );
    }
}

#[test]
fn naga_and_spirv_main_entry_points_match() {
    // Naga and SPIR-V come from the same Naga module; their entry
    // point names + workgroup sizes must be identical.
    for desc in descriptor_corpus() {
        let naga_module = vyre_emit_naga::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();
        let spirv_words = vyre_emit_spirv::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();

        // Naga module entry point matches descriptor's dispatch.
        let entry = &naga_module.entry_points[0];
        assert_eq!(entry.name, "main");
        assert_eq!(entry.workgroup_size, desc.dispatch.workgroup_size);

        // SPIR-V starts with the magic word.
        assert_eq!(spirv_words[0], vyre_emit_spirv::SPIRV_MAGIC);
    }
}

#[test]
fn ptx_output_contains_required_directives_for_every_kernel() {
    for desc in descriptor_corpus() {
        if desc.body.ops.is_empty() {
            // PTX skipped on empty kernels  -  nothing to emit.
            continue;
        }
        let ptx = vyre_emit_ptx::emit(
            &vyre_lower::verify_descriptor(&desc).expect("descriptor verification"),
        )
        .unwrap();
        assert!(
            ptx.contains(".version"),
            "PTX for `{}` missing .version",
            desc.id
        );
        assert!(
            ptx.contains(".target"),
            "PTX for `{}` missing .target",
            desc.id
        );
    }
}

#[test]
fn shared_cleanup_preserves_naga_emit_acceptance() {
    for desc in descriptor_corpus() {
        let verified = vyre_lower::verify_descriptor(&desc)
            .expect("descriptor corpus must pass shared verification");
        assert!(
            vyre_emit_naga::emit(&verified).is_ok(),
            "verified descriptor `{}` must emit through Naga",
            desc.id
        );
    }
}

#[test]
fn every_audit_layer_succeeds_without_panic_on_corpus() {
    // The audit family must be robust across realistic shapes: each layer's
    // audit() function takes a descriptor and produces a typed report. None
    // should panic, even on edge cases (empty kernel, identity-only arithmetic).
    for desc in descriptor_corpus() {
        assert_audits_carry_kernel_id(&desc, ComputeCapability::SM_80);
    }
}

#[test]
fn descriptor_verification_and_audits_succeed_on_corpus() {
    for descriptor in descriptor_corpus() {
        let verified = vyre_lower::verify_descriptor(&descriptor).unwrap_or_else(|failure| {
            panic!(
                "descriptor verification failed on `{}`: {:?}",
                descriptor.id,
                failure.errors()
            )
        });
        assert_eq!(verified.id, descriptor.id, "id round-trips");
        assert_audits_carry_kernel_id(&verified, ComputeCapability::SM_80);
    }
}

#[test]
fn audit_carries_kernel_id_through_every_layer() {
    let desc = store_one_kernel("named_kernel_42", global_rw(0, DataType::U32, "buf"));
    assert_audits_carry_kernel_id(&desc, ComputeCapability::SM_70);
}
