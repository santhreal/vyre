//! Metal emitter contracts over the public `vyre_emit_metal` surface:
//! MSL text shape, artifact binding metadata, threadgroup memory placement
//! and the errors a rejected descriptor produces.

use vyre_emit_metal::*;
use vyre_foundation::ir::{DataType, Expr, Node, Program};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_rw, lit, op, shared_rw, SlotCount,
};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn empty_kernel() -> KernelDescriptor {
    descriptor("empty").dispatch(64, 1, 1).build()
}

fn one_store_kernel() -> KernelDescriptor {
    descriptor("store_one")
        .slot(global_rw(0, DataType::U32, "out").with_count(1))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
        )
        .build()
}

#[test]
fn empty_kernel_emits_compute_msl() {
    let msl = emit(&empty_kernel()).unwrap();
    assert!(
        msl.contains("kernel"),
        "MSL source must include a Metal kernel entry point"
    );
    assert!(
        msl.contains("main"),
        "MSL source must include the canonical main entry point"
    );
}

#[test]
fn one_store_kernel_emits_buffer_binding_metadata() {
    let artifact = emit_artifact(&one_store_kernel()).unwrap();
    assert_eq!(artifact.target, "native_module");
    assert!(
        artifact.msl.contains(&artifact.entry_point),
        "artifact entry point must name the actual emitted MSL function"
    );
    assert_eq!(artifact.workgroup_size, [64, 1, 1]);
    assert_eq!(artifact.bindings.len(), 1);
    assert_eq!(artifact.bindings[0].name, "out");
    assert_eq!(artifact.bindings[0].metal_buffer_index, 0);
    assert_eq!(artifact.sizes_buffer_index, Some(1));
    assert!(artifact.threadgroup_memories.is_empty());
    assert!(!artifact.msl.is_empty());
}

#[test]
fn artifact_json_is_deterministic_for_same_descriptor() {
    let desc = one_store_kernel();
    let left = emit_artifact_json(&desc).unwrap();
    let right = emit_artifact_json(&desc).unwrap();
    assert_eq!(left, right);
}

#[test]
fn missing_entry_point_returns_actionable_error() {
    let module = vyre_emit_naga::emit(&empty_kernel()).unwrap();
    let options = MetalEmitOptions {
        entry_point: "not_main".to_string(),
        ..MetalEmitOptions::default()
    };
    let error = emit_from_naga_module(&module, &options).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("Fix:"));
    assert!(text.contains("not_main"));
}

#[test]
fn descriptor_slot_above_metal_flat_limit_is_remapped() {
    let mut desc = one_store_kernel();
    desc.bindings.slots[0].slot = 300;
    desc.body.ops[2].operands[0] = 300;
    let artifact = emit_artifact(&desc).unwrap();
    assert_eq!(artifact.bindings[0].slot, 300);
    assert_eq!(artifact.bindings[0].metal_buffer_index, 0);
    assert_eq!(artifact.sizes_buffer_index, Some(1));
}

#[test]
fn workgroup_slot_is_not_a_metal_buffer_binding() {
    let mut desc = one_store_kernel();
    desc.bindings
        .slots
        .push(shared_rw(1 << 24, DataType::U32, 4, "tile"));
    desc.bindings.slots.sort_by_key(|slot| slot.slot);
    let artifact = emit_artifact(&desc).unwrap();
    assert_eq!(artifact.bindings.len(), 1);
    assert_eq!(artifact.bindings[0].name, "out");
    assert_eq!(artifact.bindings[0].metal_buffer_index, 0);
    assert_eq!(artifact.sizes_buffer_index, Some(1));
    assert_eq!(artifact.threadgroup_memories.len(), 1);
    assert_eq!(artifact.threadgroup_memories[0].name, "tile");
    assert_eq!(artifact.threadgroup_memories[0].slot, 1 << 24);
    assert_eq!(artifact.threadgroup_memories[0].threadgroup_index, 0);
    assert_eq!(artifact.threadgroup_memories[0].byte_length, 16);
    assert_eq!(artifact.threadgroup_memories[0].aligned_byte_length, 16);
}

#[test]
fn metal_resource_count_without_sidecar_room_is_rejected() {
    let mut desc = empty_kernel();
    desc.bindings.slots = (0..=255)
        .map(|slot| global_ro(slot, DataType::U32, &format!("buf{slot}")).with_count(1))
        .collect();
    let error = emit_artifact(&desc).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("Fix:"));
    assert!(text.contains("_buffer_sizes"));
}

#[test]
fn trap_sidecar_compare_exchange_emits_msl_helper() {
    let program = Program::wrapped(vec![], [64, 1, 1], vec![Node::trap(Expr::u32(7), "fault")]);
    let desc = vyre_lower::lower_verified(&program)
        .map(|lowered| lowered.descriptor)
        .expect("Fix: trap programs must descriptor-lower");
    let artifact = emit_artifact(&desc).expect("Fix: trap descriptors must emit Metal MSL");

    assert!(
        artifact
            .msl
            .contains("naga_atomic_compare_exchange_weak_explicit"),
        "Fix: trap/CAS Metal MSL must include Naga's compare-exchange helper."
    );
    let sidecar = artifact
        .bindings
        .iter()
        .find(|binding| binding.name == vyre_lower::TRAP_SIDECAR_NAME)
        .expect("Fix: trap sidecar must stay host-bound in Metal metadata.");
    assert_eq!(
        sidecar.element_count,
        Some(vyre_lower::TRAP_SIDECAR_WORDS),
        "Fix: trap sidecar metadata must preserve the four-word runtime ABI."
    );
}

/// A read-only binding must produce `const device` in MSL, not a mutable
/// `device` pointer. Before this fix `mutable: true` was hardcoded for
/// every binding regardless of `BindingVisibility`, which prevented Naga's
/// MSL backend from emitting `const device T*` and could cause silent
/// miscompilation under Metal's strict aliasing rules.
#[test]
fn readonly_binding_emits_const_device_pointer() {
    let desc = descriptor("readonly_smoke")
        .slots([
            global_ro(0, DataType::U32, "input").with_count(16),
            global_rw(1, DataType::U32, "output").with_count(16),
        ])
        .dispatch(16, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    effect(KernelOpKind::StoreGlobal, [1, 0, 1]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let msl = emit(&desc).expect("Fix: read-only + read-write kernel must emit MSL without error.");
    // Naga's MSL backend emits a read-only storage buffer as a CONST
    // reference (`device <type> const& name`) and a read-write buffer as a
    // mutable reference (`device <type>& name`). The read-only "input"
    // binding (BindingVisibility::ReadOnly -> AddressSpace access LOAD ->
    // BindTarget.mutable = false) MUST be const-qualified; the read-write
    // "output" binding MUST NOT be. Before the fix `mutable: true` was
    // hardcoded for every binding, regressing "input" to a mutable
    // `device input_elements&` and losing Metal's read-only aliasing
    // guarantees. (Naga spells it `device T const&`, not `const device T*`.)
    assert!(
        msl.contains("input_elements const&"),
        "read-only `input` binding must emit a `device <type> const&` reference, not a mutable one. MSL excerpt:\n{}",
        &msl[..msl.len().min(800)]
    );
    assert!(
        !msl.contains("output_elements const&"),
        "read-write `output` binding must NOT be const-qualified. MSL excerpt:\n{}",
        &msl[..msl.len().min(800)]
    );
}
