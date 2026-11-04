//! Which target bindings a module's launch stages bytes into.
//!
//! # Why this suite exists
//!
//! A module's launch fills the bindings its lowered descriptor declares. Its
//! neutral `Program` declares a different list: the fused program's carried
//! value is a `Program` output, so it has no host-input slot, while the module
//! that reads it binds it as a readable descriptor slot on entry.
//!
//! A materializer that staged the `Program`'s host inputs launched the reading
//! module over an allocation nothing had filled. The fused operations in
//! `vyre-libs::security` read their stage-one bitset as zero in every case,
//! which returns a wrong answer rather than a rejection: `sink_intersection`
//! counted nothing and `aliases_dataflow` returned its seed frontier unchanged.
//!
//! # Why the cases are shaped this way
//!
//! Every case drives the three inputs the derivation reads, and the ordering
//! case is the one that goes red against the defect: it asserts a slot the
//! `Program` declares no host input for is staged, and staged at the position
//! the descriptor puts it in. What this does not catch: a descriptor whose slot
//! order disagrees with the emitted target module's own parameter order, which
//! is the emitter's contract and is proved by conformance execution.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};
use vyre_lower::{BindingSlot, BindingVisibility, KernelDescriptor, MemoryClass};
use vyre_megakernel::{
    staged_input_slots, ArtifactInputSlot, ArtifactValueId, TargetCompileError,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};

/// Element count of the value the fused program carries between its modules.
const CARRIED_ELEMENTS: u32 = 4;

/// The fused program both modules are lowered from.
///
/// `carried` is written by the first module and read by the second, so the
/// program declares it an output and no host input ever fills it.
fn fused_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("seed", 0, BufferAccess::ReadOnly, DataType::U32).with_count(2),
            BufferDecl::output("carried", 1, DataType::U32).with_count(CARRIED_ELEMENTS),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [4, 1, 1],
        Vec::new(),
    )
}

/// A descriptor binding layout over `slots`, with the body a lowering of the
/// empty program produces.
fn descriptor_over(slots: Vec<BindingSlot>) -> KernelDescriptor {
    let mut descriptor =
        vyre_lower::lower_physical(&Program::wrapped(Vec::new(), [1, 1, 1], Vec::new()))
            .expect("Fix: the empty fixture program must lower")
            .into_descriptor();
    descriptor.bindings.slots = slots;
    descriptor
}

/// One host-bound descriptor slot.
fn slot(name: &str, index: u32, visibility: BindingVisibility, count: Option<u32>) -> BindingSlot {
    BindingSlot {
        slot: index,
        element_type: DataType::U32,
        element_count: count,
        memory_class: MemoryClass::Global,
        visibility,
        name: name.to_string(),
    }
}

/// The canonical directional metadata a payload publishes for one slot.
fn canonical(resource: u32, index: u32, access: TargetResourceAccess) -> TargetResourceBinding {
    TargetResourceBinding {
        resource: ArtifactValueId(resource),
        group: 0,
        slot: index,
        memory: TargetResourceMemory::Global,
        access,
    }
}

/// The descriptor slots the second module of the fused program binds.
fn reader_slots() -> Vec<BindingSlot> {
    vec![
        slot("seed", 0, BindingVisibility::ReadOnly, Some(2)),
        slot(
            "carried",
            1,
            BindingVisibility::ReadWrite,
            Some(CARRIED_ELEMENTS),
        ),
        slot("out", 2, BindingVisibility::WriteOnly, Some(1)),
    ]
}

/// The canonical bindings that pair with [`reader_slots`].
fn reader_bindings() -> Vec<TargetResourceBinding> {
    vec![
        canonical(0, 0, TargetResourceAccess::ReadOnly),
        canonical(1, 1, TargetResourceAccess::ReadWrite),
        canonical(2, 2, TargetResourceAccess::WriteOnly),
    ]
}

/// WHY: the carried value has no host-input slot in the fused program, so a
/// derivation that walks `Program` input order omits it and the reading module
/// launches over memory nothing wrote.
#[test]
fn a_readable_binding_with_no_program_host_input_is_staged_in_descriptor_order() {
    let program = fused_program();

    let slots = staged_input_slots(
        &descriptor_over(reader_slots()),
        &reader_bindings(),
        &program,
    )
    .expect("Fix: the fused reader module must derive its staged bindings");

    assert_eq!(
        slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect::<Vec<_>>(),
        ["seed", "carried"],
        "Fix: the carried value must be staged, in the position the descriptor binds it at."
    );
    assert_eq!(
        slots[1],
        ArtifactInputSlot {
            name: "carried".to_string(),
            group: 0,
            slot: 1,
            expected_max: Some(CARRIED_ELEMENTS as usize * 4),
            launch_zeros: Some(vec![0; CARRIED_ELEMENTS as usize * 4].into_boxed_slice()),
        }
    );
}

/// WHY: a write-only slot is filled by the launch itself. Staging it would ask
/// the caller for bytes no value is bound to, which rejects a correct artifact.
#[test]
fn a_write_only_binding_is_not_staged() {
    let slots = staged_input_slots(
        &descriptor_over(reader_slots()),
        &reader_bindings(),
        &fused_program(),
    )
    .expect("Fix: the fused reader module must derive its staged bindings");

    assert!(
        slots.iter().all(|slot| slot.name != "out"),
        "Fix: a WriteOnly canonical binding must not be staged."
    );
}

/// WHY: the trap sidecar is backend-owned diagnostic storage. It carries no
/// artifact value and no `Program` buffer declares it, so staging it rejects
/// every trapping module before it runs.
#[test]
fn the_backend_owned_trap_sidecar_is_not_staged() {
    let mut slots = reader_slots();
    slots.push(slot(
        vyre_lower::TRAP_SIDECAR_NAME,
        3,
        BindingVisibility::ReadWrite,
        Some(4),
    ));

    let staged = staged_input_slots(
        &descriptor_over(slots),
        &reader_bindings(),
        &fused_program(),
    )
    .expect("Fix: a trapping module must derive its staged bindings");

    assert!(
        staged
            .iter()
            .all(|slot| slot.name != vyre_lower::TRAP_SIDECAR_NAME),
        "Fix: the trap sidecar is allocated by each driver and must not be staged."
    );
}

/// WHY: workgroup-class memory is allocated by the launch and is not published
/// in any bind group, so there is no `(group, slot)` to resolve it through.
#[test]
fn workgroup_class_bindings_are_not_staged() {
    let mut slots = reader_slots();
    for (index, class) in [MemoryClass::Shared, MemoryClass::Scratch]
        .into_iter()
        .enumerate()
    {
        let mut workgroup = slot(
            &format!("tile{index}"),
            3 + u32::try_from(index).expect("fixture index fits u32"),
            BindingVisibility::ReadWrite,
            Some(8),
        );
        workgroup.memory_class = class;
        slots.push(workgroup);
    }

    let staged = staged_input_slots(
        &descriptor_over(slots),
        &reader_bindings(),
        &fused_program(),
    )
    .expect("Fix: a module with workgroup storage must derive its staged bindings");

    assert_eq!(
        staged
            .iter()
            .map(|slot| slot.name.as_str())
            .collect::<Vec<_>>(),
        ["seed", "carried"]
    );
}

/// WHY: a runtime-sized declaration has no byte count until a caller supplies
/// one. A zero-byte launch allocation reported as its ceiling would reject
/// every bound value.
#[test]
fn a_runtime_sized_declaration_carries_no_ceiling_and_no_launch_allocation() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("seed", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("carried", 1, DataType::U32).with_count(CARRIED_ELEMENTS),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ],
        [4, 1, 1],
        Vec::new(),
    );
    let mut slots = reader_slots();
    slots[0].element_count = None;

    let staged = staged_input_slots(&descriptor_over(slots), &reader_bindings(), &program)
        .expect("Fix: a runtime-sized module must derive its staged bindings");

    assert_eq!(staged[0].expected_max, None);
    assert_eq!(staged[0].launch_zeros, None);
}

/// WHY: without canonical metadata the slot's direction is unknown, so a
/// write-only binding would be staged and a readable one would be skipped. The
/// pair it was looked up under names the repair.
#[test]
fn a_host_bound_slot_with_no_canonical_metadata_is_refused() {
    let bindings = reader_bindings()
        .into_iter()
        .filter(|binding| binding.slot != 1)
        .collect::<Vec<_>>();

    let error = staged_input_slots(
        &descriptor_over(reader_slots()),
        &bindings,
        &fused_program(),
    )
    .expect_err("Fix: a slot with no canonical directional metadata must be refused");

    let TargetCompileError::InvalidArtifact(message) = error else {
        panic!("Fix: an unresolvable binding must report an invalid artifact, found {error:?}");
    };
    assert!(
        message.contains("`carried`") && message.contains("group 0, slot 1"),
        "Fix: the rejection must name the binding and the pair it resolves under: {message}"
    );
}

/// WHY: a staged slot is filled by byte count derived from the `Program`
/// declaration. A slot naming no buffer has no declaration, and staging it
/// against a mismatched one silently truncates or overruns the launch.
#[test]
fn a_staged_slot_naming_no_program_buffer_is_refused() {
    let mut slots = reader_slots();
    slots[1].name = "absent".to_string();

    let error = staged_input_slots(
        &descriptor_over(slots),
        &reader_bindings(),
        &fused_program(),
    )
    .expect_err("Fix: a slot naming no Program buffer must be refused");

    let TargetCompileError::InvalidArtifact(message) = error else {
        panic!("Fix: an unresolvable buffer must report an invalid artifact, found {error:?}");
    };
    assert!(
        message.contains("`absent`"),
        "Fix: the rejection must name the slot that resolves to nothing: {message}"
    );
}
