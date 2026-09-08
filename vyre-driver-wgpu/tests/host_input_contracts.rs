//! Host input classification and preserved-contents contract tests.

use rustc_hash::FxHashSet;
use vyre_driver_wgpu::pipeline::descriptor_buffer_bindings;
use vyre_driver_wgpu::pipeline::host_input_slots;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, MemoryKind};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, GridIndexSpace, KernelBody,
    KernelDescriptor, MemoryClass,
};

fn slot(
    index: u32,
    name: &str,
    memory_class: MemoryClass,
    visibility: BindingVisibility,
) -> BindingSlot {
    BindingSlot {
        slot: index,
        element_type: DataType::U32,
        element_count: Some(4),
        memory_class,
        visibility,
        name: name.to_owned(),
    }
}

fn descriptor_of(slots: Vec<BindingSlot>) -> KernelDescriptor {
    KernelDescriptor {
        id: String::from("host-input-contract-test"),
        bindings: BindingLayout { slots },
        dispatch: Dispatch {
            workgroup_size: [1, 1, 1],
            grid_index: GridIndexSpace::default(),
        },
        body: KernelBody {
            ops: Vec::new(),
            child_bodies: Vec::new(),
            literals: Vec::new(),
        },
    }
}

/// A read-write binding in a non-zero bind group preserves caller-supplied initial bytes.
///
/// The host-input set and descriptor lookup derive the (group, slot) key from the same
/// descriptor mapping, so a binding in group 1 is recorded and queried under group 1.
#[test]
fn read_write_binding_in_non_zero_bind_group_preserves_input_contents() {
    let slots = vec![slot(
        0,
        "rw_uniform",
        MemoryClass::Uniform,
        BindingVisibility::ReadWrite,
    )];
    let descriptor = descriptor_of(slots);
    let buffers =
        vec![
            BufferDecl::storage("rw_uniform", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(4),
        ];
    let host_inputs = host_input_slots(&descriptor, &buffers, None)
        .expect("host input slots derivation must succeed");
    let outputs = FxHashSet::default();
    let bindings = descriptor_buffer_bindings(&descriptor, &outputs, &host_inputs)
        .expect("descriptor buffer bindings derivation must succeed");

    assert_eq!(bindings.len(), 1);
    let binding = &bindings[0];
    assert_ne!(
        binding.group, 0,
        "binding must land in a non-zero bind group"
    );
    assert_eq!(binding.group, 1);
    assert_eq!(binding.access, BufferAccess::ReadWrite);
    assert!(
        binding.consumes_host_input,
        "binding must record host input consumption"
    );
    assert!(
        binding.preserve_input_contents,
        "read-write host-input binding in non-zero bind group must preserve input contents"
    );
}

/// Recorded `consumes_host_input` matches `BufferDecl::consumes_host_input` for every binding.
///
/// A `Persistent`-kind buffer and a non-read-write `pipeline_live_out` do not consume host input.
/// Flattened re-derivations that omit `pipeline_live_out` or memory tier incorrectly return true.
#[test]
fn every_recorded_binding_matches_host_input_declaration_for_persistent_and_live_out() {
    let buffers = vec![
        BufferDecl::storage("persist", 0, BufferAccess::ReadOnly, DataType::U32)
            .with_kind(MemoryKind::Persistent)
            .with_count(4),
        BufferDecl::storage("carried", 1, BufferAccess::ReadOnly, DataType::U32)
            .with_pipeline_live_out(true)
            .with_count(4),
        BufferDecl::read("fed", 2, DataType::U32).with_count(4),
        BufferDecl::output("out", 3, DataType::U32).with_count(4),
        BufferDecl::storage("rw", 4, BufferAccess::ReadWrite, DataType::U32).with_count(4),
    ];
    let slots = vec![
        slot(
            0,
            "persist",
            MemoryClass::Global,
            BindingVisibility::ReadOnly,
        ),
        slot(
            1,
            "carried",
            MemoryClass::Global,
            BindingVisibility::ReadOnly,
        ),
        slot(2, "fed", MemoryClass::Global, BindingVisibility::ReadOnly),
        slot(3, "out", MemoryClass::Global, BindingVisibility::WriteOnly),
        slot(4, "rw", MemoryClass::Global, BindingVisibility::ReadWrite),
    ];
    let descriptor = descriptor_of(slots);
    let host_inputs = host_input_slots(&descriptor, &buffers, None)
        .expect("host input slots derivation must succeed");
    let outputs: FxHashSet<u32> = buffers
        .iter()
        .filter(|b| b.is_output())
        .map(BufferDecl::binding)
        .collect();
    let bindings = descriptor_buffer_bindings(&descriptor, &outputs, &host_inputs)
        .expect("descriptor buffer bindings derivation must succeed");

    let mut checked_count = 0usize;
    for info in &bindings {
        let decl = buffers
            .iter()
            .find(|b| b.binding() == info.binding)
            .expect("every binding must have a corresponding declaration");
        checked_count += 1;
        assert_eq!(
            info.consumes_host_input,
            decl.consumes_host_input(),
            "binding {} (`{}`) recorded consumes_host_input = {} but declaration states {}",
            info.binding,
            decl.name(),
            info.consumes_host_input,
            decl.consumes_host_input()
        );
    }
    assert_eq!(
        checked_count,
        buffers.len(),
        "every declared buffer must be checked"
    );
}
