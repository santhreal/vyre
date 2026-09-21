//! Descriptor bindings, host-visible staging, and output readback.
//!
//! One descriptor set carries every storage buffer the emitted module
//! declares, plus the reserved trap sidecar when the module emits a trap.

use ash::vk;

use vyre_driver::BackendError;

use super::device::VulkanDevice;

/// The one descriptor set this driver builds. Every storage buffer the emitted
/// module declares is decorated into it.
pub(super) const DESCRIPTOR_SET: u32 = 0;

/// One binding slot used during dispatch.
pub(super) struct DispatchBinding {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) byte_len: usize,
    pub(super) binding: u32,
}

/// One readback the caller observes: which slot of the returned vector it
/// fills, and the window of its allocation that is visible.
pub(super) struct OutputReadback {
    /// Index in the returned output vector, from `Binding::output_index`.
    slot: usize,
    /// Index into `dispatch_bindings`.
    binding_index: usize,
    /// First visible byte of the allocation.
    trim_start: usize,
    /// Visible byte count.
    read_size: usize,
}

/// Resolve the visible window of one output allocation.
///
/// A writable buffer may declare an output byte range, and every other driver
/// returns that window rather than the whole allocation. This one returned the
/// allocation: a runtime-sized output is sized from the largest input, so
/// `vyre-libs::nn::cross_entropy` returned 2048 bytes whose first 8 were the
/// result and whose remaining 2040 were the zeroed tail of the allocation.
///
/// The window comes from the driver's own output-layout walk, which is the same
/// routine the declared-count path uses, so a runtime-sized buffer and a
/// declared one cannot drift apart on where the result ends.
pub(super) fn output_readback(
    slot: usize,
    binding_index: usize,
    buffer: &vyre_foundation::ir::BufferDecl,
    allocated_bytes: usize,
) -> Result<OutputReadback, BackendError> {
    let element_size = buffer.element().size_bytes().ok_or_else(|| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Vulkan output `{}` uses a runtime-sized element type. Lower it to a fixed-width GPU storage type before SPIR-V dispatch.",
                buffer.name()
            ),
        }
    })?;
    let resolved_count = u32::try_from(allocated_bytes / element_size.max(1)).map_err(|_| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Vulkan output `{}` resolves to more elements than a u32 counts. Split the Program buffer before SPIR-V dispatch.",
                buffer.name()
            ),
        }
    })?;
    let layout = vyre_driver::output_binding_layout_parts(
        buffer.binding(),
        &buffer.name,
        &buffer.element,
        resolved_count,
        buffer.output_byte_range(),
    )?;
    Ok(OutputReadback {
        slot,
        binding_index,
        trim_start: layout.layout.trim_start,
        read_size: layout.layout.read_size,
    })
}

/// Zero one host-visible allocation and copy `input` over its prefix.
///
/// Vulkan returns whatever the memory type's free list held, and a kernel
/// writes only the elements its guard admits, so the rest of the allocation
/// stays as that content and is read back as a result. Every other driver
/// clears a writable binding it does not preserve; this clears the whole
/// allocation and restores the caller's bytes over it, which is the same
/// observable state for a preserved binding and zeroes for every other one.
///
/// # Safety
/// `memory` must be a live host-visible allocation of at least `byte_len`
/// bytes that no descriptor is reading.
pub(super) unsafe fn fill_host_buffer(
    device: &VulkanDevice,
    memory: vk::DeviceMemory,
    byte_len: usize,
    input: Option<&[u8]>,
) -> Result<(), BackendError> {
    if byte_len == 0 {
        return Ok(());
    }
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let ptr = unsafe {
        device
            .device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan memory map failed: {e}. Fix: check memory type is host-visible."
        ))
    })?;
    // SAFETY: The host memory was successfully mapped and covers `byte_len` bytes.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr.cast::<u8>(), byte_len) };
    let uploaded = match input {
        Some(input) => {
            slice[..input.len()].copy_from_slice(input);
            input.len()
        }
        None => 0,
    };
    slice[uploaded..].fill(0);
    // SAFETY: Memory was mapped successfully just above and is unmapped once the writes land.
    unsafe { device.device.unmap_memory(memory) };
    Ok(())
}

/// Bind the reserved trap-diagnostic sidecar when the module declares one.
///
/// Returns its index in `dispatch_bindings`, or `None` for a module that emits
/// no trap. The descriptor set is built from `dispatch_bindings`, so a binding
/// appended here reaches the layout, the pool, and the descriptor writes.
///
/// Any other descriptor the module declares and the Program does not is
/// refused: binding a zeroed scratch buffer under it would hand the kernel
/// storage the caller never allocated.
pub(super) fn bind_trap_sidecar(
    device: &VulkanDevice,
    spv_words: &[u32],
    dispatch_bindings: &mut Vec<DispatchBinding>,
) -> Result<Option<usize>, BackendError> {
    let declared = crate::module_bindings::module_descriptors(spv_words)?;
    let mut sidecar = None;
    for descriptor in declared {
        if descriptor.set != DESCRIPTOR_SET {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: SPIR-V module declares binding {} in descriptor set {}, but this driver binds one set ({DESCRIPTOR_SET}). Emit every storage buffer into set {DESCRIPTOR_SET}.",
                    descriptor.binding, descriptor.set
                ),
            });
        }
        if dispatch_bindings
            .iter()
            .any(|bound| bound.binding == descriptor.binding)
        {
            continue;
        }
        if sidecar.is_some() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: SPIR-V module declares binding {} that no Program buffer owns, beyond the reserved trap sidecar. Keep Program buffers synchronized with the lowered descriptor bindings.",
                    descriptor.binding
                ),
            });
        }
        let byte_len = vyre_driver::trap_record::TRAP_RECORD_BYTES;
        let vk_byte_len = vk::DeviceSize::try_from(byte_len).map_err(|_| {
            BackendError::new(
                "Vulkan trap sidecar length exceeds vk::DeviceSize. Fix: keep TRAP_RECORD_BYTES within the device size ABI.".to_string(),
            )
        })?;
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let (vk_buffer, vk_memory) = unsafe { device.create_host_buffer(vk_byte_len) }?;
        // SAFETY: The allocation was created host-visible immediately above and
        // is not yet bound to a descriptor.
        unsafe { fill_host_buffer(device, vk_memory, byte_len, None) }?;
        sidecar = Some(dispatch_bindings.len());
        dispatch_bindings.push(DispatchBinding {
            buffer: vk_buffer,
            memory: vk_memory,
            byte_len,
            binding: descriptor.binding,
        });
    }
    Ok(sidecar)
}

/// Decode the trap record one launch left in its sidecar.
///
/// The tag table is not carried here. It is produced by lowering, and a table
/// that arrives from a second lowering of the program can name a different tag
/// for the same code than the module encodes, which reports the wrong refusal;
/// the neutral decoder renders the bare code instead.
///
/// # Safety
/// `binding` must name a live host-visible allocation the completed launch
/// wrote.
pub(super) unsafe fn read_trap_record(
    device: &VulkanDevice,
    binding: &DispatchBinding,
) -> Result<Option<BackendError>, BackendError> {
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let bytes = unsafe { map_binding_bytes(device, binding) }?;
    let Some(record) = vyre_driver::trap_record::decode_trap_record(&bytes)? else {
        return Ok(None);
    };
    Ok(Some(BackendError::new(format!(
        "SPIR-V dispatch trapped: {}",
        record.describe(|_| None)
    ))))
}

/// Copy one binding's whole allocation out of device-visible memory.
///
/// # Safety
/// `binding` must name a live host-visible allocation of `binding.byte_len`
/// bytes that no launch is still writing.
pub(super) unsafe fn map_binding_bytes(
    device: &VulkanDevice,
    binding: &DispatchBinding,
) -> Result<Vec<u8>, BackendError> {
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let ptr = unsafe {
        device.device.map_memory(
            binding.memory,
            0,
            vk::WHOLE_SIZE,
            vk::MemoryMapFlags::empty(),
        )
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan memory map for readback failed: {e}. Fix: check memory type is host-visible."
        ))
    })?;
    // SAFETY: The host memory was successfully mapped and covers `byte_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), binding.byte_len) }.to_vec();
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    unsafe { device.device.unmap_memory(binding.memory) };
    Ok(bytes)
}

/// Read every declared output into the slot the ABI plan assigned it.
///
/// `BindingPlan::bindings` is ordered by VYRE binding number and the output
/// vector is ordered by `output_index`, which the plan assigns in program
/// buffer-declaration order. Pushing in iteration order silently returns the
/// right buffers in the wrong slots whenever those two orders differ:
/// `vyre-libs::nn::top_k` returned its index buffer where the caller reads
/// values, and `vyre-libs::parsing::python312_lexer` returned its match count
/// where the caller reads tokens. 21 of 41 diverging operations on an RTX 4090
/// were this and nothing else.
///
/// # Safety
/// Every allocation named by `output_bindings` must be live, host-visible, and
/// no longer written by a launch.
pub(super) unsafe fn read_output_slots(
    device: &VulkanDevice,
    dispatch_bindings: &[DispatchBinding],
    output_bindings: &[OutputReadback],
) -> Result<Vec<Vec<u8>>, BackendError> {
    let mut slots: Vec<Option<Vec<u8>>> = vec![None; output_bindings.len()];
    for readback in output_bindings {
        let b = &dispatch_bindings[readback.binding_index];
        let end = readback.trim_start + readback.read_size;
        if end > b.byte_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: binding `{}` declares a visible output window of {} bytes at offset {}, past the {} bytes allocated for it. Declare an output byte range inside the buffer.",
                    b.binding, readback.read_size, readback.trim_start, b.byte_len
                ),
            });
        }
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let bytes = unsafe { map_binding_bytes(device, b) }?;
        let slot = slots
            .get_mut(readback.slot)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: binding `{}` claims output slot {} of {} produced outputs. Rebuild BindingPlan from Program::buffers order before readback.",
                    b.binding,
                    readback.slot,
                    output_bindings.len()
                ),
            })?;
        if slot.is_some() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: two bindings claim output slot {}; one of them would be discarded. Rebuild BindingPlan from Program::buffers order before readback.",
                    readback.slot
                ),
            });
        }
        *slot = Some(bytes[readback.trim_start..end].to_vec());
    }
    slots
        .into_iter()
        .enumerate()
        .map(|(output_index, bytes)| {
            bytes.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: no binding produced output slot {output_index}; the caller would read an empty buffer as a result. Rebuild BindingPlan from Program::buffers order before readback."
                ),
            })
        })
        .collect()
}
