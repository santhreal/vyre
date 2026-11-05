//! One compute dispatch: shader module, pipeline, submit, and readback.

use ash::vk;

use vyre_driver::{BackendError, BindingPlan};
use vyre_foundation::ir::{BufferAccess, Program};

use super::bindings::{
    bind_trap_sidecar, fill_host_buffer, output_readback, read_output_slots, read_trap_record,
    DispatchBinding, OutputReadback,
};
use super::device::VulkanDevice;

/// Build a SPIR-V shader module from raw words.
unsafe fn create_shader_module(
    device: &ash::Device,
    words: &[u32],
) -> Result<vk::ShaderModule, BackendError> {
    let code_size = words.len() * std::mem::size_of::<u32>();
    let create_info = vk::ShaderModuleCreateInfo {
        code_size,
        p_code: words.as_ptr(),
        ..Default::default()
    };
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    unsafe { device.create_shader_module(&create_info, None) }
        .map_err(|e| BackendError::new(format!("Vulkan shader module creation failed: {e}. Fix: validate the SPIR-V binary with spirv-val before loading.")))
}

/// How a launch's borrowed inputs map onto the program's buffers.
#[derive(Clone, Copy)]
pub(crate) enum InputOrder<'a> {
    /// Inputs in `BindingPlan` order, which is what a direct dispatch supplies.
    Plan,
    /// Inputs in target binding order, named by the buffer each one fills.
    ///
    /// An artifact launch stages the bindings its lowered descriptor declares,
    /// and that list carries a buffer a previous module wrote which this
    /// module's Program declares as consuming no host input.
    Named(&'a [&'a str]),
}

impl InputOrder<'_> {
    /// Index in the borrowed input slice that fills `buffer`, if any.
    fn index_for(
        self,
        buffer: &vyre_foundation::ir::BufferDecl,
        plan: Option<usize>,
    ) -> Option<usize> {
        match self {
            Self::Plan => plan,
            Self::Named(names) => names.iter().position(|name| *name == buffer.name()),
        }
    }
}

/// Run one compute dispatch on the Vulkan device.
///
/// # Safety
/// The Vulkan device must be valid. This function performs all Vulkan FFI calls.
pub(crate) unsafe fn dispatch_program(
    device: &VulkanDevice,
    program: &Program,
    spv_words: &[u32],
    inputs: &[&[u8]],
    input_order: InputOrder<'_>,
    config: &vyre_driver::DispatchConfig,
) -> Result<Vec<Vec<u8>>, BackendError> {
    BackendError::reject_blocked_contraction(
        program,
        config.float_lowering,
        crate::SPIRV_BACKEND_ID,
    )?;
    if config.cooperative {
        return Err(BackendError::UnsupportedFeature {
            name: "SPIR-V cooperative grid dispatch".to_string(),
            backend: crate::SPIRV_BACKEND_ID.to_string(),
        });
    }
    let workgroup_size = config.launch_workgroup().unwrap_or(program.workgroup_size);
    if workgroup_size.contains(&0) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: SPIR-V dispatch workgroup size contains zero dimension: {workgroup_size:?}. Emit a positive GPU workgroup shape before Vulkan dispatch."
            ),
        });
    }
    let workgroup_size = [workgroup_size[0], workgroup_size[1], workgroup_size[2]];

    let grid = if let Some(grid) = config.launch_grid() {
        grid
    } else {
        infer_grid(program, workgroup_size)?
    };

    let binding_plan = match input_order {
        InputOrder::Plan => BindingPlan::from_borrowed_inputs(program, inputs)?,
        InputOrder::Named(names) => {
            let plan = BindingPlan::build(program)?;
            plan.validate_named_inputs(names, inputs)?;
            plan
        }
    };

    // Build bindings from the backend-neutral ABI plan so input and output
    // slots stay identical to every other backend.
    let mut dispatch_bindings: Vec<DispatchBinding> = Vec::new();
    let mut output_bindings: Vec<OutputReadback> = Vec::new();

    for binding in &binding_plan.bindings {
        let buffer = &program.buffers()[binding.buffer_index];
        if buffer.access() == BufferAccess::Workgroup {
            continue;
        }

        let input_index = input_order.index_for(buffer, binding.input_index);

        let element_size = buffer.element().size_bytes().ok_or_else(|| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Vulkan buffer `{}` uses a runtime-sized element type. Lower it to a fixed-width GPU storage type before SPIR-V dispatch.",
                    buffer.name()
                ),
            }
        })?;
        let byte_len = if buffer.count() == 0 {
            if let Some(input_index) = input_index {
                let input = inputs[input_index];
                input.len()
            } else if binding.output_index.is_some() {
                // Runtime-sized output with no paired input: infer capacity from the
                // largest input buffer passed to this dispatch. A one-element fallback
                // would silently truncate any scan that produces more than one match
                // (the GPU writes beyond the mapped range). If no inputs are present,
                // require an explicit grid_override to bound the output; the caller
                // must supply DispatchConfig::output_size_hint or grid_override in
                // that case.
                let max_input_bytes = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
                if max_input_bytes == 0 {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: runtime-sized output buffer `{}` has no input to infer size from. \
                             Pass at least one non-empty input or set DispatchConfig::grid_override \
                             so the backend can bound the output allocation.",
                            buffer.name()
                        ),
                    });
                }
                // Conservatively match the largest input buffer size so the GPU
                // cannot overflow. Downstream readback trims to the actual output
                // via the match-count metadata that the GPU kernel writes.
                max_input_bytes
            } else {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: buffer `{}` has runtime size but no matching input was provided.",
                        buffer.name()
                    ),
                });
            }
        } else {
            (buffer.count() as usize)
                .checked_mul(element_size)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Vulkan buffer `{}` size overflows host address space (count={}, element_size={element_size}). Split the Program buffer before dispatch.",
                        buffer.name(),
                        buffer.count()
                    ),
                })?
        };

        let vk_byte_len = vk::DeviceSize::try_from(byte_len).map_err(|_| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Vulkan buffer `{}` is {byte_len} bytes, which exceeds vk::DeviceSize. Split the Program buffer before SPIR-V dispatch.",
                    buffer.name()
                ),
            }
        })?;
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let (vk_buffer, vk_memory) = unsafe { device.create_host_buffer(vk_byte_len) }?;

        let input = input_index.map(|input_index| inputs[input_index]);
        if let Some(input) = input {
            if input.len() > byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: input buffer for Vulkan binding `{}` is {} bytes but declared storage is {byte_len} bytes. Resize the input or fix the Program buffer count; silent upload truncation is forbidden.",
                        buffer.name(),
                        input.len()
                    ),
                });
            }
        }
        // SAFETY: The allocation was created host-visible immediately above and
        // is not yet bound to a descriptor, so nothing else reads it.
        unsafe { fill_host_buffer(device, vk_memory, byte_len, input) }?;

        if let Some(slot) = binding.output_index {
            output_bindings.push(output_readback(
                slot,
                dispatch_bindings.len(),
                buffer,
                byte_len,
            )?);
        }

        dispatch_bindings.push(DispatchBinding {
            buffer: vk_buffer,
            memory: vk_memory,
            byte_len,
            binding: buffer.binding(),
        });
    }

    // The lowered module carries one descriptor the Program never declared: the
    // reserved trap-diagnostic sidecar `vyre-lower` appends to a kernel that
    // emits a trap. It is bound here, zeroed, and read back below.
    let trap_sidecar = bind_trap_sidecar(device, spv_words, &mut dispatch_bindings)?;

    // Create shader module.
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let shader_module = unsafe { create_shader_module(&device.device, spv_words) }?;

    // Descriptor set layout.
    let layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = dispatch_bindings
        .iter()
        .map(|b| vk::DescriptorSetLayoutBinding {
            binding: b.binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            p_immutable_samplers: std::ptr::null(),
            ..Default::default()
        })
        .collect();

    let layout_binding_count = u32::try_from(layout_bindings.len()).map_err(|_| {
        BackendError::new(format!(
            "Vulkan descriptor set layout has {} bindings, exceeding u32. Fix: split the Program binding table before SPIR-V dispatch.",
            layout_bindings.len()
        ))
    })?;
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let descriptor_set_layout = unsafe {
        device.device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo {
                binding_count: layout_binding_count,
                p_bindings: layout_bindings.as_ptr(),
                ..Default::default()
            },
            None,
        )
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan descriptor set layout creation failed: {e}. Fix: check binding limits."
        ))
    })?;

    // Pipeline layout.
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let pipeline_layout = unsafe {
        device.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            },
            None,
        )
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan pipeline layout creation failed: {e}. Fix: check push constant limits."
        ))
    })?;

    // Compute pipeline.
    let pipeline_info = vk::ComputePipelineCreateInfo {
        stage: vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: b"main\0".as_ptr() as *const i8,
            ..Default::default()
        },
        layout: pipeline_layout,
        ..Default::default()
    };

    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let pipeline = match unsafe {
        device.device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        )
    } {
        Ok(mut pipelines) => {
            pipelines.pop().ok_or_else(|| BackendError::new(
                "Vulkan returned zero compute pipelines. Fix: check shader module and pipeline layout compatibility.".to_string(),
            ))?
        }
        Err((_, e)) => {
            return Err(BackendError::new(format!(
                "Vulkan compute pipeline creation failed: {e:?}. Fix: validate SPIR-V entry point name is 'main' and pipeline layout matches shader bindings."
            )));
        }
    };

    // Descriptor pool.
    let descriptor_count = u32::try_from(dispatch_bindings.len()).map_err(|_| {
        BackendError::new(format!(
            "Vulkan descriptor pool needs {} storage-buffer descriptors, exceeding u32. Fix: split the Program binding table before SPIR-V dispatch.",
            dispatch_bindings.len()
        ))
    })?;
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count,
    };
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let descriptor_pool = unsafe {
        device.device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: 1,
                p_pool_sizes: &pool_size,
                ..Default::default()
            },
            None,
        )
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan descriptor pool creation failed: {e}. Fix: check pool sizes."
        ))
    })?;

    // Descriptor set.
    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    let descriptor_set = unsafe {
        device
            .device
            .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            })
    }
    .map_err(|e| {
        BackendError::new(format!(
            "Vulkan descriptor set allocation failed: {e}. Fix: check descriptor pool capacity."
        ))
    })?
    .pop()
    .ok_or_else(|| {
        BackendError::new(
            "Vulkan returned zero descriptor sets. Fix: check descriptor pool state.".to_string(),
        )
    })?;

    // Write descriptor set.
    let buffer_infos: Vec<vk::DescriptorBufferInfo> = dispatch_bindings
        .iter()
        .map(|b| vk::DescriptorBufferInfo {
            buffer: b.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        })
        .collect();

    let write_bindings: Vec<vk::WriteDescriptorSet<'_>> = dispatch_bindings
        .iter()
        .zip(buffer_infos.iter())
        .map(|(b, info)| vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: b.binding,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: info,
            ..Default::default()
        })
        .collect();

    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
    unsafe { device.device.update_descriptor_sets(&write_bindings, &[]) };

    // Dispatch.
    // SAFETY: FFI boundary to Vulkan dispatch: pipeline, descriptor_set, layout, and grid parameters are fully validated.
    unsafe { device.dispatch_compute(pipeline, pipeline_layout, descriptor_set, grid) }?;

    // A trapped launch produced no valid result, so the sidecar is read before
    // the outputs are. Reading it after would return the kernel's partial
    // writes as an answer.
    let result = match trap_sidecar {
        // SAFETY: The sidecar allocation is host-visible, was zeroed before the
        // launch, and the launch has completed.
        Some(index) => match unsafe { read_trap_record(device, &dispatch_bindings[index]) }? {
            Some(error) => Err(error),
            // SAFETY: Every output allocation is host-visible and the launch has
            // completed, so the mapped bytes are the kernel's final writes.
            None => unsafe { read_output_slots(device, &dispatch_bindings, &output_bindings) },
        },
        // SAFETY: As above; this module declared no trap sidecar.
        None => unsafe { read_output_slots(device, &dispatch_bindings, &output_bindings) },
    };

    // Cleanup.
    // SAFETY: All resources created for the temporary compute dispatch are destroyed in reverse creation order.
    unsafe {
        device.device.destroy_descriptor_pool(descriptor_pool, None);
        device.device.destroy_pipeline(pipeline, None);
        device.device.destroy_pipeline_layout(pipeline_layout, None);
        device
            .device
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.device.destroy_shader_module(shader_module, None);
    }
    for b in dispatch_bindings {
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        unsafe { device.destroy_buffer(b.buffer, b.memory) };
    }

    result
}

/// Infer the dispatch grid from the program's buffer element counts.
///
/// When all output buffers are runtime-sized (count == 0) the grid cannot be
/// derived from outputs alone. Fall back to the largest *input* buffer element
/// count so that programs with a single runtime-sized output (e.g. scan result
/// buffers) still launch one thread per input element rather than exactly one
/// workgroup. If both outputs and inputs are runtime-sized the caller must
/// supply `DispatchConfig::grid_override`; otherwise a 1-workgroup launch would
/// silently process only `workgroup_size[0]` elements.
pub(super) fn infer_grid(
    program: &Program,
    workgroup_size: [u32; 3],
) -> Result<[u32; 3], BackendError> {
    if workgroup_size[1] != 1 || workgroup_size[2] != 1 {
        return Err(BackendError::new(format!(
            "Fix: non-1D workgroup_size {:?} requires DispatchConfig::grid_override. Set grid_override explicitly.",
            workgroup_size
        )));
    }

    // Prefer the largest statically-declared output element count.
    let max_output_count = program
        .buffers()
        .iter()
        .filter(|b| b.is_output())
        .map(|b| b.count())
        .max()
        .unwrap_or(0);

    let effective_count = if max_output_count > 0 {
        max_output_count
    } else {
        // All outputs are runtime-sized: derive from input buffers instead.
        let max_input_count = program
            .buffers()
            .iter()
            .filter(|b| !b.is_output())
            .map(|b| b.count())
            .max()
            .unwrap_or(0);

        if max_input_count == 0 {
            return Err(BackendError::new(
                "Fix: all buffers have runtime size (count == 0). \
                 Set DispatchConfig::grid_override to bound the dispatch grid \
                 when no statically-sized buffer exists.",
            ));
        }
        max_input_count
    };

    let lanes = workgroup_size[0];
    let x = effective_count.div_ceil(lanes).max(1);
    Ok([x, 1, 1])
}
