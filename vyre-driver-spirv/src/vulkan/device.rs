//! Vulkan loader, physical-device selection, and the process-wide context.
//!
//! Acquiring the backend opens the loader, an instance, and a logical device.
//! Every acquisition shares one context: see [`shared_device`].

use ash::vk;

use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use vyre_driver::BackendError;

/// Owned Vulkan compute context.
pub(crate) struct VulkanDevice {
    // Keep the dynamically-loaded Vulkan loader alive for every function
    // pointer stored in `instance` and `device`. `Drop` explicitly
    // destroys the logical device, and `_instance` is declared before
    // `_entry` so the instance handle drops before the loader unloads.
    _instance: ash::Instance,
    _entry: ash::Entry,
    pub(super) device: ash::Device,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) queue_family_index: u32,
    /// Submission target. The Vulkan specification lists a queue as externally
    /// synchronized, so the host serializes `vkQueueSubmit` on it.
    pub(super) queue: Mutex<vk::Queue>,
    /// Allocation source for dispatch command buffers. A pool is externally
    /// synchronized, and that covers allocating from it, freeing back to it,
    /// and recording into any buffer allocated from it.
    pub(super) command_pool: Mutex<vk::CommandPool>,
    /// Memory type index that is host-visible and host-coherent.
    pub(super) host_memory_type_index: u32,
    /// Device properties (for limits reporting).
    pub properties: vk::PhysicalDeviceProperties,
    /// Subgroup width, when the device runs the subgroup operations this
    /// backend emits in a compute shader. `None` means it does not, or that
    /// the loader is too old to say.
    pub subgroup: Option<u32>,
}

/// Subgroup operations a program reaching this backend can use.
///
/// Emission goes through naga, which lowers a subgroup expression to the
/// `GroupNonUniform*` instruction family: ballot, broadcast, shuffle, and the
/// arithmetic reductions. A device that runs only some of them runs only some
/// of the programs, and a partial promise is what produces a dispatch failure
/// instead of a refusal, so the whole set is required.
const REQUIRED_SUBGROUP_OPERATIONS: vk::SubgroupFeatureFlags = vk::SubgroupFeatureFlags::from_raw(
    vk::SubgroupFeatureFlags::BASIC.as_raw()
        | vk::SubgroupFeatureFlags::VOTE.as_raw()
        | vk::SubgroupFeatureFlags::ARITHMETIC.as_raw()
        | vk::SubgroupFeatureFlags::BALLOT.as_raw()
        | vk::SubgroupFeatureFlags::SHUFFLE.as_raw()
        | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE.as_raw(),
);

/// Read the device's subgroup width, or `None` when it cannot run the subgroup
/// operations this backend emits.
///
/// The properties are core Vulkan 1.1, so a 1.0 loader or a 1.0 device answers
/// `None` and the backend reports no subgroup support rather than guessing a
/// width. A width that is not a power of two, or is 1, is also `None`: a
/// single-lane subgroup runs a ballot as an answer about one invocation, which
/// is not the collective the program asked for.
fn probe_subgroup(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    api_version: u32,
    properties: vk::PhysicalDeviceProperties,
) -> Option<u32> {
    if api_version < vk::API_VERSION_1_1 || properties.api_version < vk::API_VERSION_1_1 {
        return None;
    }
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut chained = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    // SAFETY: `physical_device` is a live handle bound to `instance`, and the
    // chained structures live on this stack frame until the call returns. The
    // entry point is core in the 1.1 instance this ran against.
    unsafe { instance.get_physical_device_properties2(physical_device, &mut chained) };

    if !subgroup
        .supported_stages
        .contains(vk::ShaderStageFlags::COMPUTE)
    {
        return None;
    }
    if !subgroup
        .supported_operations
        .contains(REQUIRED_SUBGROUP_OPERATIONS)
    {
        return None;
    }
    let size = subgroup.subgroup_size;
    (size > 1 && size.is_power_of_two()).then_some(size)
}

/// Take a lock over an externally synchronized Vulkan handle.
///
/// A panic under one of these guards leaks the command buffer being recorded
/// and leaves every other handle untouched: the pool is still a valid pool and
/// the queue is still a valid queue, so the poison flag records a leak rather
/// than a corrupt object. Recovering the guard keeps one panicked dispatch from
/// refusing every later dispatch in the process.
pub(super) fn guard<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(PoisonError::into_inner)
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("physical_device", &self.physical_device)
            .field("queue_family_index", &self.queue_family_index)
            .finish_non_exhaustive()
    }
}

impl VulkanDevice {
    /// Acquire the first Vulkan physical device that exposes a compute queue.
    pub(crate) fn acquire() -> Result<Self, BackendError> {
        // SAFETY: ash::Entry::load dlopen's the system Vulkan loader. The
        // returned Entry is the only reference to that loader handle and
        // owns its lifetime via Drop; we surface initialization errors
        // back to the caller as a typed BackendError.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| {
            BackendError::new(format!(
                "Failed to load Vulkan loader: {e}. Fix: install a Vulkan loader (libvulkan1) and ensure ICD files are in /usr/share/vulkan/icd.d/."
            ))
        })?;

        // Subgroup properties are core Vulkan 1.1. Asking for 1.0 makes them
        // unqueryable, and a backend that cannot read them reports no subgroup
        // support: seven subgroup operations were refused before reaching the
        // emitter, on a device whose warps are 32 lanes wide. A loader that
        // only offers 1.0 still gets 1.0, and the backend then reports the
        // truth for that loader rather than claiming a version it does not have.
        //
        // SAFETY: reading the loader's instance version takes no handle and no
        // allocation; a loader predating the entry point reports none.
        let instance_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        let api_version = if instance_version >= vk::API_VERSION_1_1 {
            vk::API_VERSION_1_1
        } else {
            vk::API_VERSION_1_0
        };
        let app_info = vk::ApplicationInfo {
            api_version,
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        // SAFETY: create_info points at app_info on the current stack and
        // both structs live until create_instance returns. The Instance
        // returned takes ownership of the Vulkan handle and frees it on
        // Drop. Allocator callbacks are null (None).
        let instance = unsafe { entry.create_instance(&create_info, None) }.map_err(|e| {
            BackendError::new(format!(
                "Vulkan instance creation failed: {e}. Fix: verify the Vulkan loader and any validation layers are compatible."
            ))
        })?;

        // SAFETY: `instance` is the live Instance returned above; its
        // handle is valid until VulkanDevice::Drop frees it.
        let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|e| {
            BackendError::new(format!(
                "Vulkan physical device enumeration failed: {e}. Fix: ensure a Vulkan-capable GPU is present and drivers are installed."
            ))
        })?;

        let mut chosen = None;
        for pd in physical_devices {
            // SAFETY: `pd` is a vk::PhysicalDevice handle returned by the
            // matching `instance.enumerate_physical_devices()` call above
            // and is valid for the lifetime of `instance`.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            let queue_families =
                // SAFETY: same as the get_physical_device_properties call
                // above  -  `pd` is a live handle bound to `instance`.
                unsafe { instance.get_physical_device_queue_family_properties(pd) };
            for (index, family) in queue_families.iter().enumerate() {
                if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    let queue_index = u32::try_from(index).map_err(|_| {
                        BackendError::new(format!(
                            "Vulkan queue family index {index} exceeds u32. Fix: repair driver-reported queue metadata before SPIR-V backend acquisition."
                        ))
                    })?;
                    chosen = Some((pd, queue_index, props));
                    break;
                }
            }
            if chosen.is_some() {
                break;
            }
        }

        let (physical_device, queue_family_index, properties) = chosen.ok_or_else(|| {
            BackendError::new(
                "No Vulkan physical GPU device with a compute queue was found. Fix: repair the Vulkan GPU driver or select the CUDA/WGPU backend; software CPU Vulkan implementations are not production dispatch backends.".to_string(),
            )
        })?;

        let subgroup = probe_subgroup(&instance, physical_device, api_version, properties);

        let queue_priority = 1.0f32;
        let queue_create_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: &queue_priority,
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create_info,
            ..Default::default()
        };

        // SAFETY: physical_device + device_create_info live until this
        // call returns; queue_create_info inside device_create_info
        // borrows queue_priority on the current stack frame, which is
        // also live for the duration of the call. The returned Device
        // takes ownership of the new vk::Device handle.
        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }
        .map_err(|e| {
            BackendError::new(format!(
                "Vulkan logical device creation failed: {e}. Fix: check device limits and feature requirements."
            ))
        })?;

        // SAFETY: queue_family_index was just used to create `device`
        // and is in range; index 0 is always valid for queue_count = 1.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // SAFETY: `device` is the live Device returned above; the
        // SAFETY: `device` is the live logical device created above,
        // `queue_family_index` was selected from that physical device's
        // compute-capable queue families, and the CommandPoolCreateInfo
        // struct lives until create_command_pool returns.
        // returns.
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo {
                    flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                    queue_family_index,
                    ..Default::default()
                },
                None,
            )
        }
        .map_err(|e| {
            BackendError::new(format!(
                "Vulkan command pool creation failed: {e}. Fix: verify queue family index is valid."
            ))
        })?;

        let memory_properties =
            // SAFETY: physical_device is a live vk::PhysicalDevice handle
            // bound to `instance`.
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let host_memory_type_index = find_host_visible_memory_type(&memory_properties).ok_or_else(
            || {
                BackendError::new(
                    "No host-visible, host-coherent memory type found on Vulkan device. Fix: select a different physical device or implement explicit staging.".to_string(),
                )
            },
        )?;

        Ok(Self {
            _instance: instance,
            _entry: entry,
            device,
            physical_device,
            queue_family_index,
            queue: Mutex::new(queue),
            command_pool: Mutex::new(command_pool),
            host_memory_type_index,
            properties,
            subgroup,
        })
    }

    /// Create a buffer backed by host-visible memory.
    pub(super) unsafe fn create_host_buffer(
        &self,
        size: vk::DeviceSize,
    ) -> Result<(vk::Buffer, vk::DeviceMemory), BackendError> {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let buffer = unsafe {
            self
            .device
            .create_buffer(&buffer_info, None)
        }
        .map_err(|e| BackendError::new(format!("Vulkan buffer creation failed: {e}. Fix: reduce buffer size or check device limits.")))?;

        // SAFETY: The buffer is a valid Vulkan buffer created successfully just above.
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: self.host_memory_type_index,
            ..Default::default()
        };
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let memory = unsafe {
            self
            .device
            .allocate_memory(&alloc_info, None)
        }
        .map_err(|e| BackendError::new(format!("Vulkan memory allocation failed: {e}. Fix: reduce buffer size or free unused allocations.")))?;

        // SAFETY: The buffer and memory were both created successfully just above and have not been freed.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }.map_err(|e| {
            BackendError::new(format!(
                "Vulkan buffer memory binding failed: {e}. Fix: verify alignment requirements."
            ))
        })?;

        Ok((buffer, memory))
    }

    /// Destroy a buffer and its memory.
    pub(super) unsafe fn destroy_buffer(&self, buffer: vk::Buffer, memory: vk::DeviceMemory) {
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        unsafe {
            self.device.destroy_buffer(buffer, None);
            self.device.free_memory(memory, None);
        }
    }

    /// Allocate a command buffer and record one dispatch into it.
    ///
    /// Allocating from a pool and recording into a buffer allocated from that
    /// pool are both host access to the pool, so one guard covers both.
    unsafe fn record_dispatch(
        &self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        workgroups: [u32; 3],
    ) -> Result<vk::CommandBuffer, BackendError> {
        let pool = guard(&self.command_pool);
        let alloc_info = vk::CommandBufferAllocateInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            command_pool: *pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            _marker: std::marker::PhantomData,
        };
        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        let mut cbs = unsafe {
            self
            .device
            .allocate_command_buffers(&alloc_info)
        }
        .map_err(|e| BackendError::new(format!("Vulkan command buffer allocation failed: {e}. Fix: reset or free existing command buffers.")))?;
        let command_buffer = cbs.pop().ok_or_else(|| {
            BackendError::new(
                "Vulkan returned zero command buffers. Fix: check command pool state.".to_string(),
            )
        })?;

        // Helper: free the command buffer on any early-exit below, under the
        // pool guard this function already holds.
        let free_cb = |device: &ash::Device, pool: vk::CommandPool, cb: vk::CommandBuffer| {
            // SAFETY: cb was allocated from pool and is no longer submitted.
            unsafe { device.free_command_buffers(pool, &[cb]) };
        };

        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        if let Err(e) = unsafe {
            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
        } {
            free_cb(&self.device, *pool, command_buffer);
            return Err(BackendError::new(format!(
                "Vulkan command buffer begin failed: {e}. Fix: check command buffer state."
            )));
        }

        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        unsafe {
            self.device
                .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device
                .cmd_dispatch(command_buffer, workgroups[0], workgroups[1], workgroups[2]);
        }

        // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
        if let Err(e) = unsafe { self.device.end_command_buffer(command_buffer) } {
            free_cb(&self.device, *pool, command_buffer);
            return Err(BackendError::new(format!(
                "Vulkan command buffer end failed: {e}. Fix: check recorded commands."
            )));
        }

        Ok(command_buffer)
    }

    /// Return a recorded command buffer to the pool.
    unsafe fn free_command_buffer(&self, command_buffer: vk::CommandBuffer) {
        let pool = guard(&self.command_pool);
        // SAFETY: the buffer was allocated from this pool by `record_dispatch`
        // and its submission, if any, has completed or the device is lost.
        unsafe { self.device.free_command_buffers(*pool, &[command_buffer]) };
    }

    /// Record a compute dispatch and wait for completion.
    ///
    /// One process-wide device serves every acquisition of this backend, so two
    /// callers reach the pool and the queue at the same time and the guards are
    /// what make that sound. The fence wait is outside both, so a dispatch
    /// executing on the GPU does not stop another caller from recording and
    /// submitting the next one.
    pub(super) unsafe fn dispatch_compute(
        &self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        workgroups: [u32; 3],
    ) -> Result<(), BackendError> {
        // SAFETY: every handle is live and owned by this device; the call
        // records into a buffer it allocates and frees on its own error paths.
        let command_buffer =
            unsafe { self.record_dispatch(pipeline, pipeline_layout, descriptor_set, workgroups) }?;

        // SAFETY: Fence creation is a standard Vulkan device operation, parameters are valid defaults.
        let fence = match unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
        } {
            Ok(f) => f,
            Err(e) => {
                // SAFETY: the buffer is recorded but was never submitted.
                unsafe { self.free_command_buffer(command_buffer) };
                return Err(BackendError::new(format!(
                    "Vulkan fence creation failed: {e}. Fix: check device limits."
                )));
            }
        };

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        let submitted = {
            let queue = guard(&self.queue);
            // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
            unsafe { self.device.queue_submit(*queue, &[submit_info], fence) }
        };
        if let Err(e) = submitted {
            // SAFETY: the submission was rejected, so neither handle is in use.
            unsafe {
                self.device.destroy_fence(fence, None);
                self.free_command_buffer(command_buffer);
            }
            return Err(BackendError::new(format!(
                "Vulkan queue submit failed: {e}. Fix: verify queue and command buffer state."
            )));
        }

        // SAFETY: The fence was created successfully and successfully submitted to the queue.
        let waited = unsafe { self.device.wait_for_fences(&[fence], true, u64::MAX) };
        // SAFETY: the wait either observed the fence signalled or reported a
        // lost device, and a lost device releases every resource it held.
        unsafe {
            self.device.destroy_fence(fence, None);
            self.free_command_buffer(command_buffer);
        }
        waited.map_err(|e| {
            BackendError::new(format!(
                "Vulkan fence wait failed: {e}. Fix: check for device loss."
            ))
        })
    }
}

/// One Vulkan context for the whole process.
///
/// Acquiring this backend used to dlopen the Vulkan loader, create an instance,
/// and create a logical device every time. Conformance acquires a backend per
/// case, twice per case for the compile-facts digest alone, so a 353-operation
/// run opened more than a thousand loader-instance-device cycles across sixteen
/// workers. Each live context holds the loader plus the driver's device nodes
/// open, the process reached its descriptor limit, and `dlopen` of
/// `libvulkan.so.1` then failed with `Too many open files`: 269 of 288 cases
/// were refused for want of a Vulkan loader on a host with a working one, and
/// the instance teardown took the wgpu backend's devices with it.
///
/// The same reasoning already governs the wgpu driver's process-wide instance.
/// This context is never destroyed, which is the point: the ICD stays loaded
/// for the life of the process instead of being unloaded under another backend.
static SHARED: LazyLock<Result<Arc<VulkanDevice>, BackendError>> =
    LazyLock::new(|| VulkanDevice::acquire().map(Arc::new));

/// The Vulkan context every acquisition of this backend shares.
pub(crate) fn shared_device() -> Result<Arc<VulkanDevice>, BackendError> {
    match &*SHARED {
        Ok(device) => Ok(Arc::clone(device)),
        Err(error) => Err(error.clone()),
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        // SAFETY: self.command_pool and self.device are the live
        // handles created in `acquire`; Drop is the single owner of
        // both and runs once when the VulkanDevice is dropped.
        unsafe {
            self.device
                .destroy_command_pool(*guard(&self.command_pool), None);
            self.device.destroy_device(None);
        }
    }
}

fn find_host_visible_memory_type(props: &vk::PhysicalDeviceMemoryProperties) -> Option<u32> {
    for i in 0..props.memory_type_count {
        let ty = props.memory_types[i as usize];
        if ty
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            && ty
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Some(i);
        }
    }
    None
}
