//! The one owner of "forward every `VyreBackend` method to an inner backend".
//!
//! A decorator that wraps a backend has to restate the whole 57-method contract
//! to forward it, and both decorators in this crate restated it by hand. That
//! is not only text: a method left out of the list does not fail to compile, it
//! falls through to the trait default. `GridSyncSplitBackend` dropped seven that
//! way, so `cooperative_grid_sync_fits`, `supports_distributed_collectives` and
//! the whole device-buffer surface answered for the wrapper instead of for the
//! backend inside it: a device-buffer-capable backend reported
//! `UnsupportedFeature` the moment the grid-sync registry wrapped it.
//!
//! # Using these
//!
//! Both macros expand inside an `impl VyreBackend for _` block and forward to a
//! field named `inner` that derefs to a `VyreBackend`. Every type in the emitted
//! signatures is fully qualified, so the calling module needs no imports beyond
//! the trait itself.
//!
//! The split between the two is the split a decorator actually cares about:
//!
//! - [`forward_vyre_backend_support`] is identity, capability queries, resident
//!   resource management, device buffers and lifecycle. Nothing here inspects a
//!   `Program`, so a decorator that changes how programs are dispatched still
//!   forwards all of it verbatim.
//! - [`forward_vyre_backend_dispatch`] is every entry point that carries a
//!   `Program`. A decorator that specializes dispatch writes these itself and
//!   must write ALL of them, or leave one on a trait default that routes back
//!   through `self`: this is the surface where forwarding one and specializing
//!   another is a semantic bug rather than a missing capability. The only
//!   transparent forwarder of this half is the test double that wants the real
//!   contract rather than the trait defaults, so it is `#[cfg(test)]`.
//!
//! `tests/vyre_backend_forwarding_closure.rs` derives the method set from this
//! crate's own trait declaration at run time and fails when a new method belongs
//! to neither macro.

/// Forward every non-dispatch `VyreBackend` method to `self.inner`.
///
/// See the [module docs](self) for the surface this covers and why it is split
/// from [`forward_vyre_backend_dispatch`].
macro_rules! forward_vyre_backend_support {
    () => {
        fn id(&self) -> &'static str {
            self.inner.id()
        }
        fn version(&self) -> &'static str {
            self.inner.version()
        }
        fn supported_ops(&self) -> &::std::collections::HashSet<::vyre_foundation::ir::OpId> {
            self.inner.supported_ops()
        }
        fn allocate_resident(
            &self,
            byte_len: usize,
        ) -> Result<$crate::backend::Resource, $crate::backend::BackendError> {
            self.inner.allocate_resident(byte_len)
        }
        fn upload_resident(
            &self,
            resource: &$crate::backend::Resource,
            bytes: &[u8],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.upload_resident(resource, bytes)
        }
        fn upload_resident_many(
            &self,
            uploads: &[(&$crate::backend::Resource, &[u8])],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.upload_resident_many(uploads)
        }
        fn upload_resident_at(
            &self,
            resource: &$crate::backend::Resource,
            dst_offset_bytes: usize,
            bytes: &[u8],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.upload_resident_at(resource, dst_offset_bytes, bytes)
        }
        fn upload_resident_at_many(
            &self,
            uploads: &[(&$crate::backend::Resource, usize, &[u8])],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.upload_resident_at_many(uploads)
        }
        fn download_resident(
            &self,
            resource: &$crate::backend::Resource,
        ) -> Result<Vec<u8>, $crate::backend::BackendError> {
            self.inner.download_resident(resource)
        }
        fn download_resident_into(
            &self,
            resource: &$crate::backend::Resource,
            out: &mut Vec<u8>,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.download_resident_into(resource, out)
        }
        fn download_resident_range(
            &self,
            resource: &$crate::backend::Resource,
            byte_offset: usize,
            byte_len: usize,
        ) -> Result<Vec<u8>, $crate::backend::BackendError> {
            self.inner.download_resident_range(resource, byte_offset, byte_len)
        }
        fn download_resident_range_into(
            &self,
            resource: &$crate::backend::Resource,
            byte_offset: usize,
            byte_len: usize,
            out: &mut Vec<u8>,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.download_resident_range_into(resource, byte_offset, byte_len, out)
        }
        fn download_resident_ranges_into(
            &self,
            ranges: &[(&$crate::backend::Resource, usize, usize)],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.download_resident_ranges_into(ranges, outputs)
        }
        fn free_resident(
            &self,
            resource: $crate::backend::Resource,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.free_resident(resource)
        }
        fn pipeline_cache_snapshot(&self) -> Option<$crate::pipeline::PipelineCacheSnapshot> {
            self.inner.pipeline_cache_snapshot()
        }
        fn backend_metric_snapshot(&self) -> Vec<(&'static str, u64)> {
            self.inner.backend_metric_snapshot()
        }
        fn supports_subgroup_ops(&self) -> bool {
            self.inner.supports_subgroup_ops()
        }
        fn supports_f16(&self) -> bool {
            self.inner.supports_f16()
        }
        fn supports_bf16(&self) -> bool {
            self.inner.supports_bf16()
        }
        fn supports_tensor_cores(&self) -> bool {
            self.inner.supports_tensor_cores()
        }
        fn supports_async_compute(&self) -> bool {
            self.inner.supports_async_compute()
        }
        fn supports_indirect_dispatch(&self) -> bool {
            self.inner.supports_indirect_dispatch()
        }
        fn supports_speculation(&self) -> bool {
            self.inner.supports_speculation()
        }
        fn supports_persistent_thread_dispatch(&self) -> bool {
            self.inner.supports_persistent_thread_dispatch()
        }
        fn supports_grid_sync(&self) -> bool {
            self.inner.supports_grid_sync()
        }
        fn cooperative_grid_sync_fits(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&[u8]],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<bool, $crate::backend::BackendError> {
            self.inner.cooperative_grid_sync_fits(program, inputs, config)
        }
        fn allows_host_grid_sync_split(&self) -> bool {
            self.inner.allows_host_grid_sync_split()
        }
        fn supports_resident_dispatch(&self) -> bool {
            self.inner.supports_resident_dispatch()
        }
        fn is_distributed(&self) -> bool {
            self.inner.is_distributed()
        }
        fn supports_distributed_collectives(&self) -> bool {
            self.inner.supports_distributed_collectives()
        }
        fn max_workgroup_size(&self) -> [u32; 3] {
            self.inner.max_workgroup_size()
        }
        fn max_compute_workgroups_per_dimension(&self) -> u32 {
            self.inner.max_compute_workgroups_per_dimension()
        }
        fn max_compute_invocations_per_workgroup(&self) -> u32 {
            self.inner.max_compute_invocations_per_workgroup()
        }
        fn subgroup_size(&self) -> Option<u32> {
            self.inner.subgroup_size()
        }
        fn max_storage_buffer_bytes(&self) -> u64 {
            self.inner.max_storage_buffer_bytes()
        }
        fn device_profile(&self) -> $crate::DeviceProfile {
            self.inner.device_profile()
        }
        fn prepare(&self) -> Result<(), $crate::backend::BackendError> {
            self.inner.prepare()
        }
        fn flush(&self) -> Result<(), $crate::backend::BackendError> {
            self.inner.flush()
        }
        fn shutdown(&self) -> Result<(), $crate::backend::BackendError> {
            self.inner.shutdown()
        }
        fn device_lost(&self) -> bool {
            self.inner.device_lost()
        }
        fn try_recover(&self) -> Result<(), $crate::backend::BackendError> {
            self.inner.try_recover()
        }
        fn allocate_device_buffer(
            &self,
            byte_len: usize,
        ) -> Result<Box<dyn $crate::backend::DeviceBuffer>, $crate::backend::BackendError> {
            self.inner.allocate_device_buffer(byte_len)
        }
        fn upload_device_buffer(
            &self,
            buffer: &mut dyn $crate::backend::DeviceBuffer,
            bytes: &[u8],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.upload_device_buffer(buffer, bytes)
        }
        fn download_device_buffer(
            &self,
            buffer: &dyn $crate::backend::DeviceBuffer,
        ) -> Result<Vec<u8>, $crate::backend::BackendError> {
            self.inner.download_device_buffer(buffer)
        }
        fn free_device_buffer(
            &self,
            buffer: Box<dyn $crate::backend::DeviceBuffer>,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.free_device_buffer(buffer)
        }
    };
}

/// Forward every `Program`-carrying `VyreBackend` method to `self.inner`.
///
/// A decorator that specializes dispatch must not use this. Nothing in the
/// shipped driver forwards this half unchanged, so it is compiled only for the
/// test double that wants the real contract rather than the trait defaults.
#[cfg(test)]
macro_rules! forward_vyre_backend_dispatch {
    () => {
        fn dispatch(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[Vec<u8>],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, $crate::backend::BackendError> {
            self.inner.dispatch(program, inputs, config)
        }
        fn dispatch_borrowed(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&[u8]],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<Vec<Vec<u8>>, $crate::backend::BackendError> {
            self.inner.dispatch_borrowed(program, inputs, config)
        }
        fn dispatch_borrowed_timed(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&[u8]],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<$crate::backend::TimedDispatchResult, $crate::backend::BackendError> {
            self.inner.dispatch_borrowed_timed(program, inputs, config)
        }
        fn dispatch_borrowed_into(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&[u8]],
            config: &$crate::backend::DispatchConfig,
            outputs: &mut $crate::backend::OutputBuffers,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.dispatch_borrowed_into(program, inputs, config, outputs)
        }
        fn dispatch_resident_timed(
            &self,
            program: &::vyre_foundation::ir::Program,
            resources: &[$crate::backend::Resource],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<$crate::backend::TimedDispatchResult, $crate::backend::BackendError> {
            self.inner.dispatch_resident_timed(program, resources, config)
        }
        fn dispatch_resident_async(
            &self,
            program: &::vyre_foundation::ir::Program,
            resources: &[$crate::backend::Resource],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<Box<dyn $crate::backend::PendingDispatch>, $crate::backend::BackendError> {
            self.inner.dispatch_resident_async(program, resources, config)
        }
        fn dispatch_resident_sequence_read_ranges_into(
            &self,
            steps: &[$crate::backend::ResidentDispatchStep<'_>],
            read_ranges: &[$crate::backend::ResidentReadRange<'_>],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.dispatch_resident_sequence_read_ranges_into(steps, read_ranges, outputs)
        }
        fn dispatch_resident_sequence_read_ranges_timed_into(
            &self,
            steps: &[$crate::backend::ResidentDispatchStep<'_>],
            read_ranges: &[$crate::backend::ResidentReadRange<'_>],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<$crate::backend::ResidentSequenceTiming, $crate::backend::BackendError> {
            self.inner
                .dispatch_resident_sequence_read_ranges_timed_into(steps, read_ranges, outputs)
        }
        fn dispatch_resident_repeated_sequence_read_ranges_into(
            &self,
            prefix_steps: &[$crate::backend::ResidentDispatchStep<'_>],
            repeated_steps: &[$crate::backend::ResidentDispatchStep<'_>],
            repeat_count: u32,
            read_ranges: &[$crate::backend::ResidentReadRange<'_>],
            outputs: &mut [&mut Vec<u8>],
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner
                .dispatch_resident_repeated_sequence_read_ranges_into(
                    prefix_steps,
                    repeated_steps,
                    repeat_count,
                    read_ranges,
                    outputs,
                )
        }
        fn dispatch_async(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[Vec<u8>],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<Box<dyn $crate::backend::PendingDispatch>, $crate::backend::BackendError> {
            self.inner.dispatch_async(program, inputs, config)
        }
        fn dispatch_borrowed_async(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&[u8]],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<Box<dyn $crate::backend::PendingDispatch>, $crate::backend::BackendError> {
            self.inner.dispatch_borrowed_async(program, inputs, config)
        }
        fn dispatch_with_device_buffers(
            &self,
            program: &::vyre_foundation::ir::Program,
            inputs: &[&dyn $crate::backend::DeviceBuffer],
            outputs: &mut [&mut dyn $crate::backend::DeviceBuffer],
            config: &$crate::backend::DispatchConfig,
        ) -> Result<(), $crate::backend::BackendError> {
            self.inner.dispatch_with_device_buffers(program, inputs, outputs, config)
        }
    };
}

pub(crate) use forward_vyre_backend_support;
#[cfg(test)]
pub(crate) use forward_vyre_backend_dispatch;
