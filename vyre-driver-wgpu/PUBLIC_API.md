pub mod vyre_driver_wgpu
pub mod vyre_driver_wgpu::buffer
pub struct vyre_driver_wgpu::buffer::BindGroupCache
impl vyre_driver_wgpu::buffer::BindGroupCache
pub fn vyre_driver_wgpu::buffer::BindGroupCache::get_or_create(&self, layout_id: usize, handles: &[vyre_driver_wgpu::buffer::GpuBufferHandle], factory: impl core::ops::function::FnOnce() -> wgpu::api::bind_group::BindGroup) -> alloc::sync::Arc<wgpu::api::bind_group::BindGroup>
pub fn vyre_driver_wgpu::buffer::BindGroupCache::new() -> Self
pub fn vyre_driver_wgpu::buffer::BindGroupCache::stats(&self) -> vyre_driver_wgpu::buffer::BindGroupCacheStats
pub fn vyre_driver_wgpu::buffer::BindGroupCache::with_cap(cap: usize) -> Self
impl core::default::Default for vyre_driver_wgpu::buffer::BindGroupCache
pub fn vyre_driver_wgpu::buffer::BindGroupCache::default() -> Self
impl core::fmt::Debug for vyre_driver_wgpu::buffer::BindGroupCache
pub fn vyre_driver_wgpu::buffer::BindGroupCache::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
pub struct vyre_driver_wgpu::buffer::BindGroupCacheStats
pub vyre_driver_wgpu::buffer::BindGroupCacheStats::entries: usize
pub vyre_driver_wgpu::buffer::BindGroupCacheStats::evictions: usize
pub vyre_driver_wgpu::buffer::BindGroupCacheStats::hits: usize
pub vyre_driver_wgpu::buffer::BindGroupCacheStats::misses: usize
pub struct vyre_driver_wgpu::buffer::BufferPool
impl vyre_driver_wgpu::buffer::BufferPool
pub fn vyre_driver_wgpu::buffer::BufferPool::acquire(&self, len: u64, usage: wgpu_types::BufferUsages) -> core::result::Result<vyre_driver_wgpu::buffer::GpuBufferHandle, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::BufferPool::device(&self) -> &wgpu::api::device::Device
pub fn vyre_driver_wgpu::buffer::BufferPool::new(device: wgpu::api::device::Device, queue: wgpu::api::queue::Queue, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> Self
pub fn vyre_driver_wgpu::buffer::BufferPool::queue(&self) -> &wgpu::api::queue::Queue
pub fn vyre_driver_wgpu::buffer::BufferPool::release(&self, handle: vyre_driver_wgpu::buffer::GpuBufferHandle)
pub fn vyre_driver_wgpu::buffer::BufferPool::stats(&self) -> vyre_driver_wgpu::buffer::BufferPoolStats
pub fn vyre_driver_wgpu::buffer::BufferPool::with_tiering(device: wgpu::api::device::Device, queue: wgpu::api::queue::Queue, config: &vyre_driver::backend::dispatch_config::DispatchConfig, tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
impl core::fmt::Debug for vyre_driver_wgpu::buffer::BufferPool
pub fn vyre_driver_wgpu::buffer::BufferPool::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
pub struct vyre_driver_wgpu::buffer::BufferPoolStats
pub vyre_driver_wgpu::buffer::BufferPoolStats::allocations: usize
pub vyre_driver_wgpu::buffer::BufferPoolStats::evictions: usize
pub vyre_driver_wgpu::buffer::BufferPoolStats::hits: usize
pub vyre_driver_wgpu::buffer::BufferPoolStats::releases: usize
pub vyre_driver_wgpu::buffer::BufferPoolStats::retained_bytes: usize
pub struct vyre_driver_wgpu::buffer::GpuBufferHandle
impl vyre_driver_wgpu::buffer::GpuBufferHandle
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::alloc(device: &wgpu::api::device::Device, len: u64, usage: wgpu_types::BufferUsages) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::allocation_len(&self) -> u64
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::buffer(&self) -> &wgpu::api::buffer::Buffer
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::buffer_arc(&self) -> alloc::sync::Arc<wgpu::api::buffer::Buffer>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::byte_len(&self) -> u64
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::element_count(&self) -> usize
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::from_resident_handle(handle: vyre_driver::backend::resource::ResidentHandle, context: &str) -> core::result::Result<core::option::Option<Self>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::from_resident_id(id: u64) -> core::option::Option<Self>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::id(&self) -> u64
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::readback(&self, device: &wgpu::api::device::Device, queue: &wgpu::api::queue::Queue, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::readback_prefix(&self, device: &wgpu::api::device::Device, queue: &wgpu::api::queue::Queue, len: u64, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::readback_range(&self, device: &wgpu::api::device::Device, queue: &wgpu::api::queue::Queue, byte_offset: u64, len: u64, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::resident_handle(&self) -> core::result::Result<vyre_driver::backend::resource::ResidentHandle, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::upload(device: &wgpu::api::device::Device, queue: &wgpu::api::queue::Queue, bytes: &[u8], usage: wgpu_types::BufferUsages) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::usage(&self) -> wgpu_types::BufferUsages
impl core::convert::From<vyre_driver_wgpu::buffer::GpuBufferHandle> for vyre_driver_wgpu::engine::graph::GpuResource
pub fn vyre_driver_wgpu::engine::graph::GpuResource::from(handle: vyre_driver_wgpu::buffer::GpuBufferHandle) -> Self
impl core::fmt::Debug for vyre_driver_wgpu::buffer::GpuBufferHandle
pub fn vyre_driver_wgpu::buffer::GpuBufferHandle::fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
pub struct vyre_driver_wgpu::buffer::StagingBufferPool
impl vyre_driver_wgpu::buffer::StagingBufferPool
pub fn vyre_driver_wgpu::buffer::StagingBufferPool::acquire(&self, device: &wgpu::api::device::Device, size: u64, usage: wgpu_types::BufferUsages) -> wgpu::api::buffer::Buffer
pub fn vyre_driver_wgpu::buffer::StagingBufferPool::new() -> Self
pub fn vyre_driver_wgpu::buffer::StagingBufferPool::release(&self, buffer: wgpu::api::buffer::Buffer, size: u64, usage: wgpu_types::BufferUsages)
pub fn vyre_driver_wgpu::buffer::StagingBufferPool::stats(&self) -> vyre_driver_wgpu::buffer::StagingBufferPoolStats
impl core::fmt::Debug for vyre_driver_wgpu::buffer::StagingBufferPool
pub fn vyre_driver_wgpu::buffer::StagingBufferPool::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
pub struct vyre_driver_wgpu::buffer::StagingBufferPoolStats
pub vyre_driver_wgpu::buffer::StagingBufferPoolStats::allocations: usize
pub vyre_driver_wgpu::buffer::StagingBufferPoolStats::hits: usize
pub mod vyre_driver_wgpu::emit
pub struct vyre_driver_wgpu::emit::WgpuBindingAssignment
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::access: vyre_spec::buffer_access::BufferAccess
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::binding: u32
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::element: vyre_spec::data_type::DataType
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::group: u32
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::kind: vyre_foundation::ir_inner::model::program::MemoryKind
pub vyre_driver_wgpu::emit::WgpuBindingAssignment::name: alloc::sync::Arc<str>
pub struct vyre_driver_wgpu::emit::WgpuDispatchGeometry
pub vyre_driver_wgpu::emit::WgpuDispatchGeometry::workgroup_size: [u32; 3]
pub vyre_driver_wgpu::emit::WgpuDispatchGeometry::workgroups: [u32; 3]
pub struct vyre_driver_wgpu::emit::WgpuProgram
pub vyre_driver_wgpu::emit::WgpuProgram::bindings: alloc::vec::Vec<vyre_driver_wgpu::emit::WgpuBindingAssignment>
pub vyre_driver_wgpu::emit::WgpuProgram::dispatch_geometry: vyre_driver_wgpu::emit::WgpuDispatchGeometry
pub vyre_driver_wgpu::emit::WgpuProgram::module: naga::ir::Module
pub vyre_driver_wgpu::emit::WgpuProgram::workgroup_size: [u32; 3]
impl vyre_driver_wgpu::emit::WgpuProgram
pub fn vyre_driver_wgpu::emit::WgpuProgram::from_program(program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig, enabled_features: &vyre_driver_wgpu::runtime::device::EnabledFeatures) -> core::result::Result<Self, vyre_foundation::lower::LoweringError>
pub fn vyre_driver_wgpu::emit::lower(program: &vyre_foundation::ir_inner::model::program::core::Program) -> core::result::Result<alloc::string::String, vyre_foundation::lower::LoweringError>
pub fn vyre_driver_wgpu::emit::lower_with_config(program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::string::String, vyre_foundation::lower::LoweringError>
pub mod vyre_driver_wgpu::engine
pub mod vyre_driver_wgpu::engine::graph
pub enum vyre_driver_wgpu::engine::graph::GpuResource
pub vyre_driver_wgpu::engine::graph::GpuResource::Borrowed(alloc::vec::Vec<u8>)
pub vyre_driver_wgpu::engine::graph::GpuResource::Resident(vyre_driver_wgpu::buffer::GpuBufferHandle)
impl core::convert::From<alloc::vec::Vec<u8>> for vyre_driver_wgpu::engine::graph::GpuResource
pub fn vyre_driver_wgpu::engine::graph::GpuResource::from(bytes: alloc::vec::Vec<u8>) -> Self
impl core::convert::From<vyre_driver_wgpu::buffer::GpuBufferHandle> for vyre_driver_wgpu::engine::graph::GpuResource
pub fn vyre_driver_wgpu::engine::graph::GpuResource::from(handle: vyre_driver_wgpu::buffer::GpuBufferHandle) -> Self
pub struct vyre_driver_wgpu::engine::graph::GpuDispatchGraph
impl vyre_driver_wgpu::engine::graph::GpuDispatchGraph
pub fn vyre_driver_wgpu::engine::graph::GpuDispatchGraph::dispatch(&self, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::graph::GpuDispatchGraph::is_empty(&self) -> bool
pub fn vyre_driver_wgpu::engine::graph::GpuDispatchGraph::len(&self) -> usize
pub fn vyre_driver_wgpu::engine::graph::GpuDispatchGraph::new() -> Self
pub fn vyre_driver_wgpu::engine::graph::GpuDispatchGraph::push(&mut self, pipeline: vyre_driver_wgpu::pipeline::WgpuPipeline, input: impl core::convert::Into<vyre_driver_wgpu::engine::graph::GpuResource>)
pub struct vyre_driver_wgpu::engine::graph::LaunchAccounting
pub vyre_driver_wgpu::engine::graph::LaunchAccounting::graph_submissions: usize
pub vyre_driver_wgpu::engine::graph::LaunchAccounting::sequential_submissions: usize
impl vyre_driver_wgpu::engine::graph::LaunchAccounting
pub fn vyre_driver_wgpu::engine::graph::LaunchAccounting::reduction_factor(self) -> usize
pub fn vyre_driver_wgpu::engine::graph::launch_accounting(op_count: usize) -> vyre_driver_wgpu::engine::graph::LaunchAccounting
pub mod vyre_driver_wgpu::engine::multi_gpu
pub struct vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem<'a>
pub vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem::config: &'a vyre_driver::backend::dispatch_config::DispatchConfig
pub vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem::cost: u64
pub vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem::id: usize
pub vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem::inputs: &'a [&'a [u8]]
pub vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem::program: &'a vyre_foundation::ir_inner::model::program::core::Program
pub struct vyre_driver_wgpu::engine::multi_gpu::DeviceLoad
pub vyre_driver_wgpu::engine::multi_gpu::DeviceLoad::device_index: usize
pub vyre_driver_wgpu::engine::multi_gpu::DeviceLoad::queued_cost: u64
pub struct vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem::config: vyre_driver::backend::dispatch_config::DispatchConfig
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem::cost: u64
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem::id: usize
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem::inputs: alloc::vec::Vec<alloc::vec::Vec<u8>>
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem::program: vyre_foundation::ir_inner::model::program::core::Program
pub struct vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput::adapter_index: usize
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput::id: usize
pub vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput::outputs: alloc::vec::Vec<alloc::vec::Vec<u8>>
pub struct vyre_driver_wgpu::engine::multi_gpu::LiveGpu
pub vyre_driver_wgpu::engine::multi_gpu::LiveGpu::adapter_index: usize
pub vyre_driver_wgpu::engine::multi_gpu::LiveGpu::info: wgpu_types::AdapterInfo
pub struct vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor
impl vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::acquire_all() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::acquire_indices(indices: &[usize]) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::adapter_indices(&self) -> alloc::vec::Vec<usize>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::dispatch_batch(&mut self, items: alloc::vec::Vec<vyre_driver_wgpu::engine::multi_gpu::GpuWorkItem>) -> core::result::Result<alloc::vec::Vec<vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::dispatch_borrowed_batch(&mut self, items: &[vyre_driver_wgpu::engine::multi_gpu::BorrowedGpuWorkItem<'_>]) -> core::result::Result<alloc::vec::Vec<core::result::Result<vyre_driver_wgpu::engine::multi_gpu::GpuWorkOutput, vyre_driver::backend::error::BackendError>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::enumerate_live_gpus() -> alloc::vec::Vec<vyre_driver_wgpu::engine::multi_gpu::LiveGpu>
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::is_empty(&self) -> bool
pub fn vyre_driver_wgpu::engine::multi_gpu::MultiGpuExecutor::len(&self) -> usize
pub struct vyre_driver_wgpu::engine::multi_gpu::Partition
pub vyre_driver_wgpu::engine::multi_gpu::Partition::device_index: usize
pub vyre_driver_wgpu::engine::multi_gpu::Partition::item_ids: alloc::vec::Vec<usize>
pub vyre_driver_wgpu::engine::multi_gpu::Partition::total_cost: u64
pub struct vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator
impl vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator
pub fn vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator::assign(&mut self, key: &[u8], cost: u64) -> core::result::Result<core::option::Option<u32>, vyre_driver_wgpu::engine::multi_gpu::stream_shard::StreamShardError>
pub fn vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator::load(&self) -> &[u64]
pub fn vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator::new(n_gpus: u32, spill_threshold: u64) -> core::result::Result<Self, vyre_driver_wgpu::engine::multi_gpu::stream_shard::StreamShardError>
pub fn vyre_driver_wgpu::engine::multi_gpu::StreamShardAllocator::seed_load(&mut self, device: u32, cost: u64) -> core::result::Result<(), vyre_driver_wgpu::engine::multi_gpu::stream_shard::StreamShardError>
pub struct vyre_driver_wgpu::engine::multi_gpu::WeightedWorkItem
pub vyre_driver_wgpu::engine::multi_gpu::WeightedWorkItem::cost: u64
pub vyre_driver_wgpu::engine::multi_gpu::WeightedWorkItem::id: usize
pub fn vyre_driver_wgpu::engine::multi_gpu::live_gpu_loads() -> core::result::Result<alloc::vec::Vec<vyre_driver_wgpu::engine::multi_gpu::DeviceLoad>, alloc::string::String>
pub fn vyre_driver_wgpu::engine::multi_gpu::partition_work_stealing(devices: &[vyre_driver_wgpu::engine::multi_gpu::DeviceLoad], items: &[vyre_driver_wgpu::engine::multi_gpu::WeightedWorkItem]) -> core::result::Result<alloc::vec::Vec<vyre_driver_wgpu::engine::multi_gpu::Partition>, alloc::string::String>
pub fn vyre_driver_wgpu::engine::multi_gpu::shard_by_blake3(key: &[u8], n_gpus: u32) -> core::result::Result<u32, vyre_driver_wgpu::engine::multi_gpu::stream_shard::StreamShardError>
pub type vyre_driver_wgpu::engine::multi_gpu::StreamShardError = vyre_driver_wgpu::engine::multi_gpu::stream_shard::StreamShardError
pub mod vyre_driver_wgpu::engine::persistent
pub struct vyre_driver_wgpu::engine::persistent::PersistentKernelReport
pub vyre_driver_wgpu::engine::persistent::PersistentKernelReport::kernel_launches: u32
pub vyre_driver_wgpu::engine::persistent::PersistentKernelReport::results: alloc::vec::Vec<vyre_driver_wgpu::engine::persistent::WorkResult>
pub struct vyre_driver_wgpu::engine::persistent::PersistentPayloadWorkItem
pub vyre_driver_wgpu::engine::persistent::PersistentPayloadWorkItem::id: u32
pub vyre_driver_wgpu::engine::persistent::PersistentPayloadWorkItem::payload: alloc::vec::Vec<u8>
pub struct vyre_driver_wgpu::engine::persistent::PersistentQueue
impl vyre_driver_wgpu::engine::persistent::PersistentQueue
pub fn vyre_driver_wgpu::engine::persistent::PersistentQueue::is_empty(&self) -> bool
pub fn vyre_driver_wgpu::engine::persistent::PersistentQueue::len(&self) -> usize
pub fn vyre_driver_wgpu::engine::persistent::PersistentQueue::new() -> Self
pub fn vyre_driver_wgpu::engine::persistent::PersistentQueue::push(&mut self, item: vyre_driver_wgpu::engine::persistent::PersistentPayloadWorkItem)
pub struct vyre_driver_wgpu::engine::persistent::WorkResult
pub vyre_driver_wgpu::engine::persistent::WorkResult::id: u32
pub vyre_driver_wgpu::engine::persistent::WorkResult::payload: alloc::vec::Vec<u8>
pub fn vyre_driver_wgpu::engine::persistent::run_persistent_kernel(backend: &vyre_driver_wgpu::WgpuBackend, program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig, queue: vyre_driver_wgpu::engine::persistent::PersistentQueue) -> core::result::Result<vyre_driver_wgpu::engine::persistent::PersistentKernelReport, vyre_driver::backend::error::BackendError>
pub mod vyre_driver_wgpu::engine::streaming
pub mod vyre_driver_wgpu::engine::streaming::async_copy
pub struct vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams
impl vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams
pub fn vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams::async_load<F>(&mut self, tag: impl core::convert::Into<alloc::string::String>, copy: F) -> core::result::Result<(), vyre_driver::backend::error::BackendError> where F: core::ops::function::FnOnce() -> core::result::Result<(), vyre_driver::backend::error::BackendError> + core::marker::Send + 'static
pub fn vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams::async_wait(&mut self, tag: &str) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams::new() -> Self
pub fn vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams::overlap_copy_compute<C, G>(&mut self, tag: impl core::convert::Into<alloc::string::String>, copy: C, compute: G) -> core::result::Result<(), vyre_driver::backend::error::BackendError> where C: core::ops::function::FnOnce() -> core::result::Result<(), vyre_driver::backend::error::BackendError> + core::marker::Send + 'static, G: core::ops::function::FnOnce() -> core::result::Result<(), vyre_driver::backend::error::BackendError>
impl core::ops::drop::Drop for vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams
pub fn vyre_driver_wgpu::engine::streaming::async_copy::AsyncCopyStreams::drop(&mut self)
pub struct vyre_driver_wgpu::engine::streaming::HostIngressStream
impl vyre_driver_wgpu::engine::streaming::HostIngressStream
pub fn vyre_driver_wgpu::engine::streaming::HostIngressStream::finish(&mut self) -> core::result::Result<core::option::Option<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::engine::streaming::HostIngressStream::from_runner<F>(runner: F, config: vyre_driver::backend::dispatch_config::DispatchConfig) -> Self where F: core::ops::function::Fn(alloc::vec::Vec<u8>, vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError> + core::marker::Send + core::marker::Sync + 'static
pub fn vyre_driver_wgpu::engine::streaming::HostIngressStream::new(pipeline: vyre_driver_wgpu::pipeline::WgpuPipeline, config: vyre_driver::backend::dispatch_config::DispatchConfig) -> Self
pub fn vyre_driver_wgpu::engine::streaming::HostIngressStream::push_chunk(&mut self, bytes: alloc::vec::Vec<u8>) -> core::result::Result<core::option::Option<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
pub mod vyre_driver_wgpu::ext
pub mod vyre_driver_wgpu::pipeline
pub use vyre_driver_wgpu::pipeline::IndirectDispatch
pub use vyre_driver_wgpu::pipeline::OutputLayout
pub use vyre_driver_wgpu::pipeline::output_layout_from_program
pub mod vyre_driver_wgpu::pipeline::persistent
pub struct vyre_driver_wgpu::pipeline::persistent::DispatchItem<'a>
pub vyre_driver_wgpu::pipeline::persistent::DispatchItem::inputs: &'a [vyre_driver_wgpu::buffer::GpuBufferHandle]
pub vyre_driver_wgpu::pipeline::persistent::DispatchItem::outputs: &'a [vyre_driver_wgpu::buffer::GpuBufferHandle]
pub vyre_driver_wgpu::pipeline::persistent::DispatchItem::params: core::option::Option<&'a vyre_driver_wgpu::buffer::GpuBufferHandle>
pub vyre_driver_wgpu::pipeline::persistent::DispatchItem::workgroups: [u32; 3]
pub struct vyre_driver_wgpu::pipeline::BindGroupCacheStats
pub vyre_driver_wgpu::pipeline::BindGroupCacheStats::entries: usize
pub vyre_driver_wgpu::pipeline::BindGroupCacheStats::evictions: usize
pub vyre_driver_wgpu::pipeline::BindGroupCacheStats::hits: usize
pub vyre_driver_wgpu::pipeline::BindGroupCacheStats::misses: usize
pub struct vyre_driver_wgpu::pipeline::DispatchItem<'a>
pub vyre_driver_wgpu::pipeline::DispatchItem::inputs: &'a [vyre_driver_wgpu::buffer::GpuBufferHandle]
pub vyre_driver_wgpu::pipeline::DispatchItem::outputs: &'a [vyre_driver_wgpu::buffer::GpuBufferHandle]
pub vyre_driver_wgpu::pipeline::DispatchItem::params: core::option::Option<&'a vyre_driver_wgpu::buffer::GpuBufferHandle>
pub vyre_driver_wgpu::pipeline::DispatchItem::workgroups: [u32; 3]
pub struct vyre_driver_wgpu::pipeline::WgpuPipeline
impl vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::bind_group_cache_stats(&self) -> vyre_driver_wgpu::buffer::BindGroupCacheStats
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent(&self, inputs: &[vyre_driver_wgpu::buffer::GpuBufferHandle], outputs: &mut [vyre_driver_wgpu::buffer::GpuBufferHandle], params: core::option::Option<&vyre_driver_wgpu::buffer::GpuBufferHandle>, workgroups: [u32; 3]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_batched(&self, items: &[vyre_driver_wgpu::pipeline::persistent::DispatchItem<'_>]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_borrowed(&self, inputs: &[&vyre_driver_wgpu::buffer::GpuBufferHandle], outputs: &[&vyre_driver_wgpu::buffer::GpuBufferHandle], params: core::option::Option<&vyre_driver_wgpu::buffer::GpuBufferHandle>, workgroups: [u32; 3]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::compile(program: &vyre_foundation::ir_inner::model::program::core::Program) -> core::result::Result<alloc::sync::Arc<Self>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::compile_with_config(program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::sync::Arc<Self>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::execution_plan(&self) -> &vyre_foundation::execution_plan::ExecutionPlan
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::push_chunk(&self, bytes: &[u8], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_coalesced(&self, inputs: &[alloc::vec::Vec<u8>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_coalesced_borrowed(&self, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_compound_v2(requests: &[(&vyre_driver_wgpu::pipeline::WgpuPipeline, vyre_driver::backend::resource::Resource)], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<alloc::vec::Vec<u8>>>, vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::prerecord_borrowed_dispatch(&self, inputs: &[&[u8]], workgroups: [u32; 3]) -> core::result::Result<vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::prerecord_persistent_dispatch(&self, inputs: &[vyre_driver_wgpu::buffer::GpuBufferHandle], outputs: &[vyre_driver_wgpu::buffer::GpuBufferHandle], params: core::option::Option<&vyre_driver_wgpu::buffer::GpuBufferHandle>, workgroups: [u32; 3]) -> core::result::Result<vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch, vyre_driver::backend::error::BackendError>
impl core::fmt::Debug for vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
impl vyre_driver::backend::compiled_pipeline::CompiledPipeline for vyre_driver_wgpu::pipeline::WgpuPipeline
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch(&self, inputs: &[alloc::vec::Vec<u8>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_borrowed(&self, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_borrowed_batched(&self, batches: &[&[&[u8]]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<vyre_driver::backend::dispatch_result::OutputBuffers>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_borrowed_batched_into(&self, batches: &[&[&[u8]]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, batch_outputs: &mut alloc::vec::Vec<vyre_driver::backend::dispatch_result::OutputBuffers>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_borrowed_into(&self, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, outputs: &mut vyre_driver::backend::dispatch_result::OutputBuffers) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_borrowed_timed(&self, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver::backend::dispatch_result::TimedDispatchResult, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_handles(&self, inputs: &[vyre_driver::backend::resource::Resource], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver::backend::dispatch_result::OutputBuffers, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_handles_batched(&self, batches: &[&[vyre_driver::backend::resource::Resource]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<vyre_driver::backend::dispatch_result::OutputBuffers>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_handles_batched_into(&self, batches: &[&[vyre_driver::backend::resource::Resource]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, batch_outputs: &mut alloc::vec::Vec<vyre_driver::backend::dispatch_result::OutputBuffers>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_handles_into(&self, inputs: &[vyre_driver::backend::resource::Resource], config: &vyre_driver::backend::dispatch_config::DispatchConfig, outputs: &mut vyre_driver::backend::dispatch_result::OutputBuffers) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_handles_timed(&self, inputs: &[vyre_driver::backend::resource::Resource], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver::backend::dispatch_result::TimedDispatchResult, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::dispatch_persistent_resource_outputs(&self, inputs: &[vyre_driver::backend::resource::Resource], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<vyre_driver::backend::resource::Resource>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::pipeline::WgpuPipeline::id(&self) -> &str
impl vyre_driver::backend::private::Sealed for vyre_driver_wgpu::pipeline::WgpuPipeline
pub mod vyre_driver_wgpu::runtime
pub mod vyre_driver_wgpu::runtime::adapter_caps_probe
pub fn vyre_driver_wgpu::runtime::adapter_caps_probe::from_backend(adapter_info: &wgpu_types::AdapterInfo, device_limits: &wgpu_types::Limits, enabled: &vyre_driver_wgpu::runtime::device::EnabledFeatures) -> vyre_foundation::optimizer::ctx::AdapterCaps
pub fn vyre_driver_wgpu::runtime::adapter_caps_probe::from_backend_profile(adapter_info: &wgpu_types::AdapterInfo, device_limits: &wgpu_types::Limits, enabled: &vyre_driver_wgpu::runtime::device::EnabledFeatures) -> vyre_driver::device_profile::DeviceProfile
pub fn vyre_driver_wgpu::runtime::adapter_caps_probe::probe(adapter: &wgpu::api::adapter::Adapter) -> vyre_foundation::optimizer::ctx::AdapterCaps
pub fn vyre_driver_wgpu::runtime::adapter_caps_probe::probe_profile(adapter: &wgpu::api::adapter::Adapter) -> vyre_driver::device_profile::DeviceProfile
pub mod vyre_driver_wgpu::runtime::aot
pub struct vyre_driver_wgpu::runtime::aot::AotArtifact
pub vyre_driver_wgpu::runtime::aot::AotArtifact::cache_hit: bool
pub vyre_driver_wgpu::runtime::aot::AotArtifact::key: alloc::string::String
pub vyre_driver_wgpu::runtime::aot::AotArtifact::wgsl: alloc::string::String
pub fn vyre_driver_wgpu::runtime::aot::backend_fingerprint() -> alloc::string::String
pub fn vyre_driver_wgpu::runtime::aot::cache_dir() -> std::path::PathBuf
pub fn vyre_driver_wgpu::runtime::aot::cache_key(spec_hash: &str, backend_fingerprint: &str) -> alloc::string::String
pub fn vyre_driver_wgpu::runtime::aot::load_or_compile(program: &vyre_foundation::ir_inner::model::program::core::Program, fingerprint: &str) -> core::result::Result<vyre_driver_wgpu::runtime::aot::AotArtifact, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::aot::load_or_compile_with_config(program: &vyre_foundation::ir_inner::model::program::core::Program, fingerprint: &str, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver_wgpu::runtime::aot::AotArtifact, vyre_driver::backend::error::BackendError>
pub mod vyre_driver_wgpu::runtime::cache
pub mod vyre_driver_wgpu::runtime::cache::lru
pub struct vyre_driver_wgpu::runtime::cache::lru::AccessMeta
pub vyre_driver_wgpu::runtime::cache::lru::AccessMeta::frequency: u32
pub vyre_driver_wgpu::runtime::cache::lru::AccessMeta::last_access: u64
pub vyre_driver_wgpu::runtime::cache::lru::AccessMeta::size: u64
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::lru::AccessTracker
impl vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::hot_set(&self, n: usize) -> alloc::vec::Vec<u64>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::new() -> Self
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::record(&mut self, key: u64)
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::stats(&self, key: u64) -> core::option::Option<vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::try_new() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
impl core::default::Default for vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::default() -> Self
pub struct vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>
impl<K, V> vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V> where K: core::hash::Hash + core::cmp::Eq + core::marker::Copy, V: core::default::Default
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::ensure(&mut self, key: K) -> &mut V
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::ensure_front(&mut self, key: K) -> &mut V
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::get(&self, key: &K) -> core::option::Option<&V>
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::hottest(&self, n: usize) -> alloc::vec::Vec<K>
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::iter_coldest(&self) -> impl core::iter::traits::iterator::Iterator<Item = (&K, &V)> + '_
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::iter_hottest(&self) -> impl core::iter::traits::iterator::Iterator<Item = (&K, &V)> + '_
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::new() -> Self
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::remove(&mut self, key: &K)
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::touch(&mut self, key: K)
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::try_new() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::try_with_capacity(capacity: usize) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::try_with_reserved_capacity(capacity: usize) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::with_capacity(capacity: usize) -> Self
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::with_reserved_capacity(capacity: usize) -> Self
impl<K, V> core::default::Default for vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V> where K: core::hash::Hash + core::cmp::Eq + core::marker::Copy, V: core::default::Default
pub fn vyre_driver_wgpu::runtime::cache::lru::IntrusiveLru<K, V>::default() -> Self
pub const vyre_driver_wgpu::runtime::cache::lru::DEFAULT_INTRUSIVE_LRU_CAPACITY: usize
pub mod vyre_driver_wgpu::runtime::cache::tiered_cache
#[non_exhaustive] pub enum vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::CapacityAccountingOverflow
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::EntryTooLarge
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::KeyNotFound
impl core::error::Error for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
impl core::fmt::Display for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats
pub vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats::frequency: u32
pub vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats::last_access: u64
pub vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats::size: u64
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry::key: u64
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry::size: u64
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry::tier: usize
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::capacity: u64
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::name: alloc::string::String
pub vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::used: u64
impl vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::new(name: impl core::convert::Into<alloc::string::String>, capacity: u64) -> Self
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::try_new(name: impl core::convert::Into<alloc::string::String>, capacity: u64) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::promote_threshold: u32
impl vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub const vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::DEFAULT_THRESHOLD: u32
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::new(promote_threshold: u32) -> Self
impl core::default::Default for vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::default() -> Self
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache
impl vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::demote(&mut self, key: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::evict_coldest(&mut self) -> core::option::Option<u64>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::get(&self, key: u64) -> core::option::Option<&vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::insert(&mut self, key: u64, size: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::promote(&mut self, key: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::record_access(&mut self, key: u64)
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::try_with_policy(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>, policy: vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::with_policy(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>, policy: vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy) -> Self
impl vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::new(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>) -> Self
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::try_new(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
#[non_exhaustive] pub enum vyre_driver_wgpu::runtime::cache::CacheError
pub vyre_driver_wgpu::runtime::cache::CacheError::CapacityAccountingOverflow
pub vyre_driver_wgpu::runtime::cache::CacheError::EntryTooLarge
pub vyre_driver_wgpu::runtime::cache::CacheError::KeyNotFound
impl core::error::Error for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
impl core::fmt::Display for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::AccessStats
pub vyre_driver_wgpu::runtime::cache::AccessStats::frequency: u32
pub vyre_driver_wgpu::runtime::cache::AccessStats::last_access: u64
pub vyre_driver_wgpu::runtime::cache::AccessStats::size: u64
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::AccessTracker
impl vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::hot_set(&self, n: usize) -> alloc::vec::Vec<u64>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::new() -> Self
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::record(&mut self, key: u64)
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::stats(&self, key: u64) -> core::option::Option<vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::try_new() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
impl core::default::Default for vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::default() -> Self
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::CacheEntry
pub vyre_driver_wgpu::runtime::cache::CacheEntry::key: u64
pub vyre_driver_wgpu::runtime::cache::CacheEntry::size: u64
pub vyre_driver_wgpu::runtime::cache::CacheEntry::tier: usize
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::CacheTier
pub vyre_driver_wgpu::runtime::cache::CacheTier::capacity: u64
pub vyre_driver_wgpu::runtime::cache::CacheTier::name: alloc::string::String
pub vyre_driver_wgpu::runtime::cache::CacheTier::used: u64
impl vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::new(name: impl core::convert::Into<alloc::string::String>, capacity: u64) -> Self
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier::try_new(name: impl core::convert::Into<alloc::string::String>, capacity: u64) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::LruPolicy
pub vyre_driver_wgpu::runtime::cache::LruPolicy::promote_threshold: u32
impl vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub const vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::DEFAULT_THRESHOLD: u32
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::new(promote_threshold: u32) -> Self
impl core::default::Default for vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::default() -> Self
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::cache::TieredCache
impl vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::demote(&mut self, key: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::evict_coldest(&mut self) -> core::option::Option<u64>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::get(&self, key: u64) -> core::option::Option<&vyre_driver_wgpu::runtime::cache::tiered_cache::CacheEntry>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::insert(&mut self, key: u64, size: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::promote(&mut self, key: u64) -> core::result::Result<(), vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::record_access(&mut self, key: u64)
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::try_with_policy(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>, policy: vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::with_policy(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>, policy: vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy) -> Self
impl vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::new(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>) -> Self
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::TieredCache::try_new(tiers: alloc::vec::Vec<vyre_driver_wgpu::runtime::cache::tiered_cache::CacheTier>) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub mod vyre_driver_wgpu::runtime::device
pub struct vyre_driver_wgpu::runtime::device::AdapterCriteria
pub vyre_driver_wgpu::runtime::device::AdapterCriteria::device_type: core::option::Option<wgpu_types::DeviceType>
pub vyre_driver_wgpu::runtime::device::AdapterCriteria::name_contains: core::option::Option<alloc::string::String>
pub vyre_driver_wgpu::runtime::device::AdapterCriteria::power: core::option::Option<wgpu_types::PowerPreference>
pub vyre_driver_wgpu::runtime::device::AdapterCriteria::vendor: core::option::Option<u32>
impl vyre_driver_wgpu::runtime::device::AdapterCriteria
pub fn vyre_driver_wgpu::runtime::device::AdapterCriteria::high_performance() -> Self
pub fn vyre_driver_wgpu::runtime::device::AdapterCriteria::low_power() -> Self
pub struct vyre_driver_wgpu::runtime::device::AdapterProbeReport
pub vyre_driver_wgpu::runtime::device::AdapterProbeReport::missing: alloc::vec::Vec<alloc::string::String>
pub vyre_driver_wgpu::runtime::device::AdapterProbeReport::probed: alloc::vec::Vec<alloc::string::String>
pub struct vyre_driver_wgpu::runtime::device::EnabledFeatures
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::indirect_first_instance: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::max_storage_buffer_binding_size: u64
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::max_subgroup_size: u32
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::max_workgroup_size: [u32; 3]
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::min_subgroup_size: u32
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::pipeline_cache: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::push_constants: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::shader_f16: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::subgroup: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::subgroup_barrier: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::timestamp_query: bool
pub vyre_driver_wgpu::runtime::device::EnabledFeatures::timestamp_query_inside_encoders: bool
pub async fn vyre_driver_wgpu::runtime::device::acquire_gpu() -> vyre_foundation::error::Result<((wgpu::api::device::Device, wgpu::api::queue::Queue), wgpu_types::AdapterInfo, vyre_driver_wgpu::runtime::device::EnabledFeatures)>
pub async fn vyre_driver_wgpu::runtime::device::acquire_gpu_for_adapter(index: usize) -> vyre_foundation::error::Result<((wgpu::api::device::Device, wgpu::api::queue::Queue), wgpu_types::AdapterInfo, vyre_driver_wgpu::runtime::device::EnabledFeatures)>
pub fn vyre_driver_wgpu::runtime::device::adapter_for_info(expected: &wgpu_types::AdapterInfo) -> vyre_foundation::error::Result<wgpu::api::adapter::Adapter>
pub fn vyre_driver_wgpu::runtime::device::adapter_index_from_env() -> vyre_foundation::error::Result<core::option::Option<usize>>
pub fn vyre_driver_wgpu::runtime::device::adapter_probe_report() -> vyre_driver_wgpu::runtime::device::AdapterProbeReport
pub fn vyre_driver_wgpu::runtime::device::cached_adapter_info() -> vyre_foundation::error::Result<&'static wgpu_types::AdapterInfo>
pub fn vyre_driver_wgpu::runtime::device::cached_device() -> vyre_foundation::error::Result<alloc::sync::Arc<(wgpu::api::device::Device, wgpu::api::queue::Queue)>>
pub fn vyre_driver_wgpu::runtime::device::enumerate_adapters() -> alloc::vec::Vec<wgpu_types::AdapterInfo>
pub fn vyre_driver_wgpu::runtime::device::has_real_gpu_adapter() -> bool
pub fn vyre_driver_wgpu::runtime::device::init_device() -> vyre_foundation::error::Result<((wgpu::api::device::Device, wgpu::api::queue::Queue), wgpu_types::AdapterInfo, vyre_driver_wgpu::runtime::device::EnabledFeatures)>
pub fn vyre_driver_wgpu::runtime::device::init_device_for_adapter(index: usize) -> vyre_foundation::error::Result<((wgpu::api::device::Device, wgpu::api::queue::Queue), wgpu_types::AdapterInfo, vyre_driver_wgpu::runtime::device::EnabledFeatures)>
pub fn vyre_driver_wgpu::runtime::device::select_adapter(criteria: &vyre_driver_wgpu::runtime::device::AdapterCriteria) -> vyre_foundation::error::Result<(usize, wgpu_types::AdapterInfo)>
pub mod vyre_driver_wgpu::runtime::indirect
pub struct vyre_driver_wgpu::runtime::indirect::IndirectArgs
pub vyre_driver_wgpu::runtime::indirect::IndirectArgs::buffer: alloc::sync::Arc<wgpu::api::buffer::Buffer>
pub vyre_driver_wgpu::runtime::indirect::IndirectArgs::offset: u64
impl vyre_driver_wgpu::runtime::indirect::IndirectArgs
pub fn vyre_driver_wgpu::runtime::indirect::IndirectArgs::from_handle(handle: &vyre_driver_wgpu::buffer::GpuBufferHandle, offset: u64) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub const vyre_driver_wgpu::runtime::indirect::INDIRECT_ARGS_BYTES: u64
pub fn vyre_driver_wgpu::runtime::indirect::dispatch_indirect<'a>(pass: &mut wgpu::api::compute_pass::ComputePass<'a>, args: &'a vyre_driver_wgpu::runtime::indirect::IndirectArgs)
pub mod vyre_driver_wgpu::runtime::prerecorded
pub struct vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::bind_groups: alloc::vec::Vec<alloc::sync::Arc<wgpu::api::bind_group::BindGroup>>
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::cb: std::sync::poison::mutex::Mutex<core::option::Option<wgpu::api::command_buffer::CommandBuffer>>
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::device: wgpu::api::device::Device
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::handles: alloc::vec::Vec<vyre_driver_wgpu::buffer::GpuBufferHandle>
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::output_handles: alloc::vec::Vec<vyre_driver_wgpu::buffer::GpuBufferHandle>
pub vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::queue: wgpu::api::queue::Queue
impl vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch
pub fn vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::read_output(&self, index: usize) -> core::result::Result<alloc::vec::Vec<u8>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::read_output_into(&self, index: usize, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::prerecorded::PrerecordedDispatch::replay(&self, queue: &wgpu::api::queue::Queue) -> core::result::Result<wgpu::api::queue::SubmissionIndex, vyre_driver::backend::error::BackendError>
pub mod vyre_driver_wgpu::runtime::readback_ring
pub enum vyre_driver_wgpu::runtime::readback_ring::SlotState
pub vyre_driver_wgpu::runtime::readback_ring::SlotState::Error
pub vyre_driver_wgpu::runtime::readback_ring::SlotState::Free
pub vyre_driver_wgpu::runtime::readback_ring::SlotState::Pending
pub vyre_driver_wgpu::runtime::readback_ring::SlotState::Ready
pub struct vyre_driver_wgpu::runtime::readback_ring::GpuSlot
pub vyre_driver_wgpu::runtime::readback_ring::GpuSlot::buffer: wgpu::api::buffer::Buffer
pub vyre_driver_wgpu::runtime::readback_ring::GpuSlot::state: alloc::sync::Arc<core::sync::atomic::AtomicU8>
pub struct vyre_driver_wgpu::runtime::readback_ring::ReadbackRing
impl vyre_driver_wgpu::runtime::readback_ring::ReadbackRing
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::arm_ticket(&self, ticket: &vyre_driver_wgpu::runtime::readback_ring::ReadbackTicket) -> core::result::Result<(crossbeam_channel::channel::Receiver<vyre_driver_wgpu::runtime::readback_ring::MapResult>, alloc::sync::Arc<core::sync::atomic::AtomicBool>), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::collect_slot(&self, device: &wgpu::api::device::Device, idx: usize) -> core::result::Result<core::option::Option<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::collect_slot_into(&self, device: &wgpu::api::device::Device, idx: usize, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<core::option::Option<usize>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::new(device: &wgpu::api::device::Device, size: usize, buffer_size: u64) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::record_copy(&self, device: &wgpu::api::device::Device, encoder: &mut wgpu::api::command_encoder::CommandEncoder, src_buffer: &wgpu::api::buffer::Buffer, src_offset: u64, byte_len: u64) -> core::result::Result<vyre_driver_wgpu::runtime::readback_ring::ReadbackTicket, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::submit_readback(&self, device: &wgpu::api::device::Device, queue: &wgpu::api::queue::Queue, src_buffer: &wgpu::api::buffer::Buffer, src_offset: u64, byte_len: u64) -> core::result::Result<usize, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRing::with_mapped_ticket<R>(&self, ticket: &vyre_driver_wgpu::runtime::readback_ring::ReadbackTicket, visitor: impl core::ops::function::FnOnce(&[u8]) -> core::result::Result<R, vyre_driver::backend::error::BackendError>) -> core::result::Result<R, vyre_driver::backend::error::BackendError>
pub struct vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet
impl vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::existing_ring_for(&self, byte_len: u64) -> core::result::Result<core::option::Option<alloc::sync::Arc<vyre_driver_wgpu::runtime::readback_ring::ReadbackRing>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::new() -> Self
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::ring_for(&self, device: &wgpu::api::device::Device, byte_len: u64) -> core::result::Result<alloc::sync::Arc<vyre_driver_wgpu::runtime::readback_ring::ReadbackRing>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::slots_per_ring(&self) -> usize
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::with_requested_slots(raw_slots: core::option::Option<&str>) -> Self
impl core::default::Default for vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet
pub fn vyre_driver_wgpu::runtime::readback_ring::ReadbackRingSet::default() -> Self
pub struct vyre_driver_wgpu::runtime::readback_ring::ReadbackTicket
pub struct vyre_driver_wgpu::runtime::readback_ring::RingStats
pub vyre_driver_wgpu::runtime::readback_ring::RingStats::dispatches: core::sync::atomic::AtomicU64
pub vyre_driver_wgpu::runtime::readback_ring::RingStats::peak_inflight: core::sync::atomic::AtomicU64
pub vyre_driver_wgpu::runtime::readback_ring::RingStats::readback_stalls: core::sync::atomic::AtomicU64
impl vyre_driver_wgpu::runtime::readback_ring::RingStats
pub fn vyre_driver_wgpu::runtime::readback_ring::RingStats::record_dispatch(&self) -> u64
pub fn vyre_driver_wgpu::runtime::readback_ring::RingStats::record_stall(&self)
pub fn vyre_driver_wgpu::runtime::readback_ring::RingStats::update_peak(&self, current: u64)
pub type vyre_driver_wgpu::runtime::readback_ring::MapResult = core::result::Result<(), wgpu::api::buffer::BufferAsyncError>
pub mod vyre_driver_wgpu::runtime::router
#[non_exhaustive] pub enum vyre_driver_wgpu::runtime::router::Override<'a>
pub vyre_driver_wgpu::runtime::router::Override::Explicit(&'a str)
pub vyre_driver_wgpu::runtime::router::Override::FromEnv
pub vyre_driver_wgpu::runtime::router::Override::None
#[non_exhaustive] pub enum vyre_driver_wgpu::runtime::router::Reason
pub vyre_driver_wgpu::runtime::router::Reason::EnvOverride
pub vyre_driver_wgpu::runtime::router::Reason::Precedence
pub struct vyre_driver_wgpu::runtime::router::BackendRouter
impl vyre_driver_wgpu::runtime::router::BackendRouter
pub fn vyre_driver_wgpu::runtime::router::BackendRouter::enumerate_by_precedence() -> alloc::vec::Vec<&'static vyre_driver::backend::registry::inventory_streams::BackendRegistration>
pub fn vyre_driver_wgpu::runtime::router::BackendRouter::new() -> Self
pub fn vyre_driver_wgpu::runtime::router::BackendRouter::pick(&self, program: &vyre_foundation::ir_inner::model::program::core::Program) -> core::result::Result<vyre_driver_wgpu::runtime::router::RouterDecision, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::runtime::router::BackendRouter::pick_with_override(&self, _program: &vyre_foundation::ir_inner::model::program::core::Program, source: vyre_driver_wgpu::runtime::router::Override<'_>) -> core::result::Result<vyre_driver_wgpu::runtime::router::RouterDecision, vyre_driver::backend::error::BackendError>
pub struct vyre_driver_wgpu::runtime::router::RouterDecision
pub vyre_driver_wgpu::runtime::router::RouterDecision::backend: &'static str
pub vyre_driver_wgpu::runtime::router::RouterDecision::reason: vyre_driver_wgpu::runtime::router::Reason
pub mod vyre_driver_wgpu::runtime::serializer
pub mod vyre_driver_wgpu::runtime::serializer::decode_parts
pub fn vyre_driver_wgpu::runtime::serializer::decode_parts::decode_parts(bytes: &[u8]) -> vyre_foundation::error::Result<alloc::vec::Vec<&[u8]>>
pub mod vyre_driver_wgpu::runtime::serializer::encode_parts
pub const vyre_driver_wgpu::runtime::serializer::encode_parts::MAX_SERIALIZED_PART_BYTES: usize
pub fn vyre_driver_wgpu::runtime::serializer::encode_parts::encode_parts(parts: &[&[u8]]) -> vyre_foundation::error::Result<alloc::vec::Vec<u8>>
pub const vyre_driver_wgpu::runtime::serializer::MAX_SERIALIZED_PART_BYTES: usize
pub fn vyre_driver_wgpu::runtime::serializer::decode_parts(bytes: &[u8]) -> vyre_foundation::error::Result<alloc::vec::Vec<&[u8]>>
pub fn vyre_driver_wgpu::runtime::serializer::encode_parts(parts: &[&[u8]]) -> vyre_foundation::error::Result<alloc::vec::Vec<u8>>
pub mod vyre_driver_wgpu::runtime::shader
pub mod vyre_driver_wgpu::runtime::shader::compile_compute_pipeline
pub fn vyre_driver_wgpu::runtime::shader::compile_compute_pipeline::compile_compute_pipeline(device: &wgpu::api::device::Device, label: &str, wgsl_source: &str, entry_point: &str) -> vyre_foundation::error::Result<wgpu::api::compute_pipeline::ComputePipeline>
pub fn vyre_driver_wgpu::runtime::shader::compile_compute_pipeline::compile_compute_pipeline_with_layout(device: &wgpu::api::device::Device, label: &str, wgsl_source: &str, entry_point: &str, layout: core::option::Option<&wgpu::api::pipeline_layout::PipelineLayout>) -> vyre_foundation::error::Result<wgpu::api::compute_pipeline::ComputePipeline>
#[non_exhaustive] pub enum vyre_driver_wgpu::runtime::CacheError
pub vyre_driver_wgpu::runtime::CacheError::CapacityAccountingOverflow
pub vyre_driver_wgpu::runtime::CacheError::EntryTooLarge
pub vyre_driver_wgpu::runtime::CacheError::KeyNotFound
impl core::error::Error for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
impl core::fmt::Display for vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::CacheError::fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::AccessStats
pub vyre_driver_wgpu::runtime::AccessStats::frequency: u32
pub vyre_driver_wgpu::runtime::AccessStats::last_access: u64
pub vyre_driver_wgpu::runtime::AccessStats::size: u64
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::AccessTracker
impl vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::hot_set(&self, n: usize) -> alloc::vec::Vec<u64>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::new() -> Self
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::record(&mut self, key: u64)
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::stats(&self, key: u64) -> core::option::Option<vyre_driver_wgpu::runtime::cache::tiered_cache::AccessStats>
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::try_new() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
impl core::default::Default for vyre_driver_wgpu::runtime::cache::lru::AccessTracker
pub fn vyre_driver_wgpu::runtime::cache::lru::AccessTracker::default() -> Self
#[non_exhaustive] pub struct vyre_driver_wgpu::runtime::LruPolicy
pub vyre_driver_wgpu::runtime::LruPolicy::promote_threshold: u32
impl vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub const vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::DEFAULT_THRESHOLD: u32
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::new(promote_threshold: u32) -> Self
impl core::default::Default for vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy
pub fn vyre_driver_wgpu::runtime::cache::tiered_cache::LruPolicy::default() -> Self
pub fn vyre_driver_wgpu::runtime::bg_entry(binding: u32, buffer: &wgpu::api::buffer::Buffer) -> wgpu::api::bind_group::BindGroupEntry<'_>
pub fn vyre_driver_wgpu::runtime::cached_adapter_info() -> vyre_foundation::error::Result<&'static wgpu_types::AdapterInfo>
pub fn vyre_driver_wgpu::runtime::cached_device() -> vyre_foundation::error::Result<alloc::sync::Arc<(wgpu::api::device::Device, wgpu::api::queue::Queue)>>
pub fn vyre_driver_wgpu::runtime::compile_compute_pipeline(device: &wgpu::api::device::Device, label: &str, wgsl_source: &str, entry_point: &str) -> vyre_foundation::error::Result<wgpu::api::compute_pipeline::ComputePipeline>
pub fn vyre_driver_wgpu::runtime::compile_compute_pipeline_with_layout(device: &wgpu::api::device::Device, label: &str, wgsl_source: &str, entry_point: &str, layout: core::option::Option<&wgpu::api::pipeline_layout::PipelineLayout>) -> vyre_foundation::error::Result<wgpu::api::compute_pipeline::ComputePipeline>
pub fn vyre_driver_wgpu::runtime::init_device() -> vyre_foundation::error::Result<((wgpu::api::device::Device, wgpu::api::queue::Queue), wgpu_types::AdapterInfo, vyre_driver_wgpu::runtime::device::EnabledFeatures)>
pub mod vyre_driver_wgpu::spirv_backend
pub struct vyre_driver_wgpu::spirv_backend::SpirvEmitter
impl vyre_driver_wgpu::spirv_backend::SpirvEmitter
pub fn vyre_driver_wgpu::spirv_backend::SpirvEmitter::default_flags() -> naga::back::spv::WriterFlags
pub fn vyre_driver_wgpu::spirv_backend::SpirvEmitter::emit(module: &naga::ir::Module, entry: &str) -> core::result::Result<alloc::vec::Vec<u32>, alloc::string::String>
pub const vyre_driver_wgpu::spirv_backend::SPIRV_BACKEND_ID: &str
pub struct vyre_driver_wgpu::DispatchArena
impl vyre_driver_wgpu::DispatchArena
pub fn vyre_driver_wgpu::DispatchArena::new(device: wgpu::api::device::Device, queue: wgpu::api::queue::Queue, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> Self
impl core::fmt::Debug for vyre_driver_wgpu::DispatchArena
pub fn vyre_driver_wgpu::DispatchArena::fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result
pub struct vyre_driver_wgpu::WgpuBackend
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::acquire() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::acquire_adapter(index: usize) -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::adapter_info(&self) -> &wgpu_types::AdapterInfo
pub fn vyre_driver_wgpu::WgpuBackend::compile_persistent(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::sync::Arc<vyre_driver_wgpu::pipeline::WgpuPipeline>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::compile_streaming(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, config: vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver_wgpu::engine::streaming::HostIngressStream, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::device_limits(&self) -> &wgpu_types::Limits
pub fn vyre_driver_wgpu::WgpuBackend::device_queue(&self) -> alloc::sync::Arc<(wgpu::api::device::Device, wgpu::api::queue::Queue)>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_batch(&self, jobs: &[(vyre_foundation::ir_inner::model::program::core::Program, alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::dispatch_config::DispatchConfig)]) -> core::result::Result<alloc::vec::Vec<core::result::Result<vyre_driver::backend::dispatch_result::OutputBuffers, vyre_driver::backend::error::BackendError>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_batch(&self, jobs: &[(&vyre_foundation::ir_inner::model::program::core::Program, &[&[u8]], &vyre_driver::backend::dispatch_config::DispatchConfig)]) -> core::result::Result<alloc::vec::Vec<core::result::Result<vyre_driver::backend::dispatch_result::OutputBuffers, vyre_driver::backend::error::BackendError>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_batch_into(&self, jobs: &[(&vyre_foundation::ir_inner::model::program::core::Program, &[&[u8]], &vyre_driver::backend::dispatch_config::DispatchConfig)], outputs: &mut [vyre_driver::backend::dispatch_result::OutputBuffers]) -> core::result::Result<alloc::vec::Vec<core::result::Result<(), vyre_driver::backend::error::BackendError>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_for_each_mapped_output<F>(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, visitor: F) -> core::result::Result<(), vyre_driver::backend::error::BackendError> where F: core::ops::function::FnMut(usize, &[u8]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_for_each_pod_output<T, F>(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, visitor: F) -> core::result::Result<(), vyre_driver::backend::error::BackendError> where T: bytemuck::pod::Pod, F: core::ops::function::FnMut(usize, &[T]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_speculative_prefilter_confirm<F>(&self, speculator: &vyre_driver::speculate::AdaptiveSpeculator, plan: vyre_driver::speculate::SpeculativeDispatchPlan<'_>, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, confirm_serial: F) -> core::result::Result<vyre_driver::speculate::SpeculativeDispatchOutcome, vyre_driver::backend::error::BackendError> where F: core::ops::function::FnMut(vyre_driver::backend::dispatch_result::OutputBuffers) -> core::result::Result<vyre_driver::backend::dispatch_result::OutputBuffers, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::force_device_lost(&self) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::invalidate_impacted_disk_cache(&self, intervention_mask: &[u32], rule_adj: &[u32], state: &[u32], join_rules: &[u32], n: u32, max_iterations: u32, pipeline_lineage_cell: &[u32], cache_keys: &[alloc::string::String]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::invalidate_impacted_pipeline_cache(&self, intervention_mask: &[u32], rule_adj: &[u32], state: &[u32], join_rules: &[u32], n: u32, max_iterations: u32, pipeline_lineage_cell: &[u32], pipeline_keys: &[[u8; 32]]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::invalidate_pipeline_cache_for_changed_op(&self, changed_op_handle: u32, pipeline_lineage_cell: &[u32], pipeline_keys: &[[u8; 32]]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::new() -> core::result::Result<Self, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::shared() -> core::result::Result<alloc::sync::Arc<Self>, vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::adapter_caps(&self) -> vyre_foundation::optimizer::ctx::AdapterCaps
pub fn vyre_driver_wgpu::WgpuBackend::device_profile(&self) -> vyre_driver::device_profile::DeviceProfile
pub fn vyre_driver_wgpu::WgpuBackend::stats(&self) -> vyre_driver_wgpu::WgpuBackendStats
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::allocate_wgpu_device_buffer(&self, byte_len: usize) -> core::result::Result<alloc::boxed::Box<dyn vyre_driver::backend::device_buffer::DeviceBuffer>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_wgpu_device_buffer(&self, buffer: &dyn vyre_driver::backend::device_buffer::DeviceBuffer) -> core::result::Result<alloc::vec::Vec<u8>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::free_wgpu_device_buffer(&self, buffer: alloc::boxed::Box<dyn vyre_driver::backend::device_buffer::DeviceBuffer>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_wgpu_device_buffer(&self, buffer: &mut dyn vyre_driver::backend::device_buffer::DeviceBuffer, bytes: &[u8]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::compile(&self, program: &vyre_foundation::ir_inner::model::program::core::Program) -> core::result::Result<vyre_driver_wgpu::WgpuIR, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_compiled(&self, compiled: &vyre_driver_wgpu::WgpuIR, inputs: &[vyre_driver::backend::capability::MemoryRef<'_>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<vyre_driver::backend::capability::Memory>, vyre_driver::backend::error::BackendError>
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_wgsl(&self, wgsl: &str, input: &[u8], output_size: usize, workgroup_size: u32) -> core::result::Result<alloc::vec::Vec<u8>, alloc::string::String>
impl vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::lower_to_backend_ir(&self, program: &vyre_foundation::ir_inner::model::program::core::Program) -> core::result::Result<vyre_driver_wgpu::emit::WgpuProgram, vyre_foundation::lower::LoweringError>
pub fn vyre_driver_wgpu::WgpuBackend::lower_to_target<'a>(&self, bir: &'a vyre_driver_wgpu::emit::WgpuProgram) -> &'a naga::ir::Module
impl vyre_driver::backend::capability::Executable for vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::dispatch(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[vyre_driver::backend::capability::MemoryRef<'_>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<vyre_driver::backend::capability::Memory>, vyre_driver::backend::error::BackendError>
impl vyre_driver::backend::private::Sealed for vyre_driver_wgpu::WgpuBackend
impl vyre_driver::backend::vyre_backend::VyreBackend for vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::allocate_device_buffer(&self, byte_len: usize) -> core::result::Result<alloc::boxed::Box<dyn vyre_driver::backend::device_buffer::DeviceBuffer>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::allocate_resident(&self, byte_len: usize) -> core::result::Result<vyre_driver::backend::resource::Resource, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::compile_native(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<core::option::Option<alloc::sync::Arc<dyn vyre_driver::backend::compiled_pipeline::CompiledPipeline>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::device_lost(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::device_profile(&self) -> vyre_driver::device_profile::DeviceProfile
pub fn vyre_driver_wgpu::WgpuBackend::dispatch(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[alloc::vec::Vec<u8>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_async(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[alloc::vec::Vec<u8>], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::boxed::Box<dyn vyre_driver::backend::pending_dispatch::PendingDispatch>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_async(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<alloc::boxed::Box<dyn vyre_driver::backend::pending_dispatch::PendingDispatch>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_into(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig, outputs: &mut vyre_driver::backend::dispatch_result::OutputBuffers) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_borrowed_timed(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&[u8]], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver::backend::dispatch_result::TimedDispatchResult, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_resident_timed(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, resources: &[vyre_driver::backend::resource::Resource], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<vyre_driver::backend::dispatch_result::TimedDispatchResult, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::dispatch_with_device_buffers(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[&dyn vyre_driver::backend::device_buffer::DeviceBuffer], outputs: &mut [&mut dyn vyre_driver::backend::device_buffer::DeviceBuffer], config: &vyre_driver::backend::dispatch_config::DispatchConfig) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_device_buffer(&self, buffer: &dyn vyre_driver::backend::device_buffer::DeviceBuffer) -> core::result::Result<alloc::vec::Vec<u8>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_resident(&self, resource: &vyre_driver::backend::resource::Resource) -> core::result::Result<alloc::vec::Vec<u8>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_resident_into(&self, resource: &vyre_driver::backend::resource::Resource, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_resident_range(&self, resource: &vyre_driver::backend::resource::Resource, byte_offset: usize, byte_len: usize) -> core::result::Result<alloc::vec::Vec<u8>, vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_resident_range_into(&self, resource: &vyre_driver::backend::resource::Resource, byte_offset: usize, byte_len: usize, out: &mut alloc::vec::Vec<u8>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::download_resident_ranges_into(&self, ranges: &[(&vyre_driver::backend::resource::Resource, usize, usize)], outputs: &mut [&mut alloc::vec::Vec<u8>]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::flush(&self) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::free_device_buffer(&self, buffer: alloc::boxed::Box<dyn vyre_driver::backend::device_buffer::DeviceBuffer>) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::free_resident(&self, resource: vyre_driver::backend::resource::Resource) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::id(&self) -> &'static str
pub fn vyre_driver_wgpu::WgpuBackend::is_distributed(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::max_compute_invocations_per_workgroup(&self) -> u32
pub fn vyre_driver_wgpu::WgpuBackend::max_compute_workgroups_per_dimension(&self) -> u32
pub fn vyre_driver_wgpu::WgpuBackend::max_storage_buffer_bytes(&self) -> u64
pub fn vyre_driver_wgpu::WgpuBackend::max_workgroup_size(&self) -> [u32; 3]
pub fn vyre_driver_wgpu::WgpuBackend::pipeline_cache_snapshot(&self) -> core::option::Option<vyre_driver::pipeline::PipelineCacheSnapshot>
pub fn vyre_driver_wgpu::WgpuBackend::subgroup_size(&self) -> core::option::Option<u32>
pub fn vyre_driver_wgpu::WgpuBackend::supported_ops(&self) -> &std::collections::hash::set::HashSet<vyre_foundation::ir_inner::model::node_kind::OpId>
pub fn vyre_driver_wgpu::WgpuBackend::supports_async_compute(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_bf16(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_f16(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_indirect_dispatch(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_persistent_thread_dispatch(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_speculation(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_subgroup_ops(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_tensor_cores(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::try_recover(&self) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_device_buffer(&self, buffer: &mut dyn vyre_driver::backend::device_buffer::DeviceBuffer, bytes: &[u8]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_resident(&self, resource: &vyre_driver::backend::resource::Resource, bytes: &[u8]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_resident_at(&self, resource: &vyre_driver::backend::resource::Resource, dst_offset_bytes: usize, bytes: &[u8]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_resident_at_many(&self, uploads: &[(&vyre_driver::backend::resource::Resource, usize, &[u8])]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::upload_resident_many(&self, uploads: &[(&vyre_driver::backend::resource::Resource, &[u8])]) -> core::result::Result<(), vyre_driver::backend::error::BackendError>
pub fn vyre_driver_wgpu::WgpuBackend::version(&self) -> &'static str
impl vyre_foundation::validate::options::BackendValidationCapabilities for vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::backend_name(&self) -> &'static str
pub fn vyre_driver_wgpu::WgpuBackend::supports_cast_target(&self, target: &vyre_spec::data_type::DataType) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_distributed_collectives(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_indirect_dispatch(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_specialization_constants(&self) -> bool
pub fn vyre_driver_wgpu::WgpuBackend::supports_subgroup_ops(&self) -> bool
impl vyre_self_substrate::optimizer::dispatcher::OptimizerDispatcher for vyre_driver_wgpu::WgpuBackend
pub fn vyre_driver_wgpu::WgpuBackend::dispatch(&self, program: &vyre_foundation::ir_inner::model::program::core::Program, inputs: &[alloc::vec::Vec<u8>], grid_override: core::option::Option<[u32; 3]>) -> core::result::Result<alloc::vec::Vec<alloc::vec::Vec<u8>>, vyre_self_substrate::optimizer::dispatcher::DispatchError>
pub struct vyre_driver_wgpu::WgpuBackendStats
pub vyre_driver_wgpu::WgpuBackendStats::adapter_name: alloc::sync::Arc<str>
pub vyre_driver_wgpu::WgpuBackendStats::persistent_pool: vyre_driver_wgpu::buffer::BufferPoolStats
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_byte_capacity: usize
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_bytes: usize
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_capacity: usize
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_entries: usize
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_evictions: u64
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_hit_rate: f64
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_hits: u64
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_insertions: u64
pub vyre_driver_wgpu::WgpuBackendStats::pipeline_cache_misses: u64
pub struct vyre_driver_wgpu::WgpuDeviceBuffer
impl vyre_driver_wgpu::WgpuDeviceBuffer
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::handle(&self) -> &vyre_driver_wgpu::buffer::GpuBufferHandle
impl vyre_driver::backend::device_buffer::DeviceBuffer for vyre_driver_wgpu::WgpuDeviceBuffer
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::as_any(&self) -> &dyn core::any::Any
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::as_any_mut(&mut self) -> &mut dyn core::any::Any
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::backend_id(&self) -> &'static str
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::byte_len(&self) -> usize
pub fn vyre_driver_wgpu::WgpuDeviceBuffer::debug_label(&self) -> core::option::Option<&str>
pub struct vyre_driver_wgpu::WgpuIR
pub vyre_driver_wgpu::WgpuIR::pipeline: vyre_driver_wgpu::pipeline::WgpuPipeline
pub const vyre_driver_wgpu::WGPU_BACKEND_ID: &str
