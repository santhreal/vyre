//! Persistent GPU buffer handles and reusable allocation pools.

mod bind_group_cache;
mod handle;
mod pool;
mod staging;

pub use bind_group_cache::{BindGroupCache, BindGroupCacheStats};
pub use handle::GpuBufferHandle;
pub(crate) use handle::{check_resident_owner, write_padded, PendingGpuBufferReadback};
pub use pool::{BufferPool, BufferPoolStats};
pub use staging::{StagingBufferPool, StagingBufferPoolStats};
