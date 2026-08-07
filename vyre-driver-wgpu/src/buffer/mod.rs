//! Persistent GPU buffer handles and reusable allocation pools.

mod handle;
mod pool;

pub(crate) use handle::{check_resident_owner, write_padded, PendingGpuBufferReadback};
pub use handle::{
    BindGroupCache, BindGroupCacheStats, GpuBufferHandle, StagingBufferPool, StagingBufferPoolStats,
};
pub use pool::{BufferPool, BufferPoolStats};
