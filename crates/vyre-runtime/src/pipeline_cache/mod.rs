//! Content-addressed neutral-artifact cache.
//!
//! [`crate::pipeline_cache::PipelineFingerprint`] derives directly from the authenticated neutral
//! [`vyre_compiler::Artifact`] identity. Dispatch inputs, target payloads,
//! device generations, and runtime policy do not enter this layer's key.
//! Persisted blobs use a versioned, digest-bound frame and stale versions miss
//! rather than being served.
//!
//! Hot paths use [`crate::pipeline_cache::InMemoryPipelineCache`], process-restart reuse uses
//! [`crate::pipeline_cache::DiskCache`], and callers compose them through [`crate::pipeline_cache::LayeredPipelineCache`].

#![allow(clippy::missing_const_for_thread_local, clippy::explicit_auto_deref)]

mod disk;
mod fingerprint;
mod in_memory;
mod layered;
mod metrics;
#[cfg(feature = "remote-cache")]
mod remote;
mod store;

pub use disk::{DiskCache, DiskCacheError};
pub use fingerprint::PipelineFingerprint;
pub use in_memory::InMemoryPipelineCache;
pub use layered::LayeredPipelineCache;
pub use metrics::PipelineCacheMetrics;
#[cfg(feature = "remote-cache")]
pub use remote::RemoteCache;
pub use store::PipelineCacheStore;
