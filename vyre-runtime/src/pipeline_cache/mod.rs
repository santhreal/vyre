//! Content-addressed neutral-artifact cache.
//!
//! [`PipelineFingerprint`] derives directly from the authenticated neutral
//! [`vyre_megakernel::Artifact`] identity. Dispatch inputs, target payloads,
//! device generations, and runtime policy do not enter this layer's key.
//! Persisted blobs use a versioned, digest-bound frame and stale versions miss
//! rather than being served.
//!
//! Hot paths use [`InMemoryPipelineCache`], process-restart reuse uses
//! [`DiskCache`], and callers compose them through [`LayeredPipelineCache`].

#![allow(clippy::missing_const_for_thread_local, clippy::explicit_auto_deref)]

mod disk;
mod fingerprint;
mod in_memory;
mod layered;
mod metrics;
#[cfg(feature = "remote-cache")]
mod remote;
mod store;

#[cfg(test)]
pub(super) mod test_helpers;

pub use disk::{DiskCache, DiskCacheDurabilityReport, DiskCacheError};
pub use fingerprint::PipelineFingerprint;
pub use in_memory::{InMemoryEvictionReason, InMemoryEvictionReport, InMemoryPipelineCache};
pub use layered::{LayeredPipelineCache, LayeredPromotionReport};
pub use metrics::{PipelineCacheMetricError, PipelineCacheMetrics};
#[cfg(feature = "remote-cache")]
pub use remote::RemoteCache;
pub use store::PipelineCacheStore;
