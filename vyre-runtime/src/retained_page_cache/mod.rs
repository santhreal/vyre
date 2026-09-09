//! Retained page cache lifecycle: radix index, immutable content identity, page allocation,
//! copy-on-write, and bounded residency.
//!
//! # Architecture
//!
//! A retained page cache accelerates pipeline execution by reusing physical pages across
//! requests that share an immutable prefix of sequence units.
//!
//! This module owns the complete lifecycle in the runtime:
//! - **Radix Trie Index**: Fast prefix lookup and sub-sequence sharing.
//! - **Immutable-Content Identity**: Multi-dimensional keying including content digest,
//!   schema digest, configuration, data type, layout, device generation, cache schema version,
//!   and tenant isolation / trust domain.
//! - **Security & Trust Domains**: Cross-tenant physical page sharing requires an
//!   explicit common trust domain. Page data is scrubbed before reassignment.
//! - **Page Slabs & Ref-Counting**: Allocation, reference counting, copy-on-write (COW)
//!   on branch mutation, and safe release.
//! - **Bounded Residency & Backpressure**: Explicit limits on pages, bytes, requests,
//!   queued units, and per-tenant usage. Admission fails closed under saturation;
//!   eviction never reclaims pinned or in-flight pages.

mod contract;
mod manager;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, MutexGuard};

pub use contract::{
    RetainedMatchResult, RetainedPageCacheError, RetainedPageCacheKey,
    RetainedPageCacheKeyFingerprint, RetainedPageLayout, RetainedPageLimits, RetainedPageMetrics,
};
pub use manager::RetainedPageCacheManager;

/// Thread-safe wrapper around [`RetainedPageCacheManager`].
#[derive(Debug, Clone)]
pub struct RetainedPageCache {
    inner: Arc<Mutex<RetainedPageCacheManager>>,
}

impl RetainedPageCache {
    /// Create a thread-safe retained page cache manager.
    #[must_use]
    pub fn new(limits: RetainedPageLimits, initial_generation: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RetainedPageCacheManager::new(
                limits,
                initial_generation,
            ))),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, RetainedPageCacheManager>, RetainedPageCacheError> {
        self.inner
            .lock()
            .map_err(|_| RetainedPageCacheError::LockPoisoned)
    }

    /// Look up a sequence prefix in the retained page cache.
    pub fn lookup(
        &self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
    ) -> Result<RetainedMatchResult, RetainedPageCacheError> {
        self.lock()?.lookup_prefix(key, sequence)
    }

    /// Insert or extend a sequence prefix.
    pub fn insert_or_extend(
        &self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, RetainedPageCacheError> {
        self.lock()?
            .insert_or_extend_prefix(key, sequence, existing_pages)
    }

    /// Pin pages.
    pub fn pin(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.lock()?.pin_pages(page_ids)
    }

    /// Unpin pages.
    pub fn unpin(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.lock()?.unpin_pages(page_ids)
    }

    /// Mark pages as in-flight during kernel execution.
    pub fn mark_in_flight(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.lock()?.mark_in_flight(page_ids)
    }

    /// Clear in-flight status after kernel completion.
    pub fn complete_in_flight(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.lock()?.clear_in_flight(page_ids)
    }

    /// Release leased page references.
    pub fn release(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.lock()?.release_pages(page_ids)
    }

    /// Invalidate device generation.
    pub fn invalidate_generation(&self, new_generation: u64) -> Result<(), RetainedPageCacheError> {
        self.lock()?.invalidate_generation(new_generation);
        Ok(())
    }

    /// Read telemetry metrics.
    pub fn metrics(&self) -> Result<RetainedPageMetrics, RetainedPageCacheError> {
        Ok(self.lock()?.metrics().clone())
    }
}
