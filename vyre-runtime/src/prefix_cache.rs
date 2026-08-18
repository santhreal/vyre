//! Prefix-cache lifecycle: radix index, immutable prefix identity, page allocation,
//! copy-on-write, and bounded residency.
//!
//! # Architecture
//!
//! A prefix cache accelerates autoregressive language model decoding by reusing
//! Key-Value (KV) cache pages across requests that share an immutable prefix of
//! tokens (e.g. system prompts, common few-shot examples, multi-turn dialogues).
//!
//! This module owns the complete lifecycle in the runtime:
//! - **Radix Trie Index**: Fast prefix lookup and sub-sequence sharing.
//! - **Immutable-Prefix Identity**: Multi-dimensional keying including model, tokenizer,
//!   weights digest, configuration, token sequence, dtype, layout, device generation,
//!   cache schema version, and tenant isolation / trust domain.
//! - **Security & Trust Domains**: Cross-tenant physical page sharing requires an
//!   explicit common trust domain. Page data is scrubbed/overwritten before reassignment.
//! - **Page Slabs & Ref-Counting**: Allocation, reference counting, copy-on-write (COW)
//!   on branch mutation, and safe release.
//! - **Bounded Residency & Backpressure**: Explicit limits on pages, bytes, requests,
//!   queued tokens, and per-tenant usage. Admission fails closed under saturation;
//!   eviction never reclaims pinned or in-flight pages.

mod manager;
mod types;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Mutex, MutexGuard};

pub use manager::PrefixCacheManager;
pub use types::{
    PrefixCacheError, PrefixCacheKey, PrefixCacheKeyFingerprint, PrefixCacheLayout,
    PrefixCacheLimits, PrefixCacheMetrics, PrefixMatchResult,
};

/// Thread-safe wrapper around [`PrefixCacheManager`].
#[derive(Debug, Clone)]
pub struct PrefixCache {
    inner: Arc<Mutex<PrefixCacheManager>>,
}

impl PrefixCache {
    /// Create a thread-safe prefix cache manager.
    #[must_use]
    pub fn new(limits: PrefixCacheLimits, initial_generation: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PrefixCacheManager::new(
                limits,
                initial_generation,
            ))),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, PrefixCacheManager>, PrefixCacheError> {
        self.inner
            .lock()
            .map_err(|_| PrefixCacheError::LockPoisoned)
    }

    /// Look up a token sequence in the prefix cache.
    pub fn lookup(
        &self,
        key: &PrefixCacheKey,
        tokens: &[u32],
    ) -> Result<PrefixMatchResult, PrefixCacheError> {
        self.lock()?.lookup_prefix(key, tokens)
    }

    /// Insert or extend a prefix.
    pub fn insert_or_extend(
        &self,
        key: &PrefixCacheKey,
        tokens: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, PrefixCacheError> {
        self.lock()?
            .insert_or_extend_prefix(key, tokens, existing_pages)
    }

    /// Pin pages.
    pub fn pin(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.pin_pages(page_ids)
    }

    /// Unpin pages.
    pub fn unpin(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.unpin_pages(page_ids)
    }

    /// Mark pages as in-flight.
    pub fn mark_in_flight(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.mark_in_flight(page_ids)
    }

    /// Clear in-flight status.
    pub fn clear_in_flight(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.clear_in_flight(page_ids)
    }

    /// Release leased pages.
    pub fn release(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.release_pages(page_ids)
    }

    /// Fetch snapshot of metrics.
    pub fn metrics(&self) -> Result<PrefixCacheMetrics, PrefixCacheError> {
        Ok(self.lock()?.metrics().clone())
    }

    /// Invalidate generation.
    pub fn invalidate_generation(&self, new_gen: u64) -> Result<(), PrefixCacheError> {
        self.lock()?.invalidate_generation(new_gen);
        Ok(())
    }
}
