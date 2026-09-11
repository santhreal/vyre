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

use std::sync::Arc;

pub use contract::{
    RetainedMatchResult, RetainedPageCacheError, RetainedPageCacheKey,
    RetainedPageCacheKeyFingerprint, RetainedPageLayout, RetainedPageLimits, RetainedPageMetrics,
};
pub use manager::RetainedPageCacheManager;
use vyre_foundation::failure_domain::{FailureDomain, RecoveryClass};

use crate::atomic_recovery::AtomicGuardedState;

/// Thread-safe wrapper around [`RetainedPageCacheManager`].
///
/// WHY: the pool records which physical pages the device holds, how many
/// leases each carries, and which are pinned or in flight. A panic under the
/// lock leaves that record half written, and every lease derived from it is
/// then unprovable, so the pool transitions to a terminal state and rejects
/// operations until a generation rebuild replaces it. The rebuild is the same
/// transition a device reset takes, because both invalidate every page the
/// pool recorded.
#[derive(Debug, Clone)]
pub struct RetainedPageCache {
    inner: Arc<AtomicGuardedState<RetainedPageCacheManager>>,
    limits: RetainedPageLimits,
}

/// The failure domain of one retained page pool.
const PAGE_CACHE_DOMAIN: FailureDomain = FailureDomain::DeviceContext;

/// The recovery class of one retained page pool.
const PAGE_CACHE_CLASS: RecoveryClass = RecoveryClass::DeviceContextFatal;

impl RetainedPageCache {
    /// Create a thread-safe retained page cache manager.
    #[must_use]
    pub fn new(limits: RetainedPageLimits, initial_generation: u64) -> Self {
        Self {
            inner: Arc::new(AtomicGuardedState::new(
                RetainedPageCacheManager::new(limits, initial_generation),
                PAGE_CACHE_DOMAIN,
                PAGE_CACHE_CLASS,
            )),
            limits,
        }
    }

    /// Look up a sequence prefix in the retained page cache.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the lookup's own error otherwise.
    pub fn lookup(
        &self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
    ) -> Result<RetainedMatchResult, RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.lookup_prefix(key, sequence))
    }

    /// Insert or extend a sequence prefix.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the insertion's own error otherwise.
    pub fn insert_or_extend(
        &self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, RetainedPageCacheError> {
        self.inner.try_with_state(|manager| {
            manager.insert_or_extend_prefix(key, sequence, existing_pages)
        })
    }

    /// Pin pages.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the pin's own error otherwise.
    pub fn pin(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.pin_pages(page_ids))
    }

    /// Unpin pages.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the unpin's own error otherwise.
    pub fn unpin(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.unpin_pages(page_ids))
    }

    /// Mark pages as in-flight during kernel execution.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the transition's own error otherwise.
    pub fn mark_in_flight(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.mark_in_flight(page_ids))
    }

    /// Clear in-flight status after kernel completion.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the transition's own error otherwise.
    pub fn complete_in_flight(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.clear_in_flight(page_ids))
    }

    /// Release leased page references.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding, and the release's own error otherwise.
    pub fn release(&self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| manager.release_pages(page_ids))
    }

    /// Enter the documented recovery, rejecting every operation until
    /// [`complete_generation_rebuild`](Self::complete_generation_rebuild)
    /// installs the new pool.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] when a rebuild is already
    /// in progress, so two callers cannot each believe they own it.
    pub fn begin_generation_rebuild(&self) -> Result<(), RetainedPageCacheError> {
        self.inner.begin_rebuild()?;
        Ok(())
    }

    /// Install a pool at `new_generation` and resume accepting operations.
    ///
    /// Every page the previous pool recorded is discarded, which is correct
    /// for both a device reset and a pool a panic left half written.
    pub fn complete_generation_rebuild(&self, new_generation: u64) {
        self.inner
            .finish_rebuild(RetainedPageCacheManager::new(self.limits, new_generation));
    }

    /// Invalidate device generation.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] when a rebuild is already
    /// in progress.
    pub fn invalidate_generation(&self, new_generation: u64) -> Result<(), RetainedPageCacheError> {
        self.begin_generation_rebuild()?;
        self.complete_generation_rebuild(new_generation);
        Ok(())
    }

    /// Read telemetry metrics.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError::Recovery`] while the pool is terminal
    /// or rebuilding.
    pub fn metrics(&self) -> Result<RetainedPageMetrics, RetainedPageCacheError> {
        self.inner
            .try_with_state(|manager| Ok(manager.metrics().clone()))
    }
}

impl crate::StateOwnerRecovery for RetainedPageCache {
    fn failure_domain(&self) -> crate::FailureDomain {
        PAGE_CACHE_DOMAIN
    }

    fn recovery_class(&self) -> crate::RecoveryClass {
        PAGE_CACHE_CLASS
    }
}
