//! Speculative state transactions, provisional retained state, and transactional verification/rollback.
//!
//! # Architecture
//!
//! Speculative execution accelerates stateful execution pipelines by predicting
//! multiple speculative future proposals ($K$ steps) concurrently.
//!
//! Key principles:
//! - **Provisional Writes**: Speculative state writes and cache reservations remain strictly
//!   provisional until verified.
//! - **Transactional Verification & Rollback**: When $m \le K$ proposals are accepted,
//!   only the verified prefix of $m$ items is committed to the cache. All unaccepted
//!   proposals ($m < k \le K$), cancellations, or device failures immediately roll back
//!   their provisional state entries and release reserved pages, without exposing
//!   uncommitted speculative state to any other request.

use thiserror::Error;

use crate::retained_page_cache::{RetainedPageCache, RetainedPageCacheError, RetainedPageCacheKey};

/// Configuration parameters for speculative state transactions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeTransactionConfig {
    /// Maximum prediction depth (number of speculative proposals $K$).
    pub max_depth: usize,
}

impl Default for SpeculativeTransactionConfig {
    fn default() -> Self {
        Self { max_depth: 3 }
    }
}

/// Errors occurring during speculative execution or verification.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SpeculativeTransactionError {
    /// Prediction depth exceeds configured maximum.
    #[error("requested speculative depth {depth} exceeds configured maximum {max_depth}")]
    DepthExceeded {
        /// Requested depth.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },
    /// Verification failed due to mismatch or invalid state.
    #[error("speculative verification error: {0}")]
    VerificationFailure(String),
    /// Underlying retained page cache error.
    #[error("retained page cache error: {0}")]
    CacheError(#[from] RetainedPageCacheError),
    /// Speculative execution was cancelled.
    #[error("speculative execution step was cancelled")]
    Cancelled,
}

/// A speculative proposal item from a speculative step.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeProposal {
    /// Proposal depth (1-indexed: 1, 2, ..., K).
    pub depth: usize,
    /// Speculative proposed unit ID.
    pub unit_id: u32,
    /// Confidence or weight score.
    pub confidence: f32,
    /// Provisional physical page IDs allocated for this speculative step.
    pub provisional_page_ids: Vec<u32>,
}

/// Provisional state held during speculative execution before verification.
#[derive(Debug, Clone)]
pub struct SpeculativeProvisionalState {
    /// Request identifier.
    pub request_id: u64,
    /// Verified base prefix units before this speculative step.
    pub base_prefix_units: Vec<u32>,
    /// Speculative proposals generated.
    pub proposals: Vec<SpeculativeProposal>,
    /// Cache key under which provisional pages are reserved.
    pub cache_key: RetainedPageCacheKey,
}

/// Outcome of speculative verification against ground-truth verification units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransactionCommitResult {
    /// Sequence verified and accepted (including base, matching proposals, and correction).
    pub accepted_units: Vec<u32>,
    /// Number of speculative proposals accepted ($0 \le m \le K$).
    pub accepted_proposal_count: usize,
    /// Number of speculative proposals rolled back.
    pub rolled_back_count: usize,
    /// Pages committed permanently into the retained cache.
    pub committed_pages: Vec<u32>,
    /// Provisional pages discarded and rolled back.
    pub released_pages: Vec<u32>,
}

/// Lifecycle and verification coordinator for speculative state transactions.
pub struct SpeculativeTransactionCoordinator {
    config: SpeculativeTransactionConfig,
    cache: RetainedPageCache,
}

impl SpeculativeTransactionCoordinator {
    /// Create a new speculative transaction coordinator.
    #[must_use]
    pub fn new(config: SpeculativeTransactionConfig, cache: RetainedPageCache) -> Self {
        Self { config, cache }
    }

    /// Access configuration.
    #[must_use]
    pub const fn config(&self) -> &SpeculativeTransactionConfig {
        &self.config
    }

    /// Stage a speculative step: reserves provisional pages for proposals.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeTransactionError`] if depth exceeds configuration or allocation fails.
    pub fn stage_speculative_step(
        &self,
        request_id: u64,
        cache_key: &RetainedPageCacheKey,
        base_prefix: &[u32],
        proposals: &[(u32, f32)], // (unit_id, confidence)
    ) -> Result<SpeculativeProvisionalState, SpeculativeTransactionError> {
        let depth = proposals.len();
        if depth > self.config.max_depth {
            return Err(SpeculativeTransactionError::DepthExceeded {
                depth,
                max_depth: self.config.max_depth,
            });
        }

        let mut full_sequence = base_prefix.to_vec();
        let mut proposal_records = Vec::with_capacity(depth);

        for (i, &(unit_id, confidence)) in proposals.iter().enumerate() {
            full_sequence.push(unit_id);
            let provisional_pages = self
                .cache
                .insert_or_extend(cache_key, &full_sequence, &[])?;

            proposal_records.push(SpeculativeProposal {
                depth: i + 1,
                unit_id,
                confidence,
                provisional_page_ids: provisional_pages,
            });
        }

        Ok(SpeculativeProvisionalState {
            request_id,
            base_prefix_units: base_prefix.to_vec(),
            proposals: proposal_records,
            cache_key: cache_key.clone(),
        })
    }

    /// Verify proposals against ground-truth verification units.
    ///
    /// Commits accepted prefix to the cache and rolls back rejected entries.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeTransactionError`] on verification or cache release failure.
    pub fn verify_and_commit(
        &self,
        provisional: SpeculativeProvisionalState,
        verified_ground_truth: &[u32],
    ) -> Result<TransactionCommitResult, SpeculativeTransactionError> {
        let mut accepted_units = provisional.base_prefix_units;
        let mut accepted_proposal_count = 0;
        let mut committed_pages = Vec::new();
        let mut released_pages = Vec::new();

        let mut mismatch_occurred = false;

        for (i, proposal) in provisional.proposals.iter().enumerate() {
            if !mismatch_occurred
                && i < verified_ground_truth.len()
                && proposal.unit_id == verified_ground_truth[i]
            {
                accepted_units.push(proposal.unit_id);
                accepted_proposal_count += 1;
                committed_pages.extend_from_slice(&proposal.provisional_page_ids);
            } else {
                mismatch_occurred = true;
                released_pages.extend_from_slice(&proposal.provisional_page_ids);
            }
        }

        if verified_ground_truth.len() > accepted_proposal_count {
            accepted_units.push(verified_ground_truth[accepted_proposal_count]);
        }

        if !released_pages.is_empty() {
            self.cache.release(&released_pages)?;
        }

        let rolled_back_count = provisional.proposals.len() - accepted_proposal_count;

        Ok(TransactionCommitResult {
            accepted_units,
            accepted_proposal_count,
            rolled_back_count,
            committed_pages,
            released_pages,
        })
    }

    /// Roll back all provisional speculative state in case of cancellation or device failure.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculativeTransactionError`] on cache release failure.
    pub fn rollback_all(
        &self,
        provisional: SpeculativeProvisionalState,
    ) -> Result<(), SpeculativeTransactionError> {
        let mut all_pages = Vec::new();
        for proposal in provisional.proposals {
            all_pages.extend_from_slice(&proposal.provisional_page_ids);
        }
        if !all_pages.is_empty() {
            self.cache.release(&all_pages)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retained_page_cache::{RetainedPageLayout, RetainedPageLimits};
    use vyre_foundation::ir::DataType;

    fn test_key() -> RetainedPageCacheKey {
        RetainedPageCacheKey {
            content_id: [10u8; 32],
            schema_id: [20u8; 32],
            payload_digest: [30u8; 32],
            config_digest: [40u8; 32],
            dtype: DataType::F32,
            layout: RetainedPageLayout {
                feature_channels: 2,
                unit_dimension: 32,
                units_per_page: 16,
            },
            device_generation: 1,
            cache_schema_version: 1,
            isolation_domain: "tenant_spec".to_string(),
            trust_domain: None,
        }
    }

    #[test]
    fn speculative_partial_acceptance_and_rollback() {
        let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
        let config = SpeculativeTransactionConfig { max_depth: 3 };
        let coordinator = SpeculativeTransactionCoordinator::new(config, cache.clone());
        let key = test_key();

        let base_prefix = vec![10, 20];
        let proposals = vec![(30, 0.9), (40, 0.8), (50, 0.7)];

        let staged = coordinator
            .stage_speculative_step(1, &key, &base_prefix, &proposals)
            .expect("stage");

        // Ground truth verification: unit 30 is accepted, but second unit is 45 (not 40).
        let ground_truth = vec![30, 45];

        let result = coordinator
            .verify_and_commit(staged, &ground_truth)
            .expect("verify");

        assert_eq!(result.accepted_proposal_count, 1);
        assert_eq!(result.rolled_back_count, 2);
        assert_eq!(result.accepted_units, vec![10, 20, 30, 45]);
        assert!(!result.released_pages.is_empty());
    }

    #[test]
    fn speculative_full_cancellation_rollback() {
        let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
        let coordinator = SpeculativeTransactionCoordinator::new(
            SpeculativeTransactionConfig::default(),
            cache.clone(),
        );
        let key = test_key();

        let staged = coordinator
            .stage_speculative_step(1, &key, &[1, 2], &[(3, 0.9), (4, 0.9)])
            .expect("stage");

        coordinator.rollback_all(staged).expect("rollback");
    }
}
