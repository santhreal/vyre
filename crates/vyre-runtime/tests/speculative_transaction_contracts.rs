//! Contracts of the speculative transaction coordinator.
//!
//! These cases ran as a `#[cfg(test)]` module beside the implementation while
//! touching nothing private, and carried a third copy of the retained cache
//! key. They assert the same contracts here against the public coordinator,
//! keyed by the shared fixture.

use crate::retained_cache_fixtures::retained_key;
use vyre_runtime::retained_page_cache::{RetainedPageCache, RetainedPageLimits};
use vyre_runtime::speculative_transaction::{
    SpeculativeTransactionConfig, SpeculativeTransactionCoordinator,
};

#[test]
fn speculative_partial_acceptance_and_rollback() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let config = SpeculativeTransactionConfig { max_depth: 3 };
    let coordinator = SpeculativeTransactionCoordinator::new(config, cache.clone());
    let key = retained_key("tenant_spec", None, 1);

    let base_prefix = vec![10, 20];
    let proposals = vec![(30, 0.9), (40, 0.8), (50, 0.7)];

    let staged = coordinator
        .stage_speculative_step(1, &key, &base_prefix, &proposals)
        .expect("stage");

    // Ground truth accepts unit 30 and diverges at the second unit.
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
    let key = retained_key("tenant_spec", None, 1);

    let staged = coordinator
        .stage_speculative_step(1, &key, &[1, 2], &[(3, 0.9), (4, 0.9)])
        .expect("stage");

    coordinator.rollback_all(staged).expect("rollback");
}
