//! Contracts of the retained page cache, asserted through its public surface.
//!
//! These cases ran as a `#[cfg(test)]` module beside the implementation while
//! touching nothing private, and each one restated the cache key the shared
//! fixture already owns. They assert the same contracts here against
//! `vyre_runtime::retained_page_cache`, with one key definition.

use crate::retained_cache_fixtures::retained_key;
use vyre_runtime::retained_page_cache::{
    RetainedPageCache, RetainedPageCacheError, RetainedPageLimits,
};

#[test]
fn retained_page_cache_cold_miss_and_warm_hit() {
    let limits = RetainedPageLimits {
        max_pages: 16,
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_units: 1024,
        per_tenant_page_limit: 16,
    };
    let cache = RetainedPageCache::new(limits, 1);
    let key = retained_key("tenant_a", None, 1);

    let sequence = vec![101, 202, 303, 404, 505];

    let miss = cache.lookup(&key, &sequence).expect("lookup");
    assert_eq!(miss.matched_units, 0);
    assert_eq!(miss.page_count, 0);

    let pages = cache
        .insert_or_extend(&key, &sequence, &[])
        .expect("insert");
    assert_eq!(pages.len(), 1);

    let hit = cache.lookup(&key, &sequence).expect("lookup");
    assert_eq!(hit.matched_units, 5);
    assert_eq!(hit.page_ids, pages);

    cache.release(&pages).expect("release");
}

#[test]
fn retained_page_cache_isolation_domain_enforcement() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key_a = retained_key("tenant_a", None, 1);
    let key_b = retained_key("tenant_b", None, 1);

    let sequence = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &sequence, &[])
        .expect("insert");

    // Tenant B looking up tenant A's sequence is an isolation violation.
    let err = cache.lookup(&key_b, &sequence).unwrap_err();
    assert!(matches!(
        err,
        RetainedPageCacheError::IsolationViolation { .. }
    ));

    cache.release(&pages_a).expect("release");
}

#[test]
fn retained_page_cache_shared_trust_domain_allowed() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key_a = retained_key("tenant_a", Some("common_trust_group"), 1);
    let key_b = retained_key("tenant_b", Some("common_trust_group"), 1);

    let sequence = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &sequence, &[])
        .expect("insert");

    // A shared trust domain allows physical sharing across distinct tenants.
    let hit = cache.lookup(&key_b, &sequence).expect("lookup");
    assert_eq!(hit.matched_units, 4);
    assert_eq!(hit.page_ids, pages_a);

    cache.release(&pages_a).expect("release");
    cache.release(&hit.page_ids).expect("release");
}

#[test]
fn retained_page_cache_stale_generation_rejected() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key_stale = retained_key("tenant_a", None, 0);

    let sequence = vec![1, 2, 3];
    let err = cache.lookup(&key_stale, &sequence).unwrap_err();
    assert!(matches!(
        err,
        RetainedPageCacheError::StaleDeviceGeneration { .. }
    ));
}

#[test]
fn retained_page_cache_eviction_protects_pinned_and_in_flight() {
    let limits = RetainedPageLimits {
        max_pages: 2,
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_units: 1024,
        per_tenant_page_limit: 8,
    };
    let cache = RetainedPageCache::new(limits, 1);
    let key = retained_key("tenant_a", None, 1);

    let p1 = cache.insert_or_extend(&key, &[1, 2], &[]).expect("p1");
    cache.pin(&p1).expect("pin");

    let p2 = cache.insert_or_extend(&key, &[3, 4], &[]).expect("p2");
    cache.mark_in_flight(&p2).expect("in-flight");

    // A third page needs an eviction, and neither the pinned nor the in-flight
    // page may be evicted.
    let err = cache.insert_or_extend(&key, &[5, 6], &[]).unwrap_err();
    assert!(matches!(
        err,
        RetainedPageCacheError::CapacityExceeded { .. }
    ));

    cache.unpin(&p1).expect("unpin");
    cache.release(&p1).expect("release");

    let p3 = cache.insert_or_extend(&key, &[5, 6], &[]).expect("p3");
    assert_eq!(p3.len(), 1);
}

#[test]
fn retained_page_cache_duplicate_release_rejected() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key = retained_key("tenant_a", None, 1);

    let p = cache.insert_or_extend(&key, &[1, 2], &[]).expect("p");
    cache.release(&p).expect("first release");

    let err = cache.release(&p).unwrap_err();
    assert!(matches!(err, RetainedPageCacheError::DuplicateRelease(_)));
}
