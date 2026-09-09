use super::*;
use vyre_foundation::ir::DataType;

fn test_key(tenant: &str, trust: Option<&str>, generation: u64) -> RetainedPageCacheKey {
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
        device_generation: generation,
        cache_schema_version: 1,
        isolation_domain: tenant.to_string(),
        trust_domain: trust.map(str::to_string),
    }
}

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
    let key = test_key("tenant_a", None, 1);

    let sequence = vec![101, 202, 303, 404, 505];

    // Cold lookup -> miss
    let miss = cache.lookup(&key, &sequence).expect("lookup");
    assert_eq!(miss.matched_units, 0);
    assert_eq!(miss.page_count, 0);

    // Insert sequence
    let pages = cache
        .insert_or_extend(&key, &sequence, &[])
        .expect("insert");
    assert_eq!(pages.len(), 1);

    // Warm lookup -> hit
    let hit = cache.lookup(&key, &sequence).expect("lookup");
    assert_eq!(hit.matched_units, 5);
    assert_eq!(hit.page_ids, pages);

    // Release references
    cache.release(&pages).expect("release");
}

#[test]
fn retained_page_cache_isolation_domain_enforcement() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key_a = test_key("tenant_a", None, 1);
    let key_b = test_key("tenant_b", None, 1);

    let sequence = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &sequence, &[])
        .expect("insert");

    // Tenant B attempts to look up Tenant A's sequence -> isolation violation
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
    let key_a = test_key("tenant_a", Some("common_trust_group"), 1);
    let key_b = test_key("tenant_b", Some("common_trust_group"), 1);

    let sequence = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &sequence, &[])
        .expect("insert");

    // Shared trust domain allows physical sharing across distinct tenants
    let hit = cache.lookup(&key_b, &sequence).expect("lookup");
    assert_eq!(hit.matched_units, 4);
    assert_eq!(hit.page_ids, pages_a);

    cache.release(&pages_a).expect("release");
    cache.release(&hit.page_ids).expect("release");
}

#[test]
fn retained_page_cache_stale_generation_rejected() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key_stale = test_key("tenant_a", None, 0);

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
    let key = test_key("tenant_a", None, 1);

    // Page 1
    let p1 = cache.insert_or_extend(&key, &[1, 2], &[]).expect("p1");
    cache.pin(&p1).expect("pin");

    // Page 2
    let p2 = cache.insert_or_extend(&key, &[3, 4], &[]).expect("p2");
    cache.mark_in_flight(&p2).expect("in-flight");

    // Attempting to allocate Page 3 when both Page 1 (pinned) and Page 2 (in-flight) cannot be evicted
    let err = cache.insert_or_extend(&key, &[5, 6], &[]).unwrap_err();
    assert!(matches!(
        err,
        RetainedPageCacheError::CapacityExceeded { .. }
    ));

    cache.unpin(&p1).expect("unpin");
    cache.release(&p1).expect("release");

    // Now p1 is unpinned and ref_count=0 -> can be evicted
    let p3 = cache.insert_or_extend(&key, &[5, 6], &[]).expect("p3");
    assert_eq!(p3.len(), 1);
}

#[test]
fn retained_page_cache_duplicate_release_rejected() {
    let cache = RetainedPageCache::new(RetainedPageLimits::default(), 1);
    let key = test_key("tenant_a", None, 1);

    let p = cache.insert_or_extend(&key, &[1, 2], &[]).expect("p");
    cache.release(&p).expect("first release");

    let err = cache.release(&p).unwrap_err();
    assert!(matches!(err, RetainedPageCacheError::DuplicateRelease(_)));
}
