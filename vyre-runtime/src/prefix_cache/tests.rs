use super::*;
use vyre_foundation::ir::DataType;

fn test_key(tenant: &str, trust: Option<&str>, gen: u64) -> PrefixCacheKey {
    PrefixCacheKey::test_sample(tenant, trust, gen)
}

#[test]
fn prefix_cache_cold_miss_and_warm_hit() {
    let limits = PrefixCacheLimits {
        max_pages: 16,
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_tokens: 1024,
        per_tenant_page_limit: 16,
    };
    let cache = PrefixCache::new(limits, 1);
    let key = test_key("tenant_a", None, 1);

    let prompt = vec![101, 202, 303, 404, 505];

    // Cold lookup -> miss
    let miss = cache.lookup(&key, &prompt).expect("lookup");
    assert_eq!(miss.matched_tokens, 0);
    assert_eq!(miss.page_count, 0);

    // Insert prompt
    let pages = cache.insert_or_extend(&key, &prompt, &[]).expect("insert");
    assert_eq!(pages.len(), 1); // 5 tokens fits in 1 block (16 tokens)

    // Warm lookup -> hit
    let hit = cache.lookup(&key, &prompt).expect("lookup");
    assert_eq!(hit.matched_tokens, 5);
    assert_eq!(hit.page_ids, pages);

    // Release references
    cache.release(&pages).expect("release");
}

#[test]
fn prefix_cache_isolation_domain_enforcement() {
    let cache = PrefixCache::new(PrefixCacheLimits::default(), 1);
    let key_a = test_key("tenant_a", None, 1);
    let key_b = test_key("tenant_b", None, 1); // Different tenant, no shared trust domain

    let prompt = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &prompt, &[])
        .expect("insert");

    // Tenant B attempts to look up Tenant A's prefix -> isolation violation
    let err = cache.lookup(&key_b, &prompt).unwrap_err();
    assert!(matches!(err, PrefixCacheError::IsolationViolation { .. }));

    cache.release(&pages_a).expect("release");
}

#[test]
fn prefix_cache_shared_trust_domain_allowed() {
    let cache = PrefixCache::new(PrefixCacheLimits::default(), 1);
    let key_a = test_key("tenant_a", Some("common_trust_group"), 1);
    let key_b = test_key("tenant_b", Some("common_trust_group"), 1);

    let prompt = vec![10, 20, 30, 40];
    let pages_a = cache
        .insert_or_extend(&key_a, &prompt, &[])
        .expect("insert");

    // Shared trust domain allows physical sharing across distinct tenants
    let hit = cache.lookup(&key_b, &prompt).expect("lookup");
    assert_eq!(hit.matched_tokens, 4);
    assert_eq!(hit.page_ids, pages_a);

    cache.release(&pages_a).expect("release");
    cache.release(&hit.page_ids).expect("release");
}

#[test]
fn prefix_cache_stale_generation_rejected() {
    let cache = PrefixCache::new(PrefixCacheLimits::default(), 1);
    let key_stale = test_key("tenant_a", None, 0); // Stale gen 0 != current 1

    let prompt = vec![1, 2, 3];
    let err = cache.lookup(&key_stale, &prompt).unwrap_err();
    assert!(matches!(
        err,
        PrefixCacheError::StaleDeviceGeneration { .. }
    ));
}

#[test]
fn prefix_cache_eviction_protects_pinned_and_in_flight() {
    let limits = PrefixCacheLimits {
        max_pages: 2, // Only 2 pages total
        max_bytes: 1024 * 1024,
        max_active_requests: 8,
        max_queued_tokens: 1024,
        per_tenant_page_limit: 8,
    };
    let cache = PrefixCache::new(limits, 1);
    let key = test_key("tenant_a", None, 1);

    // Page 1
    let p1 = cache.insert_or_extend(&key, &[1, 2], &[]).expect("p1");
    cache.pin(&p1).expect("pin");

    // Page 2
    let p2 = cache.insert_or_extend(&key, &[3, 4], &[]).expect("p2");
    cache.mark_in_flight(&p2).expect("in-flight");

    // Attempting to allocate Page 3 when both Page 1 (pinned) and Page 2 (in-flight) cannot be evicted
    let err = cache.insert_or_extend(&key, &[5, 6], &[]).unwrap_err();
    assert!(matches!(err, PrefixCacheError::CapacityExceeded { .. }));

    cache.unpin(&p1).expect("unpin");
    cache.release(&p1).expect("release");

    // Now p1 is unpinned and ref_count=0 -> can be evicted
    let p3 = cache.insert_or_extend(&key, &[5, 6], &[]).expect("p3");
    assert_eq!(p3.len(), 1);
}

#[test]
fn prefix_cache_duplicate_release_rejected() {
    let cache = PrefixCache::new(PrefixCacheLimits::default(), 1);
    let key = test_key("tenant_a", None, 1);

    let pages = cache
        .insert_or_extend(&key, &[1, 2, 3], &[])
        .expect("insert");
    cache.release(&pages).expect("first release");

    // Duplicate release must fail with DuplicateRelease error
    let err = cache.release(&pages).unwrap_err();
    assert!(matches!(err, PrefixCacheError::DuplicateRelease(_)));
}

impl PrefixCacheKey {
    /// Constructs a representative prefix cache key for testing.
    #[cfg(test)]
    pub fn test_sample(tenant: &str, trust: Option<&str>, gen: u64) -> Self {
        Self {
            model_id: [10u8; 32],
            tokenizer_id: [20u8; 32],
            weights_digest: [30u8; 32],
            config_digest: [40u8; 32],
            dtype: DataType::F32,
            layout: PrefixCacheLayout {
                kv_heads: 2,
                head_dim: 32,
                block_tokens: 16,
            },
            device_generation: gen,
            cache_schema_version: 1,
            isolation_domain: tenant.to_string(),
            trust_domain: trust.map(|s| s.to_string()),
        }
    }
}
