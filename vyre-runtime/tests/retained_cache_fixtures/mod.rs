//! The retained cache key fixture every retained-cache proof asserts against.
//!
//! A key is the cache's identity: ten fields, of which three decide what a
//! given case is about. Restating the other seven per suite means a field added
//! to the identity is added in every copy, and a copy that drifts silently stops
//! testing the same cache.

#![allow(dead_code)]

use vyre_foundation::ir::DataType;
use vyre_runtime::retained_page_cache::{RetainedPageCacheKey, RetainedPageLayout};

/// A representative key for `tenant`, optionally in `trust`, valid for device
/// generation `generation`.
pub(crate) fn retained_key(
    tenant: &str,
    trust: Option<&str>,
    generation: u64,
) -> RetainedPageCacheKey {
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
