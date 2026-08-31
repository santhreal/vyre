//! The prefix cache key every prefix-cache proof asserts against.
//!
//! A key is the cache's identity: fourteen fields, of which three decide what a
//! given case is about. Restating the other eleven per suite means a field added
//! to the identity is added in every copy, and a copy that drifts silently stops
//! testing the same cache.

// Every test binary compiles this module on its own, so a fixture a given suite
// does not ask for is unused in that binary.
#![allow(dead_code)]

use vyre_foundation::ir::DataType;
use vyre_runtime::prefix_cache::{PrefixCacheKey, PrefixCacheLayout};

/// A representative key for `tenant`, optionally in `trust`, valid for device
/// generation `generation`.
pub(crate) fn prefix_key(tenant: &str, trust: Option<&str>, generation: u64) -> PrefixCacheKey {
    PrefixCacheKey {
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
        device_generation: generation,
        cache_schema_version: 1,
        isolation_domain: tenant.to_string(),
        trust_domain: trust.map(str::to_string),
    }
}
