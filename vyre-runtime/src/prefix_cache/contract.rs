//! Values the prefix cache exchanges with a caller: key, layout, match result,
//! limits, counters, and failure.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_foundation::ir::DataType;

/// Layout and geometry for KV cache pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixCacheLayout {
    /// Number of KV attention heads.
    pub kv_heads: u32,
    /// Dimension per head.
    pub head_dim: u32,
    /// Number of tokens stored in one physical page.
    pub block_tokens: u32,
}

impl PrefixCacheLayout {
    /// Compute byte size of a single physical page for keys + values.
    #[must_use]
    pub fn page_bytes(&self, dtype: &DataType) -> u64 {
        let elem_bytes = match dtype {
            DataType::F32 | DataType::U32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => 2,
            DataType::U8 | DataType::I8 | DataType::Bool => 1,
            _ => 4,
        };
        // 2 for K and V
        2 * (self.kv_heads as u64)
            * (self.block_tokens as u64)
            * (self.head_dim as u64)
            * elem_bytes
    }
}

/// Multi-dimensional immutable prefix cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PrefixCacheKey {
    /// Model identifier hash.
    pub model_id: [u8; 32],
    /// Tokenizer vocabulary/scheme hash.
    pub tokenizer_id: [u8; 32],
    /// Exact weights content digest.
    pub weights_digest: [u8; 32],
    /// Configuration / hyperparameter digest.
    pub config_digest: [u8; 32],
    /// Element data type.
    pub dtype: DataType,
    /// KV head and block geometry.
    pub layout: PrefixCacheLayout,
    /// Device allocation generation.
    pub device_generation: u64,
    /// Cache schema version.
    pub cache_schema_version: u32,
    /// Tenant or request isolation domain.
    pub isolation_domain: String,
    /// Optional explicit trust domain required for cross-tenant sharing.
    pub trust_domain: Option<String>,
}

/// Canonical stable fingerprint of a prefix cache key for total ordering and lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrefixCacheKeyFingerprint(pub [u8; 32]);

impl PrefixCacheKey {
    /// Compute model-level fingerprint for Radix Trie indexing.
    #[must_use]
    pub fn model_fingerprint(&self) -> PrefixCacheKeyFingerprint {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.model_id);
        hasher.update(&self.tokenizer_id);
        hasher.update(&self.weights_digest);
        hasher.update(&self.config_digest);
        hasher.update(format!("{:?}", self.dtype).as_bytes());
        hasher.update(&self.layout.kv_heads.to_le_bytes());
        hasher.update(&self.layout.head_dim.to_le_bytes());
        hasher.update(&self.layout.block_tokens.to_le_bytes());
        hasher.update(&self.device_generation.to_le_bytes());
        hasher.update(&self.cache_schema_version.to_le_bytes());
        PrefixCacheKeyFingerprint(*hasher.finalize().as_bytes())
    }

    /// Compute the deterministic canonical fingerprint for this key including tenant isolation.
    #[must_use]
    pub fn fingerprint(&self) -> PrefixCacheKeyFingerprint {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.model_fingerprint().0);
        hasher.update(self.isolation_domain.as_bytes());
        if let Some(td) = &self.trust_domain {
            hasher.update(&[1u8]);
            hasher.update(td.as_bytes());
        } else {
            hasher.update(&[0u8]);
        }
        PrefixCacheKeyFingerprint(*hasher.finalize().as_bytes())
    }
}

/// Result of matching a token sequence in the prefix cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatchResult {
    /// Number of tokens matched from the prefix.
    pub matched_tokens: usize,
    /// Physical page IDs backing the matched prefix.
    pub page_ids: Vec<u32>,
    /// Number of pages backing the matched prefix.
    pub page_count: usize,
    /// Device generation of the matched pages.
    pub generation: u64,
}

/// Limits and backpressure thresholds for cache and queue residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCacheLimits {
    /// Maximum physical pages allowed in the pool.
    pub max_pages: usize,
    /// Maximum resident bytes allowed.
    pub max_bytes: u64,
    /// Maximum concurrent active requests leasing cache pages.
    pub max_active_requests: usize,
    /// Maximum queued tokens across active requests.
    pub max_queued_tokens: usize,
    /// Maximum pages allowed for a single tenant isolation domain.
    pub per_tenant_page_limit: usize,
}

impl Default for PrefixCacheLimits {
    fn default() -> Self {
        Self {
            max_pages: 1024,
            max_bytes: 1024 * 1024 * 1024, // 1 GB
            max_active_requests: 64,
            max_queued_tokens: 131_072,
            per_tenant_page_limit: 512,
        }
    }
}

/// Telemetry metrics for prefix cache lifecycle and residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefixCacheMetrics {
    /// Current number of allocated pages.
    pub allocated_pages: usize,
    /// Current number of pinned pages.
    pub pinned_pages: usize,
    /// Current number of in-flight pages.
    pub in_flight_pages: usize,
    /// Current free pages in pool.
    pub free_pages: usize,
    /// Total cumulative page evictions.
    pub evicted_pages: usize,
    /// Total cumulative bytes scrubbed.
    pub scrubbed_bytes: u64,
    /// Total cumulative cache hit lookups.
    pub cache_hits: u64,
    /// Total cumulative cache miss lookups.
    pub cache_misses: u64,
    /// Total cumulative copy-on-write page splits.
    pub cow_copies: u64,
    /// Total cumulative rejections due to stale device generation.
    pub rejected_stale_generations: u64,
    /// Total cumulative rejections due to backpressure/saturation.
    pub backpressure_rejections: u64,
}

/// Errors occurring during prefix cache operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PrefixCacheError {
    /// Token sequence is empty.
    #[error("token sequence cannot be empty for prefix cache operations")]
    EmptyTokens,
    /// Memory pool capacity exceeded.
    #[error("prefix cache pool exhausted: needed {needed} pages, but only {available} available")]
    CapacityExceeded {
        /// Pages requested.
        needed: usize,
        /// Available unpinned/free pages.
        available: usize,
    },
    /// Backpressure limit exceeded on active requests or queued tokens.
    #[error("prefix cache admission rejected: {reason}. Fix: apply backpressure or wait for requests to complete")]
    BackpressureLimitExceeded {
        /// Reason for rejection.
        reason: &'static str,
    },
    /// Tenant page quota exceeded.
    #[error("tenant {tenant} exceeded page quota {limit}")]
    TenantQuotaExceeded {
        /// Tenant identifier.
        tenant: String,
        /// Maximum allowed pages.
        limit: usize,
    },
    /// Stale device generation detected.
    #[error("stale device generation: expected {expected}, got {actual}. Fix: re-admit prefix against current device generation")]
    StaleDeviceGeneration {
        /// Current device generation.
        expected: u64,
        /// Stale generation.
        actual: u64,
    },
    /// Isolation domain violation without explicit common trust domain.
    #[error("cross-tenant physical sharing denied between {tenant_a} and {tenant_b} without common trust domain")]
    IsolationViolation {
        /// First tenant.
        tenant_a: String,
        /// Second tenant.
        tenant_b: String,
    },
    /// Page was not found in physical pool.
    #[error("physical page {0} not found")]
    PageNotFound(u32),
    /// Duplicate release of a page lease or reference underflow.
    #[error("duplicate release or reference underflow for page {0}")]
    DuplicateRelease(u32),
    /// Cannot evict a pinned or in-flight page.
    #[error("cannot evict page {0}: page is pinned or in-flight")]
    PinnedPageEvictionRejected(u32),
    /// Reassignment of unscrubbed page.
    #[error("page {0} has not been scrubbed before reassignment")]
    UnscrubbedPageReassignment(u32),
    /// Mutex lock poisoning.
    #[error("prefix cache mutex poisoned")]
    LockPoisoned,
}

/// Record of one physical page in the runtime pool.
#[derive(Debug, Clone)]
pub(super) struct PhysicalPageRecord {
    pub(super) page_id: u32,
    pub(super) generation: u64,
    pub(super) allocated: bool,
    pub(super) pinned: bool,
    pub(super) in_flight: bool,
    pub(super) ref_count: usize,
    /// Range of initialized logical token slots `[start, end]`.
    pub(super) initialized_token_range: Option<(u32, u32)>,
    pub(super) tenant_id: String,
    pub(super) trust_domain: Option<String>,
    pub(super) scrubbed: bool,
    pub(super) last_accessed_tick: u64,
}

/// Node in the Radix Trie prefix index.
#[derive(Debug, Clone)]
pub(super) struct RadixNode {
    pub(super) tokens: Vec<u32>,
    pub(super) page_ids: Vec<u32>,
    pub(super) ref_count: usize,
    pub(super) children: BTreeMap<u32, RadixNode>,
    pub(super) owner_key: PrefixCacheKey,
    pub(super) last_accessed_tick: u64,
}

impl RadixNode {
    pub(super) fn new(
        tokens: Vec<u32>,
        page_ids: Vec<u32>,
        key: PrefixCacheKey,
        tick: u64,
    ) -> Self {
        Self {
            tokens,
            page_ids,
            ref_count: 1,
            children: BTreeMap::new(),
            owner_key: key,
            last_accessed_tick: tick,
        }
    }
}
