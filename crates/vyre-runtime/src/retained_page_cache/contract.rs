//! Values the retained page cache exchanges with a caller: key, layout, match result,
//! limits, counters, and failure.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_foundation::failure_domain::TypedRecoveryError;
use vyre_foundation::ir::DataType;

/// Layout and geometry for retained cache pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetainedPageLayout {
    /// Number of feature channels or parallel lanes.
    pub feature_channels: u32,
    /// Dimension per unit/feature.
    pub unit_dimension: u32,
    /// Number of sequence units stored in one physical page.
    pub units_per_page: u32,
}

impl RetainedPageLayout {
    /// Compute byte size of a single physical page.
    #[must_use]
    pub fn page_bytes(&self, dtype: &DataType) -> u64 {
        let elem_bytes = match dtype {
            DataType::F32 | DataType::U32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => 2,
            DataType::U8 | DataType::I8 | DataType::Bool => 1,
            _ => 4,
        };
        2 * (self.feature_channels as u64)
            * (self.units_per_page as u64)
            * (self.unit_dimension as u64)
            * elem_bytes
    }
}

/// Multi-dimensional immutable retained page cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RetainedPageCacheKey {
    /// Content identifier hash.
    pub content_id: [u8; 32],
    /// Secondary schema or structural hash.
    pub schema_id: [u8; 32],
    /// Exact payload content digest.
    pub payload_digest: [u8; 32],
    /// Configuration / hyperparameter digest.
    pub config_digest: [u8; 32],
    /// Element data type.
    pub dtype: DataType,
    /// Channel and block geometry layout.
    pub layout: RetainedPageLayout,
    /// Device allocation generation.
    pub device_generation: u64,
    /// Cache schema version.
    pub cache_schema_version: u32,
    /// Tenant or request isolation domain.
    pub isolation_domain: String,
    /// Optional explicit trust domain required for cross-tenant sharing.
    pub trust_domain: Option<String>,
}

/// Canonical stable fingerprint of a cache key for total ordering and lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RetainedPageCacheKeyFingerprint(pub [u8; 32]);

impl RetainedPageCacheKey {
    /// Compute structural fingerprint for Radix Trie indexing.
    #[must_use]
    pub fn structural_fingerprint(&self) -> RetainedPageCacheKeyFingerprint {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.content_id);
        hasher.update(&self.schema_id);
        hasher.update(&self.payload_digest);
        hasher.update(&self.config_digest);
        hasher.update(format!("{:?}", self.dtype).as_bytes());
        hasher.update(&self.layout.feature_channels.to_le_bytes());
        hasher.update(&self.layout.unit_dimension.to_le_bytes());
        hasher.update(&self.layout.units_per_page.to_le_bytes());
        hasher.update(&self.device_generation.to_le_bytes());
        hasher.update(&self.cache_schema_version.to_le_bytes());
        RetainedPageCacheKeyFingerprint(*hasher.finalize().as_bytes())
    }

    /// Compute the deterministic canonical fingerprint for this key including tenant isolation.
    #[must_use]
    pub fn fingerprint(&self) -> RetainedPageCacheKeyFingerprint {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.structural_fingerprint().0);
        hasher.update(self.isolation_domain.as_bytes());
        if let Some(td) = &self.trust_domain {
            hasher.update(&[1u8]);
            hasher.update(td.as_bytes());
        } else {
            hasher.update(&[0u8]);
        }
        RetainedPageCacheKeyFingerprint(*hasher.finalize().as_bytes())
    }
}

/// Result of matching a sequence in the retained page cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetainedMatchResult {
    /// Number of units matched from the prefix.
    pub matched_units: usize,
    /// Physical page IDs backing the matched prefix.
    pub page_ids: Vec<u32>,
    /// Number of pages backing the matched prefix.
    pub page_count: usize,
    /// Device generation of the matched pages.
    pub generation: u64,
}

/// Limits and backpressure thresholds for cache and queue residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetainedPageLimits {
    /// Maximum physical pages allowed in the pool.
    pub max_pages: usize,
    /// Maximum resident bytes allowed.
    pub max_bytes: u64,
    /// Maximum concurrent active requests leasing cache pages.
    pub max_active_requests: usize,
    /// Maximum queued units across active requests.
    pub max_queued_units: usize,
    /// Maximum pages allowed for a single tenant isolation domain.
    pub per_tenant_page_limit: usize,
}

impl Default for RetainedPageLimits {
    fn default() -> Self {
        Self {
            max_pages: 1024,
            max_bytes: 1024 * 1024 * 1024, // 1 GB
            max_active_requests: 64,
            max_queued_units: 131_072,
            per_tenant_page_limit: 512,
        }
    }
}

/// Telemetry metrics for retained page cache lifecycle and residency.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RetainedPageMetrics {
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

/// Errors occurring during retained page cache operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RetainedPageCacheError {
    /// Sequence is empty.
    #[error("sequence cannot be empty for retained page cache operations")]
    EmptySequence,
    /// Memory pool capacity exceeded.
    #[error(
        "retained page cache pool exhausted: needed {needed} pages, but only {available} available"
    )]
    CapacityExceeded {
        /// Pages requested.
        needed: usize,
        /// Available unpinned/free pages.
        available: usize,
    },
    /// Backpressure limit exceeded on active requests or queued units.
    #[error("retained page cache admission rejected: {reason}. Fix: apply backpressure or wait for requests to complete")]
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
    /// The pool is not accepting operations. A panic left it terminal, or a
    /// generation rebuild is in progress, and the documented recovery is
    /// `RetainedPageCache::invalidate_generation`.
    #[error("{0}")]
    Recovery(#[from] TypedRecoveryError),
}

/// Record of one physical page in the runtime pool.
#[derive(Debug, Clone)]
pub(super) struct PhysicalPageRecord {
    pub(super) generation: u64,
    pub(super) allocated: bool,
    pub(super) pinned: bool,
    pub(super) in_flight: bool,
    pub(super) ref_count: usize,
    /// Range of initialized logical slots `[start, end]`.
    pub(super) initialized_slot_range: Option<(u32, u32)>,
    pub(super) tenant_id: String,
    pub(super) trust_domain: Option<String>,
    pub(super) scrubbed: bool,
    pub(super) last_accessed_tick: u64,
}

/// Node in the Radix Trie sequence index.
#[derive(Debug, Clone)]
pub(super) struct RadixNode {
    pub(super) sequence: Vec<u32>,
    pub(super) page_ids: Vec<u32>,
    pub(super) ref_count: usize,
    pub(super) children: BTreeMap<u32, RadixNode>,
    pub(super) owner_key: RetainedPageCacheKey,
    pub(super) last_accessed_tick: u64,
}

impl RadixNode {
    pub(super) fn new(
        sequence: Vec<u32>,
        page_ids: Vec<u32>,
        key: RetainedPageCacheKey,
        tick: u64,
    ) -> Self {
        Self {
            sequence,
            page_ids,
            ref_count: 1,
            children: BTreeMap::new(),
            owner_key: key,
            last_accessed_tick: tick,
        }
    }
}
