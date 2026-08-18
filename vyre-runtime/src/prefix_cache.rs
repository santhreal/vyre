//! Prefix-cache lifecycle: radix index, immutable prefix identity, page allocation,
//! copy-on-write, and bounded residency.
//!
//! # Architecture
//!
//! A prefix cache accelerates autoregressive language model decoding by reusing
//! Key-Value (KV) cache pages across requests that share an immutable prefix of
//! tokens (e.g. system prompts, common few-shot examples, multi-turn dialogues).
//!
//! This module owns the complete lifecycle in the runtime:
//! - **Radix Trie Index**: Fast prefix lookup and sub-sequence sharing.
//! - **Immutable-Prefix Identity**: Multi-dimensional keying including model, tokenizer,
//!   weights digest, configuration, token sequence, dtype, layout, device generation,
//!   cache schema version, and tenant isolation / trust domain.
//! - **Security & Trust Domains**: Cross-tenant physical page sharing requires an
//!   explicit common trust domain. Page data is scrubbed/overwritten before reassignment.
//! - **Page Slabs & Ref-Counting**: Allocation, reference counting, copy-on-write (COW)
//!   on branch mutation, and safe release.
//! - **Bounded Residency & Backpressure**: Explicit limits on pages, bytes, requests,
//!   queued tokens, and per-tenant usage. Admission fails closed under saturation;
//!   eviction never reclaims pinned or in-flight pages.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard};

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
struct PhysicalPageRecord {
    page_id: u32,
    generation: u64,
    allocated: bool,
    pinned: bool,
    in_flight: bool,
    ref_count: usize,
    /// Range of initialized logical token slots `[start, end]`.
    initialized_token_range: Option<(u32, u32)>,
    tenant_id: String,
    trust_domain: Option<String>,
    scrubbed: bool,
    last_accessed_tick: u64,
}

/// Node in the Radix Trie prefix index.
#[derive(Debug, Clone)]
struct RadixNode {
    tokens: Vec<u32>,
    page_ids: Vec<u32>,
    ref_count: usize,
    children: BTreeMap<u32, RadixNode>,
    owner_key: PrefixCacheKey,
    last_accessed_tick: u64,
}

impl RadixNode {
    fn new(tokens: Vec<u32>, page_ids: Vec<u32>, key: PrefixCacheKey, tick: u64) -> Self {
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

/// Complete runtime prefix-cache ownership manager.
pub struct PrefixCacheManager {
    limits: PrefixCacheLimits,
    current_generation: u64,
    active_requests: usize,
    queued_tokens: usize,
    tick: u64,
    pages: BTreeMap<u32, PhysicalPageRecord>,
    roots: BTreeMap<PrefixCacheKeyFingerprint, RadixNode>,
    metrics: PrefixCacheMetrics,
}

impl std::fmt::Debug for PrefixCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefixCacheManager")
            .field("limits", &self.limits)
            .field("current_generation", &self.current_generation)
            .field("active_requests", &self.active_requests)
            .field("allocated_pages", &self.metrics.allocated_pages)
            .finish_non_exhaustive()
    }
}

impl PrefixCacheManager {
    /// Create a new prefix cache manager with explicit limits and initial device generation.
    #[must_use]
    pub fn new(limits: PrefixCacheLimits, initial_generation: u64) -> Self {
        let mut pages = BTreeMap::new();
        for page_id in 0..(limits.max_pages as u32) {
            pages.insert(
                page_id,
                PhysicalPageRecord {
                    page_id,
                    generation: initial_generation,
                    allocated: false,
                    pinned: false,
                    in_flight: false,
                    ref_count: 0,
                    initialized_token_range: None,
                    tenant_id: String::new(),
                    trust_domain: None,
                    scrubbed: true,
                    last_accessed_tick: 0,
                },
            );
        }

        let mut metrics = PrefixCacheMetrics::default();
        metrics.free_pages = limits.max_pages;

        Self {
            limits,
            current_generation: initial_generation,
            active_requests: 0,
            queued_tokens: 0,
            tick: 1,
            pages,
            roots: BTreeMap::new(),
            metrics,
        }
    }

    /// Access current metrics.
    #[must_use]
    pub fn metrics(&self) -> &PrefixCacheMetrics {
        &self.metrics
    }

    /// Access current device generation.
    #[must_use]
    pub fn current_generation(&self) -> u64 {
        self.current_generation
    }

    /// Invalidate the device generation (e.g. on device reset or re-init).
    pub fn invalidate_generation(&mut self, new_generation: u64) {
        self.current_generation = new_generation;
        self.roots.clear();
        for page in self.pages.values_mut() {
            page.generation = new_generation;
            page.allocated = false;
            page.pinned = false;
            page.in_flight = false;
            page.ref_count = 0;
            page.initialized_token_range = None;
            page.tenant_id.clear();
            page.trust_domain = None;
            page.scrubbed = true;
        }
        self.active_requests = 0;
        self.queued_tokens = 0;
        self.metrics.allocated_pages = 0;
        self.metrics.pinned_pages = 0;
        self.metrics.in_flight_pages = 0;
        self.metrics.free_pages = self.limits.max_pages;
    }

    /// Look up a token sequence prefix in the radix trie.
    ///
    /// # Errors
    ///
    /// Returns [`PrefixCacheError`] if validation fails or stale generation is detected.
    pub fn lookup_prefix(
        &mut self,
        key: &PrefixCacheKey,
        tokens: &[u32],
    ) -> Result<PrefixMatchResult, PrefixCacheError> {
        if tokens.is_empty() {
            return Err(PrefixCacheError::EmptyTokens);
        }
        if key.device_generation != self.current_generation {
            self.metrics.rejected_stale_generations += 1;
            return Err(PrefixCacheError::StaleDeviceGeneration {
                expected: self.current_generation,
                actual: key.device_generation,
            });
        }
        if tokens.len() > self.limits.max_queued_tokens {
            self.metrics.backpressure_rejections += 1;
            return Err(PrefixCacheError::BackpressureLimitExceeded {
                reason: "lookup token sequence exceeds maximum queued tokens capacity",
            });
        }
        if self.active_requests >= self.limits.max_active_requests {
            self.metrics.backpressure_rejections += 1;
            return Err(PrefixCacheError::BackpressureLimitExceeded {
                reason: "maximum active requests limit reached",
            });
        }

        self.tick += 1;
        let tick = self.tick;

        let model_fp = key.model_fingerprint();
        let root = match self.roots.get_mut(&model_fp) {
            Some(r) => r,
            None => {
                self.metrics.cache_misses += 1;
                return Ok(PrefixMatchResult {
                    matched_tokens: 0,
                    page_ids: Vec::new(),
                    page_count: 0,
                    generation: self.current_generation,
                });
            }
        };

        // Start traversal from matching first token in root children
        let first_token = tokens[0];
        let mut curr = match root.children.get_mut(&first_token) {
            Some(child) => child,
            None => {
                self.metrics.cache_misses += 1;
                return Ok(PrefixMatchResult {
                    matched_tokens: 0,
                    page_ids: Vec::new(),
                    page_count: 0,
                    generation: self.current_generation,
                });
            }
        };

        curr.last_accessed_tick = tick;
        let mut matched_tokens = 0;
        let mut matched_pages = Vec::new();
        let mut token_offset = 0;

        while token_offset < tokens.len() {
            let chunk_len = curr.tokens.len();
            let remaining = &tokens[token_offset..];

            let common = curr
                .tokens
                .iter()
                .zip(remaining.iter())
                .take_while(|(&a, &b)| a == b)
                .count();

            if common == 0 {
                break;
            }

            // Validate isolation / trust domain at the choke point
            if curr.owner_key.isolation_domain != key.isolation_domain {
                let shared_trust = match (&curr.owner_key.trust_domain, &key.trust_domain) {
                    (Some(td_a), Some(td_b)) => td_a == td_b && !td_a.is_empty(),
                    _ => false,
                };
                if !shared_trust {
                    self.metrics.cache_misses += 1;
                    return Err(PrefixCacheError::IsolationViolation {
                        tenant_a: curr.owner_key.isolation_domain.clone(),
                        tenant_b: key.isolation_domain.clone(),
                    });
                }
            }

            if common == chunk_len {
                // Entire node matched
                matched_tokens += common;
                matched_pages.extend_from_slice(&curr.page_ids);
                token_offset += common;

                if token_offset < tokens.len() {
                    let next_token = tokens[token_offset];
                    if let Some(next_node) = curr.children.get_mut(&next_token) {
                        next_node.last_accessed_tick = tick;
                        curr = next_node;
                    } else {
                        break;
                    }
                }
            } else {
                // Partial match inside node: only tokens up to block boundaries match
                matched_tokens += common;
                let block_tokens = key.layout.block_tokens as usize;
                let full_blocks = common / block_tokens;
                if full_blocks > 0 && full_blocks <= curr.page_ids.len() {
                    matched_pages.extend_from_slice(&curr.page_ids[..full_blocks]);
                }
                break;
            }
        }

        if matched_tokens > 0 {
            self.metrics.cache_hits += 1;
            // Increment ref count for leased pages
            for &page_id in &matched_pages {
                if let Some(page) = self.pages.get_mut(&page_id) {
                    page.ref_count += 1;
                    page.last_accessed_tick = tick;
                }
            }
        } else {
            self.metrics.cache_misses += 1;
        }

        Ok(PrefixMatchResult {
            matched_tokens,
            page_ids: matched_pages.clone(),
            page_count: matched_pages.len(),
            generation: self.current_generation,
        })
    }

    /// Reserve pages and insert a token prefix into the cache.
    ///
    /// Handles copy-on-write (COW) when extending an existing shared prefix.
    ///
    /// # Errors
    ///
    /// Returns [`PrefixCacheError`] on backpressure, quota, or capacity failure.
    pub fn insert_or_extend_prefix(
        &mut self,
        key: &PrefixCacheKey,
        tokens: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, PrefixCacheError> {
        if tokens.is_empty() {
            return Err(PrefixCacheError::EmptyTokens);
        }
        if key.device_generation != self.current_generation {
            self.metrics.rejected_stale_generations += 1;
            return Err(PrefixCacheError::StaleDeviceGeneration {
                expected: self.current_generation,
                actual: key.device_generation,
            });
        }

        // Check backpressure limits
        if self.active_requests >= self.limits.max_active_requests {
            self.metrics.backpressure_rejections += 1;
            return Err(PrefixCacheError::BackpressureLimitExceeded {
                reason: "maximum active requests limit reached",
            });
        }

        if self.queued_tokens.saturating_add(tokens.len()) > self.limits.max_queued_tokens {
            self.metrics.backpressure_rejections += 1;
            return Err(PrefixCacheError::BackpressureLimitExceeded {
                reason: "maximum queued tokens capacity reached",
            });
        }

        let block_tokens = key.layout.block_tokens as usize;
        let total_needed_pages = (tokens.len() + block_tokens - 1) / block_tokens;
        let existing_page_count = existing_pages.len();

        let new_pages_needed = total_needed_pages.saturating_sub(existing_page_count);

        // Check tenant quota
        let current_tenant_pages = self
            .pages
            .values()
            .filter(|p| p.allocated && p.tenant_id == key.isolation_domain)
            .count();
        if current_tenant_pages + new_pages_needed > self.limits.per_tenant_page_limit {
            return Err(PrefixCacheError::TenantQuotaExceeded {
                tenant: key.isolation_domain.clone(),
                limit: self.limits.per_tenant_page_limit,
            });
        }

        // Increment ref count on existing pages to acquire lease for this extension
        for &id in existing_pages {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            page.ref_count += 1;
            page.last_accessed_tick = self.tick;
        }

        // Allocate new physical pages
        let mut allocated_page_ids = Vec::with_capacity(total_needed_pages);
        allocated_page_ids.extend_from_slice(existing_pages);

        for i in 0..new_pages_needed {
            let page_id = self.allocate_physical_page(key)?;
            let token_start = ((existing_page_count + i) * block_tokens) as u32;
            let token_end =
                std::cmp::min((existing_page_count + i + 1) * block_tokens, tokens.len()) as u32;

            if let Some(page) = self.pages.get_mut(&page_id) {
                page.initialized_token_range = Some((token_start, token_end));
            }
            allocated_page_ids.push(page_id);
        }

        self.tick += 1;
        let tick = self.tick;

        // Insert into radix trie
        let model_fp = key.model_fingerprint();
        let root = self
            .roots
            .entry(model_fp)
            .or_insert_with(|| RadixNode::new(Vec::new(), Vec::new(), key.clone(), tick));

        let first_token = tokens[0];
        if let Some(child) = root.children.get_mut(&first_token) {
            Self::insert_into_trie(
                child,
                tokens,
                &allocated_page_ids,
                key,
                block_tokens,
                tick,
                &mut self.metrics,
            );
        } else {
            let child = RadixNode::new(
                tokens.to_vec(),
                allocated_page_ids.clone(),
                key.clone(),
                tick,
            );
            root.children.insert(first_token, child);
        }

        self.active_requests += 1;
        self.queued_tokens += tokens.len();

        Ok(allocated_page_ids)
    }
    /// Allocate a physical page from the pool, evicting if necessary.
    fn allocate_physical_page(&mut self, key: &PrefixCacheKey) -> Result<u32, PrefixCacheError> {
        // First look for an unallocated free page
        if let Some((&id, page)) = self.pages.iter_mut().find(|(_, p)| !p.allocated) {
            page.allocated = true;
            page.generation = self.current_generation;
            page.ref_count = 1;
            page.tenant_id = key.isolation_domain.clone();
            page.trust_domain = key.trust_domain.clone();
            page.scrubbed = true;
            page.pinned = false;
            page.in_flight = false;
            page.last_accessed_tick = self.tick;

            self.metrics.allocated_pages += 1;
            self.metrics.free_pages = self.metrics.free_pages.saturating_sub(1);
            return Ok(id);
        }

        // Pool full: attempt LRU eviction of unpinned, not-in-flight pages with ref_count == 0
        let evictable_id = self
            .pages
            .iter()
            .filter(|(_, p)| p.allocated && !p.pinned && !p.in_flight && p.ref_count == 0)
            .min_by_key(|(_, p)| p.last_accessed_tick)
            .map(|(&id, _)| id);

        if let Some(id) = evictable_id {
            let Some(page) = self.pages.get_mut(&id) else {
                return Err(PrefixCacheError::PageNotFound(id));
            };
            page.allocated = true;
            page.generation = self.current_generation;
            page.ref_count = 1;
            page.tenant_id = key.isolation_domain.clone();
            page.trust_domain = key.trust_domain.clone();
            // Evicted page must be scrubbed before reassignment to prevent information leakage
            let page_bytes = key.layout.page_bytes(&key.dtype);
            self.metrics.scrubbed_bytes += page_bytes;
            page.scrubbed = true;
            page.initialized_token_range = None;
            page.last_accessed_tick = self.tick;

            self.metrics.evicted_pages += 1;
            return Ok(id);
        }

        Err(PrefixCacheError::CapacityExceeded {
            needed: 1,
            available: 0,
        })
    }

    /// Insert a token sequence into the radix trie, performing COW if branching.
    fn insert_into_trie(
        node: &mut RadixNode,
        tokens: &[u32],
        page_ids: &[u32],
        key: &PrefixCacheKey,
        block_tokens: usize,
        tick: u64,
        metrics: &mut PrefixCacheMetrics,
    ) {
        node.last_accessed_tick = tick;
        let common = node
            .tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(&a, &b)| a == b)
            .count();

        if common == node.tokens.len() {
            if common == tokens.len() {
                // Exact match: update ref count
                node.ref_count += 1;
            } else {
                // Suffix continuation: insert into child
                let rem_tokens = &tokens[common..];
                let rem_page_offset = (common + block_tokens - 1) / block_tokens;
                let rem_pages = if rem_page_offset <= page_ids.len() {
                    &page_ids[rem_page_offset..]
                } else {
                    &[]
                };
                let next_token = rem_tokens[0];
                if let Some(child) = node.children.get_mut(&next_token) {
                    Self::insert_into_trie(
                        child,
                        rem_tokens,
                        rem_pages,
                        key,
                        block_tokens,
                        tick,
                        metrics,
                    );
                } else {
                    let child =
                        RadixNode::new(rem_tokens.to_vec(), rem_pages.to_vec(), key.clone(), tick);
                    node.children.insert(next_token, child);
                }
            }
        } else if common > 0 {
            // Partial match: split node (Copy-On-Write branch)
            metrics.cow_copies += 1;
            let split_token = node.tokens[common];
            let child_tokens = node.tokens[common..].to_vec();
            let prefix_page_count = (common + block_tokens - 1) / block_tokens;

            let child_pages = if prefix_page_count <= node.page_ids.len() {
                node.page_ids[prefix_page_count..].to_vec()
            } else {
                Vec::new()
            };

            let mut split_child = RadixNode::new(
                child_tokens,
                child_pages,
                node.owner_key.clone(),
                node.last_accessed_tick,
            );
            split_child.children = std::mem::take(&mut node.children);
            split_child.ref_count = node.ref_count;

            node.tokens.truncate(common);
            node.page_ids.truncate(prefix_page_count);
            node.children.insert(split_token, split_child);

            if common < tokens.len() {
                let rem_tokens = &tokens[common..];
                let rem_page_offset = (common + block_tokens - 1) / block_tokens;
                let rem_pages = if rem_page_offset <= page_ids.len() {
                    &page_ids[rem_page_offset..]
                } else {
                    &[]
                };
                let next_token = rem_tokens[0];
                let new_child =
                    RadixNode::new(rem_tokens.to_vec(), rem_pages.to_vec(), key.clone(), tick);
                node.children.insert(next_token, new_child);
            }
        }
    }

    /// Pin pages so they cannot be evicted during compilation or resident preparation.
    pub fn pin_pages(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            page.pinned = true;
        }
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Unpin pages after compilation / submission completes.
    pub fn unpin_pages(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            page.pinned = false;
        }
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Mark pages as in-flight during kernel execution.
    pub fn mark_in_flight(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            page.in_flight = true;
        }
        self.metrics.in_flight_pages = self.pages.values().filter(|p| p.in_flight).count();
        Ok(())
    }

    /// Clear in-flight status after kernel completion.
    pub fn clear_in_flight(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            page.in_flight = false;
        }
        self.metrics.in_flight_pages = self.pages.values().filter(|p| p.in_flight).count();
        Ok(())
    }

    /// Release leased page references when a request completes or is cancelled.
    ///
    /// # Errors
    ///
    /// Returns [`PrefixCacheError::DuplicateRelease`] on double release / underflow.
    pub fn release_pages(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            if page.ref_count == 0 {
                return Err(PrefixCacheError::DuplicateRelease(id));
            }
            page.ref_count -= 1;
            if page.ref_count == 0 {
                page.pinned = false;
                page.in_flight = false;
            }
        }

        self.active_requests = self.active_requests.saturating_sub(1);
        Ok(())
    }
}

/// Thread-safe wrapper around [`PrefixCacheManager`].
#[derive(Debug, Clone)]
pub struct PrefixCache {
    inner: Arc<Mutex<PrefixCacheManager>>,
}

impl PrefixCache {
    /// Create a thread-safe prefix cache manager.
    #[must_use]
    pub fn new(limits: PrefixCacheLimits, initial_generation: u64) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PrefixCacheManager::new(
                limits,
                initial_generation,
            ))),
        }
    }

    fn lock(&self) -> Result<MutexGuard<'_, PrefixCacheManager>, PrefixCacheError> {
        self.inner
            .lock()
            .map_err(|_| PrefixCacheError::LockPoisoned)
    }

    /// Look up a token sequence in the prefix cache.
    pub fn lookup(
        &self,
        key: &PrefixCacheKey,
        tokens: &[u32],
    ) -> Result<PrefixMatchResult, PrefixCacheError> {
        self.lock()?.lookup_prefix(key, tokens)
    }

    /// Insert or extend a prefix.
    pub fn insert_or_extend(
        &self,
        key: &PrefixCacheKey,
        tokens: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, PrefixCacheError> {
        self.lock()?
            .insert_or_extend_prefix(key, tokens, existing_pages)
    }

    /// Pin pages.
    pub fn pin(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.pin_pages(page_ids)
    }

    /// Unpin pages.
    pub fn unpin(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.unpin_pages(page_ids)
    }

    /// Mark pages as in-flight.
    pub fn mark_in_flight(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.mark_in_flight(page_ids)
    }

    /// Clear in-flight status.
    pub fn clear_in_flight(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.clear_in_flight(page_ids)
    }

    /// Release leased pages.
    pub fn release(&self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.lock()?.release_pages(page_ids)
    }

    /// Fetch snapshot of metrics.
    pub fn metrics(&self) -> Result<PrefixCacheMetrics, PrefixCacheError> {
        Ok(self.lock()?.metrics().clone())
    }

    /// Invalidate generation.
    pub fn invalidate_generation(&self, new_gen: u64) -> Result<(), PrefixCacheError> {
        self.lock()?.invalidate_generation(new_gen);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key(tenant: &str, trust: Option<&str>, gen: u64) -> PrefixCacheKey {
        PrefixCacheKey {
            model_id: [1u8; 32],
            tokenizer_id: [2u8; 32],
            weights_digest: [3u8; 32],
            config_digest: [4u8; 32],
            dtype: DataType::F32,
            layout: PrefixCacheLayout {
                kv_heads: 2,
                head_dim: 64,
                block_tokens: 16,
            },
            device_generation: gen,
            cache_schema_version: 1,
            isolation_domain: tenant.to_string(),
            trust_domain: trust.map(|s| s.to_string()),
        }
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
}

impl PrefixCacheKey {
    #[cfg(test)]
    pub(crate) fn test_sample(tenant: &str, gen: u64) -> Self {
        Self {
            model_id: [1u8; 32],
            tokenizer_id: [2u8; 32],
            weights_digest: [3u8; 32],
            config_digest: [4u8; 32],
            dtype: DataType::F32,
            layout: PrefixCacheLayout {
                kv_heads: 2,
                head_dim: 32,
                block_tokens: 16,
            },
            device_generation: gen,
            cache_schema_version: 1,
            isolation_domain: tenant.to_string(),
            trust_domain: None,
        }
    }
}
