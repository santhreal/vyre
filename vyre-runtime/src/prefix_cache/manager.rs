use std::collections::BTreeMap;

use super::types::{
    PhysicalPageRecord, PrefixCacheError, PrefixCacheKey, PrefixCacheKeyFingerprint,
    PrefixCacheLimits, PrefixCacheMetrics, PrefixMatchResult, RadixNode,
};

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

    fn set_pages_flag<F>(
        &mut self,
        page_ids: &[u32],
        mut flag_setter: F,
    ) -> Result<(), PrefixCacheError>
    where
        F: FnMut(&mut PhysicalPageRecord),
    {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(PrefixCacheError::PageNotFound(id))?;
            flag_setter(page);
        }
        Ok(())
    }

    /// Pin pages so they cannot be evicted during compilation or resident preparation.
    pub fn pin_pages(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.set_pages_flag(page_ids, |p| p.pinned = true)?;
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Unpin pages after compilation / submission completes.
    pub fn unpin_pages(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.set_pages_flag(page_ids, |p| p.pinned = false)?;
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Mark pages as in-flight during kernel execution.
    pub fn mark_in_flight(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.set_pages_flag(page_ids, |p| p.in_flight = true)?;
        self.metrics.in_flight_pages = self.pages.values().filter(|p| p.in_flight).count();
        Ok(())
    }

    /// Clear in-flight status after kernel completion.
    pub fn clear_in_flight(&mut self, page_ids: &[u32]) -> Result<(), PrefixCacheError> {
        self.set_pages_flag(page_ids, |p| p.in_flight = false)?;
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
