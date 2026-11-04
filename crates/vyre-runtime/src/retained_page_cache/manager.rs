use std::collections::BTreeMap;

use super::contract::{
    PhysicalPageRecord, RadixNode, RetainedMatchResult, RetainedPageCacheError,
    RetainedPageCacheKey, RetainedPageCacheKeyFingerprint, RetainedPageLimits, RetainedPageMetrics,
};

/// Complete runtime retained-page-cache ownership manager.
pub struct RetainedPageCacheManager {
    limits: RetainedPageLimits,
    current_generation: u64,
    active_requests: usize,
    queued_units: usize,
    tick: u64,
    pages: BTreeMap<u32, PhysicalPageRecord>,
    roots: BTreeMap<RetainedPageCacheKeyFingerprint, RadixNode>,
    metrics: RetainedPageMetrics,
}

impl std::fmt::Debug for RetainedPageCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedPageCacheManager")
            .field("limits", &self.limits)
            .field("current_generation", &self.current_generation)
            .field("active_requests", &self.active_requests)
            .field("allocated_pages", &self.metrics.allocated_pages)
            .finish_non_exhaustive()
    }
}

impl RetainedPageCacheManager {
    /// Create a new retained page cache manager with explicit limits and initial device generation.
    #[must_use]
    pub fn new(limits: RetainedPageLimits, initial_generation: u64) -> Self {
        let mut pages = BTreeMap::new();
        for page_id in 0..(limits.max_pages as u32) {
            pages.insert(
                page_id,
                PhysicalPageRecord {
                    generation: initial_generation,
                    allocated: false,
                    pinned: false,
                    in_flight: false,
                    ref_count: 0,
                    initialized_slot_range: None,
                    tenant_id: String::new(),
                    trust_domain: None,
                    scrubbed: true,
                    last_accessed_tick: 0,
                },
            );
        }

        let mut metrics = RetainedPageMetrics::default();
        metrics.free_pages = limits.max_pages;

        Self {
            limits,
            current_generation: initial_generation,
            active_requests: 0,
            queued_units: 0,
            tick: 1,
            pages,
            roots: BTreeMap::new(),
            metrics,
        }
    }

    /// Access current metrics.
    #[must_use]
    pub fn metrics(&self) -> &RetainedPageMetrics {
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
            page.initialized_slot_range = None;
            page.tenant_id.clear();
            page.trust_domain = None;
            page.scrubbed = true;
        }
        self.active_requests = 0;
        self.queued_units = 0;
        self.metrics.allocated_pages = 0;
        self.metrics.pinned_pages = 0;
        self.metrics.in_flight_pages = 0;
        self.metrics.free_pages = self.limits.max_pages;
    }

    /// Look up a sequence prefix in the radix trie.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError`] if validation fails or stale generation is detected.
    pub fn lookup_prefix(
        &mut self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
    ) -> Result<RetainedMatchResult, RetainedPageCacheError> {
        if sequence.is_empty() {
            return Err(RetainedPageCacheError::EmptySequence);
        }
        if key.device_generation != self.current_generation {
            self.metrics.rejected_stale_generations += 1;
            return Err(RetainedPageCacheError::StaleDeviceGeneration {
                expected: self.current_generation,
                actual: key.device_generation,
            });
        }
        if sequence.len() > self.limits.max_queued_units {
            self.metrics.backpressure_rejections += 1;
            return Err(RetainedPageCacheError::BackpressureLimitExceeded {
                reason: "lookup sequence exceeds maximum queued units capacity",
            });
        }
        if self.active_requests >= self.limits.max_active_requests {
            self.metrics.backpressure_rejections += 1;
            return Err(RetainedPageCacheError::BackpressureLimitExceeded {
                reason: "maximum active requests limit reached",
            });
        }

        self.tick += 1;
        let tick = self.tick;

        let struct_fp = key.structural_fingerprint();
        let root = match self.roots.get_mut(&struct_fp) {
            Some(r) => r,
            None => {
                self.metrics.cache_misses += 1;
                return Ok(RetainedMatchResult {
                    matched_units: 0,
                    page_ids: Vec::new(),
                    page_count: 0,
                    generation: self.current_generation,
                });
            }
        };

        let first_unit = sequence[0];
        let mut curr = match root.children.get_mut(&first_unit) {
            Some(child) => child,
            None => {
                self.metrics.cache_misses += 1;
                return Ok(RetainedMatchResult {
                    matched_units: 0,
                    page_ids: Vec::new(),
                    page_count: 0,
                    generation: self.current_generation,
                });
            }
        };

        curr.last_accessed_tick = tick;
        let mut matched_units = 0;
        let mut matched_pages = Vec::new();
        let mut unit_offset = 0;

        while unit_offset < sequence.len() {
            let chunk_len = curr.sequence.len();
            let remaining = &sequence[unit_offset..];

            let common = curr
                .sequence
                .iter()
                .zip(remaining.iter())
                .take_while(|(&a, &b)| a == b)
                .count();

            if common == 0 {
                break;
            }

            if curr.owner_key.isolation_domain != key.isolation_domain {
                let shared_trust = match (&curr.owner_key.trust_domain, &key.trust_domain) {
                    (Some(td_a), Some(td_b)) => td_a == td_b && !td_a.is_empty(),
                    _ => false,
                };
                if !shared_trust {
                    self.metrics.cache_misses += 1;
                    return Err(RetainedPageCacheError::IsolationViolation {
                        tenant_a: curr.owner_key.isolation_domain.clone(),
                        tenant_b: key.isolation_domain.clone(),
                    });
                }
            }

            if common == chunk_len {
                matched_units += common;
                matched_pages.extend_from_slice(&curr.page_ids);
                unit_offset += common;

                if unit_offset < sequence.len() {
                    let next_unit = sequence[unit_offset];
                    if let Some(next_node) = curr.children.get_mut(&next_unit) {
                        next_node.last_accessed_tick = tick;
                        curr = next_node;
                    } else {
                        break;
                    }
                }
            } else {
                matched_units += common;
                let units_per_page = key.layout.units_per_page as usize;
                let full_pages = common / units_per_page;
                if full_pages > 0 && full_pages <= curr.page_ids.len() {
                    matched_pages.extend_from_slice(&curr.page_ids[..full_pages]);
                }
                break;
            }
        }

        if matched_units > 0 {
            self.metrics.cache_hits += 1;
            for &page_id in &matched_pages {
                if let Some(page) = self.pages.get_mut(&page_id) {
                    page.ref_count += 1;
                    page.last_accessed_tick = tick;
                }
            }
        } else {
            self.metrics.cache_misses += 1;
        }

        Ok(RetainedMatchResult {
            matched_units,
            page_ids: matched_pages.clone(),
            page_count: matched_pages.len(),
            generation: self.current_generation,
        })
    }

    /// Reserve pages and insert a sequence prefix into the cache.
    ///
    /// Handles copy-on-write (COW) when extending an existing shared prefix.
    ///
    /// # Errors
    ///
    /// Returns [`RetainedPageCacheError`] on backpressure, quota, or capacity failure.
    pub fn insert_or_extend_prefix(
        &mut self,
        key: &RetainedPageCacheKey,
        sequence: &[u32],
        existing_pages: &[u32],
    ) -> Result<Vec<u32>, RetainedPageCacheError> {
        if sequence.is_empty() {
            return Err(RetainedPageCacheError::EmptySequence);
        }
        if key.device_generation != self.current_generation {
            self.metrics.rejected_stale_generations += 1;
            return Err(RetainedPageCacheError::StaleDeviceGeneration {
                expected: self.current_generation,
                actual: key.device_generation,
            });
        }

        if self.active_requests >= self.limits.max_active_requests {
            self.metrics.backpressure_rejections += 1;
            return Err(RetainedPageCacheError::BackpressureLimitExceeded {
                reason: "maximum active requests limit reached",
            });
        }

        if self.queued_units.saturating_add(sequence.len()) > self.limits.max_queued_units {
            self.metrics.backpressure_rejections += 1;
            return Err(RetainedPageCacheError::BackpressureLimitExceeded {
                reason: "maximum queued units capacity reached",
            });
        }

        let units_per_page = key.layout.units_per_page as usize;
        let total_needed_pages = (sequence.len() + units_per_page - 1) / units_per_page;
        let existing_page_count = existing_pages.len();

        let new_pages_needed = total_needed_pages.saturating_sub(existing_page_count);

        let current_tenant_pages = self
            .pages
            .values()
            .filter(|p| p.allocated && p.tenant_id == key.isolation_domain)
            .count();
        if current_tenant_pages + new_pages_needed > self.limits.per_tenant_page_limit {
            return Err(RetainedPageCacheError::TenantQuotaExceeded {
                tenant: key.isolation_domain.clone(),
                limit: self.limits.per_tenant_page_limit,
            });
        }

        for &id in existing_pages {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(RetainedPageCacheError::PageNotFound(id))?;
            page.ref_count += 1;
            page.last_accessed_tick = self.tick;
        }

        let mut allocated_page_ids = Vec::with_capacity(total_needed_pages);
        allocated_page_ids.extend_from_slice(existing_pages);

        for i in 0..new_pages_needed {
            let page_id = self.allocate_physical_page(key)?;
            let slot_start = ((existing_page_count + i) * units_per_page) as u32;
            let slot_end = std::cmp::min(
                (existing_page_count + i + 1) * units_per_page,
                sequence.len(),
            ) as u32;

            if let Some(page) = self.pages.get_mut(&page_id) {
                page.initialized_slot_range = Some((slot_start, slot_end));
            }
            allocated_page_ids.push(page_id);
        }

        self.tick += 1;
        let tick = self.tick;

        let struct_fp = key.structural_fingerprint();
        let root = self
            .roots
            .entry(struct_fp)
            .or_insert_with(|| RadixNode::new(Vec::new(), Vec::new(), key.clone(), tick));

        let first_unit = sequence[0];
        if let Some(child) = root.children.get_mut(&first_unit) {
            Self::insert_into_trie(
                child,
                sequence,
                &allocated_page_ids,
                key,
                units_per_page,
                tick,
                &mut self.metrics,
            );
        } else {
            let child = RadixNode::new(
                sequence.to_vec(),
                allocated_page_ids.clone(),
                key.clone(),
                tick,
            );
            root.children.insert(first_unit, child);
        }

        self.active_requests += 1;
        self.queued_units += sequence.len();

        Ok(allocated_page_ids)
    }

    /// Allocate a physical page from the pool, evicting if necessary.
    fn allocate_physical_page(
        &mut self,
        key: &RetainedPageCacheKey,
    ) -> Result<u32, RetainedPageCacheError> {
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

        let evictable_id = self
            .pages
            .iter()
            .filter(|(_, p)| p.allocated && !p.pinned && !p.in_flight && p.ref_count == 0)
            .min_by_key(|(_, p)| p.last_accessed_tick)
            .map(|(&id, _)| id);

        if let Some(id) = evictable_id {
            let Some(page) = self.pages.get_mut(&id) else {
                return Err(RetainedPageCacheError::PageNotFound(id));
            };
            page.allocated = true;
            page.generation = self.current_generation;
            page.ref_count = 1;
            page.tenant_id = key.isolation_domain.clone();
            page.trust_domain = key.trust_domain.clone();
            let page_bytes = key.layout.page_bytes(&key.dtype);
            self.metrics.scrubbed_bytes += page_bytes;
            page.scrubbed = true;
            page.initialized_slot_range = None;
            page.last_accessed_tick = self.tick;

            self.metrics.evicted_pages += 1;
            return Ok(id);
        }

        Err(RetainedPageCacheError::CapacityExceeded {
            needed: 1,
            available: 0,
        })
    }

    /// Insert a sequence into the radix trie, performing COW if branching.
    fn insert_into_trie(
        node: &mut RadixNode,
        sequence: &[u32],
        page_ids: &[u32],
        key: &RetainedPageCacheKey,
        units_per_page: usize,
        tick: u64,
        metrics: &mut RetainedPageMetrics,
    ) {
        node.last_accessed_tick = tick;
        let common = node
            .sequence
            .iter()
            .zip(sequence.iter())
            .take_while(|(&a, &b)| a == b)
            .count();

        if common == node.sequence.len() {
            if common == sequence.len() {
                node.ref_count += 1;
            } else {
                let rem_units = &sequence[common..];
                let rem_page_offset = (common + units_per_page - 1) / units_per_page;
                let rem_pages = if rem_page_offset <= page_ids.len() {
                    &page_ids[rem_page_offset..]
                } else {
                    &[]
                };
                let next_unit = rem_units[0];
                if let Some(child) = node.children.get_mut(&next_unit) {
                    Self::insert_into_trie(
                        child,
                        rem_units,
                        rem_pages,
                        key,
                        units_per_page,
                        tick,
                        metrics,
                    );
                } else {
                    let child =
                        RadixNode::new(rem_units.to_vec(), rem_pages.to_vec(), key.clone(), tick);
                    node.children.insert(next_unit, child);
                }
            }
        } else if common > 0 {
            metrics.cow_copies += 1;
            let split_unit = node.sequence[common];
            let child_units = node.sequence[common..].to_vec();
            let prefix_page_count = (common + units_per_page - 1) / units_per_page;

            let child_pages = if prefix_page_count <= node.page_ids.len() {
                node.page_ids[prefix_page_count..].to_vec()
            } else {
                Vec::new()
            };

            let mut split_child = RadixNode::new(
                child_units,
                child_pages,
                node.owner_key.clone(),
                node.last_accessed_tick,
            );
            split_child.children = std::mem::take(&mut node.children);
            split_child.ref_count = node.ref_count;

            node.sequence.truncate(common);
            node.page_ids.truncate(prefix_page_count);
            node.children.insert(split_unit, split_child);

            if common < sequence.len() {
                let rem_units = &sequence[common..];
                let rem_page_offset = (common + units_per_page - 1) / units_per_page;
                let rem_pages = if rem_page_offset <= page_ids.len() {
                    &page_ids[rem_page_offset..]
                } else {
                    &[]
                };
                let next_unit = rem_units[0];
                let new_child =
                    RadixNode::new(rem_units.to_vec(), rem_pages.to_vec(), key.clone(), tick);
                node.children.insert(next_unit, new_child);
            }
        }
    }

    fn set_pages_flag<F>(
        &mut self,
        page_ids: &[u32],
        mut flag_setter: F,
    ) -> Result<(), RetainedPageCacheError>
    where
        F: FnMut(&mut PhysicalPageRecord),
    {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(RetainedPageCacheError::PageNotFound(id))?;
            flag_setter(page);
        }
        Ok(())
    }

    /// Pin pages so they cannot be evicted during compilation or resident preparation.
    pub fn pin_pages(&mut self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.set_pages_flag(page_ids, |p| p.pinned = true)?;
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Unpin pages after compilation / submission completes.
    pub fn unpin_pages(&mut self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.set_pages_flag(page_ids, |p| p.pinned = false)?;
        self.metrics.pinned_pages = self.pages.values().filter(|p| p.pinned).count();
        Ok(())
    }

    /// Mark pages as in-flight during kernel execution.
    pub fn mark_in_flight(&mut self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.set_pages_flag(page_ids, |p| p.in_flight = true)?;
        self.metrics.in_flight_pages = self.pages.values().filter(|p| p.in_flight).count();
        Ok(())
    }

    /// Clear in-flight status after kernel completion.
    pub fn clear_in_flight(&mut self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        self.set_pages_flag(page_ids, |p| p.in_flight = false)?;
        self.metrics.in_flight_pages = self.pages.values().filter(|p| p.in_flight).count();
        Ok(())
    }

    /// Release leased page references when a request completes or is cancelled.
    pub fn release_pages(&mut self, page_ids: &[u32]) -> Result<(), RetainedPageCacheError> {
        for &id in page_ids {
            let page = self
                .pages
                .get_mut(&id)
                .ok_or(RetainedPageCacheError::PageNotFound(id))?;
            if page.ref_count == 0 {
                return Err(RetainedPageCacheError::DuplicateRelease(id));
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
