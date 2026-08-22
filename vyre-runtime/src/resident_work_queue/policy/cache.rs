use super::super::lru_tick_cache::LruTickCache;
use super::{
    ResidentLaunchCacheStats, ResidentLaunchPolicy, ResidentLaunchRecommendation,
    ResidentLaunchRequest,
};
use std::cell::RefCell;

const LAUNCH_RECOMMENDATION_CACHE_CAP: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct LaunchRecommendationCacheKey {
    pub(super) policy: ResidentLaunchPolicy,
    pub(super) request: ResidentLaunchRequest,
}

pub(super) struct LaunchRecommendationCache {
    entries: LruTickCache<LaunchRecommendationCacheKey, ResidentLaunchRecommendation>,
    pub(super) hits: u64,
    pub(super) misses: u64,
}

impl LaunchRecommendationCache {
    pub(super) fn get(
        &mut self,
        key: &LaunchRecommendationCacheKey,
    ) -> Option<ResidentLaunchRecommendation> {
        let Some(recommendation) = self.entries.get(key).copied() else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.hits = self.hits.saturating_add(1);
        Some(recommendation)
    }

    pub(super) fn insert(
        &mut self,
        key: LaunchRecommendationCacheKey,
        value: ResidentLaunchRecommendation,
    ) {
        self.entries.insert(key, value);
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn stats(&self) -> ResidentLaunchCacheStats {
        ResidentLaunchCacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

impl Default for LaunchRecommendationCache {
    fn default() -> Self {
        Self {
            entries: LruTickCache::with_capacity(LAUNCH_RECOMMENDATION_CACHE_CAP),
            hits: 0,
            misses: 0,
        }
    }
}

thread_local! {
    pub(super) static LAUNCH_RECOMMENDATION_CACHE: RefCell<LaunchRecommendationCache> =
        RefCell::new(LaunchRecommendationCache::default());
}
