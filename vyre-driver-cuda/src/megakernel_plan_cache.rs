//! Bounded CUDA megakernel plan cache.
//!
//! The cache stores topology decisions keyed by stable graph layout,
//! analysis family, CUDA device feature signature, and coarse runtime-pressure
//! buckets. The first three fields are the architectural identity of a plan;
//! pressure buckets prevent a sparse first query from poisoning dense later
//! queries over the same resident graph.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

use crate::backend::ordering::sort_unstable_by_key_if_needed;
use crate::backend::staging_reserve::reserve_vec;
use crate::megakernel_scheduler::{
    select_cuda_megakernel_topology, select_cuda_megakernel_topology_stable,
    CudaMegakernelScheduleSample,
};
use vyre_driver::megakernel_execution::{
    megakernel_base_required_bytes, plan_megakernel_memory_budget, FrontierExecutionSample,
    FrontierTopology, FrontierTopologyDecision, MegakernelByteLayout, MegakernelExecutionPlan,
    MegakernelGraphShape, MegakernelMemoryBudget, MegakernelMemoryError,
};

pub use crate::megakernel_plan_cache_records::*;

/// Bounded LRU cache for CUDA megakernel topology plans.
#[derive(Debug)]
pub struct CudaMegakernelPlanCache {
    pub(crate) entries: FxHashMap<CudaMegakernelPlanCacheKey, CudaMegakernelPlanCacheEntry>,
    latest_by_identity:
        FxHashMap<CudaMegakernelPlanIdentityKey, (u64, FrontierTopology)>,
    eviction_queue: BinaryHeap<Reverse<(u64, CudaMegakernelPlanCacheKey)>>,
    max_entries: usize,
    pub(crate) serial: u64,
    pub(crate) hits: u64,
    misses: u64,
    evictions: u64,
}

impl Default for CudaMegakernelPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaMegakernelPlanCache {
    /// Create a cache with the default production entry bound.
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_entries(DEFAULT_MAX_MEGAKERNEL_PLANS)
    }

    /// Create a cache with an explicit entry bound.
    #[must_use]
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            latest_by_identity: FxHashMap::default(),
            eviction_queue: BinaryHeap::new(),
            max_entries,
            serial: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Return a cached plan or insert a newly selected topology decision.
    pub fn get_or_insert_with(
        &mut self,
        key: CudaMegakernelPlanCacheKey,
        build: impl FnOnce() -> FrontierTopologyDecision,
    ) -> Result<CudaMegakernelCachedPlan, MegakernelMemoryError> {
        let serial = self.advance_serial()?;
        if let Some(entry) = self.entries.get_mut(&key) {
            increment_plan_cache_counter(&mut self.hits, "megakernel plan-cache hit counter");
            entry.last_seen = serial;
            let plan = entry.plan;
            self.eviction_queue.push(Reverse((serial, key)));
            self.update_latest_identity(key.identity(), serial, plan.topology);
            return Ok(plan);
        }
        increment_plan_cache_counter(&mut self.misses, "megakernel plan-cache miss counter");
        if self.max_entries == 0 {
            let decision = build();
            return Ok(CudaMegakernelCachedPlan {
                topology: decision.topology,
                decision,
            });
        }
        self.evict_until_below_limit()?;
        let decision = build();
        let plan = CudaMegakernelCachedPlan {
            topology: decision.topology,
            decision,
        };
        self.entries.insert(
            key,
            CudaMegakernelPlanCacheEntry {
                plan,
                last_seen: serial,
            },
        );
        self.eviction_queue.push(Reverse((serial, key)));
        self.update_latest_identity(key.identity(), serial, plan.topology);
        Ok(plan)
    }

    /// Return a cached topology plan or select and cache one from the current
    /// CUDA telemetry sample.
    ///
    /// This is the hot-path convenience API: callers provide stable graph,
    /// analysis, device, and telemetry inputs, while the cache owns the
    /// pressure bucketing needed to avoid stale sparse/dense decisions.
    ///
    /// # Errors
    ///
    /// Returns [`MegakernelMemoryError::InvalidSample`] when an observed fact
    /// lies outside its declared domain, so a measurement defect is never
    /// bucketed as a cached decision.
    pub fn get_or_select_topology(
        &mut self,
        graph_layout_hash: u64,
        analysis_kind: CudaMegakernelAnalysisKind,
        device: CudaMegakernelDeviceKey,
        sample: CudaMegakernelScheduleSample,
        graph: MegakernelGraphShape,
        memory: MegakernelMemoryBudget,
        launch_overhead_ns: f64,
        fusion_pressure: f64,
    ) -> Result<CudaMegakernelCachedPlan, MegakernelMemoryError> {
        let frontier_sample = FrontierExecutionSample {
            dispatch_cost_ns: sample.dispatch_cost_ns,
            frontier_density: sample.frontier_density,
            readback_bytes: sample.readback_bytes,
        };
        if let Some(field) = frontier_sample.unrepresentable_fact() {
            return Err(MegakernelMemoryError::InvalidSample { field });
        }
        let effective_fusion_pressure = if device.supports_grid_sync {
            fusion_pressure
        } else {
            0.0
        };
        let key = CudaMegakernelPlanCacheKey::new(
            graph_layout_hash,
            analysis_kind,
            device,
            sample.frontier_density,
            pressure_bps(memory.required_bytes, memory.budget_bytes),
            sample.readback_bytes,
            launch_pressure_bps(sample.dispatch_cost_ns, launch_overhead_ns),
            effective_fusion_pressure,
        );
        let previous_topology =
            self.latest_topology_for_identity(graph_layout_hash, analysis_kind, device);
        self.get_or_insert_with(key, || {
            if let Some(previous_topology) = previous_topology {
                select_cuda_megakernel_topology_stable(
                    sample,
                    graph,
                    memory,
                    launch_overhead_ns,
                    effective_fusion_pressure,
                    previous_topology,
                )
            } else {
                select_cuda_megakernel_topology(
                    sample,
                    graph,
                    memory,
                    launch_overhead_ns,
                    effective_fusion_pressure,
                )
            }
        })
    }

    /// Return a cache-backed, memory-validated CUDA megakernel execution plan.
    ///
    /// The cache key uses sparse-plan memory pressure because sparse is the
    /// lower-bound resident footprint shared by every topology. That probe runs
    /// against an unbounded cap so it measures the footprint instead of failing
    /// early on a budget the caller may still satisfy under another topology. A
    /// cache hit reuses the prior topology decision, then this method validates
    /// the exact current dense/fused/sparse byte budget before returning a
    /// launchable plan. If the cached non-sparse topology no longer fits, the
    /// method downgrades to sparse only after proving the sparse plan fits the
    /// real cap.
    pub fn get_or_plan_execution(
        &mut self,
        graph_layout_hash: u64,
        analysis_kind: CudaMegakernelAnalysisKind,
        device: CudaMegakernelDeviceKey,
        sample: CudaMegakernelScheduleSample,
        graph: MegakernelGraphShape,
        bytes: MegakernelByteLayout,
        launch_overhead_ns: f64,
        fusion_pressure: f64,
    ) -> Result<MegakernelExecutionPlan, MegakernelMemoryError> {
        let base_required_bytes = megakernel_base_required_bytes(graph, bytes)?;
        let cached = self.get_or_select_topology(
            graph_layout_hash,
            analysis_kind,
            device,
            sample,
            graph,
            MegakernelMemoryBudget {
                required_bytes: base_required_bytes,
                budget_bytes: bytes.budget_bytes,
            },
            launch_overhead_ns,
            fusion_pressure,
        )?;
        match plan_megakernel_memory_budget(cached.topology, graph, bytes) {
            Ok(memory) => Ok(MegakernelExecutionPlan {
                topology: cached.topology,
                memory,
                downgraded_to_sparse: false,
            }),
            Err(MegakernelMemoryError::OverBudget { .. })
                if !cached.topology.is_baseline() =>
            {
                let memory = plan_megakernel_memory_budget(
                    cached.topology.fallback_baseline(),
                    graph,
                    bytes,
                )?;
                Ok(MegakernelExecutionPlan {
                    topology: cached.topology.fallback_baseline(),
                    memory,
                    downgraded_to_sparse: true,
                })
            }
            Err(error) => Err(error),
        }
    }

    /// Return cache counters.
    #[must_use]
    pub fn stats(&self) -> CudaMegakernelPlanCacheStats {
        CudaMegakernelPlanCacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            entries: self.entries.len(),
        }
    }

    /// Drop every cached plan and preserve counters for observability.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.latest_by_identity.clear();
        self.eviction_queue.clear();
    }

    fn latest_topology_for_identity(
        &self,
        graph_layout_hash: u64,
        analysis_kind: CudaMegakernelAnalysisKind,
        device: CudaMegakernelDeviceKey,
    ) -> Option<FrontierTopology> {
        self.latest_by_identity
            .get(&CudaMegakernelPlanIdentityKey {
                graph_layout_hash,
                analysis_kind,
                device,
            })
            .map(|(_, topology)| *topology)
    }

    fn update_latest_identity(
        &mut self,
        identity: CudaMegakernelPlanIdentityKey,
        serial: u64,
        topology: FrontierTopology,
    ) {
        match self.latest_by_identity.get(&identity) {
            Some((latest_serial, _)) if *latest_serial > serial => {}
            _ => {
                self.latest_by_identity.insert(identity, (serial, topology));
            }
        }
    }

    fn recompute_latest_identity(&mut self, identity: CudaMegakernelPlanIdentityKey) {
        let latest = self
            .entries
            .iter()
            .filter(|(key, _)| key.identity() == identity)
            .max_by_key(|(_, entry)| entry.last_seen)
            .map(|(_, entry)| (entry.last_seen, entry.plan.topology));
        if let Some(latest) = latest {
            self.latest_by_identity.insert(identity, latest);
        } else {
            self.latest_by_identity.remove(&identity);
        }
    }

    fn evict_until_below_limit(&mut self) -> Result<(), MegakernelMemoryError> {
        while self.entries.len() >= self.max_entries {
            let Some(Reverse((last_seen, lru_key))) = self.eviction_queue.pop() else {
                break;
            };
            let Some(entry) = self.entries.get(&lru_key) else {
                continue;
            };
            if entry.last_seen != last_seen {
                continue;
            }
            let identity = lru_key.identity();
            let evicted_topology = entry.plan.topology;
            self.entries.remove(&lru_key);
            if matches!(
                self.latest_by_identity.get(&identity),
                Some((latest_seen, latest_topology))
                    if *latest_seen == last_seen && *latest_topology == evicted_topology
            ) {
                self.recompute_latest_identity(identity);
            }
            increment_plan_cache_counter(
                &mut self.evictions,
                "megakernel plan-cache eviction counter",
            );
        }
        Ok(())
    }

    fn advance_serial(&mut self) -> Result<u64, MegakernelMemoryError> {
        if let Some(next) = self.serial.checked_add(1) {
            self.serial = next;
            return Ok(next);
        }
        self.rebase_lru_serials()?;
        self.serial =
            self.serial
                .checked_add(1)
                .ok_or(MegakernelMemoryError::ByteCountOverflow {
                    field: "megakernel plan-cache LRU serial after rebase",
                })?;
        Ok(self.serial)
    }

    fn rebase_lru_serials(&mut self) -> Result<(), MegakernelMemoryError> {
        let mut ordered = Vec::new();
        reserve_vec(
            &mut ordered,
            self.entries.len(),
            "megakernel plan-cache LRU rebase scratch",
        )
        .map_err(|_| MegakernelMemoryError::ByteCountOverflow {
            field: "megakernel plan-cache LRU rebase scratch",
        })?;
        for (key, entry) in &self.entries {
            ordered.push((entry.last_seen, *key));
        }
        sort_unstable_by_key_if_needed(&mut ordered, |(last_seen, key)| (*last_seen, *key));
        self.eviction_queue.clear();
        self.latest_by_identity.clear();
        let mut serial = 0_u64;
        for (_, key) in ordered {
            serial = serial
                .checked_add(1)
                .ok_or(MegakernelMemoryError::ByteCountOverflow {
                    field: "megakernel plan-cache LRU rebase serial",
                })?;
            let topology = if let Some(entry) = self.entries.get_mut(&key) {
                entry.last_seen = serial;
                Some(entry.plan.topology)
            } else {
                None
            };
            if let Some(topology) = topology {
                self.eviction_queue.push(Reverse((serial, key)));
                self.update_latest_identity(key.identity(), serial, topology);
            }
        }
        self.serial = serial;
        Ok(())
    }
}
