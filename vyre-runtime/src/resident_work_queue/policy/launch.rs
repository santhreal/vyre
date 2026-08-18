use vyre_driver::BackendError;

use super::super::planner::{ResidentGridLimits, ResidentGridRequest, ResidentSizingPolicy};
use super::super::staging_reserve::try_reserve_vec_capacity;
use super::cache;
use super::types::{
    ResidentExecutionMode, ResidentGraphBlasSwitchClass, ResidentLaunchCacheStats,
    ResidentLaunchRecommendation, ResidentLaunchRequest, ResidentPromotionEvidence,
    ResidentPromotionRoute, ResidentQueuePressure, ResidentQueueTopology, ResidentTopologyEvidence,
    HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION, TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
};

const FRONTIER_TOPOLOGY_HYSTERESIS_BPS: u16 = 250;
const MEMORY_TOPOLOGY_HYSTERESIS_BPS: u16 = 250;

/// Single policy surface for megakernel launch sizing and telemetry-driven routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentLaunchPolicy {
    /// Sizing policy for worker counts and grid geometry.
    pub sizing: ResidentSizingPolicy,
    /// Minimum capacity for sparse-hit results.
    pub min_hit_capacity: u32,
    /// Multiplier for expected hits to determine capacity.
    pub hit_capacity_multiplier: u32,
    /// Number of waves that define a saturated queue.
    pub saturated_waves: u32,
    /// Threshold for promoting hot opcodes to JIT.
    pub hot_opcode_threshold: u32,
    /// Threshold for promoting hot windows to JIT.
    pub hot_window_threshold: u32,
    /// Queue length threshold to prefer JIT over interpreter.
    pub jit_queue_len_threshold: u32,
    /// Priority age threshold to trigger aging promotions.
    pub priority_age_threshold: u32,
    /// Frontier density at or below this value uses sparse expansion.
    pub sparse_frontier_threshold_bps: u16,
    /// Frontier density at or above this value uses dense propagation.
    pub dense_frontier_threshold_bps: u16,
    /// Memory pressure at or above this value uses the memory-constrained path.
    pub memory_pressure_threshold_bps: u16,
    /// Minimum graph edge count before dense hot work is eligible for fusion.
    pub fusion_edge_threshold: u32,
    /// Conservative resident scratch bytes needed per sparse-hit entry.
    pub scratch_bytes_per_hit: u32,
}

impl Default for ResidentLaunchPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

impl ResidentLaunchPolicy {
    /// Standard launch policy used by VYRE megakernel dispatchers.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            sizing: ResidentSizingPolicy::standard(),
            min_hit_capacity: 1024,
            hit_capacity_multiplier: 2,
            saturated_waves: 4,
            hot_opcode_threshold: 8,
            hot_window_threshold: 4,
            jit_queue_len_threshold: 4096,
            priority_age_threshold: 32,
            sparse_frontier_threshold_bps: 500,
            dense_frontier_threshold_bps: 4_000,
            memory_pressure_threshold_bps: 8_500,
            fusion_edge_threshold: 65_536,
            scratch_bytes_per_hit: 16,
        }
    }

    /// Return launch recommendation cache telemetry for the current thread.
    #[must_use]
    pub fn launch_cache_stats() -> ResidentLaunchCacheStats {
        cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow().stats())
    }

    /// Clear launch recommendation cache entries and counters for this thread.
    pub fn reset_launch_cache_for_thread() {
        cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow_mut().clear());
    }

    /// Recommend geometry, hit capacity, and interpreter/JIT route.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when required adapter limits are zero or derived
    /// launch values cannot fit the u32 ring protocol.
    pub fn recommend(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        self.recommend_inner(request, None)
    }

    /// Recommend a launch and emit topology evidence for parity benches.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the underlying recommendation cannot be
    /// built from the request or adapter limits.
    pub fn recommend_with_topology_evidence(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRecommendation, ResidentTopologyEvidence), BackendError> {
        let (effective_request, recommendation) = self.recommend_with_effective_request(request)?;
        let evidence = self.topology_evidence_for(effective_request, recommendation);
        Ok((recommendation, evidence))
    }

    /// Recommend a launch and emit hot opcode/window promotion evidence.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the underlying recommendation cannot be
    /// built from the request or adapter limits.
    pub fn recommend_with_promotion_evidence(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRecommendation, ResidentPromotionEvidence), BackendError> {
        let (effective_request, recommendation) = self.recommend_with_effective_request(request)?;
        let evidence = self.promotion_evidence_for(effective_request, recommendation);
        Ok((recommendation, evidence))
    }

    /// Recommend a launch while preserving the previous topology inside a
    /// narrow hysteresis band.
    ///
    /// Resident device graphs and long-running dataflow streams should use this
    /// entry point when they can track the last successful topology. It prevents
    /// borderline frontier-density or memory-pressure telemetry from repeatedly
    /// switching kernel variants, invalidating launch plans, and disturbing
    /// cache locality at scale.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when required adapter limits are zero or derived
    /// launch values cannot fit the u32 ring protocol.
    pub fn recommend_with_previous_topology(
        &self,
        request: ResidentLaunchRequest,
        previous_topology: ResidentQueueTopology,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        self.recommend_inner(request, Some(previous_topology))
    }

    fn recommend_inner(
        &self,
        request: ResidentLaunchRequest,
        previous_topology: Option<ResidentQueueTopology>,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        let cache_key = cache::LaunchRecommendationCacheKey {
            policy: *self,
            request,
        };
        if previous_topology.is_none() {
            if let Some(cached) =
                cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow_mut().get(&cache_key))
            {
                return Ok(cached);
            }
        }

        let effective_request = self.infer_missing_scale_signals(request)?;
        let promote_hot_opcodes = effective_request.hot_opcode_count >= self.hot_opcode_threshold;
        let promote_hot_windows = effective_request.hot_window_count >= self.hot_window_threshold;
        let raw_topology =
            self.dispatch_topology_for(effective_request, promote_hot_opcodes, promote_hot_windows);
        let topology = self.stabilize_topology(
            raw_topology,
            effective_request,
            previous_topology,
            promote_hot_opcodes,
            promote_hot_windows,
        );
        let scheduled_request = self.apply_topology_worker_policy(effective_request, topology)?;
        let grid = self.sizing.calculate_optimal_grid(
            ResidentGridRequest::new(
                scheduled_request.queue_len,
                scheduled_request.requested_worker_groups,
            ),
            ResidentGridLimits::new(
                scheduled_request.max_workgroup_size_x,
                scheduled_request.max_compute_workgroups_per_dimension,
                scheduled_request.max_compute_invocations_per_workgroup,
            ),
        )?;
        let geometry = grid.geometry;
        let worker_groups = grid.worker_groups;
        let lanes = u64::from(geometry.dispatch_grid[0])
            .checked_mul(u64::from(geometry.workgroup_size_x))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel launch lane count overflowed u64. Fix: reduce dispatch grid or workgroup size.",
                )
            })?;
        let pressure = classify_pressure(
            effective_request.queue_len,
            lanes,
            effective_request.requeue_count,
            self,
        )?;
        let hit_capacity = self.hit_capacity_for(effective_request)?;
        let estimated_peak_device_bytes =
            self.estimated_peak_device_bytes(effective_request, hit_capacity)?;
        if effective_request.device_memory_budget_bytes != 0
            && estimated_peak_device_bytes > effective_request.device_memory_budget_bytes
        {
            return Err(BackendError::DeviceOutOfMemory {
                requested: estimated_peak_device_bytes,
                available: effective_request.device_memory_budget_bytes,
            });
        }
        let execution_mode = if effective_request.queue_len >= self.jit_queue_len_threshold
            || promote_hot_opcodes
            || promote_hot_windows
            || topology == ResidentQueueTopology::FusedDense
        {
            ResidentExecutionMode::Jit
        } else {
            ResidentExecutionMode::Interpreter
        };
        let age_priority_work = effective_request.requeue_count > 0
            || effective_request.max_priority_age >= self.priority_age_threshold;

        let recommendation = ResidentLaunchRecommendation {
            geometry,
            worker_groups,
            hit_capacity,
            pressure,
            execution_mode,
            topology,
            promote_hot_opcodes,
            promote_hot_windows,
            age_priority_work,
            estimated_peak_device_bytes,
            device_memory_budget_bytes: effective_request.device_memory_budget_bytes,
        };
        if previous_topology.is_none() {
            cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| {
                cache.borrow_mut().insert(cache_key, recommendation);
            });
        }
        Ok(recommendation)
    }

    fn recommend_with_effective_request(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRequest, ResidentLaunchRecommendation), BackendError> {
        let effective_request = self.infer_missing_scale_signals(request)?;
        let recommendation = self.recommend(effective_request)?;
        Ok((effective_request, recommendation))
    }

    fn topology_evidence_for(
        &self,
        request: ResidentLaunchRequest,
        recommendation: ResidentLaunchRecommendation,
    ) -> ResidentTopologyEvidence {
        ResidentTopologyEvidence {
            schema_version: TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
            queue_pressure: recommendation.pressure,
            frontier_density_bps: request.frontier_density_bps,
            semiring_frontier_density_bps: request.frontier_density_bps,
            selected_topology: recommendation.topology,
            graphblas_switch_class: Self::graphblas_switch_class_for(recommendation.topology),
            resident_device_bytes: request.resident_device_bytes,
            estimated_peak_device_bytes: recommendation.estimated_peak_device_bytes,
            output_parity_required: true,
        }
    }

    fn promotion_evidence_for(
        &self,
        request: ResidentLaunchRequest,
        recommendation: ResidentLaunchRecommendation,
    ) -> ResidentPromotionEvidence {
        ResidentPromotionEvidence {
            schema_version: HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION,
            queue_len: request.queue_len,
            jit_queue_len_threshold: self.jit_queue_len_threshold,
            hot_opcode_count: request.hot_opcode_count,
            hot_opcode_threshold: self.hot_opcode_threshold,
            hot_window_count: request.hot_window_count,
            hot_window_threshold: self.hot_window_threshold,
            execution_mode: recommendation.execution_mode,
            promote_hot_opcodes: recommendation.promote_hot_opcodes,
            promote_hot_windows: recommendation.promote_hot_windows,
            promotion_route: Self::promotion_route_for(recommendation),
            fused_descriptor_window_required: recommendation.promote_hot_windows,
            output_parity_required: true,
        }
    }

    fn promotion_route_for(recommendation: ResidentLaunchRecommendation) -> ResidentPromotionRoute {
        if recommendation.execution_mode == ResidentExecutionMode::Interpreter {
            return ResidentPromotionRoute::Interpreter;
        }
        match (
            recommendation.promote_hot_opcodes,
            recommendation.promote_hot_windows,
        ) {
            (true, true) => ResidentPromotionRoute::OpcodeAndWindowJit,
            (true, false) => ResidentPromotionRoute::OpcodeJit,
            (false, true) => ResidentPromotionRoute::WindowJit,
            (false, false) => ResidentPromotionRoute::QueueJit,
        }
    }

    fn graphblas_switch_class_for(topology: ResidentQueueTopology) -> ResidentGraphBlasSwitchClass {
        match topology {
            ResidentQueueTopology::Empty => ResidentGraphBlasSwitchClass::Empty,
            ResidentQueueTopology::SparseFrontier => ResidentGraphBlasSwitchClass::Sparse,
            ResidentQueueTopology::HybridFrontier => ResidentGraphBlasSwitchClass::Hybrid,
            ResidentQueueTopology::DenseFrontier | ResidentQueueTopology::FusedDense => {
                ResidentGraphBlasSwitchClass::Dense
            }
            ResidentQueueTopology::MemoryConstrained => {
                ResidentGraphBlasSwitchClass::MemoryConstrained
            }
        }
    }

    fn hit_capacity_for(&self, request: ResidentLaunchRequest) -> Result<u32, BackendError> {
        if request.requested_hit_capacity != 0 {
            return Ok(request.requested_hit_capacity);
        }
        let expected_hits = request.expected_hits_per_item.max(1);
        let multiplier = if request.memory_pressure_bps >= self.memory_pressure_threshold_bps {
            1
        } else {
            self.hit_capacity_multiplier
        };
        let derived = request
            .queue_len
            .checked_mul(expected_hits)
            .and_then(|value| value.checked_mul(multiplier))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel sparse-hit capacity overflowed u32. Fix: lower queue length, expected_hits_per_item, or hit_capacity_multiplier.",
                )
            })?;
        Ok(derived.max(self.min_hit_capacity))
    }

    fn estimated_peak_device_bytes(
        &self,
        request: ResidentLaunchRequest,
        hit_capacity: u32,
    ) -> Result<u64, BackendError> {
        let scratch_bytes = u64::from(hit_capacity)
            .checked_mul(u64::from(self.scratch_bytes_per_hit))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel scratch byte estimate overflowed u64. Fix: lower hit capacity or scratch_bytes_per_hit.",
                )
            })?;
        request
            .resident_device_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel peak resident byte estimate overflowed u64. Fix: reduce resident buffers or scratch capacity.",
                )
            })
    }

    fn infer_missing_scale_signals(
        &self,
        mut request: ResidentLaunchRequest,
    ) -> Result<ResidentLaunchRequest, BackendError> {
        if request.frontier_density_bps == 0
            && request.queue_len != 0
            && request.graph_node_count != 0
        {
            let active_nodes = u64::from(request.queue_len.min(request.graph_node_count));
            let density = active_nodes
                .checked_mul(10_000)
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel frontier-density numerator overflowed u64. Fix: shard the resident graph before launch.",
                    )
                })?
                .checked_div(u64::from(request.graph_node_count))
                .unwrap_or(0)
                .clamp(1, 10_000);
            request.frontier_density_bps = u16::try_from(density).map_err(|error| {
                BackendError::new(format!(
                    "megakernel frontier density cannot fit u16: {error}. Fix: clamp density before ABI encoding."
                ))
            })?;
        }
        if request.memory_pressure_bps == 0
            && request.device_memory_budget_bytes != 0
            && request.resident_device_bytes != 0
        {
            let pressure = (u128::from(request.resident_device_bytes)
                .checked_mul(10_000)
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-pressure numerator overflowed u128. Fix: reduce resident device bytes before launch.",
                    )
                })?
                / u128::from(request.device_memory_budget_bytes))
            .min(10_000);
            request.memory_pressure_bps = u16::try_from(pressure).map_err(|error| {
                BackendError::new(format!(
                    "megakernel memory pressure cannot fit u16: {error}. Fix: clamp pressure before ABI encoding."
                ))
            })?;
        }
        Ok(request)
    }

    fn apply_topology_worker_policy(
        &self,
        mut request: ResidentLaunchRequest,
        topology: ResidentQueueTopology,
    ) -> Result<ResidentLaunchRequest, BackendError> {
        if topology == ResidentQueueTopology::MemoryConstrained
            && request.memory_pressure_bps != 0
            && request.requested_worker_groups > 1
        {
            let pressure_span = u32::from(
                10_000_u16
                    .checked_sub(self.memory_pressure_threshold_bps)
                    .ok_or_else(|| {
                        BackendError::new(
                            "megakernel memory-pressure threshold exceeds 10000 bps. Fix: configure threshold in basis points.",
                        )
                    })?,
            )
            .max(1);
            let over_threshold = u32::from(
                request
                    .memory_pressure_bps
                    .saturating_sub(self.memory_pressure_threshold_bps),
            )
            .min(pressure_span);
            let shed_bps = 2_500_u32
                .checked_add(
                    over_threshold
                        .checked_mul(2_500)
                        .ok_or_else(|| {
                            BackendError::new(
                                "megakernel memory-pressure worker shed overflowed u32. Fix: lower pressure telemetry before launch.",
                            )
                        })?
                        / pressure_span,
                )
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-pressure worker shed overflowed u32. Fix: lower pressure telemetry before launch.",
                    )
                })?;
            let keep_bps = 10_000_u32.checked_sub(shed_bps).ok_or_else(|| {
                BackendError::new(
                    "megakernel memory-pressure worker keep ratio underflowed. Fix: keep shed_bps within 0..=10000.",
                )
            })?;
            let scaled = u64::from(request.requested_worker_groups)
                .checked_mul(u64::from(keep_bps))
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-constrained worker count overflowed u64. Fix: reduce requested worker groups.",
                    )
                })?
                / 10_000;
            request.requested_worker_groups = u32::try_from(scaled)
                .map_err(|error| {
                    BackendError::new(format!(
                        "megakernel memory-constrained worker count cannot fit u32: {error}. Fix: reduce requested worker groups."
                    ))
                })?
                .max(1);
        }
        if topology == ResidentQueueTopology::SparseFrontier
            && request.graph_node_count != 0
            && request.frontier_density_bps != 0
            && request.requested_worker_groups > 1
        {
            let sparse_span = u32::from(self.sparse_frontier_threshold_bps).max(1);
            let density = u32::from(request.frontier_density_bps).clamp(1, sparse_span);
            let scaled = u64::from(request.requested_worker_groups)
                .checked_mul(u64::from(density))
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel sparse-frontier worker count overflowed u64. Fix: reduce requested worker groups.",
                    )
                })?
                / u64::from(sparse_span);
            let warp_floor = request.requested_worker_groups.min(32);
            request.requested_worker_groups = u32::try_from(scaled)
                .map_err(|error| {
                    BackendError::new(format!(
                        "megakernel sparse-frontier worker count cannot fit u32: {error}. Fix: reduce requested worker groups."
                    ))
                })?
                .max(warp_floor)
                .min(request.requested_worker_groups);
        }
        Ok(request)
    }

    fn dispatch_topology_for(
        &self,
        request: ResidentLaunchRequest,
        promote_hot_opcodes: bool,
        promote_hot_windows: bool,
    ) -> ResidentQueueTopology {
        if request.queue_len == 0 {
            return ResidentQueueTopology::Empty;
        }
        if request.memory_pressure_bps >= self.memory_pressure_threshold_bps {
            return ResidentQueueTopology::MemoryConstrained;
        }
        if request.frontier_density_bps <= self.sparse_frontier_threshold_bps {
            return ResidentQueueTopology::SparseFrontier;
        }
        let dense = request.frontier_density_bps >= self.dense_frontier_threshold_bps;
        let graph_is_large =
            request.graph_node_count > 0 && request.graph_edge_count >= self.fusion_edge_threshold;
        if dense && graph_is_large && (promote_hot_opcodes || promote_hot_windows) {
            return ResidentQueueTopology::FusedDense;
        }
        if dense {
            return ResidentQueueTopology::DenseFrontier;
        }
        ResidentQueueTopology::HybridFrontier
    }

    fn stabilize_topology(
        &self,
        raw_topology: ResidentQueueTopology,
        request: ResidentLaunchRequest,
        previous_topology: Option<ResidentQueueTopology>,
        promote_hot_opcodes: bool,
        promote_hot_windows: bool,
    ) -> ResidentQueueTopology {
        if raw_topology == ResidentQueueTopology::Empty {
            return raw_topology;
        }
        if raw_topology == ResidentQueueTopology::MemoryConstrained {
            return raw_topology;
        }
        let Some(previous_topology) = previous_topology else {
            return raw_topology;
        };
        if previous_topology == ResidentQueueTopology::MemoryConstrained
            && request.memory_pressure_bps
                >= hysteresis_sub(
                    self.memory_pressure_threshold_bps,
                    MEMORY_TOPOLOGY_HYSTERESIS_BPS,
                )
        {
            return ResidentQueueTopology::MemoryConstrained;
        }

        match previous_topology {
            ResidentQueueTopology::SparseFrontier
                if raw_topology != ResidentQueueTopology::SparseFrontier
                    && request.frontier_density_bps
                        <= hysteresis_add(
                            self.sparse_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::SparseFrontier
            }
            ResidentQueueTopology::HybridFrontier
                if raw_topology == ResidentQueueTopology::SparseFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.sparse_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::HybridFrontier
            }
            ResidentQueueTopology::HybridFrontier
                if matches!(
                    raw_topology,
                    ResidentQueueTopology::DenseFrontier | ResidentQueueTopology::FusedDense
                ) && request.frontier_density_bps
                    <= hysteresis_add(
                        self.dense_frontier_threshold_bps,
                        FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                    ) =>
            {
                ResidentQueueTopology::HybridFrontier
            }
            ResidentQueueTopology::DenseFrontier
                if raw_topology == ResidentQueueTopology::HybridFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.dense_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::DenseFrontier
            }
            ResidentQueueTopology::FusedDense
                if raw_topology == ResidentQueueTopology::HybridFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.dense_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        )
                    && request.graph_edge_count >= self.fusion_edge_threshold
                    && (promote_hot_opcodes || promote_hot_windows) =>
            {
                ResidentQueueTopology::FusedDense
            }
            _ => raw_topology,
        }
    }

    /// Select the best `hit_capacity_multiplier` from a candidate set.
    ///
    /// `candidate_multipliers` are the multipliers to try; `costs[i]`
    /// is the observed dispatch latency (or any minimization metric)
    /// when `candidate_multipliers[i]` was used. Lower cost wins; the
    /// minimum observed cost selects the multiplier.
    ///
    /// Returns the chosen multiplier. If `candidate_multipliers` is
    /// empty, returns the policy's existing `hit_capacity_multiplier`.
    ///
    #[must_use]
    pub fn autotune_hit_capacity_multiplier(
        &self,
        candidate_multipliers: &[u32],
        costs: &[f64],
    ) -> u32 {
        if candidate_multipliers.is_empty() || costs.is_empty() {
            return self.hit_capacity_multiplier;
        }
        let n = candidate_multipliers.len().min(costs.len());
        best_cost_index(&costs[..n])
            .and_then(|chosen| candidate_multipliers.get(chosen).copied())
            .unwrap_or(self.hit_capacity_multiplier)
    }

    /// Select the best workgroup-size from a candidate set.
    ///
    /// `candidate_sizes[i]` is paired
    /// with `costs[i]` (lower is better). Returns the chosen size or
    /// the policy's `sizing.default_workgroup_size_x()` fallback.
    #[must_use]
    pub fn autotune_workgroup_size(
        &self,
        candidate_sizes: &[u32],
        costs: &[f64],
        current_size: u32,
    ) -> u32 {
        if candidate_sizes.is_empty() || costs.is_empty() {
            return current_size;
        }
        let n = candidate_sizes.len().min(costs.len());
        best_cost_index(&costs[..n])
            .and_then(|chosen| candidate_sizes.get(chosen).copied())
            .unwrap_or(current_size)
    }

    /// Compute the next-step parameter delta for a continuous autotune
    /// knob using a Fisher-preconditioned natural-gradient step.
    ///
    /// `m_inv_sqrt`: inverse-square-root of the Fisher block (n×n
    /// row-major). Passing an identity matrix reduces the natural
    /// gradient to plain gradient descent.
    ///
    /// `grad`: plain gradient ∂latency/∂param (length n).
    ///
    /// Returns the parameter delta `-lr · M_inv_sqrt · grad`.
    ///
    /// P-DRIVER-8: every continuous autotune knob (workgroup size,
    /// hit-capacity, fixpoint iteration count, …) should follow the
    /// natural-gradient direction by default  -  Fisher-preconditioned
    /// descent converges 5-10× faster than plain gradient on the
    /// elongated-valley latency surfaces typical of GPU autotuning.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when host staging cannot be reserved for the
    /// natural-gradient vector.
    pub fn try_natural_gradient_autotune_step(
        m_inv_sqrt: &[f64],
        grad: &[f64],
        n: u32,
        learning_rate: f64,
    ) -> Result<Vec<f64>, BackendError> {
        let mut out = Vec::new();
        Self::try_natural_gradient_autotune_step_into(
            m_inv_sqrt,
            grad,
            n,
            learning_rate,
            &mut out,
        )?;
        Ok(out)
    }

    /// Compute the natural-gradient autotune step into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when host staging cannot be reserved for the
    /// natural-gradient vector.
    pub fn try_natural_gradient_autotune_step_into(
        m_inv_sqrt: &[f64],
        grad: &[f64],
        n: u32,
        learning_rate: f64,
        out: &mut Vec<f64>,
    ) -> Result<(), BackendError> {
        let n = u32_to_usize_checked(n, "natural-gradient dimension")?;
        out.clear();
        let Some(required_matrix_len) = n.checked_mul(n) else {
            return Ok(());
        };
        if m_inv_sqrt.len() < required_matrix_len || grad.len() < n {
            return Ok(());
        }
        reserve_target_capacity(out, n, "natural-gradient output")?;
        out.resize(n, 0.0);
        for row in 0..n {
            let mut acc = 0.0;
            for col in 0..n {
                acc += m_inv_sqrt[row * n + col] * grad[col];
            }
            out[row] = -learning_rate * acc;
        }
        Ok(())
    }
}

pub(super) fn reserve_target_capacity<T>(
    out: &mut Vec<T>,
    target_capacity: usize,
    label: &'static str,
) -> Result<(), BackendError> {
    try_reserve_vec_capacity(out, target_capacity).map_err(|source| {
        BackendError::new(format!(
            "megakernel {label} reservation failed for {target_capacity} element(s): {source}. Fix: shard the policy input before launch-policy math."
        ))
    })
}

/// Index of the lowest cost, or `None` when no cost was measured.
///
/// The empty case is in the return type rather than in an assertion, because an
/// assertion compiled out of the shipped binary leaves an unchecked index at the
/// only point where the caller has nothing to select from.
pub(super) fn best_cost_index(costs: &[f64]) -> Option<usize> {
    let (first, rest) = costs.split_first()?;
    let mut best = 0;
    let mut best_cost = *first;
    for (index, &cost) in rest.iter().enumerate() {
        if cost.total_cmp(&best_cost).is_lt() {
            best = index + 1;
            best_cost = cost;
        }
    }
    Some(best)
}

fn u32_to_usize_checked(value: u32, label: &'static str) -> Result<usize, BackendError> {
    usize::try_from(value).map_err(|error| {
        BackendError::new(format!(
            "{label} cannot fit usize: {error}. Fix: shard the autotune surface."
        ))
    })
}

fn hysteresis_add(value: u16, hysteresis: u16) -> u16 {
    value.saturating_add(hysteresis)
}

fn hysteresis_sub(value: u16, hysteresis: u16) -> u16 {
    value.saturating_sub(hysteresis)
}

fn classify_pressure(
    queue_len: u32,
    lanes: u64,
    requeue_count: u64,
    policy: &ResidentLaunchPolicy,
) -> Result<ResidentQueuePressure, BackendError> {
    if queue_len == 0 {
        return Ok(ResidentQueuePressure::Empty);
    }
    let lanes = lanes.max(1);
    let queue_len = u64::from(queue_len);
    let saturated_lanes = lanes
        .checked_mul(u64::from(policy.saturated_waves))
        .ok_or_else(|| {
            BackendError::new(
                "megakernel pressure wave threshold overflowed u64. Fix: reduce worker lanes or saturated_waves.",
            )
        })?;
    if requeue_count > 0 || queue_len >= saturated_lanes {
        Ok(ResidentQueuePressure::Saturated)
    } else if queue_len >= lanes {
        Ok(ResidentQueuePressure::Balanced)
    } else {
        Ok(ResidentQueuePressure::Light)
    }
}
