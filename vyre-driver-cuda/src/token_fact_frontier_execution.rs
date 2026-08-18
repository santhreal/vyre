//! CUDA execution planner for unified token/fact graph frontier waves.

use crate::frontier_typed_ir_adapter::CudaFrontierTypedIrInput;
use crate::megakernel_barrier_planner::{
    plan_cuda_frontier_megakernel_execution_with_scratch, CudaMegakernelFrontierExecutionPlan,
    CudaMegakernelFrontierExecutionPlanError,
};
use crate::megakernel_plan_cache::{
    CudaMegakernelAnalysisKind, CudaMegakernelDeviceKey, CudaMegakernelPlanCache,
};
use crate::megakernel_scheduler::CudaMegakernelScheduleSample;
use vyre_driver::device_work_queue::{
    plan_device_work_queue_with_expansion, DeviceWorkQueueError, DeviceWorkQueueExpansionProfile,
    DeviceWorkQueuePlan, WorkQueueHostSync,
};
use vyre_driver::megakernel_barrier::MegakernelBarrierScratch;
use vyre_driver::megakernel_execution::MegakernelGraphShape;
use vyre_driver::megakernel_frontier::megakernel_frontier_fused_wave_budget_bytes;
use vyre_driver::megakernel_frontier::MegakernelFrontierWave;
use vyre_driver::ResidentGraphReuseTelemetry;
use vyre_libs::device::device_resident_token_fact_graph::{
    DeviceResidentTokenFactGraphLayout, TOKEN_FACT_DEGREE_PROFILE_BUCKETS,
    TOKEN_FACT_DEGREE_PROFILE_RANKS,
};

/// Dependency-aware CUDA execution plan for a unified token/fact graph.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaTokenFactFrontierExecutionPlan {
    /// Existing CUDA frontier execution plan.
    pub frontier: CudaMegakernelFrontierExecutionPlan,
    /// Resident device-side work queue for dependent frontier draining.
    pub work_queue: DeviceWorkQueuePlan,
    /// Resident payload bytes subtracted from the scheduler budget.
    pub resident_payload_bytes: u64,
    /// Resident work-queue bytes subtracted from the scheduler budget.
    pub resident_work_queue_bytes: u64,
    /// Total required bytes including graph records, frontier envelopes, and payload slab.
    pub total_required_bytes: u64,
}

/// Whether the token/fact graph must be uploaded for this execution plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaTokenFactGraphResidency {
    /// The graph is not resident yet; this plan includes one graph upload.
    ColdUpload,
    /// The graph is already resident on device; this plan reuses it.
    WarmResident,
}

/// CUDA token/fact execution envelope with explicit graph-residency accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct CudaTokenFactFrontierExecutionEnvelope {
    /// Device execution plan.
    pub plan: CudaTokenFactFrontierExecutionPlan,
    /// Backend-neutral cold-upload/warm-reuse graph telemetry.
    pub graph_reuse: ResidentGraphReuseTelemetry,
    /// Resident node+edge graph bytes that must remain live during execution.
    pub resident_graph_bytes: u64,
    /// Graph bytes uploaded by this plan. Zero for warm resident graphs.
    pub graph_upload_bytes: u64,
    /// Graph upload bytes avoided by reusing a warm resident graph.
    pub avoided_graph_upload_bytes: u64,
    /// Total live resident bytes required during execution.
    pub total_resident_bytes: u64,
}

/// Errors from token/fact frontier execution planning.
#[derive(Clone, Debug, PartialEq)]
pub enum CudaTokenFactFrontierExecutionError {
    /// Resident token/fact graph topology cannot be empty on the CUDA release path.
    ZeroResidentGraphBytes,
    /// The public CUDA token/fact layout reported inconsistent resident bytes.
    ResidentGraphByteEnvelopeMismatch {
        /// Node+edge+payload bytes computed from layout fields.
        expected_bytes: u64,
        /// Layout-reported resident byte total.
        actual_bytes: u64,
    },
    /// Payload alone exceeds the explicit device-memory budget.
    PayloadExceedsBudget {
        /// Resident payload bytes.
        payload_bytes: u64,
        /// Caller-provided budget.
        budget_bytes: u64,
    },
    /// Total byte arithmetic overflowed.
    ByteCountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// Frontier wave count and active-item count must match exactly.
    ActiveItemWaveCountMismatch {
        /// Number of wave memory envelopes.
        waves: usize,
        /// Number of active-item entries.
        active_items: usize,
    },
    /// Underlying frontier planner rejected the execution plan.
    FrontierPlan(CudaMegakernelFrontierExecutionPlanError),
    /// Device work-queue planning rejected the execution plan.
    WorkQueue(DeviceWorkQueueError),
}

impl std::fmt::Display for CudaTokenFactFrontierExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroResidentGraphBytes => write!(
                f,
                "CUDA token/fact frontier plan received an empty resident graph topology. Fix: build a concrete token/fact graph before CUDA execution planning."
            ),
            Self::ResidentGraphByteEnvelopeMismatch {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "CUDA token/fact frontier layout reports {actual_bytes} resident bytes but node+edge+payload fields require {expected_bytes}. Fix: rebuild the CUDA token/fact layout from the canonical adapter before planning."
            ),
            Self::PayloadExceedsBudget {
                payload_bytes,
                budget_bytes,
            } => write!(
                f,
                "CUDA token/fact frontier plan payload requires {payload_bytes} bytes but budget allows {budget_bytes}. Fix: shard the token/fact payload slab before megakernel planning."
            ),
            Self::ByteCountOverflow { field } => write!(
                f,
                "CUDA token/fact frontier planner overflowed while computing {field}. Fix: shard the resident token/fact graph before CUDA execution planning."
            ),
            Self::ActiveItemWaveCountMismatch {
                waves,
                active_items,
            } => write!(
                f,
                "CUDA token/fact frontier plan has {waves} wave envelope(s) but {active_items} active-item count(s). Fix: preserve one active-item entry per frontier wave before device work-queue planning."
            ),
            Self::FrontierPlan(err) => write!(f, "{err}"),
            Self::WorkQueue(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CudaTokenFactFrontierExecutionError {}

impl From<CudaMegakernelFrontierExecutionPlanError> for CudaTokenFactFrontierExecutionError {
    fn from(err: CudaMegakernelFrontierExecutionPlanError) -> Self {
        Self::FrontierPlan(err)
    }
}

impl From<DeviceWorkQueueError> for CudaTokenFactFrontierExecutionError {
    fn from(err: DeviceWorkQueueError) -> Self {
        Self::WorkQueue(err)
    }
}

/// Plan dependency-aware CUDA execution for frontier waves over a token/fact graph.
pub fn plan_cuda_token_fact_frontier_execution(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph_layout: DeviceResidentTokenFactGraphLayout,
    frontier_input: &CudaFrontierTypedIrInput,
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
) -> Result<CudaTokenFactFrontierExecutionPlan, CudaTokenFactFrontierExecutionError> {
    let mut barrier_scratch = MegakernelBarrierScratch::try_with_capacity(
        frontier_input.waves.len(),
        frontier_input.dependencies.len(),
    )
    .map_err(CudaMegakernelFrontierExecutionPlanError::Barrier)?;
    plan_cuda_token_fact_frontier_execution_with_scratch(
        cache,
        graph_layout_hash,
        analysis_kind,
        device,
        sample,
        graph_layout,
        frontier_input,
        budget_bytes,
        launch_overhead_ns,
        fusion_pressure,
        &mut barrier_scratch,
    )
}

/// Plan dependency-aware CUDA execution and expose explicit graph-residency
/// accounting.
pub fn plan_cuda_token_fact_frontier_execution_envelope(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph_layout: DeviceResidentTokenFactGraphLayout,
    graph_residency: CudaTokenFactGraphResidency,
    frontier_input: &CudaFrontierTypedIrInput,
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
) -> Result<CudaTokenFactFrontierExecutionEnvelope, CudaTokenFactFrontierExecutionError> {
    let mut barrier_scratch = MegakernelBarrierScratch::try_with_capacity(
        frontier_input.waves.len(),
        frontier_input.dependencies.len(),
    )
    .map_err(CudaMegakernelFrontierExecutionPlanError::Barrier)?;
    plan_cuda_token_fact_frontier_execution_envelope_with_scratch(
        cache,
        graph_layout_hash,
        analysis_kind,
        device,
        sample,
        graph_layout,
        graph_residency,
        frontier_input,
        budget_bytes,
        launch_overhead_ns,
        fusion_pressure,
        &mut barrier_scratch,
    )
}

/// Plan dependency-aware CUDA execution using caller-owned megakernel barrier scratch.
pub fn plan_cuda_token_fact_frontier_execution_with_scratch(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph_layout: DeviceResidentTokenFactGraphLayout,
    frontier_input: &CudaFrontierTypedIrInput,
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    barrier_scratch: &mut MegakernelBarrierScratch,
) -> Result<CudaTokenFactFrontierExecutionPlan, CudaTokenFactFrontierExecutionError> {
    Ok(
        plan_cuda_token_fact_frontier_execution_envelope_with_scratch(
            cache,
            graph_layout_hash,
            analysis_kind,
            device,
            sample,
            graph_layout,
            CudaTokenFactGraphResidency::ColdUpload,
            frontier_input,
            budget_bytes,
            launch_overhead_ns,
            fusion_pressure,
            barrier_scratch,
        )?
        .plan,
    )
}

/// Plan dependency-aware CUDA execution with explicit graph-residency
/// accounting.
pub fn plan_cuda_token_fact_frontier_execution_envelope_with_scratch(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph_layout: DeviceResidentTokenFactGraphLayout,
    graph_residency: CudaTokenFactGraphResidency,
    frontier_input: &CudaFrontierTypedIrInput,
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    barrier_scratch: &mut MegakernelBarrierScratch,
) -> Result<CudaTokenFactFrontierExecutionEnvelope, CudaTokenFactFrontierExecutionError> {
    if frontier_input.active_items.len() != frontier_input.waves.len() {
        return Err(
            CudaTokenFactFrontierExecutionError::ActiveItemWaveCountMismatch {
                waves: frontier_input.waves.len(),
                active_items: frontier_input.active_items.len(),
            },
        );
    }
    // The only place the neutral resident layout becomes a scheduler type. The
    // layout itself, including its byte envelope and out-degree profile, is
    // owned by vyre-libs and is not restated here.
    let graph_shape = MegakernelGraphShape {
        node_count: graph_layout.node_count,
        edge_count: graph_layout.edge_count,
    };
    let resident_graph_bytes = graph_layout
        .node_bytes
        .checked_add(graph_layout.edge_bytes)
        .ok_or(CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "resident token/fact graph bytes",
        })?;
    if resident_graph_bytes == 0 {
        return Err(CudaTokenFactFrontierExecutionError::ZeroResidentGraphBytes);
    }
    let expected_resident_bytes = resident_graph_bytes
        .checked_add(graph_layout.payload_bytes)
        .ok_or(CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "resident token/fact graph envelope bytes",
        })?;
    if expected_resident_bytes != graph_layout.resident_bytes {
        return Err(
            CudaTokenFactFrontierExecutionError::ResidentGraphByteEnvelopeMismatch {
                expected_bytes: expected_resident_bytes,
                actual_bytes: graph_layout.resident_bytes,
            },
        );
    }
    let payload_budget = budget_bytes.checked_sub(graph_layout.payload_bytes).ok_or(
        CudaTokenFactFrontierExecutionError::PayloadExceedsBudget {
            payload_bytes: graph_layout.payload_bytes,
            budget_bytes,
        },
    )?;
    let active_items = total_active_items(&frontier_input.active_items)?;
    let frontier_reserve_bytes = max_single_frontier_wave_bytes(&frontier_input.waves)?;
    let queue_budget =
        queue_residency_budget(payload_budget, resident_graph_bytes, frontier_reserve_bytes);
    let work_queue = if active_items == 0 {
        empty_device_work_queue_plan()
    } else {
        let expansion_items = estimated_queue_expansion_items(
            active_items,
            graph_shape,
            graph_layout.max_out_degree,
            graph_layout.top_out_degree_prefix_sums,
        )?;
        plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
            initial_items: active_items,
            expansion_items,
            entry_bytes: 4,
            control_bytes: 16,
            budget_bytes: queue_budget,
            host_sync: WorkQueueHostSync::FinalOnly,
        })?
    };
    let scheduler_budget = payload_budget
        .checked_sub(work_queue.resident_bytes)
        .ok_or(CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "scheduler budget after work queue",
        })?;
    let frontier = plan_cuda_frontier_megakernel_execution_with_scratch(
        cache,
        graph_layout_hash,
        analysis_kind,
        device,
        sample,
        graph_shape,
        graph_layout.node_record_bytes,
        graph_layout.edge_record_bytes,
        &frontier_input.waves,
        &frontier_input.dependencies,
        scheduler_budget,
        launch_overhead_ns,
        fusion_pressure,
        barrier_scratch,
    )?;
    let total_required_bytes = frontier
        .execution
        .memory
        .required_bytes
        .checked_add(graph_layout.payload_bytes)
        .and_then(|bytes| bytes.checked_add(work_queue.resident_bytes))
        .ok_or(CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "total required bytes",
        })?;

    let plan = CudaTokenFactFrontierExecutionPlan {
        frontier,
        work_queue,
        resident_payload_bytes: graph_layout.payload_bytes,
        resident_work_queue_bytes: work_queue.resident_bytes,
        total_required_bytes,
    };
    let graph_upload_bytes = match graph_residency {
        CudaTokenFactGraphResidency::ColdUpload => resident_graph_bytes,
        CudaTokenFactGraphResidency::WarmResident => 0,
    };
    let avoided_graph_upload_bytes = match graph_residency {
        CudaTokenFactGraphResidency::ColdUpload => 0,
        CudaTokenFactGraphResidency::WarmResident => resident_graph_bytes,
    };
    let graph_reuse = match graph_residency {
        CudaTokenFactGraphResidency::ColdUpload => {
            ResidentGraphReuseTelemetry::cold_upload(resident_graph_bytes)
        }
        CudaTokenFactGraphResidency::WarmResident => {
            ResidentGraphReuseTelemetry::warm_reuse(resident_graph_bytes)
        }
    };
    Ok(CudaTokenFactFrontierExecutionEnvelope {
        total_resident_bytes: plan.total_required_bytes,
        plan,
        graph_reuse,
        resident_graph_bytes,
        graph_upload_bytes,
        avoided_graph_upload_bytes,
    })
}

fn total_active_items(active_items: &[u64]) -> Result<u64, CudaTokenFactFrontierExecutionError> {
    let mut total = 0_u64;
    for &items in active_items {
        total = total.checked_add(items).ok_or(
            CudaTokenFactFrontierExecutionError::ByteCountOverflow {
                field: "total active frontier items",
            },
        )?;
    }
    Ok(total)
}

pub(crate) fn estimated_queue_expansion_items(
    active_items: u64,
    graph: MegakernelGraphShape,
    max_out_degree: u64,
    top_out_degree_prefix_sums: [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
) -> Result<u64, CudaTokenFactFrontierExecutionError> {
    if active_items == 0 || graph.edge_count == 0 {
        return Ok(0);
    }
    if let Some(profile_bound) =
        top_out_degree_profile_bound(active_items, graph.edge_count, top_out_degree_prefix_sums)
    {
        return Ok(profile_bound);
    }
    let expansion_degree = if max_out_degree != 0 {
        max_out_degree
    } else if graph.node_count == 0 {
        return Ok(graph.edge_count);
    } else {
        vyre_driver::numeric::checked_ceil_div_u64(graph.edge_count, graph.node_count).ok_or(
            CudaTokenFactFrontierExecutionError::ByteCountOverflow {
                field: "average token/fact graph out-degree",
            },
        )?
    };
    let projected_edges = active_items.checked_mul(expansion_degree).ok_or(
        CudaTokenFactFrontierExecutionError::ByteCountOverflow {
            field: "active frontier edge expansion",
        },
    )?;
    Ok(projected_edges.min(graph.edge_count))
}

fn top_out_degree_profile_bound(
    active_items: u64,
    edge_count: u64,
    top_out_degree_prefix_sums: [u64; TOKEN_FACT_DEGREE_PROFILE_BUCKETS],
) -> Option<u64> {
    for (rank, prefix_sum) in TOKEN_FACT_DEGREE_PROFILE_RANKS
        .iter()
        .zip(top_out_degree_prefix_sums)
    {
        if active_items <= *rank {
            if prefix_sum == 0 && edge_count != 0 {
                return None;
            }
            return Some(prefix_sum.min(edge_count));
        }
    }
    None
}

fn max_single_frontier_wave_bytes(
    waves: &[MegakernelFrontierWave],
) -> Result<u64, CudaTokenFactFrontierExecutionError> {
    let mut peak = 0_u64;
    for wave in waves {
        let bytes = megakernel_frontier_fused_wave_budget_bytes(*wave)
            .map_err(CudaMegakernelFrontierExecutionPlanError::from)
            .map_err(CudaTokenFactFrontierExecutionError::from)?;
        peak = peak.max(bytes);
    }
    Ok(peak)
}

fn queue_residency_budget(
    payload_budget: u64,
    resident_graph_bytes: u64,
    frontier_reserve_bytes: u64,
) -> u64 {
    payload_budget
        .saturating_sub(resident_graph_bytes)
        .saturating_sub(frontier_reserve_bytes)
}

fn empty_device_work_queue_plan() -> DeviceWorkQueuePlan {
    DeviceWorkQueuePlan {
        queue_bytes: 0,
        control_bytes: 0,
        resident_bytes: 0,
        initial_occupancy_bps: 0,
        final_only_host_sync: true,
    }
}
