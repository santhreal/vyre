//! CUDA telemetry adapter over the neutral megakernel frontier execution policy.
//!
//! Barrier grouping, fused-group splitting, peak byte accounting, and topology
//! selection are `vyre_driver::megakernel_frontier`. This module maps CUDA
//! telemetry and the device-local plan cache onto that policy and names the
//! results with the `Cuda*` aliases the rest of the backend already uses.

use crate::megakernel_plan_cache::{
    CudaMegakernelAnalysisKind, CudaMegakernelDeviceKey, CudaMegakernelPlanCache,
};
use crate::megakernel_scheduler::CudaMegakernelScheduleSample;
use vyre_driver::megakernel_barrier::{MegakernelBarrierScratch, MegakernelWaveDependency};
use vyre_driver::megakernel_execution::{
    MegakernelDeviceCapabilities, MegakernelExecutionPlan, MegakernelExecutionPlanner,
    MegakernelExecutionRequest, MegakernelExecutionSample, MegakernelGraphShape,
    MegakernelMemoryError,
};
use vyre_driver::megakernel_frontier::{
    plan_megakernel_frontier_execution_with_scratch, MegakernelFrontierExecutionPlan,
    MegakernelFrontierExecutionPlanError, MegakernelFrontierWave,
};

/// Dependency-aware CUDA megakernel execution plan for frontier waves.
pub type CudaMegakernelFrontierExecutionPlan = MegakernelFrontierExecutionPlan;

/// Dependency-aware CUDA frontier execution planning failure.
pub type CudaMegakernelFrontierExecutionPlanError = MegakernelFrontierExecutionPlanError;

/// Binds the device-local plan cache to the neutral execution-planner seam.
struct CudaCachedExecutionPlanner<'a> {
    cache: &'a mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
}

impl MegakernelExecutionPlanner for CudaCachedExecutionPlanner<'_> {
    fn plan_execution(
        &mut self,
        request: MegakernelExecutionRequest,
    ) -> Result<MegakernelExecutionPlan, MegakernelMemoryError> {
        self.cache.get_or_plan_execution(
            self.graph_layout_hash,
            self.analysis_kind,
            self.device,
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: request.sample.dispatch_cost_ns,
                frontier_density: request.sample.frontier_density,
                readback_bytes: request.sample.readback_bytes,
            },
            request.graph,
            request.bytes,
            request.launch_overhead_ns,
            request.fusion_pressure,
        )
    }
}

/// Device capabilities the neutral wave policy reads from a CUDA device key.
fn capabilities(device: CudaMegakernelDeviceKey) -> MegakernelDeviceCapabilities {
    MegakernelDeviceCapabilities {
        supports_device_wide_barrier: device.supports_grid_sync,
    }
}

/// Plan dependency-aware CUDA megakernel execution for frontier-typed waves.
///
/// # Errors
///
/// Returns [`CudaMegakernelFrontierExecutionPlanError`] when the neutral policy
/// rejects the dependency graph, the byte accounting, or the budget.
#[allow(clippy::too_many_arguments)]
pub fn plan_cuda_frontier_megakernel_execution(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
    waves: &[MegakernelFrontierWave],
    dependencies: &[MegakernelWaveDependency],
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
) -> Result<CudaMegakernelFrontierExecutionPlan, CudaMegakernelFrontierExecutionPlanError> {
    let mut scratch = MegakernelBarrierScratch::try_with_capacity(waves.len(), dependencies.len())?;
    plan_cuda_frontier_megakernel_execution_with_scratch(
        cache,
        graph_layout_hash,
        analysis_kind,
        device,
        sample,
        graph,
        bytes_per_node,
        bytes_per_edge,
        waves,
        dependencies,
        budget_bytes,
        launch_overhead_ns,
        fusion_pressure,
        &mut scratch,
    )
}

/// Plan dependency-aware CUDA megakernel execution using caller-owned scratch.
///
/// # Errors
///
/// Same rejections as [`plan_cuda_frontier_megakernel_execution`].
#[allow(clippy::too_many_arguments)]
pub fn plan_cuda_frontier_megakernel_execution_with_scratch(
    cache: &mut CudaMegakernelPlanCache,
    graph_layout_hash: u64,
    analysis_kind: CudaMegakernelAnalysisKind,
    device: CudaMegakernelDeviceKey,
    sample: CudaMegakernelScheduleSample,
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
    waves: &[MegakernelFrontierWave],
    dependencies: &[MegakernelWaveDependency],
    budget_bytes: u64,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    scratch: &mut MegakernelBarrierScratch,
) -> Result<CudaMegakernelFrontierExecutionPlan, CudaMegakernelFrontierExecutionPlanError> {
    let mut planner = CudaCachedExecutionPlanner {
        cache,
        graph_layout_hash,
        analysis_kind,
        device,
    };
    plan_megakernel_frontier_execution_with_scratch(
        &mut planner,
        MegakernelExecutionSample {
            dispatch_cost_ns: sample.dispatch_cost_ns,
            frontier_density: sample.frontier_density,
            readback_bytes: sample.readback_bytes,
        },
        graph,
        bytes_per_node,
        bytes_per_edge,
        waves,
        dependencies,
        budget_bytes,
        launch_overhead_ns,
        fusion_pressure,
        capabilities(device),
        scratch,
    )
}

// Inline: `vyre_driver_cuda::megakernel_barrier_planner` is `pub(crate)`, so no integration test
// can reach what this suite exercises.
#[cfg(test)]
mod tests {
    use super::plan_cuda_frontier_megakernel_execution;
    use crate::megakernel_plan_cache::{
        CudaMegakernelAnalysisKind, CudaMegakernelDeviceKey, CudaMegakernelPlanCache,
    };
    use crate::megakernel_scheduler::CudaMegakernelScheduleSample;
    use vyre_driver::megakernel_execution::FrontierTopology;
    use vyre_driver::megakernel_frontier::MegakernelFrontierWave;

    const WAVES: &[MegakernelFrontierWave] = &[
        MegakernelFrontierWave {
            frontier_bytes: 1_024,
            scratch_bytes: 512,
            output_bytes: 256,
        },
        MegakernelFrontierWave {
            frontier_bytes: 2_048,
            scratch_bytes: 1_024,
            output_bytes: 512,
        },
    ];

    fn device(supports_grid_sync: bool) -> CudaMegakernelDeviceKey {
        CudaMegakernelDeviceKey {
            sm_major: 12,
            sm_minor: 0,
            warp_size: 32,
            supports_grid_sync,
            supports_tensor_cores: true,
            max_workgroup_size: 1024,
        }
    }

    fn plan(
        cache: &mut CudaMegakernelPlanCache,
        device: CudaMegakernelDeviceKey,
        frontier_density: f64,
    ) -> FrontierTopology {
        plan_cuda_frontier_megakernel_execution(
            cache,
            42,
            CudaMegakernelAnalysisKind::ParserFrontend,
            device,
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 1_000.0,
                frontier_density,
                readback_bytes: 1 << 20,
            },
            vyre_driver::megakernel_execution::MegakernelGraphShape {
                node_count: 1_000,
                edge_count: 4_000,
            },
            16,
            8,
            WAVES,
            &[],
            128 * 1024,
            250.0,
            0.95,
        )
        .expect("Fix: frontier execution plan should fit the budget.")
        .execution
        .topology
    }

    #[test]
    fn equivalent_pressure_reuses_the_device_local_cached_topology() {
        let mut cache = CudaMegakernelPlanCache::new();

        assert_eq!(
            plan(&mut cache, device(true), 0.90),
            FrontierTopology::FusedWave
        );
        assert_eq!(
            plan(&mut cache, device(true), 0.91),
            FrontierTopology::FusedWave
        );
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn a_device_without_grid_sync_never_gets_a_fused_wave() {
        let mut cache = CudaMegakernelPlanCache::new();

        assert_ne!(
            plan(&mut cache, device(false), 0.90),
            FrontierTopology::FusedWave,
            "Fix: a fused wave crosses wave boundaries inside one launch and needs a \
             device-wide barrier; a device without cooperative grid sync cannot run it."
        );
    }
}
