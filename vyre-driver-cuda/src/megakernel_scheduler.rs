//! CUDA telemetry adapter for the scale-aware megakernel scheduler.

use vyre_driver::megakernel_execution::{
    plan_megakernel_execution, select_megakernel_topology, select_megakernel_topology_stable,
    MegakernelByteLayout, MegakernelDeviceCapabilities, MegakernelExecutionPlan,
    MegakernelExecutionSample, MegakernelExecutionTopology, MegakernelGraphShape,
    MegakernelMemoryBudget, MegakernelMemoryError, MegakernelTopologyDecision,
};

use crate::backend::CudaTelemetrySnapshot;

/// Per-candidate CUDA telemetry used to bias megakernel fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaMegakernelScheduleSample {
    /// Observed candidate dispatch cost in nanoseconds.
    pub dispatch_cost_ns: f64,
    /// Observed active-frontier density in `[0, 1]`.
    pub frontier_density: f64,
    /// Observed final readback byte volume.
    pub readback_bytes: u64,
}

impl CudaMegakernelScheduleSample {
    /// Build one scheduler sample from an observed CUDA telemetry interval.
    ///
    /// `dispatch_cost_ns` is supplied by the caller because wall/device timing
    /// belongs to the benchmark or timed-dispatch boundary. Frontier density is
    /// derived from launched logical elements over scheduled CUDA thread slots,
    /// which is the runtime proxy available for arbitrary resident kernels.
    #[must_use]
    pub fn from_telemetry_snapshot(snapshot: CudaTelemetrySnapshot, dispatch_cost_ns: f64) -> Self {
        let frontier_density = f64::from(snapshot.logical_thread_utilization_bps) / 10_000.0;
        Self {
            dispatch_cost_ns,
            frontier_density,
            readback_bytes: snapshot.readback_bytes,
        }
    }

    fn execution_sample(self) -> MegakernelExecutionSample {
        MegakernelExecutionSample {
            dispatch_cost_ns: self.dispatch_cost_ns,
            frontier_density: self.frontier_density,
            readback_bytes: self.readback_bytes,
        }
    }
}

/// These entry points take no device key, so the caller has already reduced
/// `fusion_pressure` to what its device admits. The device-wide-barrier rule
/// itself lives in `vyre_driver::megakernel_execution`; the frontier entry point
/// in `megakernel_barrier_planner` passes the real device capability there.
const CALLER_GATED: MegakernelDeviceCapabilities = MegakernelDeviceCapabilities::FUSION_CAPABLE;

/// Select the CUDA megakernel execution topology for one candidate wave.
#[must_use]
pub fn select_cuda_megakernel_topology(
    sample: CudaMegakernelScheduleSample,
    graph: MegakernelGraphShape,
    memory: MegakernelMemoryBudget,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
) -> MegakernelTopologyDecision {
    select_megakernel_topology(
        sample.execution_sample(),
        graph,
        memory,
        launch_overhead_ns,
        fusion_pressure,
        CALLER_GATED,
    )
}

/// Select CUDA megakernel topology with previous-topology hysteresis.
#[must_use]
pub fn select_cuda_megakernel_topology_stable(
    sample: CudaMegakernelScheduleSample,
    graph: MegakernelGraphShape,
    memory: MegakernelMemoryBudget,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    previous_topology: MegakernelExecutionTopology,
) -> MegakernelTopologyDecision {
    select_megakernel_topology_stable(
        sample.execution_sample(),
        graph,
        memory,
        launch_overhead_ns,
        fusion_pressure,
        previous_topology,
        CALLER_GATED,
    )
}

/// Select a CUDA megakernel topology and validate its device-memory plan.
///
/// # Errors
///
/// Returns [`MegakernelMemoryError`] when byte accounting overflows or the plan
/// does not fit the approved budget.
pub fn plan_cuda_megakernel_execution(
    sample: CudaMegakernelScheduleSample,
    graph: MegakernelGraphShape,
    bytes: MegakernelByteLayout,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
) -> Result<MegakernelExecutionPlan, MegakernelMemoryError> {
    plan_megakernel_execution(
        sample.execution_sample(),
        graph,
        bytes,
        launch_overhead_ns,
        fusion_pressure,
        CALLER_GATED,
    )
}

// Inline: covers `dispatch_cost_ns`, `frontier_density`, `readback_bytes`, which no integration
// test can name.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CudaTelemetrySnapshot;
    #[test]
    fn select_cuda_megakernel_topology_selects_expected_decision() {
        let sample = CudaMegakernelScheduleSample {
            dispatch_cost_ns: 100.0,
            frontier_density: 0.8,
            readback_bytes: 1024,
        };
        let graph = MegakernelGraphShape {
            node_count: 100,
            edge_count: 200,
        };
        let memory = MegakernelMemoryBudget {
            required_bytes: 1024,
            budget_bytes: 1024 * 1024,
        };
        let decision = select_cuda_megakernel_topology(sample, graph, memory, 10.0, 1.0);
        assert_eq!(
            decision,
            MegakernelTopologyDecision {
                topology: MegakernelExecutionTopology::DenseFrontier,
                memory_pressure_bps: 9,
                average_degree_bps: 20_000,
                launch_pressure_bps: 1_000,
            }
        );
    }
    #[test]
    fn telemetry_snapshot_maps_onto_a_scheduler_sample() {
        let sample = CudaMegakernelScheduleSample::from_telemetry_snapshot(
            CudaTelemetrySnapshot {
                readback_bytes: 4096,
                logical_thread_utilization_bps: 3750,
                ..CudaTelemetrySnapshot::default()
            },
            123.0,
        );

        assert_eq!(
            sample,
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 123.0,
                frontier_density: 0.375,
                readback_bytes: 4096,
            }
        );
    }
}
