//! CUDA telemetry adapter for the scale-aware megakernel scheduler.

use vyre_driver::megakernel_execution::{
    plan_megakernel_execution, select_megakernel_topology, select_megakernel_topology_stable,
    MegakernelByteLayout, MegakernelDeviceCapabilities, MegakernelExecutionPlan,
    MegakernelExecutionSample, MegakernelExecutionTopology, MegakernelGraphShape,
    MegakernelMemoryBudget, MegakernelMemoryError, MegakernelTopologyDecision,
};
use vyre_libs::scheduling::megakernel_schedule::{
    try_schedule_via_scale_aware_samples_into, MegakernelScaleSample, MegakernelScheduleError,
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

impl MegakernelScaleSample for CudaMegakernelScheduleSample {
    fn dispatch_cost_ns(&self) -> f64 {
        self.dispatch_cost_ns
    }

    fn frontier_density(&self) -> f64 {
        self.frontier_density
    }

    fn readback_bytes(&self) -> u64 {
        self.readback_bytes
    }
}

/// Schedule megakernel fusion pressure from CUDA telemetry samples.
///
/// # Errors
///
/// Returns [`MegakernelScheduleError`] when a sample or step count is invalid.
pub fn schedule_megakernel_from_cuda_samples(
    samples: &[CudaMegakernelScheduleSample],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
) -> Result<Vec<f64>, MegakernelScheduleError> {
    let mut out = Vec::new();
    schedule_megakernel_from_cuda_samples_into(samples, launch_overhead_ns, n_steps, dt, &mut out)?;
    Ok(out)
}

/// Schedule megakernel fusion pressure into caller-owned output storage.
///
/// # Errors
///
/// Returns [`MegakernelScheduleError`] when a sample or step count is invalid.
pub fn schedule_megakernel_from_cuda_samples_into(
    samples: &[CudaMegakernelScheduleSample],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) -> Result<(), MegakernelScheduleError> {
    try_schedule_via_scale_aware_samples_into(samples, launch_overhead_ns, n_steps, dt, out)
}

#[cfg(test)]
mod tests {
    use super::{
        schedule_megakernel_from_cuda_samples, schedule_megakernel_from_cuda_samples_into,
        CudaMegakernelScheduleSample,
    };
    use crate::backend::CudaTelemetrySnapshot;
    use vyre_libs::scheduling::megakernel_schedule::MegakernelScheduleError;

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

    #[test]
    fn scheduling_reuses_caller_owned_output_capacity() {
        let samples = [
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 10.0,
                frontier_density: 0.0,
                readback_bytes: 0,
            },
            CudaMegakernelScheduleSample {
                dispatch_cost_ns: 20.0,
                frontier_density: 1.0,
                readback_bytes: 4096,
            },
        ];
        let mut out = Vec::with_capacity(4);
        let ptr = out.as_ptr();

        schedule_megakernel_from_cuda_samples_into(&samples, 5.0, 8, 0.25, &mut out)
            .expect("Fix: valid CUDA scheduler samples must schedule");

        assert_eq!(out.len(), 2);
        assert_eq!(out.as_ptr(), ptr);
        assert!(out[1] > out[0]);
    }

    #[test]
    fn scheduling_preserves_sample_validation_errors() {
        let samples = [CudaMegakernelScheduleSample {
            dispatch_cost_ns: 10.0,
            frontier_density: 1.5,
            readback_bytes: 0,
        }];

        let error = schedule_megakernel_from_cuda_samples(&samples, 0.0, 8, 0.25)
            .expect_err("invalid frontier density must be rejected");

        assert!(matches!(
            error,
            MegakernelScheduleError::InvalidFrontierDensity { index: 0, .. }
        ));
    }
}
