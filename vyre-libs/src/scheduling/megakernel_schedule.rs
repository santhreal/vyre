//! Megakernel auto-scheduler via homotopy continuation.
//!
//! The dispatch-graph fusion-grouping problem is a 0/1 ILP. This substrate
//! exposes the continuous relaxation used before the discrete matroid
//! scheduler rounds ambiguous candidates: each program gets a fusion indicator
//! in `[0, 1]`, with high isolated dispatch cost receiving stronger fusion
//! pressure. The solver is deterministic, allocation-reusable, and rejects
//! malformed cost vectors through the `try_` entry points.

#[cfg(test)]
use crate::plumbing::host::scratch::try_reserve_vec_capacity;

/// Input-shape or numeric error from the homotopy megakernel scheduler.
#[derive(Debug, Clone, PartialEq)]
pub enum MegakernelScheduleError {
    /// `costs.len()` did not match `n`.
    CostLen {
        /// Required cost sample count.
        expected: usize,
        /// Received cost sample count.
        actual: usize,
    },
    /// A cost was negative, NaN, or infinite.
    InvalidCost {
        /// Invalid sample index.
        index: usize,
        /// Invalid cost value.
        value: f64,
    },
    /// `dt` was negative, NaN, or infinite.
    InvalidStep {
        /// Invalid homotopy step.
        value: f64,
    },
    /// `frontier_density.len()` did not match `n`.
    FrontierDensityLen {
        /// Required frontier-density sample count.
        expected: usize,
        /// Received frontier-density sample count.
        actual: usize,
    },
    /// A frontier-density sample was outside `[0, 1]`, NaN, or infinite.
    InvalidFrontierDensity {
        /// Invalid sample index.
        index: usize,
        /// Invalid frontier density.
        value: f64,
    },
    /// `readback_bytes.len()` did not match `n`.
    ReadbackBytesLen {
        /// Required readback sample count.
        expected: usize,
        /// Received readback sample count.
        actual: usize,
    },
    /// Launch overhead was negative, NaN, or infinite.
    InvalidLaunchOverhead {
        /// Invalid launch-overhead value.
        value: f64,
    },
    /// Caller-owned output storage could not reserve enough slots.
    OutputReserveFailed {
        /// Required output slot capacity.
        capacity: usize,
        /// Field being reserved.
        field: &'static str,
        /// Allocator error message.
        message: String,
    },
}

impl std::fmt::Display for MegakernelScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CostLen { expected, actual } => write!(
                f,
                "megakernel homotopy scheduler cost length {actual} does not match n={expected}. Fix: pass one non-negative finite dispatch cost per fusion candidate."
            ),
            Self::InvalidCost { index, value } => write!(
                f,
                "megakernel homotopy scheduler cost[{index}]={value} is not a non-negative finite value. Fix: sanitize dispatch-cost telemetry before scheduling."
            ),
            Self::InvalidStep { value } => write!(
                f,
                "megakernel homotopy scheduler dt={value} is not a non-negative finite value. Fix: choose a finite Euler step in [0, 1]."
            ),
            Self::FrontierDensityLen { expected, actual } => write!(
                f,
                "megakernel scale-aware scheduler frontier-density length {actual} does not match n={expected}. Fix: pass one density sample per fusion candidate."
            ),
            Self::InvalidFrontierDensity { index, value } => write!(
                f,
                "megakernel scale-aware scheduler frontier_density[{index}]={value} is not in [0, 1]. Fix: sanitize runtime frontier telemetry before scheduling."
            ),
            Self::ReadbackBytesLen { expected, actual } => write!(
                f,
                "megakernel scale-aware scheduler readback-bytes length {actual} does not match n={expected}. Fix: pass one readback-byte sample per fusion candidate."
            ),
            Self::InvalidLaunchOverhead { value } => write!(
                f,
                "megakernel scale-aware scheduler launch_overhead_ns={value} is not a non-negative finite value. Fix: derive launch overhead from valid backend telemetry."
            ),
            Self::OutputReserveFailed {
                capacity,
                field,
                message,
            } => write!(
                f,
                "megakernel scheduler could not reserve {capacity} {field} output slot(s): {message}. Fix: split the fusion candidate window before scheduling."
            ),
        }
    }
}

impl std::error::Error for MegakernelScheduleError {}

/// Runtime telemetry used by the scale-aware megakernel scheduler.
#[derive(Debug, Clone, Copy)]
pub struct MegakernelScaleTelemetry<'a> {
    /// Per-candidate active-frontier density in `[0, 1]`.
    pub frontier_density: &'a [f64],
    /// Per-candidate final readback byte volume.
    pub readback_bytes: &'a [u64],
    /// Backend launch overhead in nanoseconds for this dispatch class.
    pub launch_overhead_ns: f64,
}

/// One backend telemetry sample for scale-aware megakernel scheduling.
///
/// Backend adapters implement this over their native telemetry records so the
/// scheduler can consume a sample slice directly instead of staging parallel
/// cost/density/readback arrays on every scheduling call.
pub trait MegakernelScaleSample {
    /// Observed candidate dispatch cost in nanoseconds.
    fn dispatch_cost_ns(&self) -> f64;

    /// Observed active-frontier density in `[0, 1]`.
    fn frontier_density(&self) -> f64;

    /// Observed final readback byte volume.
    fn readback_bytes(&self) -> u64;
}

/// Solve a small fusion ILP by homotopy continuation. `costs[i]` is
/// the per-Program dispatch cost. Returns continuous fusion indicators in
/// `[0, 1]^n`; callers round or pass the result to the matroid scheduler for
/// the discrete scheduling decision.
#[cfg(test)]
#[must_use]
pub fn schedule_via_homotopy(costs: &[f64], n: u32, n_steps: u32, dt: f64) -> Vec<f64> {
    try_schedule_via_homotopy(costs, n, n_steps, dt).unwrap_or_default()
}

/// Solve a small fusion ILP by homotopy continuation into caller-owned storage.
///
/// # Panics
/// Panics on malformed schedule input. Callers that must recover use the `try_` twin.
#[cfg(test)]
pub fn schedule_via_homotopy_into(
    costs: &[f64],
    n: u32,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) {
    // Clearing to empty on failure silently yields an empty schedule, the
    // megakernel then runs no scheduled work with no signal (Law 10). Fail
    // loud; callers use try_schedule_via_homotopy_into.
    if let Err(error) = try_schedule_via_homotopy_into(costs, n, n_steps, dt, out) {
        panic!("vyre-pass-engine homotopy schedule generation failed: {error}");
    }
}

/// Fallible homotopy scheduler entry point.
#[cfg(test)]
pub fn try_schedule_via_homotopy(
    costs: &[f64],
    n: u32,
    n_steps: u32,
    dt: f64,
) -> Result<Vec<f64>, MegakernelScheduleError> {
    use crate::telemetry::{bump, megakernel_schedule_calls};
    bump(&megakernel_schedule_calls);
    let mut out = Vec::new();
    try_schedule_via_homotopy_into(costs, n, n_steps, dt, &mut out)?;
    Ok(out)
}

/// Fallible homotopy scheduler into caller-owned storage.
#[cfg(test)]
pub fn try_schedule_via_homotopy_into(
    costs: &[f64],
    n: u32,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) -> Result<(), MegakernelScheduleError> {
    let n = n as usize;
    validate_schedule_inputs(costs, n, dt)?;
    reserve_schedule_output(out, n, "homotopy")?;
    vyre_reference::composition_witness::schedule_via_homotopy_witness_into(
        costs, n_steps, dt, out,
    );
    Ok(())
}

/// Fallible scale-aware homotopy scheduler entry point.
#[cfg(test)]
pub fn try_schedule_via_scale_aware_telemetry(
    costs: &[f64],
    telemetry: MegakernelScaleTelemetry<'_>,
    n: u32,
    n_steps: u32,
    dt: f64,
) -> Result<Vec<f64>, MegakernelScheduleError> {
    use crate::telemetry::{bump, megakernel_schedule_calls};
    bump(&megakernel_schedule_calls);
    let mut out = Vec::new();
    try_schedule_via_scale_aware_telemetry_into(costs, telemetry, n, n_steps, dt, &mut out)?;
    Ok(out)
}

/// Fallible scale-aware homotopy scheduler into caller-owned storage.
#[cfg(test)]
pub fn try_schedule_via_scale_aware_telemetry_into(
    costs: &[f64],
    telemetry: MegakernelScaleTelemetry<'_>,
    n: u32,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) -> Result<(), MegakernelScheduleError> {
    let n = n as usize;
    validate_schedule_inputs(costs, n, dt)?;
    validate_scale_telemetry(telemetry, n)?;
    reserve_schedule_output(out, n, "scale-aware telemetry")?;
    vyre_reference::composition_witness::schedule_via_scale_aware_telemetry_witness_into(
        costs,
        telemetry.frontier_density,
        telemetry.readback_bytes,
        telemetry.launch_overhead_ns,
        n_steps,
        dt,
        out,
    );
    Ok(())
}

/// Fallible scale-aware scheduler over backend-native telemetry samples.
///
/// This is the hot-path form for runtime adapters: it validates and schedules
/// directly from the sample slice, avoiding the parallel staging vectors needed
/// by [`try_schedule_via_scale_aware_telemetry_into`].
#[cfg(test)]
pub fn try_schedule_via_scale_aware_samples_into<S>(
    samples: &[S],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) -> Result<(), MegakernelScheduleError>
where
    S: MegakernelScaleSample,
{
    let n = samples.len();
    validate_sample_schedule_inputs(samples, launch_overhead_ns, dt)?;
    reserve_schedule_output(out, n, "scale-aware samples")?;
    let witness_samples: Vec<vyre_reference::composition_witness::MegakernelScaleSampleWitness> =
        samples
            .iter()
            .map(
                |s| vyre_reference::composition_witness::MegakernelScaleSampleWitness {
                    dispatch_cost_ns: s.dispatch_cost_ns(),
                    frontier_density: s.frontier_density(),
                    readback_bytes: s.readback_bytes(),
                },
            )
            .collect();
    vyre_reference::composition_witness::schedule_via_scale_aware_samples_witness_into(
        &witness_samples,
        launch_overhead_ns,
        n_steps,
        dt,
        out,
    );
    Ok(())
}

#[cfg(test)]
fn reserve_schedule_output(
    out: &mut Vec<f64>,
    capacity: usize,
    field: &'static str,
) -> Result<(), MegakernelScheduleError> {
    try_reserve_vec_capacity(out, capacity).map_err(|message| {
        MegakernelScheduleError::OutputReserveFailed {
            capacity,
            field,
            message,
        }
    })
}

#[cfg(test)]
fn validate_schedule_inputs(
    costs: &[f64],
    n: usize,
    dt: f64,
) -> Result<(), MegakernelScheduleError> {
    if costs.len() != n {
        return Err(MegakernelScheduleError::CostLen {
            expected: n,
            actual: costs.len(),
        });
    }
    for (index, value) in costs.iter().copied().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(MegakernelScheduleError::InvalidCost { index, value });
        }
    }
    if !dt.is_finite() || dt < 0.0 {
        return Err(MegakernelScheduleError::InvalidStep { value: dt });
    }
    Ok(())
}

#[cfg(test)]
fn validate_sample_schedule_inputs<S>(
    samples: &[S],
    launch_overhead_ns: f64,
    dt: f64,
) -> Result<(), MegakernelScheduleError>
where
    S: MegakernelScaleSample,
{
    if !dt.is_finite() || dt < 0.0 {
        return Err(MegakernelScheduleError::InvalidStep { value: dt });
    }
    if !launch_overhead_ns.is_finite() || launch_overhead_ns < 0.0 {
        return Err(MegakernelScheduleError::InvalidLaunchOverhead {
            value: launch_overhead_ns,
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        let cost = sample.dispatch_cost_ns();
        if !cost.is_finite() || cost < 0.0 {
            return Err(MegakernelScheduleError::InvalidCost { index, value: cost });
        }
        let frontier_density = sample.frontier_density();
        if !frontier_density.is_finite() || !(0.0..=1.0).contains(&frontier_density) {
            return Err(MegakernelScheduleError::InvalidFrontierDensity {
                index,
                value: frontier_density,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_scale_telemetry(
    telemetry: MegakernelScaleTelemetry<'_>,
    n: usize,
) -> Result<(), MegakernelScheduleError> {
    if telemetry.frontier_density.len() != n {
        return Err(MegakernelScheduleError::FrontierDensityLen {
            expected: n,
            actual: telemetry.frontier_density.len(),
        });
    }
    for (index, value) in telemetry.frontier_density.iter().copied().enumerate() {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(MegakernelScheduleError::InvalidFrontierDensity { index, value });
        }
    }
    if telemetry.readback_bytes.len() != n {
        return Err(MegakernelScheduleError::ReadbackBytesLen {
            expected: n,
            actual: telemetry.readback_bytes.len(),
        });
    }
    if !telemetry.launch_overhead_ns.is_finite() || telemetry.launch_overhead_ns < 0.0 {
        return Err(MegakernelScheduleError::InvalidLaunchOverhead {
            value: telemetry.launch_overhead_ns,
        });
    }
    Ok(())
}

#[cfg(test)]
fn launch_dominance(launch_overhead_ns: f64, candidate_cost_ns: f64) -> f64 {
    vyre_reference::composition_witness::launch_dominance_witness(
        launch_overhead_ns,
        candidate_cost_ns,
    )
}

#[cfg(test)]
fn scale_aware_pressure(
    cost_pressure: f64,
    readback_pressure: f64,
    launch_pressure: f64,
    frontier_pressure: f64,
) -> f64 {
    vyre_reference::composition_witness::scale_aware_pressure_witness(
        cost_pressure,
        readback_pressure,
        launch_pressure,
        frontier_pressure,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-2 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn schedule_converges_toward_hard_solution() {
        let costs = vec![1.0, 2.0, 3.0];
        let result = schedule_via_homotopy(&costs, 3, 100, 0.2);
        for v in result {
            assert!((0.0..=1.0).contains(&v));
            assert!(v > 0.3);
        }
    }

    #[test]
    fn schedule_uses_cost_ordering() {
        let costs = vec![1.0, 4.0, 2.0];
        let result = schedule_via_homotopy(&costs, 3, 64, 0.25);
        assert!(result[1] > result[2]);
        assert!(result[2] > result[0]);
        assert!(result[1] > 0.7);
    }

    #[test]
    fn schedule_zero_steps_returns_easy() {
        let costs = vec![1.0, 2.0, 3.0];
        let result = schedule_via_homotopy(&costs, 3, 0, 0.1);
        for v in result {
            assert!(approx_eq(v, 0.0));
        }
    }

    #[test]
    fn schedule_zero_costs_remain_easy_solution() {
        let costs = vec![0.0, 0.0, 0.0];
        let result = schedule_via_homotopy(&costs, 3, 100, 0.5);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn schedule_rejects_bad_cost_shape() {
        let err = try_schedule_via_homotopy(&[1.0, 2.0], 3, 4, 0.1)
            .expect_err("bad cost length must fail");
        assert_eq!(
            err,
            MegakernelScheduleError::CostLen {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn schedule_rejects_non_finite_costs() {
        let err =
            try_schedule_via_homotopy(&[1.0, f64::NAN], 2, 4, 0.1).expect_err("NaN cost must fail");
        assert!(matches!(
            err,
            MegakernelScheduleError::InvalidCost { index: 1, .. }
        ));
    }

    #[test]
    fn schedule_rejects_negative_step() {
        let err = try_schedule_via_homotopy(&[1.0], 1, 4, -0.1).expect_err("negative dt must fail");
        assert_eq!(err, MegakernelScheduleError::InvalidStep { value: -0.1 });
    }

    #[test]
    fn scale_aware_scheduler_prefers_dense_frontier_when_costs_match() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[0.05, 0.95],
            readback_bytes: &[0, 0],
            launch_overhead_ns: 0.0,
        };
        let result = try_schedule_via_scale_aware_telemetry(&[10.0, 10.0], telemetry, 2, 64, 0.25)
            .expect("Fix: valid scale telemetry must schedule");
        assert!(
            result[1] > result[0],
            "dense-frontier candidate should receive stronger fusion pressure"
        );
    }

    #[test]
    fn scale_aware_scheduler_lifts_readback_heavy_candidate() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[0.0, 0.0],
            readback_bytes: &[1, 4096],
            launch_overhead_ns: 0.0,
        };
        let result = try_schedule_via_scale_aware_telemetry(&[10.0, 10.0], telemetry, 2, 64, 0.25)
            .expect("Fix: valid readback telemetry must schedule");
        assert!(
            result[1] > result[0],
            "readback-heavy candidate should receive stronger fusion pressure"
        );
    }

    #[test]
    fn scale_aware_scheduler_preserves_cost_ordering_without_runtime_pressure() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[0.0, 0.0, 0.0],
            readback_bytes: &[0, 0, 0],
            launch_overhead_ns: 0.0,
        };
        let result =
            try_schedule_via_scale_aware_telemetry(&[1.0, 4.0, 2.0], telemetry, 3, 64, 0.25)
                .expect("Fix: zero runtime pressure must still schedule by cost");
        assert!(result[1] > result[2]);
        assert!(result[2] > result[0]);
    }

    #[test]
    fn scale_aware_scheduler_rejects_bad_frontier_density() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[1.25],
            readback_bytes: &[0],
            launch_overhead_ns: 0.0,
        };
        let err = try_schedule_via_scale_aware_telemetry(&[1.0], telemetry, 1, 4, 0.1)
            .expect_err("density outside [0, 1] must fail");
        assert!(matches!(
            err,
            MegakernelScheduleError::InvalidFrontierDensity { index: 0, .. }
        ));
    }

    #[test]
    fn scale_aware_scheduler_rejects_bad_readback_shape() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[0.0, 0.0],
            readback_bytes: &[0],
            launch_overhead_ns: 0.0,
        };
        let err = try_schedule_via_scale_aware_telemetry(&[1.0, 2.0], telemetry, 2, 4, 0.1)
            .expect_err("readback length mismatch must fail");
        assert_eq!(
            err,
            MegakernelScheduleError::ReadbackBytesLen {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn scale_aware_scheduler_into_reuses_output_capacity() {
        let telemetry = MegakernelScaleTelemetry {
            frontier_density: &[0.0, 0.5, 1.0],
            readback_bytes: &[0, 16, 32],
            launch_overhead_ns: 25.0,
        };
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        try_schedule_via_scale_aware_telemetry_into(
            &[1.0, 2.0, 3.0],
            telemetry,
            3,
            8,
            0.25,
            &mut out,
        )
        .expect("Fix: valid scale telemetry must schedule into caller output");
        assert_eq!(out.len(), 3);
        assert_eq!(out.as_ptr(), ptr);
    }
}
