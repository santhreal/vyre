//! How a release case dispatches its resident batch and reports it.
//!
//! Three release cases hand-rolled the same sixty-line dispatch: reset the
//! output counter across every set, infer the grid, submit the batch, prove every
//! row agrees with row zero, and fall back to a single dispatch when the backend
//! has no resident pools. The copies disagreed on the reset byte count: one
//! measured the payload it uploaded, two carried a constant beside it that could
//! drift from the payload without anything noticing. The measured length is taken
//! here, so the constant has nowhere left to live.

use crate::api::case::{dispatch_config_with_inferred_grid, BenchContext, BenchError};
use crate::api::metric::MetricPoint;
use crate::api::resident::ResidentInputPool;
use vyre::ir::Program;

/// A single non-batched dispatch: what a case falls back to when the backend has
/// no resident pool, or when the pool refuses the batch.
pub(super) struct SingleSample {
    pub(super) timed: vyre_driver::TimedDispatchResult,
    pub(super) resident_used: bool,
    pub(super) reset_bytes: u64,
}

/// One dispatch of a release workload, whether it came from the batch or the
/// single fallback. `batch_wall_ns` and `batch_len` are present only for a batch.
pub(super) struct BatchSample {
    pub(super) timed: vyre_driver::TimedDispatchResult,
    pub(super) resident_used: bool,
    pub(super) reset_bytes: u64,
    pub(super) batch_wall_ns: Option<u64>,
    pub(super) batch_len: Option<u64>,
}

/// What a release case calls its resident batch, and how it clears the output
/// resource before each dispatch.
pub(super) struct BatchPlan<'a> {
    /// The prose the case names itself with in errors and metric points.
    pub(super) label: &'a str,
    pub(super) batch_size: usize,
    /// Index of the output resource the reset payload is written to.
    pub(super) reset_resource: usize,
    /// What that resource holds, for the upload's diagnostic name: a counter, a
    /// frontier, whatever the case's program declares.
    pub(super) reset_resource_kind: &'a str,
    /// The payload that clears the output resource. Its length is the reported
    /// reset byte count.
    pub(super) reset_payload: &'a [u8],
    /// The config the batch dispatches under. A case that must pin its grid
    /// sets `grid_override` here; otherwise the batch infers one from `inputs`.
    pub(super) dispatch_config: &'a vyre_driver::DispatchConfig,
}

/// Dispatch the resident batch when the backend supports one, otherwise the
/// case's own single-dispatch path.
///
/// Every row of a batch runs the same program over the same resident inputs, so
/// every row must produce the same output. A row that disagrees is a device
/// defect, not a measurement, and the batch is rejected rather than averaged.
pub(super) fn dispatch_batch_or_single(
    ctx: &BenchContext,
    program: &Program,
    inputs: &[Vec<u8>],
    resident_batch: Option<&ResidentInputPool>,
    plan: &BatchPlan<'_>,
    single: impl FnOnce() -> Result<SingleSample, BenchError>,
) -> Result<BatchSample, BenchError> {
    let Some(resident_batch) = resident_batch else {
        return Ok(from_single(single()?));
    };

    resident_batch.upload_resource_to_all_sets(
        plan.reset_resource,
        plan.reset_payload,
        &format!(
            "{} resident batch {} reset",
            plan.label, plan.reset_resource_kind
        ),
    )?;
    let config = dispatch_config_with_inferred_grid(program, inputs, plan.dispatch_config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

    match resident_batch.dispatch_artifact_batch_timed(ctx, program, plan.batch_size, &config) {
        Ok(batch) => {
            if batch.outputs.len() != plan.batch_size {
                return Err(BenchError::ExecutionFailed(format!(
                    "{} resident batch returned {} output row(s), expected {}",
                    plan.label,
                    batch.outputs.len(),
                    plan.batch_size
                )));
            }
            let first_outputs = batch.outputs.first().cloned().ok_or_else(|| {
                BenchError::ExecutionFailed(format!(
                    "{} resident batch returned no output rows",
                    plan.label
                ))
            })?;
            if let Some((index, _)) = batch
                .outputs
                .iter()
                .enumerate()
                .find(|(_, outputs)| **outputs != first_outputs)
            {
                return Err(BenchError::CorrectnessViolation(format!(
                    "{} resident batch output row {index} disagreed with row 0",
                    plan.label
                )));
            }
            Ok(BatchSample {
                timed: vyre_driver::TimedDispatchResult {
                    outputs: first_outputs,
                    wall_ns: batch.per_item_wall_ns(),
                    device_ns: batch.per_item_device_ns(),
                    enqueue_ns: None,
                    wait_ns: None,
                },
                resident_used: true,
                reset_bytes: plan.reset_payload.len() as u64,
                batch_wall_ns: Some(batch.wall_ns_total),
                batch_len: Some(batch.batch_len as u64),
            })
        }
        Err(vyre_driver::BackendError::UnsupportedFeature { .. }) => Ok(from_single(single()?)),
        Err(error) => Err(BenchError::BackendFailed(error.to_string())),
    }
}

/// The single-dispatch fallback every release case shares: no reset, no resident
/// pool, straight through the benchmark backend.
pub(super) fn dispatch_single(
    ctx: &BenchContext,
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<SingleSample, BenchError> {
    let timed = ctx
        .dispatch_timed(program, inputs, &ctx.dispatch_config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    Ok(SingleSample {
        timed,
        resident_used: false,
        reset_bytes: 0,
    })
}

/// The metric points a resident batch sample publishes, under the case's prefix.
pub(super) fn batch_metric_points(prefix: &str, sample: &BatchSample) -> Vec<MetricPoint> {
    let mut points = vec![
        MetricPoint {
            name: format!("{prefix}_resident_buffers"),
            value: u64::from(sample.resident_used),
        },
        MetricPoint {
            name: format!("{prefix}_resident_reset_bytes"),
            value: sample.reset_bytes,
        },
    ];
    if let Some(wall_ns) = sample.batch_wall_ns {
        points.push(MetricPoint {
            name: format!("{prefix}_resident_batch_wall_ns"),
            value: wall_ns,
        });
    }
    if let Some(len) = sample.batch_len {
        points.push(MetricPoint {
            name: format!("{prefix}_resident_batch_len"),
            value: len,
        });
    }
    points
}

fn from_single(single: SingleSample) -> BatchSample {
    BatchSample {
        timed: single.timed,
        resident_used: single.resident_used,
        reset_bytes: single.reset_bytes,
        batch_wall_ns: None,
        batch_len: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{batch_metric_points, BatchSample};

    fn sample(batch: bool) -> BatchSample {
        BatchSample {
            timed: vyre_driver::TimedDispatchResult {
                outputs: vec![vec![0u8; 4]],
                wall_ns: 10,
                device_ns: None,
                enqueue_ns: None,
                wait_ns: None,
            },
            resident_used: batch,
            reset_bytes: if batch { 4 } else { 0 },
            batch_wall_ns: batch.then_some(160),
            batch_len: batch.then_some(16),
        }
    }

    /// A batch sample publishes all four points under the case's own prefix, so
    /// two release cases in one report cannot collide.
    #[test]
    fn batch_publishes_all_four_points_under_the_prefix() {
        let points = batch_metric_points("sparse", &sample(true));
        let names: Vec<&str> = points.iter().map(|point| point.name.as_str()).collect();

        assert_eq!(
            names,
            vec![
                "sparse_resident_buffers",
                "sparse_resident_reset_bytes",
                "sparse_resident_batch_wall_ns",
                "sparse_resident_batch_len",
            ]
        );
        assert_eq!(points[0].value, 1);
        assert_eq!(points[1].value, 4);
        assert_eq!(points[2].value, 160);
        assert_eq!(points[3].value, 16);
    }

    /// A fallback sample publishes only the two points it has evidence for. A
    /// zero batch length would read as a measured empty batch.
    #[test]
    fn fallback_omits_the_batch_points() {
        let points = batch_metric_points("metadata", &sample(false));
        let names: Vec<&str> = points.iter().map(|point| point.name.as_str()).collect();

        assert_eq!(
            names,
            vec!["metadata_resident_buffers", "metadata_resident_reset_bytes"]
        );
        assert_eq!(points[0].value, 0);
        assert_eq!(points[1].value, 0);
    }
}
