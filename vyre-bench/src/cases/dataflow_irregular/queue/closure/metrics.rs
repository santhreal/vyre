use crate::api::metric::MetricPoint;

use super::{DataflowIfdsSkewedQueueClosurePrepared, QUEUE_CLOSURE_WORKGROUP_SIZE};
use crate::cases::dataflow_irregular::closure::CLOSURE_MAX_ITERS;
use crate::cases::dataflow_irregular::metrics::{
    ifds_closure_baseline_metric_points, ifds_closure_metric_points,
};

pub(super) fn queue_closure_metric_points(
    prepared: &DataflowIfdsSkewedQueueClosurePrepared,
    wall_ns: u64,
    resident_used: bool,
) -> Vec<MetricPoint> {
    let mut metrics = ifds_closure_metric_points(
        prepared.stats,
        prepared.closure_iterations,
        prepared.closure_changed,
        prepared.baseline_wall_ns,
        wall_ns,
        resident_used,
        0,
        true,
        prepared.closure_iterations,
        CLOSURE_MAX_ITERS,
        QUEUE_CLOSURE_WORKGROUP_SIZE[0],
    );
    append_queue_closure_points(prepared, &mut metrics);
    metrics
}

pub(super) fn queue_closure_baseline_metric_points(
    prepared: &DataflowIfdsSkewedQueueClosurePrepared,
) -> Vec<MetricPoint> {
    let mut metrics = ifds_closure_baseline_metric_points(
        prepared.stats,
        prepared.closure_iterations,
        prepared.closure_changed,
        prepared.closure_iterations,
        CLOSURE_MAX_ITERS,
    );
    append_queue_closure_points(prepared, &mut metrics);
    metrics
}

fn append_queue_closure_points(
    prepared: &DataflowIfdsSkewedQueueClosurePrepared,
    metrics: &mut Vec<MetricPoint>,
) {
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_queue_capacity".to_string(),
        value: u64::from(prepared.queue_capacity),
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_seed_queue_len".to_string(),
        value: u64::from(prepared.seed_queue_len),
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_dispatch_count".to_string(),
        value: u64::from(1 + prepared.closure_iterations.saturating_mul(2)),
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_total_queue_pops".to_string(),
        value: prepared.total_queue_pops,
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_max_wave_queue_len".to_string(),
        value: u64::from(prepared.max_wave_queue_len),
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_queue_delta".to_string(),
        value: 1,
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_closure_seed_scan_elided".to_string(),
        value: 1,
    });
}
