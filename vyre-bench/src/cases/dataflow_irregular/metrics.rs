use crate::api::metric::MetricPoint;

use super::IfdsSkewedStats;

pub(super) fn ifds_skewed_metric_points(
    stats: IfdsSkewedStats,
    baseline_wall_ns: u64,
    wall_ns: u64,
    resident_used: bool,
    workgroup_size_x: u32,
) -> Vec<MetricPoint> {
    let mut metrics = ifds_skewed_baseline_metric_points(stats);
    metrics.push(MetricPoint {
        name: "dataflow_ifds_skewed_resident_buffers".to_string(),
        value: u64::from(resident_used),
    });
    metrics.push(MetricPoint {
        name: "dataflow_ifds_skewed_workgroup_size_x".to_string(),
        value: u64::from(workgroup_size_x),
    });
    if wall_ns > 0 {
        metrics.push(MetricPoint {
            name: "dataflow_ifds_skewed_speedup_x1000".to_string(),
            value: (u128::from(baseline_wall_ns) * 1000 / u128::from(wall_ns))
                .min(u128::from(u64::MAX)) as u64,
        });
    }
    metrics
}

pub(super) fn ifds_skewed_baseline_metric_points(stats: IfdsSkewedStats) -> Vec<MetricPoint> {
    vec![
        MetricPoint {
            name: "dataflow_ifds_skewed_nodes".to_string(),
            value: u64::from(stats.nodes),
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_edges".to_string(),
            value: u64::from(stats.edges),
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_frontier_words".to_string(),
            value: u64::from(stats.frontier_words),
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_active_sources".to_string(),
            value: stats.active_sources,
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_allowed_edges".to_string(),
            value: stats.allowed_edges_from_active,
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_filtered_edges".to_string(),
            value: stats.filtered_edges_from_active,
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_output_words_set".to_string(),
            value: stats.output_words_set,
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_max_degree".to_string(),
            value: u64::from(stats.max_degree),
        },
        MetricPoint {
            name: "dataflow_ifds_skewed_high_degree_sources".to_string(),
            value: stats.high_degree_sources,
        },
    ]
}
