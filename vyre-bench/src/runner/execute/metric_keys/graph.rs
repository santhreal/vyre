pub(super) fn custom_graph_metric_key(prefix: &'static str, name: &str) -> Option<&'static str> {
    match (prefix, name) {
        ("", "graph_csr_nodes") => Some("graph_csr_nodes"),
        ("", "graph_csr_edges") => Some("graph_csr_edges"),
        ("", "graph_csr_frontier_words") => Some("graph_csr_frontier_words"),
        ("", "graph_csr_active_sources") => Some("graph_csr_active_sources"),
        ("", "graph_csr_allowed_edges") => Some("graph_csr_allowed_edges"),
        ("", "graph_csr_output_words_set") => Some("graph_csr_output_words_set"),
        ("", "graph_csr_max_degree") => Some("graph_csr_max_degree"),
        ("", "graph_csr_high_degree_sources") => Some("graph_csr_high_degree_sources"),
        ("", "graph_csr_resident_buffers") => Some("graph_csr_resident_buffers"),
        ("", "graph_csr_workgroup_size_x") => Some("graph_csr_workgroup_size_x"),
        ("", "graph_csr_skewed_speedup_x1000") => Some("graph_csr_skewed_speedup_x1000"),
        ("", "graph_csr_queue_materializer") => Some("graph_csr_queue_materializer"),
        ("", "graph_csr_queue_capacity") => Some("graph_csr_queue_capacity"),
        ("", "graph_csr_queue_row_strided") => Some("graph_csr_queue_row_strided"),
        ("", "graph_csr_queue_lane_reduction_x1000") => {
            Some("graph_csr_queue_lane_reduction_x1000")
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::custom_graph_metric_key;

    #[test]
    fn custom_graph_metric_key_keeps_queue_materialization_visible() {
        for name in [
            "graph_csr_nodes",
            "graph_csr_edges",
            "graph_csr_active_sources",
            "graph_csr_queue_materializer",
            "graph_csr_queue_capacity",
            "graph_csr_queue_row_strided",
            "graph_csr_queue_lane_reduction_x1000",
        ] {
            assert_eq!(custom_graph_metric_key("", name), Some(name));
        }
    }
}
