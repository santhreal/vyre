//! Whether a before/after metric pair proves an optimization actually fired.
//!
//! A pair only counts as a win when the after value is a real improvement
//! rather than a repeat of the before value.

use serde_json::Value;

use super::json_reader::metric_p50_f64;

pub(crate) fn benchmark_before_after_semantic_win(
    case_id: &str,
    metrics: Option<&serde_json::Map<String, Value>>,
) -> bool {
    let Some(metrics) = metrics else {
        return false;
    };
    match case_id {
        "foundation.optimizer.impact" => metric_p50_f64(metrics.get("optimizer_nodes_eliminated"))
            .is_some_and(|value| value > 0.0),
        _ => false,
    }
}
