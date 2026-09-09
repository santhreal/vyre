//! Causal receipt and critical-path inspection report (Row 117).

use std::collections::BTreeMap;
use vyre_foundation::causal::CausalReceipt;

/// Structured inspection report of a causal receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalReceiptReport {
    /// Global trace identifier.
    pub trace_id: String,
    /// Total wall duration in milliseconds.
    pub total_wall_ms: f64,
    /// Total device execution duration in milliseconds.
    pub total_device_ms: f64,
    /// Wall duration per compiler lifecycle phase in milliseconds.
    pub phase_durations_ms: BTreeMap<String, f64>,
    /// Number of critical path spans.
    pub critical_path_span_count: usize,
    /// Number of counterfactual schedule decisions recorded.
    pub counterfactual_decision_count: usize,
    /// Human-readable critical path summary.
    pub critical_path_stages: Vec<String>,
}

impl CausalReceiptReport {
    /// Generate a summary report from a causal receipt.
    pub fn from_receipt(receipt: &CausalReceipt) -> Self {
        let mut phase_durations_ms: BTreeMap<String, f64> = BTreeMap::new();

        for event in &receipt.events {
            let entry = phase_durations_ms
                .entry(event.phase.name().to_string())
                .or_default();
            *entry += (event.wall_time_ns as f64) / 1_000_000.0;
        }

        let mut critical_path_stages = Vec::new();
        for &span_id in &receipt.critical_path {
            if let Some(event) = receipt.events.iter().find(|e| e.span_id == span_id) {
                critical_path_stages.push(format!(
                    "[{}] {} ({:.3} ms)",
                    event.phase.name(),
                    event.stage_name,
                    (event.wall_time_ns as f64) / 1_000_000.0
                ));
            }
        }

        Self {
            trace_id: format!("{}", receipt.trace_id),
            total_wall_ms: (receipt.total_wall_time_ns as f64) / 1_000_000.0,
            total_device_ms: (receipt.total_device_time_ns as f64) / 1_000_000.0,
            phase_durations_ms,
            critical_path_span_count: receipt.critical_path.len(),
            counterfactual_decision_count: receipt.counterfactual_summary.len(),
            critical_path_stages,
        }
    }
}
