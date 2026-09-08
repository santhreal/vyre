//! Contract tests for Causal Receipt Diagnostics & Reporting (Row 117).

use vyre_debug::CausalReceiptReport;
use vyre_foundation::causal::{CausalEvent, CausalPhase, CausalReceipt, CausalSpanId, TraceId};

#[test]
fn causal_receipt_report_formats_metrics_and_critical_path() {
    let trace_id = TraceId::new(0xABCDEF);
    let root = CausalSpanId::new(1);
    let mut receipt = CausalReceipt::new(trace_id, root);

    let mut ev1 = CausalEvent::new(root, None, CausalPhase::SemanticFrontend, "parse");
    ev1.wall_time_ns = 2_000_000; // 2.0 ms
    receipt.record_event(ev1);

    let child = CausalSpanId::new(2);
    let mut ev2 = CausalEvent::new(child, Some(root), CausalPhase::TargetLowering, "lower_ptx");
    ev2.wall_time_ns = 3_500_000; // 3.5 ms
    ev2.device_time_ns = Some(1_000_000);
    receipt.record_event(ev2);

    let _ = receipt.reconstruct_critical_path();

    let report = CausalReceiptReport::from_receipt(&receipt);
    assert_eq!(report.total_wall_ms, 5.5);
    assert_eq!(report.total_device_ms, 1.0);
    assert_eq!(report.critical_path_span_count, 2);
    assert_eq!(report.critical_path_stages.len(), 2);
    assert!(report.critical_path_stages[0].contains("parse"));
    assert!(report.critical_path_stages[1].contains("lower_ptx"));
}
