//! Contract tests for Causal Introspection & Unified Causal-Span Schema (Row 117).

use vyre_foundation::causal::*;

#[test]
fn all_causal_phases_are_derived_dynamically_and_exhaustive() {
    assert_eq!(CausalPhase::ALL.len(), 6);
    for (i, &phase) in CausalPhase::ALL.iter().enumerate() {
        assert_eq!(phase.level_tier() as usize, i);
        assert_eq!(exhaustiveness_check_causal_phase(phase) as usize, i);
        assert!(!phase.name().is_empty());
    }
}

#[test]
fn causal_receipt_records_events_and_reconstructs_critical_path() {
    let trace_id = TraceId::new(0x12345678_abcdef01_23456789_abcdef01);
    let root_span = CausalSpanId::new(1);
    let mut receipt = CausalReceipt::new(trace_id, root_span);

    // Span 1: Frontend
    let mut ev1 = CausalEvent::new(root_span, None, CausalPhase::SemanticFrontend, "ast_parse");
    ev1.wall_time_ns = 1_000_000; // 1ms
    ev1.allocations = 10;
    ev1.retained_bytes = 4096;
    receipt.record_event(ev1);

    // Span 2: Optimizer (child of 1)
    let span2 = CausalSpanId::new(2);
    let mut ev2 = CausalEvent::new(span2, Some(root_span), CausalPhase::SemanticOptimizer, "canonicalize");
    ev2.wall_time_ns = 2_000_000; // 2ms
    receipt.record_event(ev2);

    // Span 3: Lowering (child of 2)
    let span3 = CausalSpanId::new(3);
    let mut ev3 = CausalEvent::new(span3, Some(span2), CausalPhase::TargetLowering, "physical_lower");
    ev3.wall_time_ns = 5_000_000; // 5ms
    receipt.record_event(ev3);

    // Span 4: Fast branch (child of 1)
    let span4 = CausalSpanId::new(4);
    let mut ev4 = CausalEvent::new(span4, Some(root_span), CausalPhase::SemanticOptimizer, "constant_fold");
    ev4.wall_time_ns = 500_000; // 0.5ms
    receipt.record_event(ev4);

    let critical_path = receipt.reconstruct_critical_path().expect("critical path reconstruction failed");
    assert_eq!(critical_path, &[root_span, span2, span3]);
    assert_eq!(receipt.total_wall_time_ns, 8_500_000);
    assert_eq!(receipt.total_allocations, 10);
    assert_eq!(receipt.total_retained_bytes, 4096);
}

#[test]
fn counterfactual_decision_recording_and_lookup() {
    let trace_id = TraceId::new(100);
    let root_span = CausalSpanId::new(1);
    let mut receipt = CausalReceipt::new(trace_id, root_span);

    let mut event = CausalEvent::new(root_span, None, CausalPhase::MegakernelCompilation, "schedule_search");
    event.counterfactual_decision = Some(CounterfactualDecision {
        chosen_schedule: "tiled_fused_gemm".to_string(),
        chosen_cost: 42.5,
        winning_reason: "lowest shared memory bank conflicts and highest occupancy".to_string(),
        alternatives: vec![AlternativeSchedule {
            schedule_name: "naive_unfused".to_string(),
            estimated_cost: 110.0,
            rejection_reason: "excessive global memory round-trips".to_string(),
        }],
    });

    receipt.record_event(event);

    let decision = receipt.explain_decision(root_span).expect("decision must exist");
    assert_eq!(decision.chosen_schedule, "tiled_fused_gemm");
    assert_eq!(decision.alternatives.len(), 1);
    assert_eq!(decision.alternatives[0].schedule_name, "naive_unfused");
}

#[test]
fn disabled_tracer_allocates_zero_events() {
    let mut tracer = CausalTracer::disabled();
    assert!(!tracer.is_active());
    assert_eq!(tracer.mode(), CausalTraceMode::Off);

    let event = CausalEvent::new(CausalSpanId::new(1), None, CausalPhase::DriverSubmission, "queue_submit");
    tracer.record_event(event);

    let receipt = tracer.finalize();
    assert!(receipt.events.is_empty());
}

#[test]
fn causal_receipt_json_toml_roundtrip() {
    let trace_id = TraceId::new(42);
    let root_span = CausalSpanId::new(1);
    let mut receipt = CausalReceipt::new(trace_id, root_span);

    let mut event = CausalEvent::new(root_span, None, CausalPhase::RuntimeExecution, "kernel_execute");
    event.wall_time_ns = 12345;
    event.device_time_ns = Some(9876);
    event.work_units = 512;
    receipt.record_event(event);

    // JSON round-trip
    let json = receipt.to_json().expect("json serialize");
    let from_json = CausalReceipt::from_json(&json).expect("json deserialize");
    assert_eq!(receipt, from_json);

    // TOML round-trip
    let toml = receipt.to_toml().expect("toml serialize");
    let from_toml = CausalReceipt::from_toml(&toml).expect("toml deserialize");
    assert_eq!(receipt, from_toml);
}

#[test]
fn stale_causal_receipt_schema_fails_closed() {
    let mut receipt = CausalReceipt::new(TraceId::new(1), CausalSpanId::new(1));
    receipt.schema_version = 999; // Future or stale unsupported version

    let json = serde_json::to_string(&receipt).unwrap();
    let err = CausalReceipt::from_json(&json).unwrap_err();
    assert!(matches!(err, CausalError::StaleSchemaVersion { expected: 1, found: 999 }));
}
