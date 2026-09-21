//! Tests for VyreBackend trait default implementations.

use vyre_foundation::ir::Program;

use crate::backend::{sealed, BackendError, DispatchConfig, VyreBackend};

struct TelemetryBackend;

impl sealed::Sealed for TelemetryBackend {}

impl VyreBackend for TelemetryBackend {
    fn id(&self) -> &'static str {
        "telemetry-test"
    }

    fn dispatch_borrowed(
        &self,
        _program: &Program,
        _inputs: &[&[u8]],
        _config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok(vec![vec![1, 2], vec![3, 4]])
    }
}

#[test]
fn default_borrowed_into_dispatch_records_runtime_telemetry() {
    let _guard = crate::observability::audit_events_test_lock();
    let before = crate::observability::snapshot_dispatch_telemetry();
    let backend = TelemetryBackend;
    let mut outputs = vec![Vec::with_capacity(4), Vec::with_capacity(1)];

    backend
        .dispatch_borrowed_into(
            &Program::empty(),
            &[&[9, 8, 7]],
            &DispatchConfig::default(),
            &mut outputs,
        )
        .expect("Fix: default borrowed-into dispatch must succeed");

    let telemetry = crate::observability::snapshot_dispatch_telemetry();
    assert!(telemetry.launches > before.launches);
    assert!(telemetry.input_bytes >= before.input_bytes + 3);
    assert!(telemetry.output_bytes >= before.output_bytes + 4);
    assert!(telemetry.output_slots >= before.output_slots + 2);
    assert!(telemetry.output_slots_reused > before.output_slots_reused);
    assert!(telemetry.output_slots_moved > before.output_slots_moved);
    assert!(telemetry.output_slots_appended >= before.output_slots_appended);
}
