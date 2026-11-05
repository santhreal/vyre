//! Opt-in stderr timing traces for proof pairs, per-backend passes, and whole runs.

fn proof_timing_enabled() -> bool {
    std::env::var("VYRE_CONFORM_PROOF_TIMING")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

fn proof_millis(elapsed: std::time::Duration) -> u128 {
    elapsed.as_millis()
}

fn proof_pair_timing_threshold_ms() -> u128 {
    std::env::var("VYRE_CONFORM_PROOF_PAIR_TIMING_MS")
        .ok()
        .and_then(|value| value.parse::<u128>().ok())
        .unwrap_or(250)
}

fn proof_pair_start_timing_enabled() -> bool {
    std::env::var("VYRE_CONFORM_PROOF_PAIR_START")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

pub(crate) fn emit_pair_proof_start(backend_id: &str, op_id: &str) {
    if proof_timing_enabled() && proof_pair_start_timing_enabled() {
        eprintln!("vyre-conform proof pair start: backend={backend_id} op={op_id}");
    }
}

pub(crate) fn emit_pair_proof_timing(
    backend_id: &str,
    op_id: &str,
    passed: bool,
    elapsed: std::time::Duration,
) {
    if !proof_timing_enabled() {
        return;
    }
    let elapsed_ms = proof_millis(elapsed);
    if elapsed_ms >= proof_pair_timing_threshold_ms() {
        eprintln!(
            "vyre-conform proof pair timing: backend={backend_id} op={op_id} passed={passed} elapsed_ms={elapsed_ms}"
        );
    }
}

pub(crate) fn emit_backend_proof_timing(
    backend_id: &str,
    pair_count: usize,
    worker_count: usize,
    elapsed: std::time::Duration,
) {
    if proof_timing_enabled() {
        eprintln!(
            "vyre-conform proof backend timing: backend={backend_id} pairs={pair_count} workers={worker_count} elapsed_ms={}",
            proof_millis(elapsed)
        );
    }
}

pub(crate) struct ProofTimingReport<'a> {
    pub(crate) out: &'a str,
    pub(crate) backend_count: usize,
    pub(crate) selected_op_count: usize,
    pub(crate) prepared_op_count: usize,
    pub(crate) pair_count: usize,
    pub(crate) worker_count: usize,
    pub(crate) prepare_elapsed: std::time::Duration,
    pub(crate) backend_elapsed: std::time::Duration,
    pub(crate) signing_elapsed: std::time::Duration,
    pub(crate) total_elapsed: std::time::Duration,
}

pub(crate) fn emit_proof_timing(report: ProofTimingReport<'_>) {
    if proof_timing_enabled() {
        eprintln!(
            "vyre-conform proof timing: out={} backends={} selected_ops={} prepared_ops={} pairs={} workers={} prepare_ms={} backend_ms={} signing_ms={} total_ms={}",
            report.out,
            report.backend_count,
            report.selected_op_count,
            report.prepared_op_count,
            report.pair_count,
            report.worker_count,
            proof_millis(report.prepare_elapsed),
            proof_millis(report.backend_elapsed),
            proof_millis(report.signing_elapsed),
            proof_millis(report.total_elapsed),
        );
    }
}
