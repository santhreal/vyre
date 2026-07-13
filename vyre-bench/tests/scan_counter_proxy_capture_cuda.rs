//! W3-6 scan-counter evidence: the proxies are SOURCED, not just schema-declared.
//!
//! `SCAN_COUNTER_EVIDENCE.toml` requires each backend to supply `memory_bytes`,
//! `occupancy_proxy`, `candidate_count`, and (for cuda) `branch_divergence_proxy`,
//! OR a stated `unavailable_reason`. The precise Nsight-Compute counters
//! (`dram__bytes`, `sm__throughput`) are admin-only on this host
//! (`RmProfilingAdminOnly=1`), so they are legitimately unavailable to a non-root
//! run. This test proves the RUNTIME-TELEMETRY PROXIES for those counters are
//! real and capturable from a live CUDA scan, an honest, non-root counter source
//!, so the evidence is not vacuous schema. It runs on the real GPU; skips cleanly
//! when no CUDA device is present.

use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};
use vyre_libs::scan::GpuLiteralSet;

/// The four scan-counter proxies captured from one real CUDA scan, in the units
/// the evidence table records. Mirrors the `required_counters` of the cuda backend
/// row in `SCAN_COUNTER_EVIDENCE.toml`.
struct ScanCounterProxies {
    /// `memory_bytes` proxy: host<->device bytes actually moved for the scan.
    memory_bytes: u64,
    /// `occupancy_proxy`: mean driver-measured achieved occupancy (basis points).
    occupancy_bps: u32,
    /// `branch_divergence_proxy`: empty scheduled thread slots (basis points), a
    /// higher value means more threads carried no logical element, a divergence/
    /// tail-effect proxy.
    branch_divergence_bps: u32,
    /// `candidate_count`: positioned matches the scan produced.
    candidate_count: u64,
}

#[test]
fn cuda_scan_sources_all_scan_counter_proxies_from_runtime_telemetry() {
    // `CudaBackendRegistration` is the `VyreBackend` wrapper a scan runs on; it
    // also exposes the CUDA telemetry snapshot the proxies come from.
    let backend = match CudaBackend::acquire() {
        Ok(backend) => CudaBackendRegistration::new(backend),
        Err(error) => {
            eprintln!("no CUDA backend ({error}); skipping scan-counter proxy capture");
            return;
        }
    };

    // A corpus with a known number of matches so `candidate_count` is checkable
    // against a real value, not just non-empty.
    let patterns: &[&[u8]] = &[b"AKIA", b"secret", b"token"];
    let matcher = GpuLiteralSet::compile(patterns);
    // 5 planted matches: AKIA x1, secret x2, token x2.
    let haystack: &[u8] =
        b"prefix AKIA gap secret middle token more secret trailing token end filler bytes here";
    let expected_matches = 5u64;

    // Fresh telemetry epoch so the proxies reflect exactly this scan.
    backend.reset_telemetry();
    let matches = matcher
        .scan_all(&backend, haystack)
        .expect("CUDA literal-set scan must succeed");

    let snapshot = backend.telemetry_snapshot();
    let proxies = ScanCounterProxies {
        memory_bytes: snapshot.host_to_device_bytes + snapshot.device_to_host_bytes,
        occupancy_bps: snapshot.mean_occupancy_bps(),
        branch_divergence_bps: snapshot.logical_thread_waste_bps,
        candidate_count: matches.len() as u64,
    };

    println!(
        "cuda scan-counter proxies: memory_bytes={} occupancy_bps={} branch_divergence_bps={} candidate_count={}",
        proxies.memory_bytes,
        proxies.occupancy_bps,
        proxies.branch_divergence_bps,
        proxies.candidate_count
    );

    // candidate_count: the real match count, asserted against the planted total.
    assert_eq!(
        proxies.candidate_count, expected_matches,
        "candidate_count proxy must equal the planted match count"
    );

    // memory_bytes: the scan moved the haystack to the device and results back, so
    // this must be at least the haystack length (a real, non-zero transfer).
    assert!(
        proxies.memory_bytes >= haystack.len() as u64,
        "memory_bytes proxy ({}) must cover at least the {}-byte haystack transfer",
        proxies.memory_bytes,
        haystack.len()
    );

    // occupancy_proxy: the scan launched at least one kernel whose occupancy was
    // measured, so the mean is a real fraction in (0, 10000] bps.
    assert!(
        snapshot.occupancy_measured_launches > 0,
        "the scan must have measured occupancy on at least one kernel launch"
    );
    assert!(
        proxies.occupancy_bps > 0 && proxies.occupancy_bps <= 10_000,
        "occupancy_proxy ({}) must be a real fraction in (0, 10000] bps",
        proxies.occupancy_bps
    );

    // branch_divergence_proxy: a basis-point value (0..=10000). Zero is legitimate
    // (a perfectly packed launch), so the contract is the valid range, not >0.
    assert!(
        proxies.branch_divergence_bps <= 10_000,
        "branch_divergence_proxy ({}) must be a basis-point value",
        proxies.branch_divergence_bps
    );
}
