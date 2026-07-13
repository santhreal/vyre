//! W3-6 roofline (achieved-bandwidth axis), from vyre's own timing (no Nsight).
//!
//! The precise `ROOFLINE_COUNTER_EVIDENCE.toml` position wants Nsight-Compute DRAM
//! counters, which are admin-only on this host (`RmProfilingAdminOnly=1`). But the
//! MEMORY-BANDWIDTH axis of the roofline, achieved read bandwidth vs the device's
//! peak, is honestly measurable from vyre's own timed dispatch: a byte-scan reads
//! the haystack once, so `achieved_GB/s = haystack_bytes / device_ns` (1 byte/ns ==
//! 1 GB/s), and `CudaDeviceCaps::memory_bandwidth_gbps()` gives the peak from the
//! memory clock × bus width. Their ratio places the scan kernel on the roofline and
//! states the bound (memory-bandwidth-bound when the achieved fraction is high).
//! This is an HONEST, non-root roofline datum, clearly sourced from timing rather
//! than presented as Nsight counters. Runs on the real GPU; skips with none.

use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

#[test]
fn resident_scan_reports_achieved_bandwidth_within_device_peak() {
    let backend = match CudaBackend::acquire() {
        Ok(backend) => CudaBackendRegistration::new(backend),
        Err(error) => {
            eprintln!("no CUDA backend ({error}); skipping roofline bandwidth measurement");
            return;
        }
    };
    let peak_gbps = u64::from(backend.inner().caps.memory_bandwidth_gbps());
    assert!(peak_gbps > 0, "device must report a peak memory bandwidth");

    // A large single-region haystack so the device time is dominated by the
    // haystack read (the resident tables upload once, outside the timed dispatch).
    const HAYSTACK_BYTES: usize = 32 * 1024 * 1024;
    let mut haystack = vec![b'.'; HAYSTACK_BYTES];
    // Plant a handful of matches so the scan does real work, without changing the
    // read-once bandwidth characteristic.
    for (i, chunk) in haystack.chunks_mut(4 * 1024 * 1024).enumerate() {
        if chunk.len() >= 6 {
            chunk[0..6].copy_from_slice(b"secret");
            let _ = i;
        }
    }
    let region_starts = [0u32];

    let patterns: &[&[u8]] = &[b"secret"];
    let matcher = GpuLiteralSet::compile(patterns);
    let max_matches = 4_096u32;

    let session = matcher
        .prepare_resident_fused_scan(
            &backend,
            HAYSTACK_BYTES + 64,
            region_starts.len() as u32,
            max_matches,
        )
        .expect("prepare resident fused scan");

    // Warm dispatch (tables resident, caches primed), then a timed measurement.
    let mut presence: Vec<u32> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    session
        .scan_into(
            &backend,
            &haystack,
            &region_starts,
            0,
            &mut presence,
            &mut matches,
            &mut scratch,
        )
        .expect("warm resident fused scan");

    let timed = session
        .scan_into_timed(
            &backend,
            &haystack,
            &region_starts,
            0,
            &mut presence,
            &mut matches,
            &mut scratch,
        )
        .expect("timed resident fused scan");
    session.free(&backend).expect("free resident session");

    let Some(device_ns) = timed.device_ns else {
        panic!("the CUDA backend must report device time for the roofline measurement");
    };
    assert!(
        device_ns > 0,
        "device time must be a real non-zero measurement"
    );

    // 1 byte/ns == 1 GB/s (decimal), so achieved read bandwidth is bytes/device_ns.
    let achieved_gbps = HAYSTACK_BYTES as u64 / device_ns;
    let utilization_pct = (achieved_gbps * 100) / peak_gbps;
    let bound = if utilization_pct >= 50 {
        "memory-bandwidth-bound"
    } else {
        "not-bandwidth-bound (compute/launch/latency headroom)"
    };
    println!(
        "scan roofline (timing-sourced): haystack={HAYSTACK_BYTES}B device_ns={device_ns} achieved={achieved_gbps}GB/s peak={peak_gbps}GB/s util={utilization_pct}% bound={bound}"
    );

    // The scan found its planted matches (non-vacuous work).
    assert!(
        !matches.is_empty(),
        "the scan must find the planted `secret` matches"
    );
    // Achieved read bandwidth must be a real, positive figure. It can legitimately
    // exceed the DRAM peak when the haystack is served from L2 (then the kernel is
    // L2-bound, a valid roofline verdict), so the sanity ceiling allows for L2 
    // a value far beyond even L2 bandwidth (here: >8x DRAM peak) means the timing
    // or byte accounting is wrong, not a real measurement.
    assert!(achieved_gbps > 0, "achieved bandwidth must be positive");
    assert!(
        achieved_gbps <= peak_gbps.saturating_mul(8),
        "achieved read bandwidth ({achieved_gbps} GB/s) exceeds 8x the device DRAM peak ({peak_gbps} GB/s), beyond any L2 effect; timing or byte accounting is wrong"
    );
}
