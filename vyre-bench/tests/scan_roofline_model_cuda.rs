//! W3-6 roofline MODEL, both ceilings + ridge point + the measured operating point,
//! all without Nsight-Compute (admin-only here, `RmProfilingAdminOnly=1`).
//!
//! The bandwidth axis alone (`scan_roofline_bandwidth_cuda`) places a kernel on the
//! MEMORY ceiling. A roofline needs the second ceiling too: peak COMPUTE throughput.
//! That is now honestly derivable from device caps 
//! `CudaDeviceCaps::peak_compute_ops_per_sec()` = `SM_count × 4 warp-schedulers ×
//! warp_size × core_clock`, where "4 schedulers/SM" is a universal published NVIDIA
//! constant (Volta→Blackwell), NOT a fabricated per-generation cores table. With both
//! ceilings the model has a **ridge point** (the operational intensity, ops/byte, at
//! which a kernel flips from memory-bound to compute-bound) and the scan's measured
//! bandwidth point places it on the memory side, yielding an honest BOUND verdict.
//!
//! What still needs Nsight: the scan's own achieved COMPUTE point (its executed
//! op-count, `sm__inst_executed`), so the compute-axis operating point stays
//! ncu-gated. The model (both ceilings + ridge) and the memory-axis operating point +
//! bound verdict are complete and honest here. Runs on the real GPU; skips with none.

use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

#[test]
fn resident_scan_roofline_model_has_both_ceilings_and_states_the_bound() {
    let backend = match CudaBackend::acquire() {
        Ok(backend) => CudaBackendRegistration::new(backend),
        Err(error) => {
            eprintln!("no CUDA backend ({error}); skipping roofline model measurement");
            return;
        }
    };
    let caps = &backend.inner().caps;

    // --- The two roofline ceilings, both from device caps (no ncu) ---
    let peak_gbps = u64::from(caps.memory_bandwidth_gbps());
    let peak_compute_ops = caps.peak_compute_ops_per_sec();
    assert!(peak_gbps > 0, "device must report a peak memory bandwidth");
    assert!(
        peak_compute_ops > 0,
        "device must report a peak compute throughput (SM count, warp size, core clock)"
    );

    // Peak bytes/sec from the bandwidth ceiling (1 GB/s == 1e9 bytes/s, decimal).
    let peak_bytes_per_sec = peak_gbps * 1_000_000_000;
    // Ridge point: operational intensity (ops/byte, scaled ×1000 for integer math) at
    // which the compute ceiling meets the memory ceiling. A kernel with intensity below
    // the ridge is memory-bound; above it, compute-bound.
    let ridge_ops_per_kbyte = (peak_compute_ops.saturating_mul(1_000)) / peak_bytes_per_sec;
    assert!(
        ridge_ops_per_kbyte > 0,
        "the roofline ridge intensity must be positive (both ceilings are real)"
    );

    // --- The scan's measured operating point on the MEMORY axis ---
    const HAYSTACK_BYTES: usize = 32 * 1024 * 1024;
    let mut haystack = vec![b'.'; HAYSTACK_BYTES];
    for chunk in haystack.chunks_mut(4 * 1024 * 1024) {
        if chunk.len() >= 6 {
            chunk[0..6].copy_from_slice(b"secret");
        }
    }
    let region_starts = [0u32];
    let matcher = GpuLiteralSet::compile(&[b"secret".as_slice()]);
    let max_matches = 4_096u32;

    let session = matcher
        .prepare_resident_fused_scan(
            &backend,
            HAYSTACK_BYTES + 64,
            region_starts.len() as u32,
            max_matches,
        )
        .expect("prepare resident fused scan");

    let mut presence: Vec<u32> = Vec::new();
    let mut matches: Vec<Match> = Vec::new();
    let mut scratch: Vec<u8> = Vec::new();
    // Warm dispatch (tables resident, caches primed), then the timed measurement.
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
    assert!(device_ns > 0, "device time must be a real measurement");
    assert!(
        !matches.is_empty(),
        "the scan must find its planted matches (non-vacuous work)"
    );

    // 1 byte/ns == 1 GB/s (decimal): achieved read bandwidth is bytes/device_ns.
    let achieved_gbps = HAYSTACK_BYTES as u64 / device_ns;
    let bandwidth_util_pct = (achieved_gbps * 100) / peak_gbps;
    assert!(achieved_gbps > 0, "achieved bandwidth must be positive");

    // --- The BOUND verdict, from the measured memory-axis position ---
    // A byte-scan reads each byte a small constant number of times, so its operational
    // intensity is far below the ridge, it lives on the memory-bound side of the
    // roofline. Whether it is *bandwidth-bound* (near the memory ceiling) or has
    // headroom (launch/latency-bound) is the measured util fraction.
    let bound = if bandwidth_util_pct >= 50 {
        "memory-bandwidth-bound (near the memory ceiling)"
    } else {
        "memory-side with bandwidth headroom (launch/latency-bound, not compute-bound)"
    };
    println!(
        "scan roofline model (caps + timing, no ncu): \
         peak_memory={peak_gbps}GB/s peak_compute={}TOPS ridge={}ops/KiB | \
         achieved_bandwidth={achieved_gbps}GB/s util={bandwidth_util_pct}% -> {bound}",
        peak_compute_ops / 1_000_000_000_000,
        ridge_ops_per_kbyte
    );

    // Model self-consistency: the achieved point sits under the memory ceiling (with an
    // L2-over-DRAM allowance, same as the bandwidth test), and the compute ceiling is a
    // physically sane tens-of-TOPS figure for the device (so the roofline is well-formed).
    assert!(
        achieved_gbps <= peak_gbps.saturating_mul(8),
        "achieved read bandwidth ({achieved_gbps} GB/s) exceeds 8x the DRAM peak, timing/byte accounting is wrong, not a real roofline point"
    );
    let peak_tops = peak_compute_ops as f64 / 1e12;
    assert!(
        (1.0..1000.0).contains(&peak_tops),
        "peak compute {peak_tops:.1} TOPS is outside any sane GPU range, the compute ceiling model is wrong"
    );
}
