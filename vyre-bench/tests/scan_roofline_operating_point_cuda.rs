//! W3-6 roofline OPERATING POINT, the scan kernel's measured position on the roofline,
//! without Nsight-Compute (admin-only here).
//!
//! The roofline model (`scan_roofline_model_cuda`) has both ceilings + the ridge; the
//! achieved memory-axis point is timing-sourced. The remaining coordinate is the scan's
//! operational INTENSITY (ops/byte), normally read from ncu `sm__inst_executed`. It is
//! instead measured HONESTLY and non-root here: the reference interpreter executes the
//! SAME literal-scan IR with the SAME data-dependent control flow the GPU does, so the
//! arithmetic IR-op count it reports (`vyre_reference::count_ops`) for the scan over a
//! haystack equals the GPU's dynamic IR-op count for that haystack. Intensity =
//! IR-ops / bytes; combined with the GPU's measured achieved bandwidth it places the
//! scan on the roofline and confirms the bound.
//!
//! Granularity note (no overclaiming): this is a vyre-**IR** operation count, coarser
//! than hardware SASS instructions. The ncu-SASS dynamic count is the finer refinement
//! and remains the only root-gated piece; the IR-level operating point is complete and
//! honest here. Runs on the real GPU; skips with none.

use vyre_driver_cuda::{CudaBackend, CudaBackendRegistration};
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

fn planted_haystack(bytes: usize) -> Vec<u8> {
    let mut haystack = vec![b'.'; bytes];
    for chunk in haystack.chunks_mut(1024) {
        if chunk.len() >= 6 {
            chunk[0..6].copy_from_slice(b"secret");
        }
    }
    haystack
}

#[test]
fn scan_roofline_operating_point_is_memory_bound_under_both_ceilings() {
    let backend = match CudaBackend::acquire() {
        Ok(backend) => CudaBackendRegistration::new(backend),
        Err(error) => {
            eprintln!("no CUDA backend ({error}); skipping roofline operating-point measurement");
            return;
        }
    };
    let caps = &backend.inner().caps;
    let peak_gbps = u64::from(caps.memory_bandwidth_gbps());
    let peak_compute_ops = caps.peak_compute_ops_per_sec();
    assert!(
        peak_gbps > 0 && peak_compute_ops > 0,
        "device must report both ceilings"
    );

    let matcher = GpuLiteralSet::compile(&[b"secret".as_slice()]);

    // --- Operational INTENSITY (IR-ops / byte), measured via the interpreter ---
    // A modest haystack keeps the (slow) interpreter fast; ops/byte is size-independent.
    const INTENSITY_BYTES: usize = 8 * 1024;
    let intensity_haystack = planted_haystack(INTENSITY_BYTES);
    let (scan_result, ir_ops) =
        vyre_reference::count_ops(|| matcher.scan_all(&CpuRefBackend, &intensity_haystack));
    let cpu_matches = scan_result.expect("reference scan_all must succeed for op counting");
    assert!(
        !cpu_matches.is_empty(),
        "the reference scan must find planted matches (non-vacuous work)"
    );
    assert!(
        ir_ops > 0,
        "the literal scan must execute arithmetic IR ops; got zero, op counting is not wired"
    );
    // ops per byte, scaled ×1000 for integer reporting.
    let intensity_milli_ops_per_byte = ir_ops.saturating_mul(1000) / INTENSITY_BYTES as u64;
    let intensity_ops_per_byte = ir_ops as f64 / INTENSITY_BYTES as f64;

    // --- Achieved BANDWIDTH (memory-axis point), measured on the GPU ---
    const BANDWIDTH_BYTES: usize = 32 * 1024 * 1024;
    let gpu_haystack = planted_haystack(BANDWIDTH_BYTES);
    let mut gpu_matches: Vec<Match> = Vec::new();
    // Warm, then timed.
    matcher
        .scan_all_into(&backend, &gpu_haystack, &mut gpu_matches)
        .expect("warm GPU scan");
    let timed = matcher
        .scan_all_timed(&backend, &gpu_haystack, &mut gpu_matches)
        .expect("timed GPU scan");
    let Some(device_ns) = timed.timed.device_ns else {
        panic!("the CUDA backend must report device time for the roofline operating point");
    };
    assert!(device_ns > 0, "device time must be a real measurement");
    let achieved_gbps = BANDWIDTH_BYTES as u64 / device_ns; // 1 byte/ns == 1 GB/s
    assert!(achieved_gbps > 0, "achieved bandwidth must be positive");
    let achieved_bytes_per_sec = achieved_gbps as f64 * 1e9;

    // --- The operating point (intensity, achieved compute throughput) ---
    // achieved_compute = intensity[ops/byte] × achieved_bandwidth[bytes/sec].
    let achieved_compute_ops_per_sec = intensity_ops_per_byte * achieved_bytes_per_sec;
    // Ridge intensity (ops/byte) where the two ceilings meet.
    let ridge_ops_per_byte = peak_compute_ops as f64 / (peak_gbps as f64 * 1e9);

    println!(
        "scan roofline operating point (IR-op count via interpreter, bandwidth via timing; no ncu): \
         intensity={intensity_ops_per_byte:.3} IR-ops/byte ({intensity_milli_ops_per_byte} milli-ops/byte, {ir_ops} ops over {INTENSITY_BYTES}B) | \
         ridge={ridge_ops_per_byte:.2} ops/byte | achieved_bandwidth={achieved_gbps} GB/s | \
         achieved_compute={:.2} G-IR-ops/s | peak_compute={} TOPS",
        achieved_compute_ops_per_sec / 1e9,
        peak_compute_ops / 1_000_000_000_000
    );

    // The scan lives on the MEMORY-BOUND side of the roofline: its intensity is far below
    // the ridge (a byte-scan does O(1) ops per byte, the ridge is tens of ops/byte).
    assert!(
        intensity_ops_per_byte < ridge_ops_per_byte,
        "a literal byte-scan must be left of the ridge (memory-bound): intensity {intensity_ops_per_byte:.3} >= ridge {ridge_ops_per_byte:.2} ops/byte"
    );
    // The operating point sits under the compute ceiling (with the same L2-over-DRAM
    // allowance the bandwidth test uses, since achieved bandwidth can exceed DRAM peak).
    assert!(
        achieved_compute_ops_per_sec <= peak_compute_ops as f64 * 8.0,
        "achieved compute throughput {achieved_compute_ops_per_sec:.0} exceeds 8x the peak compute ceiling {peak_compute_ops}, the operating point is malformed"
    );
    // Non-vacuous: the GPU scan found the planted matches too.
    assert!(
        !gpu_matches.is_empty(),
        "the GPU scan must find planted matches"
    );
}
