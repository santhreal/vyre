//! A FIXED synthetic device envelope for context-free estimator tests.
//!
//! This is a test fixture, not a description of any real GPU. It exists so
//! occupancy, autotune, planner, and megakernel-cache arithmetic can be exercised
//! without opening a CUDA context, and its values are held CONSTANT on purpose:
//! tests that pin `2048 / 256 = 8` are checking the estimator's division, not a
//! hardware fact, and they must not churn when the hardware under the desk
//! changes.
//!
//! Its numbers are deliberately NOT this machine's. Several of them differ from
//! the local RTX 5090, measured with `cuDeviceGetAttribute`: this envelope says
//! 2048 threads per SM where the device reports 1536, 256 KiB of shared memory
//! per SM where the device reports 100 KiB, and 128 KiB of shared memory per
//! block where the device reports 48 KiB (with an opt-in maximum of 99 KiB, so
//! the envelope's figure is unreachable at any setting). Every one of those
//! OVERSTATES the hardware, so an occupancy figure derived from this fixture is
//! optimistic.
//!
//! Therefore: NEVER derive a real-hardware decision from this. No shipping path
//! may read it, and none does; probe [`crate::device::CudaDeviceCaps`] from a live
//! context instead. Treating a fixed fixture as a hardware source of truth is how
//! a number that reads as measured turns out not to be.

use crate::device::CudaDeviceCaps;

/// Default synthetic VRAM for the fixed envelope.
pub const SYNTHETIC_SM120_DEFAULT_MEMORY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Construct the fixed synthetic SM_120 envelope.
///
/// The caller supplies total memory so tests can exercise both high-VRAM planning
/// and low-VRAM pressure behavior without duplicating the rest of the envelope.
/// The remaining fields are fixed; see the module doc for why they are not this
/// machine's values and must not be used as if they were.
#[must_use]
pub fn synthetic_sm120_envelope(total_memory: u64) -> CudaDeviceCaps {
    CudaDeviceCaps {
        // Deliberately not a real product name: this envelope is a fixture and
        // must not be mistaken for a probe of the local device.
        name: "synthetic sm_120 envelope (test fixture, not real hardware)".to_string(),
        ordinal: 0,
        compute_capability: (12, 0),
        total_memory,
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [i32::MAX, 65_535, 65_535],
        shared_memory_per_block: 128 * 1024,
        shared_memory_per_sm: 256 * 1024,
        warp_size: 32,
        cooperative_launch: true,
        concurrent_kernels: true,
        async_engine_count: 2,
        multi_processor_count: 170,
        l2_cache_bytes: 96 * 1024 * 1024,
        memory_clock_rate_khz: 14_000_000,
        core_clock_rate_khz: 2_410_000,
        global_memory_bus_width_bits: 512,
        max_registers_per_block: 65_536,
        max_registers_per_sm: 65_536,
        max_threads_per_sm: 2048,
        // Fixed like the rest of the envelope, and chosen so the cooperative
        // block-cap clamp is exercised without a CUDA context: at width 32 the
        // thread budget alone would allow 2048/32 = 64 blocks per SM and this
        // holds it to 24. Real devices report the cap through
        // CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR; this is not a probe.
        max_blocks_per_sm: 24,
    }
}

/// Construct the canonical fixed synthetic SM_120 envelope.
#[must_use]
pub fn synthetic_sm120_envelope_default() -> CudaDeviceCaps {
    synthetic_sm120_envelope(SYNTHETIC_SM120_DEFAULT_MEMORY_BYTES)
}

#[cfg(test)]
mod tests {
    use super::{synthetic_sm120_envelope, synthetic_sm120_envelope_default};

    #[test]
    fn synthetic_envelope_preserves_architecture_fields() {
        let caps = synthetic_sm120_envelope_default();

        assert_eq!(caps.compute_capability, (12, 0));
        assert_eq!(caps.warp_size, 32);
        assert_eq!(caps.multi_processor_count, 170);
        assert_eq!(caps.shared_memory_per_block, 128 * 1024);
        assert_eq!(caps.shared_memory_per_sm, 256 * 1024);
        assert_eq!(caps.l2_cache_bytes, 96 * 1024 * 1024);
        assert!(caps.cooperative_launch);
        assert!(caps.concurrent_kernels);
    }

    #[test]
    fn synthetic_envelope_peak_compute_matches_scheduler_issue_model() {
        let caps = synthetic_sm120_envelope_default();
        // SM_count × 4 warp schedulers × warp_size × core_clock_hz.
        let expected = 170u64 * 4 * 32 * 2_410_000 * 1_000;
        assert_eq!(
            caps.peak_compute_ops_per_sec(),
            expected,
            "peak compute must follow the universal 4-scheduler issue model exactly"
        );
        // Sanity bound on the envelope's own arithmetic (about 52 TOPS at these
        // fixed clocks), not a claim about any real part.
        let tops = caps.peak_compute_ops_per_sec() as f64 / 1e12;
        assert!(
            (40.0..80.0).contains(&tops),
            "peak int throughput {tops:.1} TOPS is outside the range this envelope's fixed clocks \
             and SM count can produce"
        );
    }

    #[test]
    fn synthetic_envelope_keeps_memory_pressure_parametric() {
        let low_vram = synthetic_sm120_envelope(512 * 1024 * 1024);
        let high_vram = synthetic_sm120_envelope_default();

        assert_eq!(low_vram.total_memory, 512 * 1024 * 1024);
        assert_eq!(high_vram.total_memory, 32 * 1024 * 1024 * 1024);
        assert_eq!(low_vram.compute_capability, high_vram.compute_capability);
        assert_eq!(
            low_vram.max_threads_per_block,
            high_vram.max_threads_per_block
        );
    }
}
