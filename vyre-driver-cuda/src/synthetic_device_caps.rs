//! A FIXED synthetic device envelope for context-free estimator tests.
//!
//! This is a test fixture, not a description of any real GPU. It exists so
//! occupancy, autotune, planner, and megakernel-cache arithmetic can be exercised
//! without opening a CUDA context, and its values are held CONSTANT on purpose:
//! tests that pin `2048 / 256 = 8` are checking the estimator's division, not a
//! hardware fact, and they must not churn when the hardware under test
//! changes.
//!
//! Its numbers are deliberately synthetic. Several of them differ from
//! physical hardware (such as an SM_120 reference device), measured with `cuDeviceGetAttribute`: this envelope says
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
/// The remaining fields are fixed; see the module doc for why they are synthetic
/// and must not be used as if they were a live probe.
#[must_use]
pub fn synthetic_sm120_envelope(total_memory: u64) -> CudaDeviceCaps {
    CudaDeviceCaps {
        // Deliberately not a real product name: this envelope is a fixture and
        // must not be mistaken for a probe of a live device.
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
