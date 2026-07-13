//! Synthetic CUDA device capability profiles for offline planning.
//!
//! These profiles are not a substitute for live CUDA probing. They give
//! planner, autotune, occupancy, and megakernel-cache tests one source of
//! truth for architecture envelopes that must be exercised without opening
//! a CUDA context.

use crate::device::CudaDeviceCaps;

/// Default synthetic VRAM for the local Blackwell release-path profile.
pub const BLACKWELL_SM120_DEFAULT_MEMORY_BYTES: u64 = 32 * 1024 * 1024 * 1024;

/// Construct a synthetic Blackwell SM_120 capability envelope.
///
/// The caller supplies total memory so tests can exercise both high-VRAM
/// release-path planning and low-VRAM pressure behavior without duplicating
/// the rest of the device envelope.
#[must_use]
pub fn blackwell_sm120_caps(total_memory: u64) -> CudaDeviceCaps {
    CudaDeviceCaps {
        name: "NVIDIA GeForce RTX 5090".to_string(),
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
    }
}

/// Construct the canonical synthetic Blackwell SM_120 release-path profile.
#[must_use]
pub fn blackwell_sm120_caps_default() -> CudaDeviceCaps {
    blackwell_sm120_caps(BLACKWELL_SM120_DEFAULT_MEMORY_BYTES)
}

#[cfg(test)]
mod tests {
    use super::{blackwell_sm120_caps, blackwell_sm120_caps_default};

    #[test]
    fn blackwell_profile_preserves_release_path_architecture_fields() {
        let caps = blackwell_sm120_caps_default();

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
    fn blackwell_profile_peak_compute_matches_scheduler_issue_model() {
        let caps = blackwell_sm120_caps_default();
        // SM_count × 4 warp schedulers × warp_size × core_clock_hz.
        let expected = 170u64 * 4 * 32 * 2_410_000 * 1_000;
        assert_eq!(
            caps.peak_compute_ops_per_sec(),
            expected,
            "peak compute must follow the universal 4-scheduler issue model exactly"
        );
        // Sanity: a Blackwell RTX 5090's peak int32 throughput is tens of TOPS
        // (≈52 TOPS here), consistent with its ~105 TFLOP32 FMA figure.
        let tops = caps.peak_compute_ops_per_sec() as f64 / 1e12;
        assert!(
            (40.0..80.0).contains(&tops),
            "peak int throughput {tops:.1} TOPS is outside the sane Blackwell range"
        );
    }

    #[test]
    fn blackwell_profile_keeps_memory_pressure_parametric() {
        let low_vram = blackwell_sm120_caps(512 * 1024 * 1024);
        let high_vram = blackwell_sm120_caps_default();

        assert_eq!(low_vram.total_memory, 512 * 1024 * 1024);
        assert_eq!(high_vram.total_memory, 32 * 1024 * 1024 * 1024);
        assert_eq!(low_vram.compute_capability, high_vram.compute_capability);
        assert_eq!(
            low_vram.max_threads_per_block,
            high_vram.max_threads_per_block
        );
    }
}
