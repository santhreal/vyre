//! Contracts for `vyre_driver_cuda::occupancy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::validation::blocks_per_compute_unit;
use vyre_driver_cuda::occupancy::{
    can_launch_concurrently, cooperative_thread_residency_block_limit, estimate_occupancy,
    pick_workgroup_size_for_occupancy, ConcurrentLaunchBlocker, ConcurrentLaunchDecision,
    KernelResourceUsage, OccupancyEstimate,
};
use vyre_driver_cuda::synthetic_device_caps::synthetic_sm120_envelope_default;

#[test]
fn estimate_zero_when_workgroup_exceeds_max_threads_per_block() {
    let caps = synthetic_sm120_envelope_default();
    let usage = KernelResourceUsage {
        regs_per_thread: 32,
        shared_bytes_per_block: 0,
    };
    let est = estimate_occupancy(&caps, usage, 4096);
    assert_eq!(est, OccupancyEstimate::ZERO);
}

#[test]
fn estimate_zero_when_register_pressure_too_high() {
    let caps = synthetic_sm120_envelope_default();
    // 256 regs/thread * 256 threads = 65_536 → fits exactly per block.
    // 256 regs/thread * 257 threads = 65_792 → busts per-block ceiling.
    let usage = KernelResourceUsage {
        regs_per_thread: 256,
        shared_bytes_per_block: 0,
    };
    let busts = estimate_occupancy(&caps, usage, 257);
    assert_eq!(busts, OccupancyEstimate::ZERO);
    let fits = estimate_occupancy(&caps, usage, 256);
    assert!(fits.is_runnable());
}

#[test]
fn estimate_zero_when_register_requirement_overflows() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.max_threads_per_block = i32::MAX;
    caps.max_threads_per_sm = i32::MAX;
    caps.max_registers_per_block = i32::MAX;
    caps.max_registers_per_sm = i32::MAX;
    let usage = KernelResourceUsage {
        regs_per_thread: u32::MAX,
        shared_bytes_per_block: 0,
    };
    let est = estimate_occupancy(&caps, usage, 2);
    assert_eq!(
        est,
        OccupancyEstimate::ZERO,
        "Fix: CUDA occupancy must reject overflowing register products instead of saturating them into plausible resource pressure."
    );
}

#[test]
fn estimate_full_occupancy_on_lightweight_kernel() {
    let caps = synthetic_sm120_envelope_default();
    // 16 regs/thread, no shared. At 256 threads → blocks-by-regs =
    // 65_536 / (16*256) = 16; blocks-by-threads = 2048/256 = 8 →
    // 8 blocks/SM. Warps/SM = 8 * 8 = 64 = max_threads_per_sm/warp =
    // 2048/32 = 64 → 100% occupancy.
    let usage = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    let est = estimate_occupancy(&caps, usage, 256);
    assert_eq!(est.blocks_per_sm, 8);
    assert_eq!(est.warps_per_sm, 64);
    assert_eq!(est.occupancy_bps, 10_000);
}

#[test]
fn picker_chooses_smaller_size_on_tie() {
    let caps = synthetic_sm120_envelope_default();
    let usage = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    // 128 and 256 both reach 100% occupancy; picker should choose 128.
    let chosen = pick_workgroup_size_for_occupancy(&caps, usage, &[128, 256, 512]);
    assert_eq!(chosen, Some(128));
}

#[test]
fn picker_returns_none_when_no_candidate_runnable() {
    let caps = synthetic_sm120_envelope_default();
    // 65_537 regs/thread per block is impossible at any block size > 0.
    let usage = KernelResourceUsage {
        regs_per_thread: 65_537,
        shared_bytes_per_block: 0,
    };
    let chosen = pick_workgroup_size_for_occupancy(&caps, usage, &[32, 64, 128]);
    assert_eq!(chosen, None);
}

#[test]
fn estimate_zero_when_shared_memory_exceeds_per_block_limit() {
    let caps = synthetic_sm120_envelope_default();
    let usage = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 256 * 1024,
    };
    let est = estimate_occupancy(&caps, usage, 64);
    assert_eq!(est, OccupancyEstimate::ZERO);
}

#[test]
fn estimate_uses_probed_per_sm_shared_memory_not_block_multiplier() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.shared_memory_per_block = 128 * 1024;
    caps.shared_memory_per_sm = 192 * 1024;
    let usage = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 96 * 1024,
    };

    let est = estimate_occupancy(&caps, usage, 256);

    assert_eq!(
        est.blocks_per_sm, 2,
        "Fix: CUDA occupancy must use probed per-SM shared memory instead of assuming a 4x per-block budget."
    );
}

#[test]
fn occupancy_bps_is_proportional_to_warps_per_sm() {
    let caps = synthetic_sm120_envelope_default();
    // High-pressure kernel: 64 regs/thread, 256 threads. Blocks/SM =
    // min(2048/256, 65536/(64*256)) = min(8, 4) = 4.
    // Warps/SM = 4 * 8 = 32. max_warps_per_sm = 64.
    // occupancy_bps = (32 * 10000) / 64 = 5000.
    let usage = KernelResourceUsage {
        regs_per_thread: 64,
        shared_bytes_per_block: 0,
    };
    let est = estimate_occupancy(&caps, usage, 256);
    assert_eq!(est.blocks_per_sm, 4);
    assert_eq!(est.warps_per_sm, 32);
    assert_eq!(est.occupancy_bps, 5_000);
}

#[test]
fn picker_prefers_higher_occupancy_over_smaller_size() {
    let caps = synthetic_sm120_envelope_default();
    // At 32 threads, 64 regs/thread → blocks_by_regs = 65536/2048 = 32,
    // blocks_by_threads = 2048/32 = 64 → 32 blocks * 1 warp = 32 warps/SM = 50%.
    // At 256 threads, 64 regs/thread → 32 warps/SM = 50% (computed above).
    // Tie → picker prefers smaller size (32).
    let usage = KernelResourceUsage {
        regs_per_thread: 64,
        shared_bytes_per_block: 0,
    };
    let chosen = pick_workgroup_size_for_occupancy(&caps, usage, &[32, 256]);
    assert_eq!(chosen, Some(32));
}

#[test]
fn cooperative_residency_limit_uses_sm_thread_ceiling() {
    let caps = synthetic_sm120_envelope_default();
    assert_eq!(
        cooperative_thread_residency_block_limit(&caps, 256),
        1_360,
        "Fix: CUDA cooperative launch preflight must reject grids larger than blocks_per_sm * sm_count before calling cuLaunchCooperativeKernel."
    );
    assert_eq!(cooperative_thread_residency_block_limit(&caps, 0), 0);
}

/// The block cap must clamp the cooperative limit at narrow widths.
///
/// # The bug this locks out
///
/// The limit was computed from the per-SM THREAD budget alone. Hardware
/// separately caps blocks per SM, and at narrow widths that cap binds first,
/// so the preflight admitted grids the driver then refused: on the local
/// device, width 32 yielded 48 blocks/SM and 8160 blocks admitted against a
/// real 24 and 4080, a factor of two. A preflight that says "fits" where
/// `cuLaunchCooperativeKernel` fails is the exact predicate/driver
/// disagreement that sharing one residency definition exists to prevent.
///
/// The envelope's 2048 threads/SM and 24-block cap make this checkable with
/// no CUDA context: 2048/32 = 64 blocks by threads, clamped to 24.
#[test]
fn cooperative_residency_limit_clamps_by_the_block_cap_at_narrow_widths() {
    let caps = synthetic_sm120_envelope_default();
    let sms = u64::from(caps.multi_processor_count_u32());
    assert_eq!(
        caps.max_blocks_per_sm_u32(),
        Some(24),
        "Fix: this envelope must report a block cap, or the clamp below is vacuous."
    );
    assert_eq!(
        cooperative_thread_residency_block_limit(&caps, 32),
        24 * sms,
        "Fix: at width 32 the thread budget allows 2048/32 = 64 blocks/SM but the hardware \
         holds 24, so the limit must be 24 per SM. Admitting 64 lets the preflight pass a grid \
         cuLaunchCooperativeKernel refuses."
    );
    assert!(
        cooperative_thread_residency_block_limit(&caps, 32)
            < u64::from(blocks_per_compute_unit(caps.max_threads_per_sm_u32(), 32)) * sms,
        "Fix: the clamped limit must be strictly below the thread-only figure at width 32, or \
         the clamp is not being applied."
    );
}

/// The clamp must not lower a limit the thread budget already bounds.
///
/// # The bug this locks out
///
/// A clamp applied as a floor, or applied unconditionally, would cut the
/// admissible grid at the widths real cooperative kernels use and silently
/// route them to the host-split path, losing the native grid barrier for no
/// reason. Only the narrow end may move.
#[test]
fn cooperative_residency_limit_is_unchanged_where_threads_bind_first() {
    let caps = synthetic_sm120_envelope_default();
    let sms = u64::from(caps.multi_processor_count_u32());
    for (width, blocks_per_sm) in [(256_u32, 8_u64), (512, 4), (1024, 2)] {
        assert_eq!(
            cooperative_thread_residency_block_limit(&caps, width),
            blocks_per_sm * sms,
            "Fix: at width {width} the thread budget yields {blocks_per_sm} blocks/SM, below \
             the 24-block cap, so the cap must not change it."
        );
    }
}

/// An unreported cap must apply no clamp.
///
/// # The bug this locks out
///
/// A driver that cannot answer the attribute reports 0. Reading 0 as a real
/// ceiling would clamp every cooperative launch to zero blocks and refuse the
/// native grid-sync route outright on that device, turning a missing
/// refinement into a total loss of the feature.
#[test]
fn an_unreported_block_cap_applies_no_clamp() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.max_blocks_per_sm = 0;
    assert_eq!(
        caps.max_blocks_per_sm_u32(),
        None,
        "Fix: 0 means unreported and must read as None, not as a zero-block ceiling."
    );
    let sms = u64::from(caps.multi_processor_count_u32());
    assert_eq!(
        cooperative_thread_residency_block_limit(&caps, 32),
        64 * sms,
        "Fix: with no reported cap the limit must stay at the thread-only figure, preserving \
         behavior on a driver that cannot answer."
    );
    assert!(
        cooperative_thread_residency_block_limit(&caps, 256) > 0,
        "Fix: an unreported cap must not refuse cooperative launches."
    );
}

/// A negative cap must be treated as unreported, not as a huge unsigned one.
///
/// # The bug this locks out
///
/// The field is `i32` because the CUDA attribute API returns `int`. Casting a
/// negative value with `as u32` would produce a cap near 4 billion, which
/// clamps nothing and hides the driver's bad answer instead of ignoring it.
#[test]
fn a_negative_block_cap_is_treated_as_unreported() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.max_blocks_per_sm = -1;
    assert_eq!(
        caps.max_blocks_per_sm_u32(),
        None,
        "Fix: a negative cap must read as unreported; `as u32` would turn -1 into 4294967295 \
         and clamp nothing."
    );
    let sms = u64::from(caps.multi_processor_count_u32());
    assert_eq!(
        cooperative_thread_residency_block_limit(&caps, 32),
        64 * sms,
        "Fix: an unusable cap must fall back to the thread-only figure."
    );
}

// ── D5: concurrent-launch decision policy tests ─────────────────

#[test]
fn co_launch_two_kernels_with_headroom_fits_concurrently() {
    let caps = synthetic_sm120_envelope_default();
    let light = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    let decision = can_launch_concurrently(&caps, light, 256, light, 256);
    assert_eq!(decision, ConcurrentLaunchDecision::Concurrent);
}

#[test]
fn co_launch_two_full_occupancy_kernels_overflows_warp_cap() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.max_threads_per_sm = 512;
    let full = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    let decision = can_launch_concurrently(&caps, full, 512, full, 512);
    assert_eq!(
        decision,
        ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::WarpResidency
        }
    );
}

#[test]
fn co_launch_register_heavy_kernels_serializes_on_register_pressure() {
    let caps = synthetic_sm120_envelope_default();
    let heavy = KernelResourceUsage {
        regs_per_thread: 129,
        shared_bytes_per_block: 0,
    };
    let decision = can_launch_concurrently(&caps, heavy, 256, heavy, 256);
    assert_eq!(
        decision,
        ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::RegisterPressure
        }
    );
}

#[test]
fn co_launch_with_unrunnable_kernel_returns_kernel_unrunnable() {
    let caps = synthetic_sm120_envelope_default();
    let runnable = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    let too_big = KernelResourceUsage {
        regs_per_thread: 65_537, // exceeds per-block register cap
        shared_bytes_per_block: 0,
    };
    let decision = can_launch_concurrently(&caps, runnable, 128, too_big, 256);
    assert_eq!(
        decision,
        ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::KernelUnrunnable
        }
    );
}

#[test]
fn co_launch_on_device_without_concurrency_short_circuits() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.concurrent_kernels = false;
    let usage = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 0,
    };
    let decision = can_launch_concurrently(&caps, usage, 64, usage, 64);
    assert_eq!(
        decision,
        ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::DeviceUnsupported
        }
    );
}

#[test]
fn co_launch_with_shared_memory_headroom_fits() {
    let caps = synthetic_sm120_envelope_default();
    let shared = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 96 * 1024,
    };
    let decision = can_launch_concurrently(&caps, shared, 128, shared, 128);
    assert_eq!(decision, ConcurrentLaunchDecision::Concurrent);
}

#[test]
fn co_launch_shared_memory_uses_exact_per_sm_limit() {
    let mut caps = synthetic_sm120_envelope_default();
    caps.shared_memory_per_sm = 160 * 1024;
    let shared = KernelResourceUsage {
        regs_per_thread: 16,
        shared_bytes_per_block: 96 * 1024,
    };

    let decision = can_launch_concurrently(&caps, shared, 128, shared, 128);

    assert_eq!(
        decision,
        ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::SharedMemory
        },
        "Fix: CUDA concurrent-launch policy must reject co-resident shared-memory pressure using the probed SM budget, not a guessed multiplier."
    );
}
