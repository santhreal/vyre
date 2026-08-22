//! I4 substrate: occupancy-aware empirical autotuning.
//!
//! Given a probed [`CudaDeviceCaps`] snapshot and a kernel's measured
//! per-thread register pressure plus per-block shared-memory usage, compute
//! the expected hardware occupancy at a candidate workgroup size. The
//! workgroup-size picker chooses the candidate that maximises blocks/SM
//! within the device's hard limits (max_threads_per_block, warp alignment,
//! register and shared-memory ceilings).
//!
//! The estimator is intentionally pure (takes a [`CudaDeviceCaps`] by
//! reference, returns a value type) so it can be unit-tested without a
//! live CUDA context. Live ptxas register counts feed the
//! `regs_per_thread` parameter; `shared_bytes_per_block` is read directly
//! from the descriptor's shared bindings.
//!
//! # Measured occupancy on RTX 5090 reference fixture
//!
//! You can read these numbers instead of re-deriving them. They are OBSERVED, not
//! calculated: each row comes from `cuOccupancyMaxActiveBlocksPerMultiprocessor`
//! on a real emitted vyre storage kernel at that width, on an RTX 5090 reference device (compute
//! capability 12.0, 170 SMs, 1536 threads per SM, 24 blocks per SM, driver
//! 570.211.01, CUDA 12.8). The kernel used 10 registers per thread and zero
//! static shared memory, and the element count was a multiple of every width, so
//! thread slots were genuinely the binding constraint and tail waste could not
//! skew the result.
//!
//! | width | blocks/SM | resident threads/SM | device-wide threads |
//! |---|---|---|---|
//! | 32 | 24 | 768 | 130,560 |
//! | 64 | 24 | 1,536 | 261,120 |
//! | 128 | 12 | 1,536 | 261,120 |
//! | 256 | 6 | 1,536 | 261,120 |
//! | 512 | 3 | 1,536 | 261,120 |
//! | 1024 | 1 | 1,024 | 174,080 |
//!
//! Three things follow, and each one had been guessed wrong before it was
//! measured:
//!
//! Width 1024 costs a third of the device. One block per SM leaves 512 of every
//! SM's 1536 slots idle for the launch's duration, so a cooperative ceiling at
//! 1024 wide is 174,080 lanes rather than 261,120.
//!
//! Width 64 is the NARROWEST width that still reaches full occupancy, and it does
//! so with zero margin: 1536/64 = 24 is exactly the blocks-per-SM cap. That makes
//! the cap load-bearing rather than incidental, and a device with a lower cap
//! would lose width 64 first.
//!
//! Below 64 the thread budget stops predicting anything. At width 32 the thread
//! arithmetic claims 48 blocks and 1536 resident threads; the hardware gives 24
//! and 768, a factor of two. This is why
//! [`crate::occupancy::cooperative_thread_residency_block_limit`] clamps by the reported block cap:
//! without it the cooperative preflight admits grids the driver then refuses.

use vyre_driver::validation::blocks_per_compute_unit;

use crate::device::CudaDeviceCaps;

/// Per-kernel resource pressure required to compute occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelResourceUsage {
    /// 32-bit registers used by each thread, as reported by ptxas
    /// `--ptxas-options=-v` for the JIT-compiled module.
    pub regs_per_thread: u32,
    /// Static shared memory bytes the kernel allocates per block.
    pub shared_bytes_per_block: u32,
}

/// Estimated occupancy at a given workgroup size on a given device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OccupancyEstimate {
    /// Active blocks per streaming multiprocessor at this workgroup size.
    /// Zero when the workgroup configuration cannot run at all (exceeds
    /// per-block register or shared-memory ceiling).
    pub blocks_per_sm: u32,
    /// Active warps per SM (`blocks_per_sm * workgroup_size / warp_size`).
    pub warps_per_sm: u32,
    /// `warps_per_sm` as a fraction of the device's `max_warps_per_sm`,
    /// expressed in basis points (0..=10000) so the value is integer-only
    /// and comparable across configurations without floating-point.
    pub occupancy_bps: u32,
}

impl OccupancyEstimate {
    /// Sentinel for "this workgroup size cannot execute on this device."
    pub const ZERO: Self = Self {
        blocks_per_sm: 0,
        warps_per_sm: 0,
        occupancy_bps: 0,
    };

    /// Whether the configuration achieves at least one resident block.
    #[must_use]
    pub fn is_runnable(&self) -> bool {
        self.blocks_per_sm > 0
    }
}

/// Compute the occupancy estimate for `workgroup_size` threads/block on
/// `caps` given measured `usage`.
///
/// Returns [`OccupancyEstimate::ZERO`] when the workgroup is fundamentally
/// unrunnable (exceeds per-block register or shared-memory limits, or
/// exceeds `max_threads_per_block`). Otherwise the estimator takes the
/// minimum of:
///   - register-pressure cap: `max_registers_per_sm / (regs_per_thread * workgroup_size)`
///   - shared-memory cap: `shared_per_sm / shared_bytes_per_block`
///   - thread-residence cap: `max_threads_per_sm / workgroup_size`
#[must_use]
pub fn estimate_occupancy(
    caps: &CudaDeviceCaps,
    usage: KernelResourceUsage,
    workgroup_size: u32,
) -> OccupancyEstimate {
    let warp = match caps.warp_size_u32() {
        Some(w) if w > 0 => w,
        _ => return OccupancyEstimate::ZERO,
    };
    if workgroup_size == 0 || workgroup_size > caps.max_threads_per_block_u32() {
        return OccupancyEstimate::ZERO;
    }
    let max_regs_block = caps.max_registers_per_block_u32();
    let max_regs_sm = caps.max_registers_per_sm_u32();
    let max_threads_sm = caps.max_threads_per_sm_u32();
    let shared_per_block = caps.shared_memory_per_block_bytes();

    if max_regs_block == 0 || max_regs_sm == 0 || max_threads_sm == 0 {
        return OccupancyEstimate::ZERO;
    }

    // Per-block register requirement.
    let Some(regs_per_block) = usage.regs_per_thread.checked_mul(workgroup_size) else {
        return OccupancyEstimate::ZERO;
    };
    if regs_per_block > max_regs_block {
        return OccupancyEstimate::ZERO;
    }
    if usage.shared_bytes_per_block > shared_per_block {
        return OccupancyEstimate::ZERO;
    }

    let blocks_by_threads = max_threads_sm / workgroup_size;
    let blocks_by_regs = if regs_per_block == 0 {
        u32::MAX
    } else {
        max_regs_sm / regs_per_block
    };
    let blocks_by_shared = if usage.shared_bytes_per_block == 0 {
        u32::MAX
    } else {
        caps.shared_memory_per_sm_bytes() / usage.shared_bytes_per_block
    };

    let blocks_per_sm = blocks_by_threads.min(blocks_by_regs).min(blocks_by_shared);
    if blocks_per_sm == 0 {
        return OccupancyEstimate::ZERO;
    }

    occupancy_estimate_from_blocks(caps, workgroup_size, blocks_per_sm)
}

/// Finish an occupancy estimate from an already-computed `blocks_per_sm`: the
/// warp-fraction tail shared by the theoretical [`estimate_occupancy`] (which
/// derives `blocks_per_sm` from register/shared/thread limits) and the
/// driver-measured launch path (which gets `blocks_per_sm` from
/// `cuOccupancyMaxActiveBlocksPerMultiprocessor`). Keeping this in ONE PLACE means
/// both occupancy numbers are the same fraction of `max_warps_per_sm`, expressed
/// in basis points, so they are directly comparable. Returns
/// [`OccupancyEstimate::ZERO`] when the device reports no warp size or the warp
/// math overflows.
#[must_use]
pub(crate) fn occupancy_estimate_from_blocks(
    caps: &CudaDeviceCaps,
    workgroup_size: u32,
    blocks_per_sm: u32,
) -> OccupancyEstimate {
    let warp = match caps.warp_size_u32() {
        Some(w) if w > 0 => w,
        _ => return OccupancyEstimate::ZERO,
    };
    if blocks_per_sm == 0 || workgroup_size == 0 {
        return OccupancyEstimate::ZERO;
    }
    let max_threads_sm = caps.max_threads_per_sm_u32();
    if max_threads_sm == 0 {
        return OccupancyEstimate::ZERO;
    }
    let warps_per_block = workgroup_size.div_ceil(warp);
    let Some(warps_per_sm) = blocks_per_sm.checked_mul(warps_per_block) else {
        return OccupancyEstimate::ZERO;
    };
    let max_warps_per_sm = max_threads_sm / warp;
    let occupancy_bps = crate::numeric::CUDA_NUMERIC
        .ratio_basis_points_u64(
            u64::from(warps_per_sm),
            u64::from(max_warps_per_sm),
            0,
            "occupancy estimator",
        )
        .min(10_000);

    OccupancyEstimate {
        blocks_per_sm,
        warps_per_sm,
        occupancy_bps,
    }
}

/// Pick the workgroup size from `candidates` that maximises occupancy on
/// `caps` for the measured `usage`. Ties resolve toward the smaller size
/// so launch latency stays low when occupancy is identical. Returns
/// `None` when no candidate is runnable.
#[must_use]
pub fn pick_workgroup_size_for_occupancy(
    caps: &CudaDeviceCaps,
    usage: KernelResourceUsage,
    candidates: &[u32],
) -> Option<u32> {
    let mut best: Option<(u32, OccupancyEstimate)> = None;
    for &candidate in candidates {
        let est = estimate_occupancy(caps, usage, candidate);
        if !est.is_runnable() {
            continue;
        }
        match best {
            None => best = Some((candidate, est)),
            Some((_, current)) if est.occupancy_bps > current.occupancy_bps => {
                best = Some((candidate, est))
            }
            Some((current_size, current))
                if est.occupancy_bps == current.occupancy_bps && candidate < current_size =>
            {
                best = Some((candidate, est))
            }
            _ => {}
        }
    }
    best.map(|(size, _)| size)
}

/// Maximum whole-grid block count that can be resident for a cooperative launch.
///
/// CUDA cooperative kernels require every block in the grid to be resident at
/// once. Register and shared-memory pressure can tighten this further at
/// module-load time, but the thread ceiling and the block ceiling are both
/// available from the probed device caps and catch impossible grids before the
/// release path crosses the FFI boundary into `cuLaunchCooperativeKernel`.
///
/// # The bug this locks out
///
/// Two independent per-SM ceilings apply and this must respect BOTH. The thread
/// budget gives `max_threads_per_sm / workgroup`; the hardware separately caps
/// blocks per SM (`CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR`, e.g. 24 on sm_120).
/// At narrow widths the block cap binds first and the thread budget
/// alone over-admits: at workgroup 32 the thread math yields 48 blocks/SM and
/// 8160 blocks device-wide, while the hardware holds 24 per SM and 4080 total.
/// Admitting on threads alone returns `true` from the preflight and then fails
/// the launch inside the driver, which is the preflight-disagrees-with-the-driver
/// split that sharing one predicate exists to prevent. Widths of 64 and up are
/// unaffected here because 1536/64 = 24 is exactly the cap.
///
/// An unreported cap (`None`) applies no clamp, so a driver that cannot answer
/// keeps the thread-only behavior rather than losing cooperative launches.
#[must_use]
pub fn cooperative_thread_residency_block_limit(caps: &CudaDeviceCaps, workgroup_size: u32) -> u64 {
    if workgroup_size == 0 || !caps.cooperative_launch || caps.compute_capability < (6, 0) {
        return 0;
    }
    let by_threads = blocks_per_compute_unit(caps.max_threads_per_sm_u32(), workgroup_size);
    let blocks_per_sm = match caps.max_blocks_per_sm_u32() {
        Some(cap) => by_threads.min(cap),
        None => by_threads,
    };
    u64::from(blocks_per_sm) * u64::from(caps.multi_processor_count_u32())
}

/// Decision returned by [`can_launch_concurrently`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentLaunchDecision {
    /// The two kernels can launch concurrently on the same SM with
    /// neither one's per-SM resource budget exceeded.
    Concurrent,
    /// At least one resource (registers, threads, or shared memory)
    /// would be over-subscribed; the dispatcher should serialize.
    Serialize {
        /// Human-readable reason naming the over-subscribed resource.
        reason: ConcurrentLaunchBlocker,
    },
}

/// Reason a co-launch was rejected. Useful for telemetry / diagnostics
/// so operators can understand why concurrency wasn't achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrentLaunchBlocker {
    /// Device does not support concurrent kernels at all
    /// (`CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS == 0`).
    DeviceUnsupported,
    /// Either kernel alone would not run (occupancy estimate ZERO).
    KernelUnrunnable,
    /// Combined warps/SM exceed the device's hardware ceiling.
    WarpResidency,
    /// Combined registers/SM exceed the per-SM register file.
    RegisterPressure,
    /// Combined per-block shared bytes exceed the per-block ceiling
    /// (each kernel still has to fit its own block's shared budget).
    SharedMemory,
}

/// Decide whether two kernels can launch concurrently on the same CUDA
/// device under the same SM resources. Pure decision  -  does not perform
/// the launch, only validates that the device + measured per-kernel
/// `KernelResourceUsage` would fit a co-resident schedule.
///
/// Resource model: concurrent kernels need at least one block from each
/// kernel to be co-resident on an SM. Full single-kernel occupancy is not
/// required for overlap; CUDA can interleave blocks as resources free up.
/// This check therefore first proves each kernel is individually runnable,
/// then checks the combined one-block register, warp, and shared-memory
/// footprint against per-SM caps.
///
/// `concurrent_kernels = false` on the device short-circuits to
/// `Serialize { DeviceUnsupported }`.
#[must_use]
pub fn can_launch_concurrently(
    caps: &CudaDeviceCaps,
    usage_a: KernelResourceUsage,
    workgroup_a: u32,
    usage_b: KernelResourceUsage,
    workgroup_b: u32,
) -> ConcurrentLaunchDecision {
    if !caps.concurrent_kernels {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::DeviceUnsupported,
        };
    }

    let est_a = estimate_occupancy(caps, usage_a, workgroup_a);
    let est_b = estimate_occupancy(caps, usage_b, workgroup_b);
    if !est_a.is_runnable() || !est_b.is_runnable() {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::KernelUnrunnable,
        };
    }

    let warp = match caps.warp_size_u32() {
        Some(w) if w > 0 => w,
        _ => {
            return ConcurrentLaunchDecision::Serialize {
                reason: ConcurrentLaunchBlocker::DeviceUnsupported,
            };
        }
    };
    let max_threads_sm = caps.max_threads_per_sm_u32();
    let max_warps_sm = max_threads_sm / warp;
    let warps_a = workgroup_a.div_ceil(warp);
    let warps_b = workgroup_b.div_ceil(warp);
    let Some(total_warps) = warps_a.checked_add(warps_b) else {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::WarpResidency,
        };
    };
    if total_warps > max_warps_sm {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::WarpResidency,
        };
    }

    let max_regs_sm = caps.max_registers_per_sm_u32();
    let Some(regs_a) = usage_a.regs_per_thread.checked_mul(workgroup_a) else {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::RegisterPressure,
        };
    };
    let Some(regs_b) = usage_b.regs_per_thread.checked_mul(workgroup_b) else {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::RegisterPressure,
        };
    };
    let Some(total_regs) = regs_a.checked_add(regs_b) else {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::RegisterPressure,
        };
    };
    if total_regs > max_regs_sm {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::RegisterPressure,
        };
    }

    let shared_per_sm = caps.shared_memory_per_sm_bytes();
    let shared_a = usage_a.shared_bytes_per_block;
    let shared_b = usage_b.shared_bytes_per_block;
    let Some(total_shared) = shared_a.checked_add(shared_b) else {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::SharedMemory,
        };
    };
    if total_shared > shared_per_sm {
        return ConcurrentLaunchDecision::Serialize {
            reason: ConcurrentLaunchBlocker::SharedMemory,
        };
    }

    ConcurrentLaunchDecision::Concurrent
}
