//! Keys, cached plan, and counters the CUDA megakernel plan cache stores.

use vyre_driver::megakernel_execution::{MegakernelExecutionTopology, MegakernelTopologyDecision};

use crate::device::CudaDeviceCaps;

pub(crate) const DEFAULT_MAX_MEGAKERNEL_PLANS: usize = 256;
pub(crate) const PRESSURE_BUCKET_BPS: u32 = 1_000;
pub(crate) const DENSITY_BUCKETS: u16 = 16;
pub(crate) const READBACK_BUCKET_SHIFT: u32 = 12;

/// Analysis family for a cached CUDA megakernel plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum CudaMegakernelAnalysisKind {
    /// Generic graph dataflow wave.
    Dataflow,
    /// IFDS/IDE-style exploded-supergraph propagation.
    Ifds,
    /// Reaching-definitions propagation.
    ReachingDefinitions,
    /// Live-variable propagation.
    Liveness,
    /// Points-to propagation.
    PointsTo,
    /// Source-token or parser-frontier wave.
    ParserFrontend,
    /// Caller-owned analysis family identified by a stable numeric tag.
    Custom(u64),
}

/// CUDA device feature signature that invalidates cached megakernel plans.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CudaMegakernelDeviceKey {
    /// CUDA SM major version.
    pub sm_major: u16,
    /// CUDA SM minor version.
    pub sm_minor: u16,
    /// Hardware warp size.
    pub warp_size: u16,
    /// Whether cooperative grid synchronization is available.
    pub supports_grid_sync: bool,
    /// Whether tensor-core lowering is available for this backend session.
    pub supports_tensor_cores: bool,
    /// Maximum threads accepted for one workgroup/block.
    pub max_workgroup_size: u32,
}

impl From<&CudaDeviceCaps> for CudaMegakernelDeviceKey {
    fn from(caps: &CudaDeviceCaps) -> Self {
        Self {
            sm_major: caps.compute_capability.0.min(u32::from(u16::MAX)) as u16,
            sm_minor: caps.compute_capability.1.min(u32::from(u16::MAX)) as u16,
            warp_size: caps.required_warp_size_u32().min(u32::from(u16::MAX)) as u16,
            supports_grid_sync: caps.compute_capability >= (6, 0) && caps.cooperative_launch,
            supports_tensor_cores: caps.hardware_supports_tensor_cores(),
            max_workgroup_size: caps.max_threads_per_block_u32(),
        }
    }
}

/// Stable key for cached CUDA megakernel plans.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct CudaMegakernelPlanCacheKey {
    /// Stable hash of the normalized resident graph layout.
    pub graph_layout_hash: u64,
    /// Analysis family consuming the graph layout.
    pub analysis_kind: CudaMegakernelAnalysisKind,
    /// CUDA device feature signature.
    pub device: CudaMegakernelDeviceKey,
    /// Coarse active-frontier density bucket.
    pub frontier_density_bucket: u16,
    /// Coarse memory-pressure bucket in basis points.
    pub memory_pressure_bucket: u32,
    /// Coarse output/readback pressure bucket.
    pub readback_pressure_bucket: u16,
    /// Coarse launch-over-dispatch pressure bucket in basis points.
    pub launch_pressure_bucket: u32,
    /// Coarse caller-provided fusion-pressure bucket.
    pub fusion_pressure_bucket: u32,
}

impl CudaMegakernelPlanCacheKey {
    /// Build a cache key from stable identity fields and runtime pressure.
    #[must_use]
    pub fn new(
        graph_layout_hash: u64,
        analysis_kind: CudaMegakernelAnalysisKind,
        device: CudaMegakernelDeviceKey,
        frontier_density: f64,
        memory_pressure_bps: u32,
        readback_bytes: u64,
        launch_pressure_bps: u32,
        fusion_pressure: f64,
    ) -> Self {
        Self {
            graph_layout_hash,
            analysis_kind,
            device,
            frontier_density_bucket: density_bucket(frontier_density),
            memory_pressure_bucket: pressure_bucket(memory_pressure_bps),
            readback_pressure_bucket: readback_bucket(readback_bytes),
            launch_pressure_bucket: pressure_bucket(launch_pressure_bps),
            fusion_pressure_bucket: fusion_bucket(fusion_pressure),
        }
    }

    pub(crate) fn identity(self) -> CudaMegakernelPlanIdentityKey {
        CudaMegakernelPlanIdentityKey {
            graph_layout_hash: self.graph_layout_hash,
            analysis_kind: self.analysis_kind,
            device: self.device,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct CudaMegakernelPlanIdentityKey {
    pub(crate) graph_layout_hash: u64,
    pub(crate) analysis_kind: CudaMegakernelAnalysisKind,
    pub(crate) device: CudaMegakernelDeviceKey,
}

/// Cached CUDA megakernel plan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CudaMegakernelCachedPlan {
    /// Selected topology for this key.
    pub topology: MegakernelExecutionTopology,
    /// Full decision telemetry used when the plan was inserted.
    pub decision: MegakernelTopologyDecision,
}

/// Runtime counters for [`CudaMegakernelPlanCache`](crate::CudaMegakernelPlanCache).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaMegakernelPlanCacheStats {
    /// Cache lookup hits.
    pub hits: u64,
    /// Cache lookup misses.
    pub misses: u64,
    /// Entries evicted by the bounded LRU policy.
    pub evictions: u64,
    /// Current entry count.
    pub entries: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaMegakernelPlanCacheEntry {
    pub(crate) plan: CudaMegakernelCachedPlan,
    pub(crate) last_seen: u64,
}

pub(crate) fn increment_plan_cache_counter(counter: &mut u64, field: &'static str) {
    vyre_driver::accounting::pinning_increment_u64(counter, || {
        tracing::error!(
            "CUDA megakernel {field} overflowed u64; pinning counter at u64::MAX. Fix: scrape metrics more frequently or shard the cache."
        );
    });
}

pub(crate) fn density_bucket(frontier_density: f64) -> u16 {
    if !frontier_density.is_finite() {
        return 0;
    }
    let clamped = frontier_density.clamp(0.0, 1.0);
    rounded_f64_to_u16_bucket(
        clamped * f64::from(DENSITY_BUCKETS - 1),
        "frontier-density bucket",
    )
}

pub(crate) fn pressure_bucket(memory_pressure_bps: u32) -> u32 {
    memory_pressure_bps / PRESSURE_BUCKET_BPS
}

pub(crate) fn pressure_bps(numerator: u64, denominator: u64) -> u32 {
    crate::numeric::CUDA_NUMERIC.ratio_basis_points_u64(
        numerator,
        denominator,
        if numerator == 0 { 0 } else { u32::MAX },
        "megakernel pressure",
    )
}

pub(crate) fn launch_pressure_bps(dispatch_cost_ns: f64, launch_overhead_ns: f64) -> u32 {
    crate::numeric::CUDA_NUMERIC.finite_f64_ratio_basis_points_trunc(
        launch_overhead_ns,
        dispatch_cost_ns,
        u32::MAX,
        0,
        "launch-pressure basis-points",
    )
}

pub(crate) fn readback_bucket(readback_bytes: u64) -> u16 {
    if readback_bytes == 0 {
        return 0;
    }
    let shifted = readback_bytes >> READBACK_BUCKET_SHIFT;
    let bucket = u64::BITS - shifted.leading_zeros();
    bucket.min(u32::from(u16::MAX)) as u16
}

pub(crate) fn fusion_bucket(fusion_pressure: f64) -> u32 {
    pressure_bucket(
        crate::numeric::CUDA_NUMERIC.finite_f64_unit_basis_points_trunc(
            fusion_pressure,
            0,
            "fusion-pressure basis-points",
        ),
    )
}

pub(crate) fn rounded_f64_to_u16_bucket(value: f64, label: &'static str) -> u16 {
    let rounded = value.round();
    if !rounded.is_finite() || rounded < 0.0 || rounded > f64::from(u16::MAX) {
        tracing::error!(
            "CUDA megakernel {label} value {rounded} cannot fit u16. Fix: reduce bucket resolution or shard cache domains."
        );
        return if rounded.is_sign_negative() {
            0
        } else {
            u16::MAX
        };
    }
    rounded as u16
}
