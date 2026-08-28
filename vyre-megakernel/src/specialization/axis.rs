//! The classes of fact a compiled variant may specialize on.
//!
//! An axis is a typed fact, never a caller's vocabulary. A compile that
//! specialized on a supplied identifier would carry the caller's naming into
//! artifact identity, and two callers that spell the same workload differently
//! would compile twice while two that spell different workloads alike would
//! share one artifact. Application information reaches the compiler as the
//! configuration digest and as graph identity; what specializes here is a graph
//! dimension the graph itself declares, a property of a graph value, a stated
//! submission arrangement, or an authenticated target fact.
//!
//! [`SpecializationAxis`] is closed and every reader matches it exhaustively, so
//! a new axis does not compile until each stage records what it does with it.

use std::fmt;

use serde::{Deserialize, Serialize};
use vyre_foundation::validate::BackendCapabilities;

use crate::identity::Digest;
use crate::DeviceFacts;

/// A class of fact one compiled variant may be selected by.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "axis", rename_all = "snake_case")]
pub enum SpecializationAxis {
    /// Exact value bound to one symbolic graph dimension.
    SymbolicDimension {
        /// Dimension name the graph declares.
        dimension: String,
    },
    /// Element layout class of one graph value.
    ValueLayout {
        /// Graph value whose layout class specializes.
        value: u32,
    },
    /// Density class of one graph value: dense, sparse, or ragged.
    ValueDensity {
        /// Graph value whose density class specializes.
        value: u32,
    },
    /// Whether the caller retains state across submissions.
    RetainedState,
    /// Launches the caller submits against one artifact.
    LaunchBatch,
    /// Submissions the caller keeps in flight at once.
    Concurrency,
    /// Content identity of one immutable graph value.
    ConstantIdentity {
        /// Graph value whose content identity specializes.
        value: u32,
    },
    /// One authenticated boolean target capability.
    TargetCapability {
        /// Capability the variant is selected by.
        capability: TargetCapabilityAxis,
    },
    /// One authenticated target resource extent.
    TargetResource {
        /// Resource extent the variant is selected by.
        resource: TargetResourceAxis,
    },
}

impl SpecializationAxis {
    /// Whether values on this axis are content identities rather than scalars.
    #[must_use]
    pub const fn is_identity_axis(&self) -> bool {
        matches!(self, Self::ConstantIdentity { .. })
    }

    /// Stable field path used in diagnostics.
    #[must_use]
    pub fn field(&self) -> String {
        match self {
            Self::SymbolicDimension { dimension } => {
                format!("specialization.axis.symbolic_dimension[{dimension}]")
            }
            Self::ValueLayout { value } => {
                format!("specialization.axis.value_layout[{value}]")
            }
            Self::ValueDensity { value } => {
                format!("specialization.axis.value_density[{value}]")
            }
            Self::RetainedState => "specialization.axis.retained_state".to_string(),
            Self::LaunchBatch => "specialization.axis.launch_batch".to_string(),
            Self::Concurrency => "specialization.axis.concurrency".to_string(),
            Self::ConstantIdentity { value } => {
                format!("specialization.axis.constant_identity[{value}]")
            }
            Self::TargetCapability { capability } => {
                format!(
                    "specialization.axis.target_capability[{}]",
                    capability.name()
                )
            }
            Self::TargetResource { resource } => {
                format!("specialization.axis.target_resource[{}]", resource.name())
            }
        }
    }
}

impl fmt::Display for SpecializationAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.field())
    }
}

/// One authenticated boolean capability a variant may be selected by.
///
/// The reader below destructures [`BackendCapabilities`] completely, so a
/// capability added to the validator's snapshot stops this crate compiling until
/// it is either given an axis or explicitly declined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetCapabilityAxis {
    /// Subgroup arithmetic, ballot, and shuffle are lowerable.
    SubgroupOps,
    /// Indirect dispatch is lowerable.
    IndirectDispatch,
    /// Specialization constants compile.
    SpecializationConstants,
    /// Distributed collectives are lowerable.
    DistributedCollectives,
    /// Native unsigned multiply-high exists.
    MulHigh,
    /// Integer and float pipelines issue together.
    DualIssueFp32Int32,
    /// Integer tensor-core matrix multiply exists.
    TensorCoreInt,
    /// Native half-precision arithmetic runs at useful throughput.
    NativeF16,
    /// Warp-level shuffle exists.
    WarpShuffle,
    /// Shared memory with explicit barriers exists.
    SharedMemory,
    /// Bounded polynomial transcendentals are emittable.
    TranscendentalPolynomialEmit,
    /// Tensor-core matrix instructions exist.
    TensorCores,
    /// The whole grid can be held resident for a cooperative launch.
    CooperativeLaunch,
    /// Device timestamps can be read.
    DeviceTimestamps,
    /// The device can be partitioned spatially.
    SpatialPartitioning,
}

impl TargetCapabilityAxis {
    /// Every declared capability axis.
    pub const ALL: &'static [Self] = &[
        Self::SubgroupOps,
        Self::IndirectDispatch,
        Self::SpecializationConstants,
        Self::DistributedCollectives,
        Self::MulHigh,
        Self::DualIssueFp32Int32,
        Self::TensorCoreInt,
        Self::NativeF16,
        Self::WarpShuffle,
        Self::SharedMemory,
        Self::TranscendentalPolynomialEmit,
        Self::TensorCores,
        Self::CooperativeLaunch,
        Self::DeviceTimestamps,
        Self::SpatialPartitioning,
    ];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SubgroupOps => "subgroup_ops",
            Self::IndirectDispatch => "indirect_dispatch",
            Self::SpecializationConstants => "specialization_constants",
            Self::DistributedCollectives => "distributed_collectives",
            Self::MulHigh => "mul_high",
            Self::DualIssueFp32Int32 => "dual_issue_fp32_int32",
            Self::TensorCoreInt => "tensor_core_int",
            Self::NativeF16 => "native_f16",
            Self::WarpShuffle => "warp_shuffle",
            Self::SharedMemory => "shared_memory",
            Self::TranscendentalPolynomialEmit => "transcendental_polynomial_emit",
            Self::TensorCores => "tensor_cores",
            Self::CooperativeLaunch => "cooperative_launch",
            Self::DeviceTimestamps => "device_timestamps",
            Self::SpatialPartitioning => "spatial_partitioning",
        }
    }

    /// Read this capability out of authenticated device facts.
    #[must_use]
    pub fn read(self, device: DeviceFacts) -> bool {
        let BackendCapabilities {
            supports_subgroup_ops,
            supports_indirect_dispatch,
            supports_specialization_constants,
            supports_distributed_collectives,
            has_mul_high,
            has_dual_issue_fp32_int32,
            has_tensor_core_int,
            has_native_f16,
            has_warp_shuffle,
            has_shared_memory,
            has_transcendental_polynomial_emit,
            max_native_int_width: _,
            max_shared_memory_bytes: _,
            regs_per_thread_max: _,
            subgroup_size: _,
            supports_tensor_cores,
        } = device.capabilities();
        match self {
            Self::SubgroupOps => supports_subgroup_ops,
            Self::IndirectDispatch => supports_indirect_dispatch,
            Self::SpecializationConstants => supports_specialization_constants,
            Self::DistributedCollectives => supports_distributed_collectives,
            Self::MulHigh => has_mul_high,
            Self::DualIssueFp32Int32 => has_dual_issue_fp32_int32,
            Self::TensorCoreInt => has_tensor_core_int,
            Self::NativeF16 => has_native_f16,
            Self::WarpShuffle => has_warp_shuffle,
            Self::SharedMemory => has_shared_memory,
            Self::TranscendentalPolynomialEmit => has_transcendental_polynomial_emit,
            Self::TensorCores => supports_tensor_cores,
            Self::CooperativeLaunch => device.supports_cooperative_launch(),
            Self::DeviceTimestamps => device.supports_device_timestamps(),
            Self::SpatialPartitioning => device.supports_spatial_partitioning(),
        }
    }
}

/// One authenticated target resource extent a variant may be selected by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetResourceAxis {
    /// Invocations one workgroup may hold.
    MaxInvocationsPerWorkgroup,
    /// Shared scratch bytes one workgroup may reserve.
    SharedScratchBytesPerWorkgroup,
    /// Registers one invocation may allocate.
    RegistersPerInvocation,
    /// Hardware compute units.
    ComputeUnits,
    /// Queues that accept concurrent submissions.
    ConcurrentQueues,
    /// Invocations one subgroup holds.
    SubgroupSize,
    /// Last-level cache capacity in bytes.
    CacheCapacityBytes,
}

impl TargetResourceAxis {
    /// Every declared resource axis.
    pub const ALL: &'static [Self] = &[
        Self::MaxInvocationsPerWorkgroup,
        Self::SharedScratchBytesPerWorkgroup,
        Self::RegistersPerInvocation,
        Self::ComputeUnits,
        Self::ConcurrentQueues,
        Self::SubgroupSize,
        Self::CacheCapacityBytes,
    ];

    /// Stable identifier used in diagnostics and evidence.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MaxInvocationsPerWorkgroup => "max_invocations_per_workgroup",
            Self::SharedScratchBytesPerWorkgroup => "shared_scratch_bytes_per_workgroup",
            Self::RegistersPerInvocation => "registers_per_invocation",
            Self::ComputeUnits => "compute_units",
            Self::ConcurrentQueues => "concurrent_queues",
            Self::SubgroupSize => "subgroup_size",
            Self::CacheCapacityBytes => "cache_capacity_bytes",
        }
    }

    /// Read this extent out of authenticated device facts.
    #[must_use]
    pub const fn read(self, device: DeviceFacts) -> u64 {
        match self {
            Self::MaxInvocationsPerWorkgroup => device.max_invocations_per_workgroup() as u64,
            Self::SharedScratchBytesPerWorkgroup => {
                device.shared_scratch_bytes_per_workgroup() as u64
            }
            Self::RegistersPerInvocation => device.hardware_registers_per_invocation() as u64,
            Self::ComputeUnits => device.compute_units() as u64,
            Self::ConcurrentQueues => device.concurrent_queues() as u64,
            Self::SubgroupSize => device.subgroup_size() as u64,
            Self::CacheCapacityBytes => device.cache_capacity_bytes(),
        }
    }
}

/// One value on one axis.
///
/// Scalar axes are compared and ordered; an identity axis is compared only for
/// equality, because a content digest has no neighbourhood.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisValue {
    /// A scalar fact: an extent, a count, a class code, or a boolean as zero or one.
    Scalar(u64),
    /// A content identity.
    Identity(Digest),
}

impl AxisValue {
    /// The scalar this value carries, or `None` on an identity axis.
    #[must_use]
    pub const fn scalar(self) -> Option<u64> {
        match self {
            Self::Scalar(scalar) => Some(scalar),
            Self::Identity(_) => None,
        }
    }

    /// The identity this value carries, or `None` on a scalar axis.
    #[must_use]
    pub const fn identity(self) -> Option<Digest> {
        match self {
            Self::Scalar(_) => None,
            Self::Identity(digest) => Some(digest),
        }
    }
}
