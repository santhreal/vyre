//! Backend-neutral device capability profile.
//!
//! Concrete backend crates probe their native device/API surfaces and project
//! them into this value object. Shared optimizer, validation, launch, and
//! strategy code consume projections of this profile instead of carrying
//! independent capability records that can drift.

use vyre_foundation::optimizer::AdapterCaps;
use vyre_foundation::validate;
use vyre_foundation::{CooperativeWidth, ElementPolicy, GeometryRequirements, Uniformity};

/// Quality class for backend timing data exposed through [`DeviceProfile`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceTimingQuality {
    /// The backend reports host wall-clock timing only.
    HostOnly,
    /// The backend can split host enqueue and host wait timing, but not trusted device elapsed time.
    HostEnqueueWait,
    /// The backend can report device elapsed time through timestamp queries or events.
    DeviceTimestamps,
    /// The backend can report device elapsed time plus hardware counter samples.
    HardwareCounters,
}

impl DeviceTimingQuality {
    /// Stable report/config string for timing-quality evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HostOnly => "host_only",
            Self::HostEnqueueWait => "host_enqueue_wait",
            Self::DeviceTimestamps => "device_timestamps",
            Self::HardwareCounters => "hardware_counters",
        }
    }
}

/// Device capability snapshot used across driver-shared planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceProfile {
    /// Stable backend identifier.
    pub backend: &'static str,
    /// The device and lowering path support subgroup intrinsics.
    pub supports_subgroup_ops: bool,
    /// The backend supports indirect dispatch.
    pub supports_indirect_dispatch: bool,
    /// The backend lowers distributed collective communication nodes.
    pub supports_distributed_collectives: bool,
    /// The device can launch a cooperative grid, so a whole-grid fence runs
    /// inside one kernel instead of needing a launch boundary per fence.
    pub supports_cooperative_launch: bool,
    /// Measured host cost of one kernel launch in nanoseconds, or `0` when the
    /// backend has not measured it.
    pub per_launch_overhead_ns: u64,
    /// Measured one-time cost of bringing up persistent execution in
    /// nanoseconds, or `0` when the backend has not measured it.
    pub persistent_setup_overhead_ns: u64,
    /// The backend supports compile-time specialization constants.
    pub supports_specialization_constants: bool,
    /// The backend lowers binary16 natively.
    pub supports_f16: bool,
    /// The backend lowers bfloat16 natively.
    pub supports_bf16: bool,
    /// The backend preserves explicit trap propagation.
    pub supports_trap_propagation: bool,
    /// The backend lowers matrix-engine operations for supported shapes.
    pub supports_tensor_cores: bool,
    /// Native unsigned multiply-high is available to lowering strategies.
    pub has_mul_high: bool,
    /// Integer and float pipelines can issue concurrently.
    pub has_dual_issue_fp32_int32: bool,
    /// Subgroup shuffle-like communication is available.
    pub has_subgroup_shuffle: bool,
    /// Explicit workgroup/shared memory is available.
    pub has_shared_memory: bool,
    /// Maximum native integer width in bits.
    pub max_native_int_width: u32,
    /// Maximum workgroup dimensions.
    pub max_workgroup_size: [u32; 3],
    /// Maximum invocations in one workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Shared memory per workgroup in bytes.
    pub max_shared_memory_bytes: u32,
    /// Maximum single storage-buffer binding in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// Native subgroup size, or `0` when unknown.
    pub subgroup_size: u32,
    /// Physical compute-unit count, or `0` when unknown.
    pub compute_units: u32,
    /// Maximum registers per thread, or `0` when unknown.
    pub regs_per_thread_max: u32,
    /// 32-bit registers one compute unit holds across all resident
    /// invocations, or `0` when unknown.
    ///
    /// Separate from [`Self::regs_per_thread_max`], which is the architectural
    /// ceiling one invocation may allocate. The compiler needs both: the
    /// ceiling says what the device refuses to launch, and this figure divided
    /// by the resident invocation count says how many registers an invocation
    /// holds before the device schedules fewer of them.
    pub max_registers_per_compute_unit: u32,
    /// Invocations resident on one compute unit at full occupancy, or `0` when
    /// unknown.
    pub max_invocations_per_compute_unit: u32,
    /// L1 cache size in bytes, or `0` when unknown.
    pub l1_cache_bytes: u32,
    /// L2 cache size in bytes, or `0` when unknown.
    pub l2_cache_bytes: u32,
    /// Peak memory bandwidth in GB/s, or `0` when unknown.
    pub mem_bw_gbps: u32,
    /// Timing-data quality exposed by this backend/device.
    pub timing_quality: DeviceTimingQuality,
    /// Device timestamp queries/events are available for dispatch timing.
    pub supports_device_timestamps: bool,
    /// Hardware counter sampling is available for benchmark telemetry.
    pub supports_hardware_counters: bool,
    /// Device-profile preferred unroll depth, or `0` when unknown.
    pub ideal_unroll_depth: u32,
    /// Device-profile preferred vector pack width in bits, or `0` when unknown.
    pub ideal_vector_pack_bits: u32,
    /// Device-profile preferred workgroup tile, or `[0, 0, 0]` when unknown.
    pub ideal_workgroup_tile: [u32; 3],
    /// Shared-memory bank count, or `0` when unknown.
    pub shared_memory_bank_count: u32,
    /// Shared-memory bank width in bytes, or `0` when unknown.
    pub shared_memory_bank_width_bytes: u32,
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self::conservative("unknown")
    }
}

impl DeviceProfile {
    /// Conservative profile for a backend that has not probed a device.
    #[must_use]
    pub const fn conservative(backend: &'static str) -> Self {
        Self {
            backend,
            supports_subgroup_ops: false,
            supports_indirect_dispatch: false,
            supports_distributed_collectives: false,
            supports_cooperative_launch: false,
            per_launch_overhead_ns: 0,
            persistent_setup_overhead_ns: 0,
            supports_specialization_constants: false,
            supports_f16: false,
            supports_bf16: false,
            supports_trap_propagation: false,
            supports_tensor_cores: false,
            has_mul_high: false,
            has_dual_issue_fp32_int32: false,
            has_subgroup_shuffle: false,
            has_shared_memory: false,
            max_native_int_width: 32,
            max_workgroup_size: [1, 1, 1],
            max_invocations_per_workgroup: 1,
            max_shared_memory_bytes: 0,
            max_storage_buffer_binding_size: 0,
            subgroup_size: 0,
            compute_units: 0,
            regs_per_thread_max: 0,
            max_registers_per_compute_unit: 0,
            max_invocations_per_compute_unit: 0,
            l1_cache_bytes: 0,
            l2_cache_bytes: 0,
            mem_bw_gbps: 0,
            timing_quality: DeviceTimingQuality::HostOnly,
            supports_device_timestamps: false,
            supports_hardware_counters: false,
            ideal_unroll_depth: 0,
            ideal_vector_pack_bits: 0,
            ideal_workgroup_tile: [0, 0, 0],
            shared_memory_bank_count: 0,
            shared_memory_bank_width_bytes: 0,
        }
    }

    /// Build a profile from the stable backend trait capability methods.
    ///
    /// This is the neutral profile: every fact the trait can answer, and a
    /// conservative value for every fact it cannot. A backend that knows more
    /// overrides the fields it knows and takes the rest from here, rather than
    /// respelling forty fields; one that spelled them all had already lost two
    /// of them.
    #[must_use]
    pub fn from_backend<B: crate::backend::VyreBackend + ?Sized>(backend: &B) -> Self {
        let max_workgroup_size = backend.max_workgroup_size();
        Self {
            backend: backend.id(),
            supports_subgroup_ops: backend.supports_subgroup_ops(),
            supports_indirect_dispatch: backend.supports_indirect_dispatch(),
            supports_distributed_collectives: backend.supports_distributed_collectives(),
            supports_specialization_constants: false,
            supports_f16: backend.supports_f16(),
            supports_bf16: backend.supports_bf16(),
            supports_cooperative_launch: backend.supports_grid_sync(),
            per_launch_overhead_ns: 0,
            persistent_setup_overhead_ns: 0,
            supports_trap_propagation: false,
            supports_tensor_cores: backend.supports_tensor_cores(),
            has_mul_high: false,
            has_dual_issue_fp32_int32: false,
            has_subgroup_shuffle: backend.supports_subgroup_ops(),
            has_shared_memory: false,
            max_native_int_width: 32,
            max_workgroup_size,
            max_invocations_per_workgroup: backend.max_compute_invocations_per_workgroup(),
            max_shared_memory_bytes: 0,
            max_storage_buffer_binding_size: backend.max_storage_buffer_bytes(),
            subgroup_size: backend.subgroup_size().unwrap_or(0),
            compute_units: 0,
            regs_per_thread_max: 0,
            max_registers_per_compute_unit: 0,
            max_invocations_per_compute_unit: 0,
            l1_cache_bytes: 0,
            l2_cache_bytes: 0,
            mem_bw_gbps: 0,
            timing_quality: DeviceTimingQuality::HostOnly,
            supports_device_timestamps: false,
            supports_hardware_counters: false,
            ideal_unroll_depth: 0,
            ideal_vector_pack_bits: 0,
            ideal_workgroup_tile: [0, 0, 0],
            shared_memory_bank_count: 0,
            shared_memory_bank_width_bytes: 0,
        }
    }

    /// Validation capability projection.
    #[must_use]
    pub const fn validation_capabilities(self) -> validate::BackendCapabilities {
        validate::BackendCapabilities {
            supports_subgroup_ops: self.supports_subgroup_ops,
            supports_indirect_dispatch: self.supports_indirect_dispatch,
            supports_specialization_constants: self.supports_specialization_constants,
            has_mul_high: self.has_mul_high,
            has_dual_issue_fp32_int32: self.has_dual_issue_fp32_int32,
            has_tensor_core_int: self.supports_tensor_cores,
            has_native_f16: self.supports_f16,
            has_subgroup_shuffle: self.has_subgroup_shuffle,
            has_shared_memory: self.has_shared_memory,
            has_transcendental_polynomial_emit: true,
            supports_distributed_collectives: self.supports_distributed_collectives,
            max_native_int_width: self.max_native_int_width,
            supports_tensor_cores: self.supports_tensor_cores,
            max_shared_memory_bytes: self.max_shared_memory_bytes,
            regs_per_thread_max: self.regs_per_thread_max,
            subgroup_size: self.subgroup_size,
        }
    }

    /// Whole-program compile facts.
    ///
    /// The compiler validates and ranks against these, so every field is what the
    /// backend reported. A zero means the backend measured nothing, and the
    /// compiler treats a zero budget or a zero cost as unknown rather than as a
    /// limit of zero.
    ///
    /// Every fact a backend probes is forwarded. Dropping one here does not make
    /// the compiler cautious, it makes the omitted term zero: a profile that
    /// reported a compute-unit count, a cache capacity and a memory bandwidth
    /// was ranked as though it had reported none, so the model priced traffic
    /// against launches on a device it had the bandwidth for.
    #[must_use]
    pub fn compile_facts(self) -> vyre_megakernel::DeviceFacts {
        vyre_megakernel::DeviceFacts::new(
            self.validation_capabilities(),
            self.max_invocations_per_workgroup,
        )
        .with_cooperative_launch(self.supports_cooperative_launch)
        .with_device_timestamps(self.supports_device_timestamps)
        .with_occupancy(
            self.registers_per_invocation_at_full_occupancy(),
            self.max_shared_memory_bytes,
        )
        .with_architectural_register_limit(self.regs_per_thread_max)
        .with_launch_costs(
            self.per_launch_overhead_ns,
            self.persistent_setup_overhead_ns,
        )
        .with_compute_units(self.compute_units)
        .with_subgroup_size(self.subgroup_size)
        .with_cache_capacity(self.cache_capacity_bytes())
        .with_bandwidth_facts(u64::from(self.mem_bw_gbps), 0)
    }

    /// 32-bit registers one invocation holds while the device stays fully
    /// occupied, or `0` when the backend reports no per-compute-unit figures.
    ///
    /// The architectural ceiling is not this number. A device that allows 255
    /// registers per invocation runs one invocation group per compute unit at
    /// that allocation, so ranking against the ceiling reports every candidate
    /// as resident and the occupancy term never fires.
    #[must_use]
    pub const fn registers_per_invocation_at_full_occupancy(self) -> u32 {
        if self.max_registers_per_compute_unit == 0 || self.max_invocations_per_compute_unit == 0 {
            return 0;
        }
        self.max_registers_per_compute_unit / self.max_invocations_per_compute_unit
    }

    /// Bytes of cache a second pass over a working set can be served from, or
    /// `0` when the backend reports no cache capacity.
    ///
    /// The device-wide level is the one that matters to a whole-program plan: a
    /// group re-reading its own output reaches it across workgroups, which a
    /// per-compute-unit level cannot serve.
    #[must_use]
    pub const fn cache_capacity_bytes(self) -> u64 {
        self.l2_cache_bytes as u64
    }

    /// Optimizer capability projection.
    #[must_use]
    pub const fn adapter_caps(self) -> AdapterCaps {
        AdapterCaps {
            backend: self.backend,
            supports_subgroup_ops: self.supports_subgroup_ops,
            supports_indirect_dispatch: self.supports_indirect_dispatch,
            supports_specialization_constants: self.supports_specialization_constants,
            max_workgroup_size: self.max_workgroup_size,
            max_invocations_per_workgroup: self.max_invocations_per_workgroup,
            max_shared_memory_bytes: self.max_shared_memory_bytes,
            max_storage_buffer_binding_size: self.max_storage_buffer_binding_size,
            subgroup_size: self.subgroup_size,
            compute_units: self.compute_units,
            regs_per_thread_max: self.regs_per_thread_max,
            l1_cache_bytes: self.l1_cache_bytes,
            l2_cache_bytes: self.l2_cache_bytes,
            mem_bw_gbps: self.mem_bw_gbps,
            ideal_unroll_depth: self.ideal_unroll_depth,
            ideal_vector_pack_bits: self.ideal_vector_pack_bits,
            ideal_workgroup_tile: self.ideal_workgroup_tile,
            shared_memory_bank_count: self.shared_memory_bank_count,
            shared_memory_bank_width_bytes: self.shared_memory_bank_width_bytes,
        }
    }

    /// Strategy capability projection.
    #[must_use]
    pub const fn strategy_capabilities(self) -> validate::BackendCapabilities {
        self.validation_capabilities()
    }

    /// Workgroups a grid-stride kernel should ask for to keep the device busy.
    ///
    /// `compute_units` is `0` when the backend cannot report it, and one
    /// workgroup is not what unknown means: a one-million-element reduction
    /// launched at one workgroup measured 0.08x of a multithreaded CPU
    /// baseline. An unknown count is therefore no cap at all, and the receiving
    /// builder clamps the request down to what the shape admits, so the value
    /// is a ceiling and never an index.
    #[must_use]
    pub const fn grid_stride_workgroups(self) -> u32 {
        if self.compute_units == 0 {
            u32::MAX
        } else {
            self.compute_units
        }
    }
}

impl From<DeviceProfile> for AdapterCaps {
    #[inline]
    fn from(profile: DeviceProfile) -> Self {
        profile.adapter_caps()
    }
}

impl From<DeviceProfile> for validate::BackendCapabilities {
    #[inline]
    fn from(profile: DeviceProfile) -> Self {
        profile.validation_capabilities()
    }
}

impl DeviceProfile {
    /// Every workgroup width this profile admits for `requirements`, ascending.
    ///
    /// A width in the list is legal on this device, not preferred. Ordering
    /// candidates is `vyre-megakernel`'s decision under the compile objective,
    /// so a preference order returned here would be a second cost model. An
    /// empty list states that no width satisfies the requirements.
    #[must_use]
    pub fn admissible_workgroup_widths(&self, requirements: &GeometryRequirements) -> Vec<u32> {
        if requirements.min_shared_bytes > 0 && !self.has_shared_memory {
            return Vec::new();
        }
        if (requirements.requires_cooperative_launch
            || requirements
                .memory_ordering
                .is_some_and(vyre_foundation::ir::MemoryOrdering::requires_grid_sync))
            && !self.supports_cooperative_launch
        {
            return Vec::new();
        }
        if requirements.subgroup_uniformity == Uniformity::SubgroupUniform
            && self.subgroup_size == 0
        {
            return Vec::new();
        }
        match requirements.subgroup_width {
            CooperativeWidth::Agnostic => {}
            CooperativeWidth::AtLeast(minimum) => {
                if minimum == 0 || self.subgroup_size < minimum {
                    return Vec::new();
                }
            }
            CooperativeWidth::Exactly(exact) => {
                if exact == 0 || self.subgroup_size != exact {
                    return Vec::new();
                }
            }
        }
        if matches!(
            requirements.per_invocation_elements,
            ElementPolicy::Multiple(0)
        ) {
            return Vec::new();
        }
        if requirements.min_shared_bytes > self.max_shared_memory_bytes
            && self.max_shared_memory_bytes > 0
        {
            return Vec::new();
        }

        let max_x = self
            .max_invocations_per_workgroup
            .min(self.max_workgroup_size[0])
            .max(1);
        let subgroup_floor = if self.subgroup_size > 0 {
            self.subgroup_size.min(max_x).max(1)
        } else {
            1
        };

        match requirements.cooperative_width {
            CooperativeWidth::Exactly(exact) => {
                if exact == 0 || exact > max_x {
                    Vec::new()
                } else {
                    vec![exact]
                }
            }
            CooperativeWidth::AtLeast(minimum) => {
                if minimum > max_x {
                    return Vec::new();
                }
                powers_of_two_through(minimum.max(1).next_power_of_two(), max_x)
            }
            CooperativeWidth::Agnostic => {
                // A single invocation is admitted whatever the subgroup size: a
                // one-lane launch performs no cooperative operation.
                let mut widths = vec![1];
                widths.extend(powers_of_two_through(
                    subgroup_floor.next_power_of_two(),
                    max_x,
                ));
                widths.dedup();
                widths
            }
        }
    }
}

/// Ascending powers of two from `first` through `last` inclusive.
fn powers_of_two_through(first: u32, last: u32) -> Vec<u32> {
    let mut widths = Vec::new();
    let mut width = first;
    while width <= last {
        widths.push(width);
        match width.checked_mul(2) {
            Some(next) if next > width => width = next,
            _ => break,
        }
    }
    widths
}

// Inline: `vyre_driver::device_profile` is `pub(crate)`, so no integration test can reach what this
// suite exercises.
#[cfg(test)]
mod tests {
    use super::{DeviceProfile, DeviceTimingQuality};

    #[test]
    fn timing_quality_has_stable_report_strings() {
        assert_eq!(DeviceTimingQuality::HostOnly.as_str(), "host_only");
        assert_eq!(
            DeviceTimingQuality::HostEnqueueWait.as_str(),
            "host_enqueue_wait"
        );
        assert_eq!(
            DeviceTimingQuality::DeviceTimestamps.as_str(),
            "device_timestamps"
        );
        assert_eq!(
            DeviceTimingQuality::HardwareCounters.as_str(),
            "hardware_counters"
        );
    }

    #[test]
    fn projections_share_the_same_feature_bits() {
        let profile = DeviceProfile {
            backend: "test",
            supports_subgroup_ops: true,
            supports_indirect_dispatch: true,
            supports_distributed_collectives: true,
            supports_cooperative_launch: true,
            per_launch_overhead_ns: 5_000,
            persistent_setup_overhead_ns: 25_000,
            supports_specialization_constants: true,
            supports_f16: true,
            supports_bf16: false,
            supports_trap_propagation: true,
            supports_tensor_cores: true,
            has_mul_high: true,
            has_dual_issue_fp32_int32: true,
            has_subgroup_shuffle: true,
            has_shared_memory: true,
            max_native_int_width: 64,
            max_workgroup_size: [256, 1, 1],
            max_invocations_per_workgroup: 256,
            max_shared_memory_bytes: 48 * 1024,
            max_storage_buffer_binding_size: 1 << 30,
            subgroup_size: 32,
            compute_units: 128,
            regs_per_thread_max: 255,
            max_registers_per_compute_unit: 65_536,
            max_invocations_per_compute_unit: 1_536,
            l1_cache_bytes: 128 * 1024,
            l2_cache_bytes: 64 * 1024 * 1024,
            mem_bw_gbps: 1700,
            timing_quality: super::DeviceTimingQuality::HardwareCounters,
            supports_device_timestamps: true,
            supports_hardware_counters: true,
            ideal_unroll_depth: 8,
            ideal_vector_pack_bits: 128,
            ideal_workgroup_tile: [16, 16, 1],
            shared_memory_bank_count: 32,
            shared_memory_bank_width_bytes: 4,
        };

        let validation = profile.validation_capabilities();
        let adapter = profile.adapter_caps();
        let strategy = profile.strategy_capabilities();

        assert!(validation.supports_subgroup_ops);
        assert!(validation.supports_distributed_collectives);
        assert!(adapter.supports_subgroup_ops);
        assert!(strategy.has_subgroup_shuffle);
        assert_eq!(adapter.max_invocations_per_workgroup, 256);
        assert_eq!(adapter.ideal_unroll_depth, 8);
        assert_eq!(adapter.ideal_vector_pack_bits, 128);
        assert_eq!(adapter.ideal_workgroup_tile, [16, 16, 1]);
        assert_eq!(strategy.max_native_int_width, 64);
        assert_eq!(
            profile.timing_quality,
            super::DeviceTimingQuality::HardwareCounters
        );
        assert!(profile.supports_device_timestamps);
        assert!(profile.supports_hardware_counters);
    }
}
