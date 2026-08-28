//! Live device facts a plan is selected against, and the admission gate that
//! refuses a graph the device cannot execute.

use std::sync::Arc;

use vyre_foundation::ir::{BufferAccess, Ident, Program, ProgramGraph};
use vyre_foundation::program_caps;
use vyre_foundation::validate::BackendCapabilities;

use crate::error::{failure, CompileError, CompilerFailureKind};
use crate::grid_sync;

/// Live device facts the whole-program compiler selects against.
///
/// Every field is a fact about the device that will run the artifact. A zero
/// occupancy budget or launch cost means the backend reported no number for
/// it, and the cost term that field feeds is then omitted rather than guessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceFacts {
    pub(crate) capabilities: BackendCapabilities,
    supports_cooperative_launch: bool,
    supports_device_timestamps: bool,
    supports_spatial_partitioning: bool,
    compute_units: u32,
    concurrent_queues: u32,
    max_invocations_per_workgroup: u32,
    registers_per_invocation: u32,
    shared_scratch_bytes_per_workgroup: u32,
    per_launch_overhead_ns: u64,
    persistent_setup_overhead_ns: u64,
    peak_bandwidth_bytes_per_ns: u64,
    calibrated_materialization_throughput_bytes_per_ns: u64,
    architectural_registers_per_invocation: u32,
    cache_capacity_bytes: u64,
    compute_throughput_ops_per_ns: u64,
    tensor_throughput_ops_per_ns: u64,
    barrier_ns: u64,
    grid_sync_ns: u64,
    subgroup_size: u32,
    calibration_version: u16,
}

/// Serializable projection of every fact a plan is selected against.
///
/// The conversion below destructures [`DeviceFacts`] completely, so a fact added
/// there stops this crate compiling until it is projected here. That is the
/// point: a fact the compiler selects against and the identity omits lets two
/// devices that disagree share one artifact, and the calibration version is
/// exactly such a fact, because a recalibrated device prices every candidate
/// differently while reporting the same capabilities.
#[derive(serde::Serialize)]
pub(crate) struct DeviceIdentity {
    supports_subgroup_ops: bool,
    supports_indirect_dispatch: bool,
    supports_specialization_constants: bool,
    supports_distributed_collectives: bool,
    has_mul_high: bool,
    has_dual_issue_fp32_int32: bool,
    has_tensor_core_int: bool,
    has_native_f16: bool,
    has_warp_shuffle: bool,
    has_shared_memory: bool,
    has_transcendental_polynomial_emit: bool,
    max_native_int_width: u32,
    max_shared_memory_bytes: u32,
    regs_per_thread_max: u32,
    capability_subgroup_size: u32,
    supports_tensor_cores: bool,
    supports_cooperative_launch: bool,
    supports_device_timestamps: bool,
    supports_spatial_partitioning: bool,
    compute_units: u32,
    concurrent_queues: u32,
    max_invocations_per_workgroup: u32,
    registers_per_invocation: u32,
    shared_scratch_bytes_per_workgroup: u32,
    per_launch_overhead_ns: u64,
    persistent_setup_overhead_ns: u64,
    peak_bandwidth_bytes_per_ns: u64,
    calibrated_materialization_throughput_bytes_per_ns: u64,
    architectural_registers_per_invocation: u32,
    cache_capacity_bytes: u64,
    compute_throughput_ops_per_ns: u64,
    tensor_throughput_ops_per_ns: u64,
    barrier_ns: u64,
    grid_sync_ns: u64,
    subgroup_size: u32,
    calibration_version: u16,
}

impl From<DeviceFacts> for DeviceIdentity {
    fn from(facts: DeviceFacts) -> Self {
        let DeviceFacts {
            capabilities,
            supports_cooperative_launch,
            supports_device_timestamps,
            supports_spatial_partitioning,
            compute_units,
            concurrent_queues,
            max_invocations_per_workgroup,
            registers_per_invocation,
            shared_scratch_bytes_per_workgroup,
            per_launch_overhead_ns,
            persistent_setup_overhead_ns,
            peak_bandwidth_bytes_per_ns,
            calibrated_materialization_throughput_bytes_per_ns,
            architectural_registers_per_invocation,
            cache_capacity_bytes,
            compute_throughput_ops_per_ns,
            tensor_throughput_ops_per_ns,
            barrier_ns,
            grid_sync_ns,
            subgroup_size,
            calibration_version,
        } = facts;
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
            max_native_int_width,
            max_shared_memory_bytes,
            regs_per_thread_max,
            subgroup_size: capability_subgroup_size,
            supports_tensor_cores,
        } = capabilities;
        Self {
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
            max_native_int_width,
            max_shared_memory_bytes,
            regs_per_thread_max,
            capability_subgroup_size,
            supports_tensor_cores,
            supports_cooperative_launch,
            supports_device_timestamps,
            supports_spatial_partitioning,
            compute_units,
            concurrent_queues,
            max_invocations_per_workgroup,
            registers_per_invocation,
            shared_scratch_bytes_per_workgroup,
            per_launch_overhead_ns,
            persistent_setup_overhead_ns,
            peak_bandwidth_bytes_per_ns,
            calibrated_materialization_throughput_bytes_per_ns,
            architectural_registers_per_invocation,
            cache_capacity_bytes,
            compute_throughput_ops_per_ns,
            tensor_throughput_ops_per_ns,
            barrier_ns,
            grid_sync_ns,
            subgroup_size,
            calibration_version,
        }
    }
}

impl DeviceFacts {
    /// Facts for a caller that has no device.
    ///
    /// Every capability is absent and every budget is zero, so validation grants
    /// nothing: a program that needs a gated capability is rejected instead of
    /// being compiled against an assumed device. A zero budget is unknown rather
    /// than a limit of zero, so no size gate fires and no cost term is charged.
    /// Use this only where no backend is reachable; a caller holding a backend
    /// passes its live facts.
    #[must_use]
    pub const fn unknown() -> Self {
        Self::new(
            BackendCapabilities {
                supports_subgroup_ops: false,
                supports_indirect_dispatch: false,
                supports_specialization_constants: false,
                supports_distributed_collectives: false,
                has_mul_high: false,
                has_dual_issue_fp32_int32: false,
                has_tensor_core_int: false,
                has_native_f16: false,
                has_warp_shuffle: false,
                has_shared_memory: false,
                has_transcendental_polynomial_emit: false,
                max_native_int_width: 0,
                supports_tensor_cores: false,
                max_shared_memory_bytes: 0,
                regs_per_thread_max: 0,
                subgroup_size: 0,
            },
            0,
        )
    }

    /// Construct facts from the live capability snapshot and invocation limit.
    ///
    /// Cooperative launch, launch timestamps, occupancy budgets, and launch
    /// costs start absent. A backend that measures one supplies it through the
    /// matching `with_` method.
    #[must_use]
    pub const fn new(
        capabilities: BackendCapabilities,
        max_invocations_per_workgroup: u32,
    ) -> Self {
        let subgroup_size = capabilities.subgroup_size;
        Self {
            capabilities,
            supports_cooperative_launch: false,
            supports_device_timestamps: false,
            supports_spatial_partitioning: false,
            compute_units: 0,
            concurrent_queues: 0,
            max_invocations_per_workgroup,
            registers_per_invocation: 0,
            shared_scratch_bytes_per_workgroup: 0,
            per_launch_overhead_ns: 0,
            persistent_setup_overhead_ns: 0,
            peak_bandwidth_bytes_per_ns: 0,
            calibrated_materialization_throughput_bytes_per_ns: 0,
            architectural_registers_per_invocation: 0,
            cache_capacity_bytes: 0,
            compute_throughput_ops_per_ns: 0,
            tensor_throughput_ops_per_ns: 0,
            barrier_ns: 0,
            grid_sync_ns: 0,
            subgroup_size,
            calibration_version: 0,
        }
    }

    /// Record which version of the calibrated fact set these figures came from.
    ///
    /// Zero means uncalibrated: the throughput, latency and capacity figures are
    /// whatever the backend reported without a calibration run behind them. A
    /// recalibration that changes any priced figure advances this version, which
    /// is what allows a later measurement session to replace a winner an earlier
    /// session authenticated. Leaving it unchanged makes the two sessions
    /// comparable, and the recorded winner then stands.
    #[must_use]
    pub const fn with_calibration_version(mut self, calibration_version: u16) -> Self {
        self.calibration_version = calibration_version;
        self
    }

    /// Version of the calibrated fact set behind these figures, zero when
    /// uncalibrated.
    #[must_use]
    pub const fn calibration_version(&self) -> u16 {
        self.calibration_version
    }

    /// Record whether the device can launch a cooperative grid.
    #[must_use]
    pub const fn with_cooperative_launch(mut self, supported: bool) -> Self {
        self.supports_cooperative_launch = supported;
        self
    }

    /// Record whether the device timestamps a launch on the device itself.
    #[must_use]
    pub const fn with_device_timestamps(mut self, supported: bool) -> Self {
        self.supports_device_timestamps = supported;
        self
    }
    /// Record the number of hardware compute units on the device.
    #[must_use]
    pub const fn with_compute_units(mut self, compute_units: u32) -> Self {
        self.compute_units = compute_units;
        self
    }

    /// Record the number of independent concurrent hardware queues/streams.
    #[must_use]
    pub const fn with_concurrent_queues(mut self, concurrent_queues: u32) -> Self {
        self.concurrent_queues = concurrent_queues;
        self
    }

    /// Record whether the target hardware/driver exposes enforceable spatial partitioning capability.
    #[must_use]
    pub const fn with_spatial_partitioning(mut self, supported: bool) -> Self {
        self.supports_spatial_partitioning = supported;
        self
    }

    /// Record the per-invocation register budget and the per-workgroup
    /// shared-scratch budget.
    #[must_use]
    pub const fn with_occupancy(
        mut self,
        registers_per_invocation: u32,
        shared_scratch_bytes_per_workgroup: u32,
    ) -> Self {
        self.registers_per_invocation = registers_per_invocation;
        self.shared_scratch_bytes_per_workgroup = shared_scratch_bytes_per_workgroup;
        self
    }

    /// Record the architectural register ceiling one invocation may allocate.
    ///
    /// The occupancy budget recorded by [`Self::with_occupancy`] is what an
    /// invocation holds before the device schedules fewer of them; this is what
    /// the device refuses to launch beyond. Ranking needs both, because a plan
    /// between the two spills and still runs, and a plan above the ceiling has
    /// no execution at all.
    #[must_use]
    pub const fn with_architectural_register_limit(mut self, registers: u32) -> Self {
        self.architectural_registers_per_invocation = registers;
        self
    }

    /// Record the device-wide cache capacity a repeated pass can be served from.
    #[must_use]
    pub const fn with_cache_capacity(mut self, bytes: u64) -> Self {
        self.cache_capacity_bytes = bytes;
        self
    }

    /// Record measured instruction and matrix-engine throughput.
    ///
    /// Both are operations retired per nanosecond across the whole device. A
    /// zero rate means the backend measured none, and the term it feeds is
    /// omitted rather than priced at a rate nothing observed.
    #[must_use]
    pub const fn with_compute_throughput(
        mut self,
        compute_ops_per_ns: u64,
        tensor_ops_per_ns: u64,
    ) -> Self {
        self.compute_throughput_ops_per_ns = compute_ops_per_ns;
        self.tensor_throughput_ops_per_ns = tensor_ops_per_ns;
        self
    }

    /// Record measured workgroup-barrier and whole-grid rendezvous costs in
    /// nanoseconds.
    #[must_use]
    pub const fn with_synchronization_costs(mut self, barrier_ns: u64, grid_sync_ns: u64) -> Self {
        self.barrier_ns = barrier_ns;
        self.grid_sync_ns = grid_sync_ns;
        self
    }

    /// Record measured host launch cost and persistent-mode setup cost.
    #[must_use]
    pub const fn with_launch_costs(
        mut self,
        per_launch_overhead_ns: u64,
        persistent_setup_overhead_ns: u64,
    ) -> Self {
        self.per_launch_overhead_ns = per_launch_overhead_ns;
        self.persistent_setup_overhead_ns = persistent_setup_overhead_ns;
        self
    }

    /// Record measured peak memory bandwidth and calibrated materialization throughput.
    #[must_use]
    pub const fn with_bandwidth_facts(
        mut self,
        peak_bandwidth_bytes_per_ns: u64,
        calibrated_materialization_throughput_bytes_per_ns: u64,
    ) -> Self {
        self.peak_bandwidth_bytes_per_ns = peak_bandwidth_bytes_per_ns;
        self.calibrated_materialization_throughput_bytes_per_ns =
            calibrated_materialization_throughput_bytes_per_ns;
        self
    }

    /// Record explicit subgroup size.
    #[must_use]
    pub const fn with_subgroup_size(mut self, subgroup_size: u32) -> Self {
        self.subgroup_size = subgroup_size;
        self
    }

    /// Peak memory bandwidth in bytes per nanosecond, or zero when unknown.
    #[must_use]
    pub const fn peak_bandwidth_bytes_per_ns(&self) -> u64 {
        self.peak_bandwidth_bytes_per_ns
    }

    /// Calibrated materialization throughput in bytes per nanosecond, or zero when unmeasured.
    #[must_use]
    pub const fn calibrated_materialization_throughput_bytes_per_ns(&self) -> u64 {
        self.calibrated_materialization_throughput_bytes_per_ns
    }

    /// Hardware subgroup size in lanes, or zero when unmeasured.
    #[must_use]
    pub const fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    /// Live IR capability snapshot advertised by the device.
    #[must_use]
    pub const fn capabilities(&self) -> BackendCapabilities {
        self.capabilities
    }

    /// Whether a whole-grid fence can run inside one kernel on this device.
    #[must_use]
    pub const fn supports_cooperative_launch(&self) -> bool {
        self.supports_cooperative_launch
    }

    /// Whether a search measurement can carry a device timestamp.
    #[must_use]
    pub const fn supports_device_timestamps(&self) -> bool {
        self.supports_device_timestamps
    }
    /// Number of hardware compute units on the device, or zero when unknown.
    #[must_use]
    pub const fn compute_units(&self) -> u32 {
        self.compute_units
    }

    /// Number of independent concurrent hardware queues/streams, or zero when unknown.
    #[must_use]
    pub const fn concurrent_queues(&self) -> u32 {
        self.concurrent_queues
    }

    /// Whether the device exposes an enforceable hardware spatial partitioning capability.
    #[must_use]
    pub const fn supports_spatial_partitioning(&self) -> bool {
        self.supports_spatial_partitioning
    }

    /// Largest legal invocation count in one workgroup.
    #[must_use]
    pub const fn max_invocations_per_workgroup(&self) -> u32 {
        self.max_invocations_per_workgroup
    }

    /// Registers one invocation holds before the target compiler spills, or zero
    /// when the backend reports no budget.
    #[must_use]
    pub const fn registers_per_invocation(&self) -> u32 {
        self.registers_per_invocation
    }

    /// Shared scratch bytes one workgroup holds, or zero when the backend
    /// reports no budget.
    #[must_use]
    pub const fn shared_scratch_bytes_per_workgroup(&self) -> u32 {
        self.shared_scratch_bytes_per_workgroup
    }

    /// Registers one invocation may allocate before the device refuses the
    /// launch, or zero when the backend reports no ceiling.
    #[must_use]
    pub const fn architectural_registers_per_invocation(&self) -> u32 {
        self.architectural_registers_per_invocation
    }

    /// Registers per invocation above which no execution exists, or zero when
    /// the backend reports no ceiling.
    ///
    /// Allocating above [`Self::registers_per_invocation`] costs occupancy and
    /// spill traffic, both of which the cost model prices. Allocating above
    /// this ceiling has no launch to price, so legality rejects it. A backend
    /// that reports only the occupancy budget is held to that budget, because a
    /// ceiling nothing reported cannot admit anything.
    #[must_use]
    pub const fn hardware_registers_per_invocation(&self) -> u32 {
        if self.architectural_registers_per_invocation > 0 {
            self.architectural_registers_per_invocation
        } else {
            self.registers_per_invocation
        }
    }

    /// Device-wide cache bytes a repeated pass can be served from, or zero when
    /// the backend reports no capacity.
    #[must_use]
    pub const fn cache_capacity_bytes(&self) -> u64 {
        self.cache_capacity_bytes
    }

    /// Instructions the device retires per nanosecond, or zero when unmeasured.
    #[must_use]
    pub const fn compute_throughput_ops_per_ns(&self) -> u64 {
        self.compute_throughput_ops_per_ns
    }

    /// Matrix-engine operations the device retires per nanosecond, or zero when
    /// unmeasured.
    #[must_use]
    pub const fn tensor_throughput_ops_per_ns(&self) -> u64 {
        self.tensor_throughput_ops_per_ns
    }

    /// Cost of one workgroup barrier in nanoseconds, or zero when unmeasured.
    #[must_use]
    pub const fn barrier_ns(&self) -> u64 {
        self.barrier_ns
    }

    /// Cost of one whole-grid rendezvous in nanoseconds, or zero when
    /// unmeasured.
    #[must_use]
    pub const fn grid_sync_ns(&self) -> u64 {
        self.grid_sync_ns
    }

    /// Host cost of one kernel launch in nanoseconds, or zero when unmeasured.
    #[must_use]
    pub const fn per_launch_overhead_ns(&self) -> u64 {
        self.per_launch_overhead_ns
    }

    /// One-time cost of bringing up persistent execution in nanoseconds, or
    /// zero when unmeasured.
    #[must_use]
    pub const fn persistent_setup_overhead_ns(&self) -> u64 {
        self.persistent_setup_overhead_ns
    }
}

/// Reject a graph the live device cannot execute.
///
/// Foundation node validation covers the capability bits it knows about:
/// subgroup expressions and distributed collectives. This gate covers the rest
/// of the live snapshot, plus the two device facts no instruction expresses.
///
/// A whole-grid fence is a launch property, not an instruction property, so a
/// program that fences the grid on a device that cannot launch a cooperative
/// grid has no correct execution and is refused here instead of deadlocking at
/// dispatch. The declared workgroup is checked against the live invocation and
/// shared-scratch limits for the same reason: a group the device will not accept
/// is a compile-time fact, not a dispatch failure.
pub(crate) fn validate_device_support(
    graph: &ProgramGraph,
    device: DeviceFacts,
) -> Result<(), CompileError> {
    let capabilities = device.capabilities;
    for node in graph.nodes() {
        let path = format!("request.graph.nodes[{}].program", node.id.0);
        if grid_sync::requires_grid_sync(&node.program) && !device.supports_cooperative_launch {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                "program fences the whole grid but the device cannot launch a cooperative grid",
                "split the program at the grid fence into one node per segment, or compile for a device that reports cooperative launch",
            ));
        }
        let required = program_caps::scan(&node.program);
        let shared_scratch_bytes = workgroup_scratch_bytes(&node.program);
        let unmet = [
            (
                required.tensor_ops && !capabilities.has_tensor_core_int,
                "program uses tensor-core operands but the device reports no tensor-core integer support",
                "lower the tensor operation to scalar arithmetic, or compile for a device with tensor cores",
            ),
            (
                required.f16 && !capabilities.has_native_f16,
                "program uses binary16 operands but the device reports no native f16 arithmetic",
                "widen the f16 operands to f32, or compile for a device with native f16",
            ),
            (
                required.subgroup_ops && !capabilities.has_warp_shuffle,
                "program uses subgroup operations but the device reports no warp shuffle",
                "remove the subgroup operation, or compile for a device with warp-level shuffle",
            ),
            (
                required.indirect_dispatch && !capabilities.supports_indirect_dispatch,
                "program dispatches indirectly but the device reports no indirect dispatch",
                "resolve the dispatch extent on the host, or compile for a device with indirect dispatch",
            ),
            (
                shared_scratch_bytes > 0 && !capabilities.has_shared_memory,
                "program declares workgroup-scoped scratch but the device reports no shared memory",
                "move the scratch buffer to global memory, or compile for a device with shared memory",
            ),
        ];
        if let Some((_, message, fix)) = unmet.into_iter().find(|(unmet, _, _)| *unmet) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                message,
                fix,
            ));
        }
        let declared = node.program.workgroup_size;
        let invocations = u64::from(declared[0])
            .saturating_mul(u64::from(declared[1]))
            .saturating_mul(u64::from(declared[2]));
        if device.max_invocations_per_workgroup > 0
            && invocations > u64::from(device.max_invocations_per_workgroup)
        {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                format!(
                    "program declares {invocations} invocations per workgroup; the device accepts {}",
                    device.max_invocations_per_workgroup
                ),
                "declare a workgroup within the live device invocation limit",
            ));
        }
        if device.shared_scratch_bytes_per_workgroup > 0
            && shared_scratch_bytes > u64::from(device.shared_scratch_bytes_per_workgroup)
        {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                path,
                format!(
                    "program declares {shared_scratch_bytes} workgroup scratch bytes; the device accepts {}",
                    device.shared_scratch_bytes_per_workgroup
                ),
                "reduce the workgroup-scoped scratch to the live device budget",
            ));
        }
    }
    Ok(())
}

/// Workgroup-scoped scratch one program declares, one entry per buffer.
///
/// Fusion unions buffers by name: `merge_programs_shared` keeps one
/// declaration per name and takes the larger count, so two fused arms that
/// name the same tile share it. A group's scratch is therefore the union of
/// its members' declarations, and summing per-member totals charges a shared
/// tile once per member.
pub(crate) fn workgroup_scratch_declarations(
    program: &Program,
) -> impl Iterator<Item = (Ident, u64)> + '_ {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access == BufferAccess::Workgroup)
        .map(|buffer| {
            let count = usize::try_from(buffer.count).unwrap_or(usize::MAX);
            let bytes = buffer
                .element
                .packed_size_bytes(count)
                .ok()
                .flatten()
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(0);
            (Ident::new(Arc::clone(&buffer.name)), bytes)
        })
}

/// Workgroup-scoped scratch bytes one program declares.
///
/// One program declares each name once, so its own total is the sum.
pub(crate) fn workgroup_scratch_bytes(program: &Program) -> u64 {
    workgroup_scratch_declarations(program)
        .fold(0_u64, |total, (_, bytes)| total.saturating_add(bytes))
}
