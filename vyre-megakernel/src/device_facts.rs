//! Live device facts a plan is selected against, and the admission gate that
//! refuses a graph the device cannot execute.

use vyre_foundation::ir::{BufferAccess, Program, ProgramGraph};
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
    max_invocations_per_workgroup: u32,
    registers_per_invocation: u32,
    shared_scratch_bytes_per_workgroup: u32,
    per_launch_overhead_ns: u64,
    persistent_setup_overhead_ns: u64,
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
        Self {
            capabilities,
            supports_cooperative_launch: false,
            supports_device_timestamps: false,
            max_invocations_per_workgroup,
            registers_per_invocation: 0,
            shared_scratch_bytes_per_workgroup: 0,
            per_launch_overhead_ns: 0,
            persistent_setup_overhead_ns: 0,
        }
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

/// Workgroup-scoped scratch bytes one program declares.
pub(crate) fn workgroup_scratch_bytes(program: &Program) -> u64 {
    program
        .buffers()
        .iter()
        .filter(|buffer| buffer.access == BufferAccess::Workgroup)
        .fold(0_u64, |total, buffer| {
            let count = usize::try_from(buffer.count).unwrap_or(usize::MAX);
            let bytes = buffer
                .element
                .packed_size_bytes(count)
                .ok()
                .flatten()
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(0);
            total.saturating_add(bytes)
        })
}
