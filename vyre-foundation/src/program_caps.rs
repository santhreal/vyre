//! Program → required-capability analysis.
//!
//! Scan a `Program` and report the hardware capabilities its lowering will
//! need. Callers (backends, conformance harnesses, certificate emitters)
//! compare the required set against what a backend advertises and surface
//! `MissingCapability` *before* handing the kernel to the device, avoiding
//! panics inside `create_shader_module` / `createComputePipeline`.
//!
//! The scanner is strictly syntactic: it walks every `Expr` and `Node` in
//! the program and checks the IR surface. It intentionally does **not**
//! know anything about backend-specific lowering rules  -  that would make it
//! a circular dependency of the very thing it is supposed to gate.

use std::fmt;

use crate::ir::Program;

/// Capabilities a `Program` needs from whichever backend executes it.
///
/// This is a structured replacement for hardcoded "exempt op" lists. A
/// universal diff harness asks `scan(program)` which bits the program
/// needs, asks the backend which bits it advertises, and skips the pair
/// when they disagree. The result reasons are attached for telemetry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct RequiredCapabilities {
    /// The program invokes `Expr::SubgroupAdd`, `SubgroupBallot`, or
    /// `SubgroupShuffle`. Lowering paths need the SUBGROUP / wave-op
    /// feature on the target device.
    pub subgroup_ops: bool,
    /// The program uses any IEEE 754 binary16 operand.
    pub f16: bool,
    /// The program uses any bfloat16 operand.
    pub bf16: bool,
    /// The program uses 64-bit floats.
    pub f64: bool,
    /// The program dispatches async DMA (`Node::AsyncLoad` / `AsyncStore`).
    pub async_dispatch: bool,
    /// The program emits `Node::IndirectDispatch`.
    pub indirect_dispatch: bool,
    /// The program reaches into tensor / tensor-core operand types.
    pub tensor_ops: bool,
    /// The program uses a `Node::Trap`  -  backend needs trap propagation.
    pub trap: bool,
    /// The program contains a grid-scope barrier. The backend must offer a
    /// cooperative launch; a workgroup-scoped barrier cannot stand in for one,
    /// so a target without it cannot emit the program at any geometry.
    pub grid_sync: bool,
    /// The program uses collective communication nodes that require transport.
    pub distributed_collectives: bool,
    /// Count of collective nodes that can lower to local single-rank IR.
    pub local_single_rank_collectives: usize,
    /// Count of collective nodes that require real multi-rank transport.
    pub transport_collectives: usize,
    /// Maximum workgroup size declared by the program across all axes.
    pub max_workgroup_size: [u32; 3],
    /// Sum of `BufferDecl::count * sizeof(DataType)` across every buffer
    /// whose size can be computed statically. `0` means every buffer has
    /// dynamic size.
    pub static_storage_bytes: u64,
}

impl RequiredCapabilities {
    /// Empty capability set.
    pub const NONE: Self = Self {
        subgroup_ops: false,
        f16: false,
        bf16: false,
        f64: false,
        async_dispatch: false,
        indirect_dispatch: false,
        tensor_ops: false,
        trap: false,
        grid_sync: false,
        distributed_collectives: false,
        local_single_rank_collectives: 0,
        transport_collectives: 0,
        max_workgroup_size: [0, 0, 0],
        static_storage_bytes: 0,
    };

    /// Empty set  -  the Program needs nothing beyond the minimum substrate.
    #[must_use]
    pub const fn none() -> Self {
        Self::NONE
    }

    /// Enable subgroup operations.
    #[must_use]
    pub const fn with_subgroup_ops(mut self) -> Self {
        self.subgroup_ops = true;
        self
    }
    /// Strongest applicable capability set (all capability flags active, maximum limits).
    #[must_use]
    pub fn all() -> Self {
        Self {
            subgroup_ops: true,
            f16: true,
            bf16: true,
            f64: true,
            async_dispatch: true,
            indirect_dispatch: true,
            tensor_ops: true,
            trap: true,
            grid_sync: true,
            distributed_collectives: true,
            local_single_rank_collectives: 1,
            transport_collectives: 1,
            max_workgroup_size: [1024, 1024, 64],
            static_storage_bytes: u64::MAX,
        }
    }

    /// Monotonic lattice join for transitive call-graph fixed-point propagation.
    ///
    /// Computes the supremum of two capability requirements using boolean OR
    /// and component-wise maximums, guaranteeing idempotence (`a.join(a) == a`)
    /// and finite monotonic convergence on cyclic and multi-path call graphs.
    #[must_use]
    pub fn join(mut self, other: RequiredCapabilities) -> Self {
        // Destructured exhaustively: a new capability field fails to compile
        // here rather than being silently dropped from the lattice.
        let RequiredCapabilities {
            subgroup_ops,
            f16,
            bf16,
            f64,
            async_dispatch,
            indirect_dispatch,
            tensor_ops,
            trap,
            grid_sync,
            distributed_collectives,
            local_single_rank_collectives,
            transport_collectives,
            max_workgroup_size,
            static_storage_bytes,
        } = other;
        self.subgroup_ops |= subgroup_ops;
        self.f16 |= f16;
        self.bf16 |= bf16;
        self.f64 |= f64;
        self.async_dispatch |= async_dispatch;
        self.indirect_dispatch |= indirect_dispatch;
        self.tensor_ops |= tensor_ops;
        self.trap |= trap;
        self.grid_sync |= grid_sync;
        self.distributed_collectives |= distributed_collectives;
        self.local_single_rank_collectives = self
            .local_single_rank_collectives
            .max(local_single_rank_collectives);
        self.transport_collectives = self.transport_collectives.max(transport_collectives);
        for axis in 0..3 {
            self.max_workgroup_size[axis] =
                self.max_workgroup_size[axis].max(max_workgroup_size[axis]);
        }
        self.static_storage_bytes = self.static_storage_bytes.max(static_storage_bytes);
        self
    }

    /// Build the union of two capability sets (field-wise `OR` and `max`).
    #[must_use]
    pub fn union(mut self, other: RequiredCapabilities) -> Self {
        // Destructured exhaustively: see `join`.
        let RequiredCapabilities {
            subgroup_ops,
            f16,
            bf16,
            f64,
            async_dispatch,
            indirect_dispatch,
            tensor_ops,
            trap,
            grid_sync,
            distributed_collectives,
            local_single_rank_collectives,
            transport_collectives,
            max_workgroup_size,
            static_storage_bytes,
        } = other;
        self.subgroup_ops |= subgroup_ops;
        self.f16 |= f16;
        self.bf16 |= bf16;
        self.f64 |= f64;
        self.async_dispatch |= async_dispatch;
        self.indirect_dispatch |= indirect_dispatch;
        self.tensor_ops |= tensor_ops;
        self.trap |= trap;
        self.grid_sync |= grid_sync;
        self.distributed_collectives |= distributed_collectives;
        self.local_single_rank_collectives = self
            .local_single_rank_collectives
            .saturating_add(local_single_rank_collectives);
        self.transport_collectives = self
            .transport_collectives
            .saturating_add(transport_collectives);
        for axis in 0..3 {
            self.max_workgroup_size[axis] =
                self.max_workgroup_size[axis].max(max_workgroup_size[axis]);
        }
        self.static_storage_bytes = self
            .static_storage_bytes
            .saturating_add(static_storage_bytes);
        self
    }
}

/// The reason a backend cannot execute a program.
///
/// Returned by [`check_backend_capabilities`] when the scan finds a
/// capability the backend did not advertise. Carries every missing bit
/// so callers can emit one actionable error instead of bisecting.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MissingCapability {
    /// Backend identifier that was asked to run the program.
    pub backend: String,
    /// Flat list of human-readable capability names the backend lacks.
    /// Workgroup-axis violations include the stable `"workgroup_size"`
    /// category plus `"workgroup_size axis N (requested R, max M)"`
    /// detail so callers can both match the category and point at the
    /// specific axis.
    pub missing: Vec<String>,
}

impl fmt::Display for MissingCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "backend `{}` is missing required capabilities: {}. \
             Fix: pick a GPU backend that advertises these capabilities \
             or lower the program requirements before dispatch.",
            self.backend,
            self.missing.join(", ")
        )
    }
}

impl std::error::Error for MissingCapability {}

/// Walk the program and collect the union of capabilities it requires.
#[must_use]
pub fn scan(program: &Program) -> RequiredCapabilities {
    let stats = program.stats();
    let collective_plan = crate::transform::collectives::collective_transport_plan(program);
    RequiredCapabilities {
        subgroup_ops: stats.subgroup_ops(),
        f16: stats.f16(),
        bf16: stats.bf16(),
        f64: stats.f64(),
        async_dispatch: stats.async_dispatch(),
        indirect_dispatch: stats.indirect_dispatch(),
        tensor_ops: stats.tensor_ops(),
        trap: stats.trap(),
        grid_sync: stats.grid_sync(),
        distributed_collectives: collective_plan.requires_transport(),
        local_single_rank_collectives: collective_plan.local_single_rank_collectives(),
        transport_collectives: collective_plan.transport_collectives(),
        max_workgroup_size: program.workgroup_size,
        static_storage_bytes: stats.static_storage_bytes,
    }
}

/// What a backend advertises, as named fields.
///
/// This used to be eight `bool` parameters in a row on
/// [`check_backend_capabilities`]. Every call site had to get eight
/// same-typed arguments in the right order, and transposing two of them
/// compiles, runs, and silently admits a program the device cannot execute or
/// refuses one it can. Naming them makes the transposition unspellable.
// Deliberately neither `#[non_exhaustive]` nor `Default`. Every backend crate
// builds this literal, so an added capability must fail to compile at each one
// until that backend states its answer. `#[non_exhaustive]` would forbid the
// literal outright and force a default-then-assign that absorbs a new field in
// silence, which is the failure this struct exists to prevent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackendSupport {
    /// Wave/subgroup collectives are available.
    pub subgroup_ops: bool,
    /// IEEE 754 binary16 operands are available.
    pub half_precision: bool,
    /// bfloat16 operands are available.
    pub brain_float: bool,
    /// `Node::IndirectDispatch` can be lowered.
    pub indirect_dispatch: bool,
    /// A `Node::Trap` propagates out of the kernel.
    pub trap_propagation: bool,
    /// Collective communication has real transport behind it.
    pub distributed_collectives: bool,
    /// A grid-scope barrier can be executed by some route the backend permits.
    ///
    /// Either the backend lowers a cooperative launch itself, or it accepts
    /// the host split that turns one barrier into sequential dispatches. This
    /// is not the same question as which route to take: it is whether any
    /// exists. A backend that lowers no whole-grid barrier and also refuses
    /// the split cannot run the program at any geometry, and a workgroup
    /// barrier cannot stand in for one, so refusing it here is what makes
    /// grid synchronisation a planning decision rather than an emitter
    /// failure discovered after a target has been chosen.
    pub grid_sync: bool,
    /// Largest workgroup the device accepts, per axis. A zero axis is unknown
    /// and never fires the size check.
    pub max_workgroup_size: [u32; 3],
}

/// Return `Ok(())` when a backend that advertises `support` can run a program
/// whose required set is `required`, otherwise return the missing-capability
/// explanation.
pub fn check_backend_capabilities(
    backend_id: &str,
    support: &BackendSupport,
    required: &RequiredCapabilities,
) -> Result<(), MissingCapability> {
    let BackendSupport {
        subgroup_ops: supports_subgroup_ops,
        half_precision: supports_half_precision,
        brain_float: supports_brain_float,
        indirect_dispatch: supports_indirect_dispatch,
        trap_propagation: supports_trap_propagation,
        distributed_collectives: supports_distributed_collectives,
        grid_sync: supports_grid_sync,
        max_workgroup_size,
    } = *support;
    let mut missing: Vec<String> = Vec::new();
    if required.subgroup_ops && !supports_subgroup_ops {
        missing.push("subgroup_ops".to_string());
    }
    if required.f16 && !supports_half_precision {
        missing.push("f16".to_string());
    }
    if required.bf16 && !supports_brain_float {
        missing.push("bf16".to_string());
    }
    if required.indirect_dispatch && !supports_indirect_dispatch {
        missing.push("indirect_dispatch".to_string());
    }
    if required.trap && !supports_trap_propagation {
        missing.push("trap_propagation".to_string());
    }
    if required.grid_sync && !supports_grid_sync {
        missing.push("grid_sync".to_string());
    }
    if required.distributed_collectives && !supports_distributed_collectives {
        missing.push("distributed_collectives".to_string());
        missing.push(format!(
            "distributed_collectives transport_collectives={} local_single_rank_collectives={}",
            required.transport_collectives, required.local_single_rank_collectives
        ));
    }
    for (axis, (req_size, max_size)) in required
        .max_workgroup_size
        .iter()
        .zip(max_workgroup_size.iter())
        .enumerate()
    {
        if *req_size > *max_size && *max_size != 0 {
            if !missing.iter().any(|item| item == "workgroup_size") {
                missing.push("workgroup_size".to_string());
            }
            missing.push(format!(
                "workgroup_size axis {axis} (requested {req_size}, max {max_size})"
            ));
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        Err(MissingCapability {
            backend: backend_id.to_string(),
            missing,
        })
    }
}

// The capability contract suite lives in
// `vyre-foundation/tests/capability_contracts.rs`: `scan`,
// `check_backend_capabilities`, `RequiredCapabilities` and `MissingCapability`
// are all public, so an inline copy would test the same surface twice.
