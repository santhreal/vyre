//! Immutable dispatch policy supplied by callers before backend execution.

use std::time::Duration;

use crate::backend::BackendError;

/// Immutable execution policy supplied by the caller before dispatch.
///
/// `DispatchConfig` is an additive, non-exhaustive struct so that new backend
/// options (conformance profiles, adapter hints, etc.) can be added without
/// breaking the frozen `VyreBackend::dispatch` signature. Backends must treat
/// every field as read-only policy and must not assume the presence of any
/// particular option.
///
/// # Examples
///
/// ```
/// use vyre_driver::DispatchConfig;
///
/// // DispatchConfig is `#[non_exhaustive]`; construct it through
/// // `default()` and overwrite the fields you want to change.
/// let mut config = DispatchConfig::default();
/// config.profile = Some("stress".to_string());
/// config.ulp_budget = None;
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DispatchConfig {
    /// Optional stable profile identifier such as `default`, `stress`, or a
    /// backend-defined conformance mode.
    pub profile: Option<String>,
    /// Optional maximum ULP error budget for approximate transcendental lowering.
    ///
    /// `None` and `Some(0)` require the strict target-text intrinsic path. A positive
    /// budget allows backends to select fast approximate intrinsic wrappers only
    /// when the wrapper contract is bounded by the supplied ULP ceiling.
    pub ulp_budget: Option<u8>,
    /// Optional timeout for the dispatch.
    pub timeout: Option<Duration>,
    /// Optional label for the dispatch (for debugging/profiling).
    pub label: Option<String>,
    /// Optional maximum output byte limit.
    pub max_output_bytes: Option<usize>,
    /// The complete launch a compiled artifact recorded, or a caller stated.
    ///
    /// A frozen launch is authoritative: the workgroup, the grid and the shared
    /// byte requirement are submitted exactly, no tuner sees the launch, and
    /// nothing infers a grid from buffer shapes. Stating it alongside any of the
    /// four override fields below is rejected rather than resolved, because the
    /// resolution would pick one authority and drop the other silently.
    pub launch: Option<crate::launch_directive::LaunchDirective>,
    /// Optional workgroup size override.
    ///
    /// When `Some`, the backend uses the supplied `[x, y, z]` workgroup size
    /// instead of the one declared on the [`vyre_foundation::ir::Program`].
    /// This lets callers tune workgroup sizing at dispatch time without
    /// cloning the program metadata. When `None` (the default), the backend
    /// falls back to `Program::workgroup_size`.
    pub workgroup_override: Option<[u32; 3]>,
    /// Optional grid size override (number of workgroups).
    ///
    /// When set, the backend launches the supplied workgroup count instead of
    /// the one inferred from the program's output buffer size. This is
    /// required for megakernels where the work queue length is managed through
    /// storage buffers rather than the primary output slot.
    pub grid_override: Option<[u32; 3]>,
    /// True per-invocation element/byte coverage count for an element-grid
    /// dispatch (e.g. a one-lane-per-byte scan: `Some(haystack_len)`).
    ///
    /// This exists SEPARATELY from [`grid_override`](Self::grid_override) because
    /// that field is OVERLOADED: for an element-grid dispatch it is the workgroup
    /// count derived from the input size, but for a MEGAKERNEL it is a work-queue
    /// length managed through storage buffers, the two cannot be told apart from
    /// the `[u32; 3]` alone. Backends that infer their dispatch coverage from
    /// buffer SHAPES rather than from a real GPU grid (the CPU reference
    /// interpreter, [`CpuRefBackend`](../../../vyre_driver_reference/index.html))
    /// cannot see the runtime scan length, so a byte-scan program would be
    /// under-dispatched to `haystack_len / 4` invocations and SILENTLY skip high
    /// positions (a Law-10 recall regression). An element-grid caller sets this to
    /// the true coverage so such a backend dispatches exactly what the GPU would;
    /// `None` (the default, and every megakernel) means "infer from buffer shapes"
    ///: so a megakernel is never over-run by a byte count that is not its grid.
    pub dispatch_elements: Option<u32>,
    /// True per-workgroup-axis dispatch grid `[x, y, z]` for a multi-dimensional
    /// element dispatch.
    ///
    /// This is the N-dimensional counterpart of
    /// [`dispatch_elements`](Self::dispatch_elements) (a 1-D floor). A backend that
    /// infers its coverage from buffer SHAPES rather than a real GPU grid (the CPU
    /// reference interpreter,
    /// [`CpuRefBackend`](../../../vyre_driver_reference/index.html)) distributes the
    /// dispatch only across workgroup axes whose size is greater than one, so a
    /// program that fans a `[256, 1, 1]` workgroup across `grid.y` (batched
    /// persistent-BFS runs one query per `grid.y` block) would collapse to
    /// `grid.y == 1` and SILENTLY compute only the first query (a Law-10
    /// under-coverage). A caller that knows the real grid, e.g. one block per
    /// query alongside the node domain the program's guard admits, sets it here so
    /// the interpreter covers every workgroup the GPU would. `None` (the default)
    /// keeps buffer-shape inference. When both this and `dispatch_elements` are set,
    /// this wins because it fully specifies the grid.
    pub dispatch_grid: Option<[u32; 3]>,
    /// Maximum back-to-back dispatch iterations the backend should run on
    /// the same persistent input/output handles before reading back the
    /// final outputs.
    ///
    /// `None` means one iteration. `Some(0)` is invalid: backends must reject
    /// it instead of silently rewriting caller policy.
    pub fixpoint_iterations: Option<u32>,
    /// Optional speculation policy.
    pub speculation: Option<crate::speculate::SpeculationMode>,
    /// Optional persistent-thread dispatch policy.
    pub persistent_thread: Option<crate::persistent::PersistentThreadMode>,
    /// Whether the backend should launch through its cooperative-grid API.
    ///
    /// A backend MUST reject `cooperative = true` with `UnsupportedFeature`
    /// when its `VyreBackend::supports_grid_sync()` returns `false`.
    pub cooperative: bool,
}

impl DispatchConfig {
    /// Construct a `DispatchConfig` from explicit fields in one call.
    /// Complement to `DispatchConfig::default()` for external crates
    /// that want all optional fields set up front.
    #[must_use]
    pub fn new(
        profile: Option<String>,
        ulp_budget: Option<u8>,
        timeout: Option<Duration>,
        label: Option<String>,
    ) -> Self {
        Self {
            profile,
            ulp_budget,
            timeout,
            label,
            max_output_bytes: None,
            launch: None,
            workgroup_override: None,
            grid_override: None,
            dispatch_elements: None,
            dispatch_grid: None,
            fixpoint_iterations: None,
            speculation: None,
            persistent_thread: None,
            cooperative: false,
        }
    }

    /// Workgroup shape this dispatch launches, when one is stated.
    ///
    /// A frozen launch answers first, so a consumer reads one field instead of
    /// deciding between a recorded shape and a tuner override.
    #[must_use]
    pub fn launch_workgroup(&self) -> Option<[u32; 3]> {
        match &self.launch {
            Some(launch) => Some(launch.workgroup()),
            None => self.workgroup_override,
        }
    }

    /// Workgroup count this dispatch launches, when one is stated.
    #[must_use]
    pub fn launch_grid(&self) -> Option<[u32; 3]> {
        match &self.launch {
            Some(launch) => Some(launch.grid()),
            None => self.grid_override,
        }
    }

    /// Grid a backend that infers coverage from buffer shapes must cover.
    #[must_use]
    pub fn coverage_grid(&self) -> Option<[u32; 3]> {
        match &self.launch {
            Some(launch) => Some(launch.grid()),
            None => self.dispatch_grid,
        }
    }

    /// Workgroup-shared bytes the launch reserves, zero when none is stated.
    #[must_use]
    pub fn launch_dynamic_shared_bytes(&self) -> u32 {
        match &self.launch {
            Some(launch) => launch.dynamic_shared_bytes(),
            None => 0,
        }
    }

    /// Reject a dispatch that states two launch authorities.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when a frozen launch arrives
    /// beside a tuner override. Resolving the pair would pick one shape and drop
    /// the other without saying so, and one of the two is a kernel nothing
    /// compiled.
    pub fn validate_launch_authority(&self, backend: &str) -> Result<(), BackendError> {
        // Destructured field by field: a new dispatch-shape field cannot be
        // added without deciding here whether it competes with a frozen launch.
        let Self {
            profile: _,
            ulp_budget: _,
            timeout: _,
            label: _,
            max_output_bytes: _,
            launch,
            workgroup_override,
            grid_override,
            dispatch_elements,
            dispatch_grid,
            fixpoint_iterations: _,
            speculation: _,
            persistent_thread: _,
            cooperative: _,
        } = self;
        let Some(launch) = launch else {
            return Ok(());
        };
        for (field, stated) in [
            ("workgroup_override", workgroup_override.is_some()),
            ("grid_override", grid_override.is_some()),
            ("dispatch_elements", dispatch_elements.is_some()),
            ("dispatch_grid", dispatch_grid.is_some()),
        ] {
            if stated {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: backend `{backend}` received a dispatch stating both the frozen launch {:?}/{:?} and `DispatchConfig::{field}`. \
                         Submit the frozen launch alone, or drop it and state the override alone.",
                        launch.workgroup(),
                        launch.grid(),
                    ),
                });
            }
        }
        Ok(())
    }
}
