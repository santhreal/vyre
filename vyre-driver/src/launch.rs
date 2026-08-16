//! Backend-neutral dispatch launch preparation.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use vyre_foundation::ir::{MemoryKind, Node, Program};

use crate::binding::Binding;
use crate::program_walks::{
    dispatch_element_count_for_program, infer_dispatch_grid_for_count,
    program_uses_launch_geometry_ids, try_dispatch_param_words_into,
};
use crate::tuner::{
    identity_fisher_q16, Mode, NaturalGradientPolicy, Tuner, TunerCache, TuningMeasurement,
    WORKGROUP_CANDIDATES,
};
use crate::validation::{validate_launch_geometry, LaunchGeometryLimits};
use crate::{BackendError, DispatchConfig};

const COLD_START_GRID_STEP_NS: u64 = 1_024;
const COLD_START_IDLE_LANE_NS: u64 = 8;
const COLD_START_TEMPERATURE_NS: u64 = 4_096;
const MAX_NATURAL_LAUNCH_CACHE_ENTRIES: usize = 4_096;

static NATURAL_LAUNCH_CACHE: OnceLock<Mutex<BTreeMap<NaturalLaunchCacheKey, NaturalLaunchEntry>>> =
    OnceLock::new();

/// Fully prepared launch metadata shared by concrete drivers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LaunchPlan {
    /// Logical element count passed to the lowered kernel.
    pub element_count: u32,
    /// Effective workgroup/block shape after dispatch overrides.
    pub workgroup: [u32; 3],
    /// Effective grid shape after dispatch overrides or inference.
    pub grid: [u32; 3],
    /// Per-buffer element-count metadata uploaded as the shared params buffer.
    pub param_words: Vec<u32>,
    /// Maximum preferred alignment across all launch bindings.
    ///
    /// Concrete drivers use this to pick upload staging and device-buffer
    /// allocation paths without re-inspecting Program buffer declarations.
    pub max_binding_alignment: usize,
}

impl LaunchPlan {
    /// Empty launch plan with reusable parameter-word storage.
    #[must_use]
    pub fn new() -> Self {
        Self {
            element_count: 1,
            workgroup: [1, 1, 1],
            grid: [1, 1, 1],
            param_words: Vec::new(),
            max_binding_alignment: 1,
        }
    }

    /// Prepare dispatch geometry and parameter words from a validated binding plan.
    ///
    /// # Errors
    ///
    /// Returns when caller overrides produce zero dimensions, overflow the
    /// logical launch element count, or exceed backend-reported launch limits.
    pub fn from_bindings(
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
    ) -> Result<Self, BackendError> {
        let mut plan = Self::new();
        plan.prepare_into(program, bindings, config, limits)?;
        Ok(plan)
    }

    /// Prepare dispatch geometry and parameter words, reusing this plan's buffers.
    ///
    /// # Errors
    ///
    /// Returns when caller overrides produce zero dimensions, overflow the
    /// logical launch element count, or exceed backend-reported launch limits.
    pub fn prepare_into(
        &mut self,
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
    ) -> Result<(), BackendError> {
        self.prepare_into_for_mode(program, bindings, config, limits, Mode::from_env())
    }

    fn prepare_into_for_mode(
        &mut self,
        program: &Program,
        bindings: &[Binding],
        config: &DispatchConfig,
        limits: LaunchGeometryLimits,
        mode: Mode,
    ) -> Result<(), BackendError> {
        let workgroup =
            effective_launch_workgroup_for_mode(program, bindings, config, limits, mode);
        validate_launch_geometry(workgroup, [1, 1, 1], limits)?;
        let element_count = launch_element_count(program, bindings, workgroup, config, limits)?;
        let grid = match config.grid_override {
            Some(grid) => grid,
            None => {
                // Non-1D workgroups need an explicit grid_override  -
                // there's no single right way to map an unknown
                // element_count across N×M (or N×M×K) thread tiles,
                // and silently picking one produces silently-wrong
                // results. Force the caller to make the choice.
                if workgroup[1] != 1 || workgroup[2] != 1 {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: backend `{}` requires DispatchConfig::grid_override for non-1D workgroups. \
                             workgroup={:?} has no unambiguous default grid; set grid_override to the logical [x, y, z] you want.",
                            limits.backend, workgroup,
                        ),
                    });
                }
                infer_dispatch_grid_for_count(element_count, workgroup)?
            }
        };
        validate_launch_geometry(workgroup, grid, limits)?;
        self.element_count = element_count;
        self.workgroup = workgroup;
        self.grid = grid;
        self.max_binding_alignment = bindings
            .iter()
            .map(|binding| binding.preferred_alignment)
            .max()
            .unwrap_or(1);
        try_dispatch_param_words_into(bindings, element_count, &mut self.param_words).map_err(
            |error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {}: dispatch ABI parameter staging failed: {error}",
                    limits.backend
                ),
            },
        )?;
        Ok(())
    }
}

impl Default for LaunchPlan {
    fn default() -> Self {
        Self::new()
    }
}

fn launch_element_count(
    program: &Program,
    bindings: &[Binding],
    workgroup: [u32; 3],
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
) -> Result<u32, BackendError> {
    let inferred = dispatch_element_count_for_program(program, bindings);
    let Some(grid) = config.grid_override else {
        return Ok(inferred);
    };
    if workgroup.contains(&0) || grid.contains(&0) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} grid_override and workgroup dimensions must all be non-zero.",
                limits.backend
            ),
        });
    }
    grid[0]
        .checked_mul(workgroup[0])
        .filter(|count| *count != 0)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {} grid_override.x * workgroup_size.x must fit in u32.",
                limits.backend
            ),
        })
}

fn effective_launch_workgroup_for_mode(
    program: &Program,
    bindings: &[Binding],
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    mode: Mode,
) -> [u32; 3] {
    let element_count = dispatch_element_count_for_program(program, bindings);
    resolve_launch_workgroup_for_mode(program, config, limits, element_count, mode)
}

/// Resolve the backend-visible workgroup shape for a dispatch.
///
/// Explicit caller overrides remain authoritative. When no override is
/// supplied and `VYRE_AUTOTUNER` resolves to natural-gradient mode, eligible
/// 1D storage-only kernels receive a deterministic natural-gradient cold-start
/// workgroup before grid inference.
#[must_use]
pub fn resolve_launch_workgroup(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
) -> [u32; 3] {
    resolve_launch_workgroup_for_mode(program, config, limits, element_count, Mode::from_env())
}

/// Where a launch's workgroup shape comes from.
///
/// A program compiled through the whole-program compiler carries a geometry the
/// compiler searched for and recorded in the artifact, and the emitted module
/// declares that shape. Launching such a module at another width runs a kernel
/// nobody compiled, so the recorded geometry is authoritative and the launch
/// tuner never sees the launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaunchGeometry {
    /// No compiled artifact governs this launch, so the tuner may choose a width.
    Untracked,
    /// The artifact recorded this workgroup for the node being launched.
    Compiled([u32; 3]),
}

impl LaunchGeometry {
    /// Read the geometry a target descriptor recorded for one artifact node.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when the record is absent. A
    /// descriptor with a zero extent recorded nothing, and falling back to a
    /// declared or tuned width there would launch a shape the artifact never
    /// authenticated.
    pub fn from_recorded(workgroup: [u32; 3], backend: &str) -> Result<Self, BackendError> {
        if workgroup.contains(&0) {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: backend `{backend}` received an authenticated target whose descriptor records no workgroup geometry ({workgroup:?}). \
                     Recompile the artifact with a compiler that records the selected geometry for every node; the driver must not choose one."
                ),
            });
        }
        Ok(Self::Compiled(workgroup))
    }
}

/// Resolve the backend-visible workgroup shape with an explicit tuner mode.
///
/// This is public so backends whose shader/pipeline compilation must include
/// the selected workgroup size can derive the same shape before lowering.
#[must_use]
pub fn resolve_launch_workgroup_for_mode(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    mode: Mode,
) -> [u32; 3] {
    resolve_launch_workgroup_for_geometry(
        program,
        config,
        limits,
        element_count,
        mode,
        LaunchGeometry::Untracked,
    )
}

/// Resolve the backend-visible workgroup shape against a launch's geometry source.
///
/// A recorded compiled geometry outranks every dispatch override and the launch
/// tuner. Only an untracked launch reaches the tuner, so a compiled group's
/// workgroup cannot change between compilation and dispatch.
#[must_use]
pub fn resolve_launch_workgroup_for_geometry(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    mode: Mode,
    geometry: LaunchGeometry,
) -> [u32; 3] {
    if let LaunchGeometry::Compiled(workgroup) = geometry {
        return workgroup;
    }
    if let Some(workgroup) = config.workgroup_override {
        return workgroup;
    }
    let declared = program.workgroup_size();
    if mode != Mode::NaturalGradient || config.grid_override.is_some() {
        return declared;
    }
    natural_gradient_cold_start_workgroup(program, declared, element_count, limits)
        .unwrap_or(declared)
}

/// Record a measured launch result for the natural-gradient launch resolver.
///
/// Backends should call this only after a real dispatch timing is available.
/// The function returns `true` when the measurement was accepted into the
/// bounded feedback cache. Explicit caller overrides, explicit grid launches,
/// non-natural tuner modes, non-1D kernels, workgroup-local scratch kernels,
/// zero timings, and out-of-limit candidates are ignored so measured feedback
/// never changes kernel semantics.
#[must_use]
pub fn record_launch_measurement(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    observed_workgroup: [u32; 3],
    elapsed_ns: u64,
) -> bool {
    record_launch_measurement_for_mode(
        program,
        config,
        limits,
        element_count,
        observed_workgroup,
        elapsed_ns,
        Mode::from_env(),
    )
}

fn record_launch_measurement_for_mode(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    observed_workgroup: [u32; 3],
    elapsed_ns: u64,
    mode: Mode,
) -> bool {
    record_launch_measurement_for_mode_with_store(
        program,
        config,
        limits,
        element_count,
        observed_workgroup,
        elapsed_ns,
        mode,
        None,
    )
}

fn record_launch_measurement_for_mode_with_store(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    observed_workgroup: [u32; 3],
    elapsed_ns: u64,
    mode: Mode,
    persistent_path: Option<&Path>,
) -> bool {
    if mode != Mode::NaturalGradient
        || elapsed_ns == 0
        || config.workgroup_override.is_some()
        || config.grid_override.is_some()
        || observed_workgroup[1] != 1
        || observed_workgroup[2] != 1
        || !candidate_x_fits_limits(observed_workgroup[0], limits)
    {
        return false;
    }
    let declared = program.workgroup_size();
    if !is_natural_gradient_launch_tunable(program, declared, element_count) {
        return false;
    }
    let cache_key = NaturalLaunchCacheKey::new(program, declared, element_count, limits);
    let mut measurements = natural_launch_cache_measurements(cache_key).unwrap_or_default();
    measurements
        .entry(observed_workgroup)
        .and_modify(|best_ns| *best_ns = (*best_ns).min(elapsed_ns))
        .or_insert(elapsed_ns);
    let Some(selected) =
        select_natural_launch_workgroup(declared, element_count, limits, Some(&measurements))
    else {
        return false;
    };
    natural_launch_cache_set(
        cache_key,
        NaturalLaunchEntry {
            selected,
            measurements,
        },
    );
    if let Err(error) =
        persist_natural_launch_selection(cache_key, limits, selected, persistent_path)
    {
        tracing::debug!(
            error,
            "natural-gradient launch feedback accepted in memory but could not persist"
        );
    }
    true
}

fn natural_gradient_cold_start_workgroup(
    program: &Program,
    declared: [u32; 3],
    element_count: u32,
    limits: LaunchGeometryLimits,
) -> Option<[u32; 3]> {
    natural_gradient_cold_start_workgroup_with_store(program, declared, element_count, limits, None)
}

fn natural_gradient_cold_start_workgroup_with_store(
    program: &Program,
    declared: [u32; 3],
    element_count: u32,
    limits: LaunchGeometryLimits,
    persistent_path: Option<&Path>,
) -> Option<[u32; 3]> {
    if !is_natural_gradient_launch_tunable(program, declared, element_count) {
        return None;
    }
    let cache_key = NaturalLaunchCacheKey::new(program, declared, element_count, limits);
    if let Some(cached) = natural_launch_cache_get(cache_key) {
        return Some(cached);
    }
    if let Some(persisted) = natural_launch_cache_get_persisted(cache_key, limits, persistent_path)
    {
        natural_launch_cache_set(
            cache_key,
            NaturalLaunchEntry {
                selected: persisted,
                measurements: BTreeMap::new(),
            },
        );
        return Some(persisted);
    }

    let selected = select_natural_launch_workgroup(declared, element_count, limits, None)?;
    natural_launch_cache_set(
        cache_key,
        NaturalLaunchEntry {
            selected,
            measurements: BTreeMap::new(),
        },
    );
    Some(selected)
}

fn select_natural_launch_workgroup(
    declared: [u32; 3],
    element_count: u32,
    limits: LaunchGeometryLimits,
    measurements: Option<&BTreeMap<[u32; 3], u64>>,
) -> Option<[u32; 3]> {
    let peak_resident = peak_resident_threads_per_compute_unit(declared[0], limits);
    let mut samples = Vec::with_capacity(WORKGROUP_CANDIDATES.len() + 1);
    for candidate_x in WORKGROUP_CANDIDATES
        .iter()
        .copied()
        .chain(std::iter::once(declared[0]))
    {
        if !candidate_x_fits_limits(candidate_x, limits)
            || samples
                .iter()
                .any(|sample: &TuningMeasurement| sample.workgroup_size[0] == candidate_x)
        {
            continue;
        }
        let workgroup_size = [candidate_x, 1, 1];
        let elapsed_ns = match measurements.and_then(|measured| measured.get(&workgroup_size)) {
            Some(&measured_ns) => measured_ns,
            None if cold_start_admits_width(candidate_x, limits, peak_resident) => {
                estimate_cold_start_latency_ns(element_count, candidate_x)
            }
            None => continue,
        };
        samples.push(TuningMeasurement {
            workgroup_size,
            elapsed_ns,
        });
    }
    if let Some(measured) = measurements {
        for (&workgroup_size, &elapsed_ns) in measured {
            if workgroup_size[1] != 1
                || workgroup_size[2] != 1
                || elapsed_ns == 0
                || !candidate_x_fits_limits(workgroup_size[0], limits)
                || samples
                    .iter()
                    .any(|sample| sample.workgroup_size == workgroup_size)
            {
                continue;
            }
            samples.push(TuningMeasurement {
                workgroup_size,
                elapsed_ns,
            });
        }
    }

    if samples.len() < 2 {
        return None;
    }
    NaturalGradientPolicy {
        temperature_ns: COLD_START_TEMPERATURE_NS,
    }
    .suggest(&samples, &identity_fisher_q16(samples.len()))
    .ok()
    .map(|step| step.selected_workgroup_size)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct NaturalLaunchCacheKey {
    fingerprint: [u8; 32],
    declared: [u32; 3],
    element_count: u32,
    max_threads_per_block: u32,
    max_block_dim: [u32; 3],
    max_grid_dim: [u32; 3],
    max_threads_per_sm: u32,
}

impl NaturalLaunchCacheKey {
    fn new(
        program: &Program,
        declared: [u32; 3],
        element_count: u32,
        limits: LaunchGeometryLimits,
    ) -> Self {
        Self {
            fingerprint: program.fingerprint(),
            declared,
            element_count,
            max_threads_per_block: limits.max_threads_per_block,
            max_block_dim: limits.max_block_dim,
            max_grid_dim: limits.max_grid_dim,
            max_threads_per_sm: limits.max_threads_per_sm,
        }
    }

    fn persistent_key(self) -> String {
        let mut hasher = blake3::Hasher::new();
        // v2 adds the per-compute-unit thread budget. It selects the width, so
        // a v1 entry may record a choice made without it and must not be read
        // back as if it had been.
        hasher.update(b"vyre-natural-launch-feedback-v2\0");
        hasher.update(&self.fingerprint);
        for axis in self.declared {
            hasher.update(&axis.to_le_bytes());
        }
        hasher.update(&self.element_count.to_le_bytes());
        hasher.update(&self.max_threads_per_block.to_le_bytes());
        for axis in self.max_block_dim {
            hasher.update(&axis.to_le_bytes());
        }
        for axis in self.max_grid_dim {
            hasher.update(&axis.to_le_bytes());
        }
        hasher.update(&self.max_threads_per_sm.to_le_bytes());
        let digest = hasher.finalize();
        let mut key = String::with_capacity(74);
        key.push_str("launch-v2-");
        crate::pipeline::hashing::push_lower_hex(digest.as_bytes(), &mut key);
        key
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]

struct NaturalLaunchEntry {
    selected: [u32; 3],
    measurements: BTreeMap<[u32; 3], u64>,
}

fn natural_launch_cache_get(key: NaturalLaunchCacheKey) -> Option<[u32; 3]> {
    let cache = NATURAL_LAUNCH_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    guard.get(&key).map(|entry| entry.selected)
}

fn natural_launch_cache_measurements(
    key: NaturalLaunchCacheKey,
) -> Option<BTreeMap<[u32; 3], u64>> {
    let cache = NATURAL_LAUNCH_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    guard.get(&key).map(|entry| entry.measurements.clone())
}

fn natural_launch_cache_set(key: NaturalLaunchCacheKey, value: NaturalLaunchEntry) {
    let cache = NATURAL_LAUNCH_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|poison| poison.into_inner());
    if guard.len() >= MAX_NATURAL_LAUNCH_CACHE_ENTRIES && !guard.contains_key(&key) {
        if let Some(oldest) = guard.keys().next().copied() {
            guard.remove(&oldest);
        }
    }
    guard.insert(key, value);
}

#[cfg(test)]
fn natural_launch_cache_remove(key: NaturalLaunchCacheKey) {
    if let Some(cache) = NATURAL_LAUNCH_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.remove(&key);
        }
    }
}

fn natural_launch_cache_get_persisted(
    key: NaturalLaunchCacheKey,
    limits: LaunchGeometryLimits,
    persistent_path: Option<&Path>,
) -> Option<[u32; 3]> {
    let path = persistent_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| natural_launch_persistent_cache_path(limits));
    let selected = TunerCache::load(&path).ok()?.get(&key.persistent_key())?;
    valid_persisted_launch_selection(selected, limits).then_some(selected)
}

fn persist_natural_launch_selection(
    key: NaturalLaunchCacheKey,
    limits: LaunchGeometryLimits,
    selected: [u32; 3],
    persistent_path: Option<&Path>,
) -> Result<(), String> {
    let path = persistent_path
        .map(Path::to_path_buf)
        .unwrap_or_else(|| natural_launch_persistent_cache_path(limits));
    persist_natural_launch_selection_to_path(key, selected, &path)
}

fn persist_natural_launch_selection_to_path(
    key: NaturalLaunchCacheKey,
    selected: [u32; 3],
    path: &Path,
) -> Result<(), String> {
    let mut cache = TunerCache::load(path)?;
    while cache.entries.len() >= MAX_NATURAL_LAUNCH_CACHE_ENTRIES {
        let Some(oldest) = cache.entries.keys().next().cloned() else {
            break;
        };
        cache.entries.remove(&oldest);
    }
    cache.set(key.persistent_key(), selected);
    cache.save(path)
}

fn natural_launch_persistent_cache_path(limits: LaunchGeometryLimits) -> PathBuf {
    Tuner::cache_path_for_adapter(&natural_launch_persistent_adapter_key(limits))
}

fn natural_launch_persistent_adapter_key(limits: LaunchGeometryLimits) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"vyre-natural-launch-adapter-v2\0");
    hasher.update(limits.backend.as_bytes());
    hasher.update(&limits.max_threads_per_block.to_le_bytes());
    for axis in limits.max_block_dim {
        hasher.update(&axis.to_le_bytes());
    }
    for axis in limits.max_grid_dim {
        hasher.update(&axis.to_le_bytes());
    }
    hasher.update(&limits.max_threads_per_sm.to_le_bytes());
    let digest = hasher.finalize();
    let mut key = String::with_capacity(92);
    key.push_str("natural-launch-feedback-v2-");
    crate::pipeline::hashing::push_lower_hex(digest.as_bytes(), &mut key);
    key
}

fn valid_persisted_launch_selection(selected: [u32; 3], limits: LaunchGeometryLimits) -> bool {
    selected[1] == 1 && selected[2] == 1 && candidate_x_fits_limits(selected[0], limits)
}

fn is_natural_gradient_launch_tunable(
    program: &Program,
    declared: [u32; 3],
    element_count: u32,
) -> bool {
    declared[0] != 0
        && declared[1] == 1
        && declared[2] == 1
        && element_count != 0
        && program
            .entry
            .iter()
            .any(|node| !matches!(node, Node::Return))
        && !program.non_composable_with_self
        && !program_uses_launch_geometry_ids(program)
        && program
            .buffers
            .iter()
            .all(|buffer| buffer.kind() != MemoryKind::Shared)
}

fn candidate_x_fits_limits(candidate_x: u32, limits: LaunchGeometryLimits) -> bool {
    candidate_x != 0
        && candidate_x <= limits.max_threads_per_block
        && candidate_x <= limits.max_block_dim[0]
}

/// Highest resident thread count per compute unit that any admissible width
/// reaches on this device.
///
/// `None` when the backend reports no per-unit thread budget. That is the
/// inert case: no width is preferred over another on residency grounds and
/// cold start selects exactly what it selected before residency entered this
/// decision.
fn peak_resident_threads_per_compute_unit(
    declared_x: u32,
    limits: LaunchGeometryLimits,
) -> Option<u32> {
    WORKGROUP_CANDIDATES
        .iter()
        .copied()
        .chain(std::iter::once(declared_x))
        .filter(|&candidate_x| candidate_x_fits_limits(candidate_x, limits))
        .filter_map(|candidate_x| limits.resident_threads_per_compute_unit(candidate_x))
        .max()
}

/// Whether cold start may propose `candidate_x` with no measurement behind it.
///
/// Residency ranks ahead of the latency estimate because the estimate cannot
/// see occupancy at all: it counts workgroups and idle tail lanes, so it always
/// favours the widest candidate and the tail penalty vanishes entirely when the
/// element count is a multiple of that width. Resident threads per unit is
/// `(max_threads_per_sm / width) * width` with an integral division, so against
/// a 1536-thread unit a 1024-wide group hosts one block and strands 512 slots,
/// while 32 through 512 all host enough blocks to fill all 1536. Only widths
/// tying for the peak survive here; the latency estimate then breaks that tie,
/// and it breaks it toward the widest survivor.
///
/// Ranking widest-first is also what keeps the thread-only residency model
/// sound. It ignores the device-reported cap on blocks per unit, so it
/// overstates residency at the narrow end: measured on an RTX 5090, whose cap
/// is 24, a 32-wide group gets 24 blocks and 768 resident threads where this
/// model predicts 48 and 1536, a factor of two. Selecting the widest survivor
/// never reaches that regime, and on this device the pick lands at 512 with 3
/// blocks per SM, well clear of the cap. A future tie-break toward narrower
/// widths would need the block cap as a second input.
///
/// This gate applies to cold start only. A width carrying a real measurement
/// bypasses it entirely, so measured feedback can still choose a width cold
/// start would never propose.
fn cold_start_admits_width(
    candidate_x: u32,
    limits: LaunchGeometryLimits,
    peak_resident: Option<u32>,
) -> bool {
    let Some(peak_resident) = peak_resident else {
        return true;
    };
    limits
        .resident_threads_per_compute_unit(candidate_x)
        .is_none_or(|resident| resident >= peak_resident)
}

fn estimate_cold_start_latency_ns(element_count: u32, candidate_x: u32) -> u64 {
    let groups = u64::from(element_count.div_ceil(candidate_x));
    let scheduled_lanes = groups.saturating_mul(u64::from(candidate_x));
    let idle_lanes = scheduled_lanes.saturating_sub(u64::from(element_count));
    groups
        .saturating_mul(COLD_START_GRID_STEP_NS)
        .saturating_add(idle_lanes.saturating_mul(COLD_START_IDLE_LANE_NS))
}

/// Compute the shared VSA program fingerprint used by backend caches.
#[must_use]
pub fn program_vsa_fingerprint(program: &Program) -> Vec<u32> {
    program_vsa_fingerprint_words(program).to_vec()
}

/// Compute the shared VSA program fingerprint without heap allocation.
#[must_use]
pub fn program_vsa_fingerprint_words(program: &Program) -> [u32; 8] {
    let fingerprint = program.fingerprint();
    let mut words = [0u32; 8];
    for (word, chunk) in words.iter_mut().zip(fingerprint.chunks_exact(4)) {
        *word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    words
}

// Inline: the suite drives the `#[cfg(test)]` `natural_launch_cache_remove`, which an integration
// test does not compile.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::BindingRole;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

    #[test]
    fn program_vsa_fingerprint_words_match_wire_decoder() {
        let program = Program::wrapped(vec![], [64, 1, 1], vec![]);
        let words = program_vsa_fingerprint_words(&program);
        let fingerprint = program.fingerprint();

        for (index, chunk) in fingerprint.chunks_exact(4).enumerate() {
            assert_eq!(
                words[index],
                u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
            );
        }
        assert_eq!(program_vsa_fingerprint(&program), words.to_vec());
    }

    #[test]
    fn launch_plan_prepare_into_reuses_param_words() {
        let program = Program::wrapped(vec![], [64, 1, 1], vec![]);
        let bindings = vec![Binding {
            name: std::sync::Arc::from("input"),
            binding: 0,
            buffer_index: 0,
            role: BindingRole::Input,
            element_size: 4,
            preferred_alignment: 64,
            element_count: 7,
            static_byte_len: Some(28),
            input_index: Some(0),
            output_index: None,
        }];
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let mut plan = LaunchPlan {
            param_words: Vec::with_capacity(8),
            ..LaunchPlan::new()
        };
        let ptr = plan.param_words.as_ptr();
        plan.prepare_into(&program, &bindings, &DispatchConfig::default(), limits)
            .unwrap();
        assert_eq!(plan.element_count, 7);
        assert_eq!(plan.grid, [1, 1, 1]);
        assert_eq!(plan.param_words, vec![7, 7]);
        assert_eq!(plan.max_binding_alignment, 64);
        assert_eq!(plan.param_words.as_ptr(), ptr);
    }

    #[test]
    fn natural_gradient_launch_tunes_safe_1d_storage_program() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4096)],
            [32, 1, 1],
            vec![],
        );
        let bindings = vec![Binding {
            name: std::sync::Arc::from("out"),
            binding: 0,
            buffer_index: 0,
            role: BindingRole::Output,
            element_size: 4,
            preferred_alignment: 128,
            element_count: 4096,
            static_byte_len: Some(16_384),
            input_index: None,
            output_index: Some(0),
        }];
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let mut plan = LaunchPlan::new();

        plan.prepare_into_for_mode(
            &program,
            &bindings,
            &DispatchConfig::default(),
            limits,
            Mode::NaturalGradient,
        )
        .expect("Fix: safe 1D storage launch should accept natural-gradient cold start");

        assert_eq!(
            plan.workgroup,
            [512, 1, 1],
            "Fix: cold start must pick the widest width that keeps every resident thread slot usable. Was [1024,1,1], which is 1536/1024 = 1 block per SM and 512 stranded slots on every SM."
        );
        assert_eq!(
            limits.resident_threads_per_compute_unit(plan.workgroup[0]),
            Some(1536),
            "Fix: the chosen width must strand no per-SM thread slot when a candidate dividing 1536 evenly exists."
        );
        assert_eq!(plan.grid, [8, 1, 1]);
        assert_eq!(plan.element_count, 4096);
    }

    #[test]
    fn natural_gradient_launch_preserves_declared_shape_for_local_workgroup_ids() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out_local_ids", 0, DataType::U32).with_count(4096)],
            [1024, 1, 1],
            vec![
                Node::let_bind("lane", Expr::LocalId { axis: 0 }),
                Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
                Node::let_bind(
                    "global",
                    Expr::add(
                        Expr::mul(Expr::var("block"), Expr::u32(1024)),
                        Expr::var("lane"),
                    ),
                ),
                Node::store("out_local_ids", Expr::var("global"), Expr::var("lane")),
            ],
        );
        let bindings = vec![Binding {
            name: std::sync::Arc::from("out_local_ids"),
            binding: 0,
            buffer_index: 0,
            role: BindingRole::Output,
            element_size: 4,
            preferred_alignment: 128,
            element_count: 4096,
            static_byte_len: Some(16_384),
            input_index: None,
            output_index: Some(0),
        }];
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };

        assert_eq!(
            effective_launch_workgroup_for_mode(
                &program,
                &bindings,
                &DispatchConfig::default(),
                limits,
                Mode::NaturalGradient,
            ),
            [1024, 1, 1],
            "Fix: automatic launch tuning must not change kernels whose LocalId/WorkgroupId arithmetic makes workgroup shape semantic."
        );
    }

    #[test]
    fn measured_launch_feedback_overrides_heuristic_cold_start() {
        let dir = tempfile::tempdir()
            .expect("Fix: measured launch feedback test needs an isolated tuner cache");
        let path = dir.path().join("launch-feedback.toml");
        let program = Program::wrapped(
            vec![BufferDecl::output("out_feedback_isolated", 0, DataType::U32).with_count(8192)],
            [32, 1, 1],
            vec![],
        );
        let config = DispatchConfig::default();
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 8192, limits);
        natural_launch_cache_remove(key);

        assert_eq!(
            natural_gradient_cold_start_workgroup_with_store(
                &program,
                [32, 1, 1],
                8192,
                limits,
                Some(&path),
            ),
            Some([512, 1, 1]),
            "Fix: this pins the cold-start selector's output, not a required constant. It was [1024,1,1] under a heuristic with no occupancy term at all, so the old message's claim of an occupancy-efficient shape described the opposite of what it selected."
        );
        assert!(
            record_launch_measurement_for_mode_with_store(
                &program,
                &config,
                limits,
                8192,
                [64, 1, 1],
                1,
                Mode::NaturalGradient,
                Some(&path),
            ),
            "Fix: natural-gradient resolver must accept measured backend timing for safe 1D launches."
        );
        assert_eq!(
            natural_gradient_cold_start_workgroup_with_store(
                &program,
                [32, 1, 1],
                8192,
                limits,
                Some(&path),
            ),
            Some([64, 1, 1]),
            "Fix: measured launch feedback must steer future automatic launch choices."
        );
    }

    #[test]
    fn persisted_launch_feedback_rehydrates_measured_selection() {
        let dir = tempfile::tempdir()
            .expect("Fix: launch feedback persistence test needs a temporary cache directory");
        let path = dir.path().join("launch-feedback.toml");
        let program = Program::wrapped(
            vec![BufferDecl::output("out_persisted", 0, DataType::U32).with_count(16_384)],
            [32, 1, 1],
            vec![],
        );
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 16_384, limits);
        natural_launch_cache_remove(key);

        persist_natural_launch_selection_to_path(key, [64, 1, 1], &path)
            .expect("Fix: measured launch feedback should persist through the tuner cache format");

        assert_eq!(
            natural_gradient_cold_start_workgroup_with_store(
                &program,
                [32, 1, 1],
                16_384,
                limits,
                Some(&path),
            ),
            Some([64, 1, 1]),
            "Fix: automatic launch resolution must rehydrate measured feedback from the bounded tuner cache before falling back to heuristics."
        );
    }

    #[test]
    fn natural_gradient_launch_preserves_explicit_and_shared_memory_shapes() {
        let program = Program::wrapped(
            vec![
                BufferDecl::output("out", 0, DataType::U32).with_count(4096),
                BufferDecl::workgroup("scratch", 64, DataType::U32),
            ],
            [64, 1, 1],
            vec![],
        );
        let bindings = vec![Binding {
            name: std::sync::Arc::from("out"),
            binding: 0,
            buffer_index: 0,
            role: BindingRole::Output,
            element_size: 4,
            preferred_alignment: 128,
            element_count: 4096,
            static_byte_len: Some(16_384),
            input_index: None,
            output_index: Some(0),
        }];
        let limits = LaunchGeometryLimits {
            backend: "test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        };
        let mut config = DispatchConfig::default();
        config.workgroup_override = Some([256, 1, 1]);

        assert_eq!(
            effective_launch_workgroup_for_mode(
                &program,
                &bindings,
                &config,
                limits,
                Mode::NaturalGradient,
            ),
            [256, 1, 1],
            "Fix: explicit dispatch workgroup overrides must remain authoritative."
        );

        let default_config = DispatchConfig::default();
        assert_eq!(
            effective_launch_workgroup_for_mode(
                &program,
                &bindings,
                &default_config,
                limits,
                Mode::NaturalGradient,
            ),
            [64, 1, 1],
            "Fix: workgroup-local scratch kernels must keep their declared shape."
        );
    }

    // Reproducing test for: launch-cache-mutex-poison-silent-fallback
    // Before fix: .lock().ok() silently returned None on mutex poison, causing a silent
    // fallback from feedback-informed to cold-start workgroup selection.
    // After fix: .unwrap_or_else(|p| p.into_inner()) recovers the guard and preserves
    // accumulated timing data even after a thread panics while holding the lock.
    // Reproducing test for: launch-cache-measurements-unwrap-or-default-silent-feedback-loss
    // Before fix: natural_launch_cache_measurements returned None on mutex poison and
    // record_launch_measurement_for_mode_with_store would .unwrap_or_default() that None,
    // overwriting all prior measurement history with a single-sample empty map.
    // After fix (driven by the mutex fix): None from cache_measurements means genuinely
    // no prior entry, not a poison-induced data loss. The measurement path correctly starts
    // from an empty map only when no prior measurements exist.
    #[test]
    fn record_launch_measurement_starts_fresh_only_when_no_prior_history_exists() {
        let dir = tempfile::tempdir()
            .expect("Fix: measurement history test needs a temporary cache directory");
        let path = dir.path().join("measurements-test.toml");
        let program = Program::wrapped(
            vec![BufferDecl::output("out_meas_history", 0, DataType::U32).with_count(4096)],
            [32, 1, 1],
            vec![],
        );
        let config = DispatchConfig::default();
        let limits = LaunchGeometryLimits {
            backend: "test-measurements",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 0,
        };
        let key = NaturalLaunchCacheKey::new(&program, [32, 1, 1], 4096, limits);
        natural_launch_cache_remove(key);

        // First measurement accepted (starts from empty, correct).
        assert!(
            record_launch_measurement_for_mode_with_store(
                &program,
                &config,
                limits,
                4096,
                [256, 1, 1],
                100,
                Mode::NaturalGradient,
                Some(&path),
            ),
            "Fix: first measurement must be accepted into the cache"
        );

        // Read back the selection (must be [256, 1, 1] (only candidate with real timing)).
        let after_first = natural_launch_cache_get(key);
        assert!(
            after_first.is_some(),
            "Fix: cache must hold a selection after the first measurement"
        );

        // Second measurement with a *faster* timing for a different candidate.
        // The history from the first must be preserved (not replaced by an empty map).
        assert!(
            record_launch_measurement_for_mode_with_store(
                &program,
                &config,
                limits,
                4096,
                [128, 1, 1],
                50,
                Mode::NaturalGradient,
                Some(&path),
            ),
            "Fix: second measurement must be accepted into the cache"
        );

        let measurements = natural_launch_cache_measurements(key)
            .expect("Fix: cache must hold measurements after two records");
        assert!(
            measurements.len() >= 2,
            "Fix: measurement history must accumulate across calls, got {} entries, expected >= 2",
            measurements.len()
        );
        assert_eq!(
            measurements.get(&[256, 1, 1]),
            Some(&100),
            "Fix: first measurement (workgroup=[256,1,1], 100ns) must be retained in history"
        );
        assert_eq!(
            measurements.get(&[128, 1, 1]),
            Some(&50),
            "Fix: second measurement (workgroup=[128,1,1], 50ns) must be present in history"
        );
    }

    /// Streaming multiprocessors on the RTX 5090 this defect was measured on.
    const RTX_5090_SM_COUNT: u32 = 170;

    /// Launch limits shaped like that RTX 5090: 1024 threads per block and a
    /// 1536-thread per-SM residency budget, which 1024 does not divide.
    fn blackwell_5090_limits() -> LaunchGeometryLimits {
        LaunchGeometryLimits {
            backend: "blackwell-5090-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 1536,
        }
    }

    /// A 1-D storage-only program the natural-gradient resolver treats as
    /// tunable: no `LocalId`/`WorkgroupId` arithmetic, no workgroup scratch,
    /// and composable with itself, so none of the early-out gates fire.
    ///
    /// Each caller passes a distinct output name because the resolver memoizes
    /// on the program fingerprint.
    fn tunable_1d_program(output: &'static str, element_count: u32, declared: [u32; 3]) -> Program {
        Program::wrapped(
            vec![BufferDecl::output(output, 0, DataType::U32).with_count(element_count)],
            declared,
            vec![],
        )
    }

    /// Cold start must never choose a width that strands per-SM thread slots
    /// while a width dividing the budget evenly is available.
    ///
    /// Blocks per SM is an integral division, so on a 1536-thread SM a
    /// 1024-wide group hosts exactly one block and leaves 512 of every SM's
    /// 1536 slots unusable, a third of the device idle by arithmetic. Every
    /// candidate from 32 to 512 divides 1536 exactly. The element counts below
    /// span both sides of the old estimate's blind spot: multiples of 1024,
    /// where its idle-lane penalty vanishes and the widest candidate won
    /// outright, and counts with a tail.
    #[test]
    fn cold_start_never_strands_resident_thread_slots_when_an_even_divisor_exists() {
        let limits = blackwell_5090_limits();
        for (output, element_count) in [
            ("out_no_strand_1k", 1024u32),
            ("out_no_strand_4k", 4096),
            ("out_no_strand_64k", 65_536),
            ("out_no_strand_tail", 4097),
            ("out_no_strand_100k", 100_000),
        ] {
            let program = tunable_1d_program(output, element_count, [32, 1, 1]);
            let resolved = resolve_launch_workgroup_for_mode(
                &program,
                &DispatchConfig::default(),
                limits,
                element_count,
                Mode::NaturalGradient,
            );
            let resident = limits.resident_threads_per_compute_unit(resolved[0]);
            assert_eq!(
                resident,
                Some(1536),
                "Fix: cold start chose {resolved:?} for {element_count} elements, leaving {} of every SM's 1536 thread slots unusable. Prefer a width that divides the per-SM budget evenly.",
                1536 - resident.unwrap_or(1536)
            );
        }
    }

    /// The fix is a residency rule, not a hardcoded rejection of 1024.
    ///
    /// On a device whose per-SM budget is 2048 threads, 1024 hosts two whole
    /// blocks and reaches every slot, so it ties with every narrower candidate
    /// on residency and the latency estimate breaks the tie toward the widest.
    /// A fix that simply banned 1024 would fail here.
    #[test]
    fn cold_start_still_admits_1024_where_the_per_sm_budget_divides_evenly() {
        let limits = LaunchGeometryLimits {
            backend: "even-divisor-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 2048,
        };
        let program = tunable_1d_program("out_even_divisor", 65_536, [32, 1, 1]);

        assert_eq!(
            limits.resident_threads_per_compute_unit(1024),
            Some(2048),
            "Fix: 1024 must reach every thread slot on a 2048-thread SM, otherwise this test's premise is wrong."
        );
        assert_eq!(
            resolve_launch_workgroup_for_mode(
                &program,
                &DispatchConfig::default(),
                limits,
                65_536,
                Mode::NaturalGradient,
            ),
            [1024, 1, 1],
            "Fix: residency-aware cold start must stay a residency rule. A width that strands nothing has to remain selectable on every device."
        );
    }

    /// A backend reporting no per-SM thread budget keeps its previous cold
    /// start bit for bit.
    ///
    /// WebGPU exposes no such number, so wgpu reports `max_threads_per_sm: 0`.
    /// Zero must make the residency preference inert rather than derive an
    /// opinion from a budget the backend never supplied: every candidate stays
    /// eligible and the latency estimate alone decides, which is [1024,1,1] for
    /// each count below. Multiples of 1024 are the interesting ones, because
    /// that is where the estimate's idle-lane penalty vanishes entirely. If
    /// someone later makes 0 mean "guess a budget", this fails loudly.
    #[test]
    fn unreported_per_sm_budget_leaves_cold_start_byte_identical() {
        let limits = LaunchGeometryLimits {
            backend: "unreported-residency-test",
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            max_threads_per_sm: 0,
        };
        assert_eq!(
            limits.resident_threads_per_compute_unit(1024),
            None,
            "Fix: an unreported per-SM budget must answer `unknown`, never a guessed number."
        );

        for (output, element_count) in [
            ("out_inert_1k", 1024u32),
            ("out_inert_4k", 4096),
            ("out_inert_64k", 65_536),
            ("out_inert_1000", 1_000),
            ("out_inert_4097", 4097),
            ("out_inert_100k", 100_000),
        ] {
            let program = tunable_1d_program(output, element_count, [32, 1, 1]);
            assert_eq!(
                resolve_launch_workgroup_for_mode(
                    &program,
                    &DispatchConfig::default(),
                    limits,
                    element_count,
                    Mode::NaturalGradient,
                ),
                [1024, 1, 1],
                "Fix: residency-aware cold start must be inert for a backend that reports no per-SM budget. {element_count} elements resolved differently than they did before residency entered this decision."
            );
        }
    }

    /// Both explicit geometry pins outrank residency-aware cold start.
    ///
    /// `workgroup_override` is authoritative and `grid_override` returns the
    /// declared shape, because a caller that pinned its geometry is telling the
    /// driver the shape is load bearing. `exatok` sets both, which is why this
    /// defect never reached it.
    #[test]
    fn explicit_geometry_pins_outrank_residency_aware_cold_start() {
        let limits = blackwell_5090_limits();
        let declared = [256, 1, 1];
        let program = tunable_1d_program("out_pinned_geometry", 262_144, declared);

        let mut pinned_workgroup = DispatchConfig::default();
        pinned_workgroup.workgroup_override = Some([64, 1, 1]);
        assert_eq!(
            resolve_launch_workgroup_for_mode(
                &program,
                &pinned_workgroup,
                limits,
                262_144,
                Mode::NaturalGradient,
            ),
            [64, 1, 1],
            "Fix: an explicit workgroup override stays authoritative even when residency prefers another width."
        );

        let mut pinned_grid = DispatchConfig::default();
        pinned_grid.grid_override = Some([1024, 1, 1]);
        assert_eq!(
            resolve_launch_workgroup_for_mode(
                &program,
                &pinned_grid,
                limits,
                262_144,
                Mode::NaturalGradient,
            ),
            declared,
            "Fix: an explicit grid override must keep the declared workgroup, since the caller sized the grid against it."
        );
    }

    /// The cooperative residency bound follows the width the tuner RESOLVES,
    /// never the width the program DECLARES.
    ///
    /// This is the test whose failure exposed the defect and it must stay. A
    /// preflight boundary written against a declared 256 expected the flip at
    /// 1021 blocks (1020 = 6 blocks/SM x 170 SMs is the last grid that fits)
    /// and observed it at 681, because the tuner had silently resolved 1024:
    /// 681 x 256 = 174,336 lanes, exactly one lane past 170 x 1024 = 174,080.
    /// Earlier over-residency tests missed this because 1024 blocks of 256
    /// exceeds the ceiling under BOTH widths, so they were green for the wrong
    /// reason. Past the ceiling a grid-sync program stops fitting one
    /// cooperative launch and takes the host split route, so a width choice
    /// turns one launch into many.
    #[test]
    fn cooperative_lane_ceiling_follows_the_resolved_width_not_the_declared_one() {
        let limits = blackwell_5090_limits();
        let declared = [256, 1, 1];
        let program = tunable_1d_program("out_resolved_ceiling", 262_144, declared);
        let lane_ceiling = |width: u32| -> u64 {
            u64::from(
                limits
                    .resident_threads_per_compute_unit(width)
                    .expect("Fix: this device model reports a per-SM thread budget"),
            ) * u64::from(RTX_5090_SM_COUNT)
        };

        let resolved = resolve_launch_workgroup_for_mode(
            &program,
            &DispatchConfig::default(),
            limits,
            262_144,
            Mode::NaturalGradient,
        );
        assert_ne!(
            resolved, declared,
            "Fix: this program is tunable, so a bound taken from the declared width would bound a width nothing launches."
        );
        assert_eq!(
            lane_ceiling(1024),
            174_080,
            "Fix: 1024 wide is 1 block/SM x 170 SMs x 1024 lanes. This is the ceiling the defect produced."
        );
        assert_eq!(
            lane_ceiling(resolved[0]),
            261_120,
            "Fix: the resolved width must reach the device's full cooperative capacity, 1536 resident threads x 170 SMs. Seeing 174,080 here means the tuner resolved 1024 again."
        );

        let mut pinned = DispatchConfig::default();
        pinned.workgroup_override = Some(declared);
        assert_eq!(
            resolve_launch_workgroup_for_mode(
                &program,
                &pinned,
                limits,
                262_144,
                Mode::NaturalGradient,
            ),
            declared,
            "Fix: a pinned width must resolve to itself so the declared and resolved bounds coincide."
        );
        assert_eq!(lane_ceiling(declared[0]), 261_120);
    }

    /// Measured feedback still outranks the residency preference.
    ///
    /// The residency rule governs the choice made with no measurements. Once a
    /// real timing says 1024 is faster for a given program, the tuner must be
    /// free to take it even though cold start would never have proposed it.
    #[test]
    fn measured_feedback_can_still_select_a_width_cold_start_would_reject() {
        let dir =
            tempfile::tempdir().expect("Fix: measured feedback test needs an isolated tuner cache");
        let path = dir.path().join("residency-feedback.toml");
        let limits = blackwell_5090_limits();
        let declared = [32, 1, 1];
        let program = tunable_1d_program("out_measured_beats_residency", 65_536, declared);
        let key = NaturalLaunchCacheKey::new(&program, declared, 65_536, limits);
        natural_launch_cache_remove(key);

        assert_eq!(
            natural_gradient_cold_start_workgroup_with_store(
                &program,
                declared,
                65_536,
                limits,
                Some(&path),
            ),
            Some([512, 1, 1]),
            "Fix: with no measurements the residency preference decides."
        );
        natural_launch_cache_remove(key);
        assert!(
            record_launch_measurement_for_mode_with_store(
                &program,
                &DispatchConfig::default(),
                limits,
                65_536,
                [1024, 1, 1],
                1,
                Mode::NaturalGradient,
                Some(&path),
            ),
            "Fix: a real timing for a residency-poor width must still be accepted."
        );
        assert_eq!(
            natural_gradient_cold_start_workgroup_with_store(
                &program,
                declared,
                65_536,
                limits,
                Some(&path),
            ),
            Some([1024, 1, 1]),
            "Fix: residency governs the cold start only. Measured feedback must remain able to choose a width cold start would never propose."
        );
    }
}
