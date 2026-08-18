//! Natural-gradient launch tuning, caching, and workgroup resolution.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use vyre_foundation::ir::{MemoryKind, Node, Program};

use crate::launch::LaunchGeometry;
use crate::program_walks::program_uses_launch_geometry_ids;
use crate::tuner::{
    identity_fisher_q16, Mode, NaturalGradientPolicy, Tuner, TunerCache, TuningMeasurement,
    WORKGROUP_CANDIDATES,
};
use crate::validation::LaunchGeometryLimits;
use crate::DispatchConfig;

const COLD_START_GRID_STEP_NS: u64 = 1_024;
const COLD_START_IDLE_LANE_NS: u64 = 8;
const COLD_START_TEMPERATURE_NS: u64 = 4_096;
const MAX_NATURAL_LAUNCH_CACHE_ENTRIES: usize = 4_096;

static NATURAL_LAUNCH_CACHE: LazyLock<Mutex<BTreeMap<NaturalLaunchCacheKey, NaturalLaunchEntry>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

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

pub(crate) fn record_launch_measurement_for_mode(
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

pub(crate) fn record_launch_measurement_for_mode_with_store(
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

pub(crate) fn natural_gradient_cold_start_workgroup(
    program: &Program,
    declared: [u32; 3],
    element_count: u32,
    limits: LaunchGeometryLimits,
) -> Option<[u32; 3]> {
    natural_gradient_cold_start_workgroup_with_store(program, declared, element_count, limits, None)
}

pub(crate) fn natural_gradient_cold_start_workgroup_with_store(
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
pub(crate) struct NaturalLaunchCacheKey {
    fingerprint: [u8; 32],
    declared: [u32; 3],
    element_count: u32,
    max_threads_per_block: u32,
    max_block_dim: [u32; 3],
    max_grid_dim: [u32; 3],
    max_threads_per_sm: u32,
}

impl NaturalLaunchCacheKey {
    pub(crate) fn new(
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
    let guard = NATURAL_LAUNCH_CACHE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.get(&key).map(|entry| entry.selected)
}

fn natural_launch_cache_measurements(
    key: NaturalLaunchCacheKey,
) -> Option<BTreeMap<[u32; 3], u64>> {
    let guard = NATURAL_LAUNCH_CACHE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.get(&key).map(|entry| entry.measurements.clone())
}

fn natural_launch_cache_set(key: NaturalLaunchCacheKey, value: NaturalLaunchEntry) {
    let mut guard = NATURAL_LAUNCH_CACHE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if guard.len() >= MAX_NATURAL_LAUNCH_CACHE_ENTRIES && !guard.contains_key(&key) {
        if let Some(oldest) = guard.keys().next().copied() {
            guard.remove(&oldest);
        }
    }
    guard.insert(key, value);
}

#[cfg(test)]
pub(crate) fn natural_launch_cache_remove(key: NaturalLaunchCacheKey) {
    if let Ok(mut guard) = NATURAL_LAUNCH_CACHE.lock() {
        guard.remove(&key);
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

pub(crate) fn persist_natural_launch_selection_to_path(
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

#[path = "launch_natural_tests.rs"]
#[cfg(test)]
mod launch_natural_tests;
