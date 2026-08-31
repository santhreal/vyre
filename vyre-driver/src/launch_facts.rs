//! Launch geometry realization, and the launch measurements a compile can read.
//!
//! A launch shape is a compile decision. The artifact records the workgroup the
//! search selected for every node, and this module realizes it: a recorded
//! geometry is used as recorded, a caller override is honored as a diagnostic
//! pin, and anything else launches at the width the program declares. Nothing
//! here ranks widths, because a driver that re-picks one runs bytes no artifact
//! identity covers.
//!
//! What the driver does own is what it observed. A dispatch that produced a
//! device timing records the width and the elapsed nanoseconds as a fact, and
//! [`launch_width_measurements`] reports the table so schedule selection can
//! rank against measured figures instead of guessing at them. Facts live for
//! the process that measured them: a selection persisted across runs would
//! outlive the artifact that authorized it.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex};

use vyre_foundation::ir::{MemoryKind, Node, Program};

use crate::launch::LaunchGeometry;
use crate::program_walks::program_uses_launch_geometry_ids;
use crate::validation::LaunchGeometryLimits;
use crate::DispatchConfig;

/// Programs the measurement table holds facts for at once.
const MAX_MEASURED_PROGRAMS: usize = 4_096;

static LAUNCH_MEASUREMENTS: LazyLock<Mutex<BTreeMap<LaunchFactKey, BTreeMap<[u32; 3], u64>>>> =
    LazyLock::new(|| Mutex::new(BTreeMap::new()));

/// Resolve the backend-visible workgroup shape for an untracked dispatch.
///
/// A launch with no recorded artifact geometry runs at the width its program
/// declares, widened only where the inferred grid would exceed what the target
/// admits.
#[must_use]
pub fn resolve_launch_workgroup(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
) -> [u32; 3] {
    resolve_launch_workgroup_for_geometry(
        program,
        config,
        limits,
        element_count,
        LaunchGeometry::Untracked,
    )
}

/// Resolve the backend-visible workgroup shape against a launch's geometry source.
///
/// A recorded compiled geometry outranks every dispatch override: the emitted
/// module declares that shape, so launching it at another width runs a kernel
/// nobody compiled. A caller override outranks the declared width, and an
/// explicit grid override keeps the declared width because the caller sized the
/// grid against it.
///
/// Whatever survives is then judged against the target's per-axis ceiling.
/// Ranking is a performance question and legality is not: a block whose
/// inferred grid the device refuses is not a slow choice, it is a launch that
/// cannot run, so a width-free launch is widened into the ceiling.
#[must_use]
pub fn resolve_launch_workgroup_for_geometry(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    geometry: LaunchGeometry,
) -> [u32; 3] {
    if let LaunchGeometry::Compiled(workgroup) = geometry {
        return workgroup;
    }
    if let Some(workgroup) = config.launch_workgroup() {
        return workgroup;
    }
    let declared = program.workgroup_size();
    if config.launch_grid().is_some() {
        return declared;
    }
    widen_into_grid_ceiling(program, declared, element_count, limits)
}

/// Record a measured launch result as a fact of this process.
///
/// Backends call this only after a real dispatch timing is available. The
/// function returns `true` when the timing entered the bounded fact table.
/// A zero timing measures nothing, a pinned launch measures a shape the caller
/// chose rather than one a selector could compare, and a program whose result
/// depends on its block width has no comparable timings across widths; each is
/// refused so the table only ever holds figures a selection may rank.
#[must_use]
pub fn record_launch_measurement(
    program: &Program,
    config: &DispatchConfig,
    limits: LaunchGeometryLimits,
    element_count: u32,
    observed_workgroup: [u32; 3],
    elapsed_ns: u64,
) -> bool {
    if elapsed_ns == 0
        || config.launch_workgroup().is_some()
        || config.launch_grid().is_some()
        || observed_workgroup[1] != 1
        || observed_workgroup[2] != 1
        || !width_fits_limits(observed_workgroup[0], limits)
    {
        return false;
    }
    let declared = program.workgroup_size();
    if !launch_width_is_free(program, declared, element_count) {
        return false;
    }
    let key = LaunchFactKey::new(program, declared, element_count, limits);
    let mut guard = LAUNCH_MEASUREMENTS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    while guard.len() >= MAX_MEASURED_PROGRAMS && !guard.contains_key(&key) {
        let Some(oldest) = guard.keys().next().copied() else {
            break;
        };
        guard.remove(&oldest);
    }
    guard
        .entry(key)
        .or_default()
        .entry(observed_workgroup)
        .and_modify(|best_ns| *best_ns = (*best_ns).min(elapsed_ns))
        .or_insert(elapsed_ns);
    true
}

/// Report the fastest measured nanoseconds per launch width for one program.
///
/// The table is what this process observed, keyed by the program, its declared
/// width, the launch element count, and the target limits the timings were
/// taken under. An empty table means nothing was measured, never that a width
/// is slow.
#[must_use]
pub fn launch_width_measurements(
    program: &Program,
    limits: LaunchGeometryLimits,
    element_count: u32,
) -> BTreeMap<[u32; 3], u64> {
    let key = LaunchFactKey::new(program, program.workgroup_size(), element_count, limits);
    let guard = LAUNCH_MEASUREMENTS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.get(&key).cloned().unwrap_or_default()
}

/// Which program, launch size, and target one measurement table describes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct LaunchFactKey {
    fingerprint: [u8; 32],
    declared: [u32; 3],
    element_count: u32,
    max_threads_per_block: u32,
    max_block_dim: [u32; 3],
    max_grid_dim: [u32; 3],
    max_threads_per_sm: u32,
}

impl LaunchFactKey {
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
}

/// Whether a program computes the same result at any 1D block width.
///
/// A program that reads its workgroup or local id observes the block width, so
/// its result depends on it, and one holding workgroup-shared memory partitions
/// by block. Neither may be launched at another width, and timings taken at
/// different widths do not describe the same computation.
fn launch_width_is_free(program: &Program, declared: [u32; 3], element_count: u32) -> bool {
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

/// Whether a 1D block of `width` threads is one the target admits.
fn width_fits_limits(width: u32, limits: LaunchGeometryLimits) -> bool {
    width != 0 && width <= limits.max_threads_per_block && width <= limits.max_block_dim[0]
}

/// Whether a 1D launch of `element_count` lanes in blocks of `width` asks for a
/// grid the target admits.
///
/// A zero ceiling means no device was probed, and an empty launch has no grid to
/// judge; both pass rather than refusing what the target accepts.
fn grid_fits(width: u32, element_count: u32, limits: LaunchGeometryLimits) -> bool {
    if width == 0 || element_count == 0 || limits.max_grid_dim[0] == 0 {
        return true;
    }
    element_count.div_ceil(width) <= limits.max_grid_dim[0]
}

/// Widen a 1D block until its inferred grid fits the target's per-axis ceiling.
///
/// A program that declares one output element per lane asks for one workgroup
/// per block of lanes, so a large launch reaches a graphics-derived ceiling long
/// before it reaches any memory limit: 16.8 million lanes in blocks of 256 ask
/// for 65536 workgroups, one past what the API admits. The same lane space in
/// blocks of 1024 asks for 16384 and runs. Refusing it instead would take every
/// large launch off the backend over a block width nobody chose.
///
/// The widened width is the smallest power of two that clears the ceiling, so
/// the choice is arithmetic rather than a ranked candidate: a launch that fits
/// keeps its declared width, and one that cannot fit at any admissible width
/// keeps it too and is refused downstream by name.
fn widen_into_grid_ceiling(
    program: &Program,
    declared: [u32; 3],
    element_count: u32,
    limits: LaunchGeometryLimits,
) -> [u32; 3] {
    if declared[1] != 1
        || declared[2] != 1
        || grid_fits(declared[0], element_count, limits)
        || !launch_width_is_free(program, declared, element_count)
    {
        return declared;
    }
    let ceiling = limits.max_grid_dim[0];
    let Some(widened) = element_count
        .div_ceil(ceiling.max(1))
        .checked_next_power_of_two()
        .map(|width| width.max(declared[0]))
    else {
        return declared;
    };
    if width_fits_limits(widened, limits) && grid_fits(widened, element_count, limits) {
        [widened, 1, 1]
    } else {
        declared
    }
}

#[cfg(test)]
pub(crate) fn forget_launch_measurements(
    program: &Program,
    limits: LaunchGeometryLimits,
    element_count: u32,
) {
    let key = LaunchFactKey::new(program, program.workgroup_size(), element_count, limits);
    let mut guard = LAUNCH_MEASUREMENTS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    guard.remove(&key);
}

#[path = "launch_facts_tests.rs"]
#[cfg(test)]
mod launch_facts_tests;
