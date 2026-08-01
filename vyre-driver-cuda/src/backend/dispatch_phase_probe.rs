//! Per-dispatch decomposition of the timed CUDA dispatch window.
//!
//! # Why this exists
//!
//! The `exatok` encode profile charges a `vyre` row of 12.5 ms to its `cjk`
//! and `code` shapes against 3.4 ms for `prose`, on the same 262144 input
//! bytes. That row is dominated by the ENQUEUE half of vyre's own timed
//! window: 12.56 ms over 6 dispatches for `cjk` against 1.88 ms over 4 for
//! `prose`, so 2.09 ms per dispatch against 0.47 ms. A flat per-dispatch cost
//! cannot produce that, and neither can input size, which is held fixed.
//!
//! Ten separate whole-program or whole-PTX walks run on that path, spread over
//! four crates, so no wall figure could be charged to code without splitting
//! it. This module splits it. It reports, per dispatch, the host nanoseconds
//! in each named phase beside the counted quantities that phase should scale
//! with, so "scales with program size" and "fixed per dispatch" become
//! distinguishable rather than both being consistent with the total.
//!
//! # Why it also reports kernel-only device time
//!
//! The dispatch path records its outer timing events around a region that
//! contains host work: PTX function resolution, the grid-barrier lease (which
//! scans the PTX text), and the post-launch release (which synchronizes the
//! stream and reads the arrival counter back). CUDA event elapsed time is a
//! difference of two DEVICE timestamps, so every stretch where the stream
//! drains and the host has not yet enqueued the next item is inside that
//! difference. The outer window therefore reports host starvation as device
//! time. The probe records a second event pair immediately around the launch
//! loop, and `kernel_ns` against `device_window_ns` prices that error directly
//! instead of leaving it as an unquantified caveat on every device figure.
//!
//! # Running it
//!
//! ```text
//! VYRE_CUDA_DISPATCH_PHASE_PROBE=1 <binary that dispatches>
//! ```
//!
//! One `vyre-cuda-dispatch-phase` line per dispatch goes to stderr, emitted
//! AFTER the dispatch's own timing has been captured so the write cannot land
//! inside a measured region. Disabled, every entry point below is a load of
//! one cached bool followed by the unmodified call.

use std::cell::RefCell;
use std::sync::LazyLock;
use std::time::Instant;

use vyre_foundation::ir::Program;

/// Named host phases of one timed dispatch. Disjoint leaves that add.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Phase {
    /// `prepare_resident_dispatch`: both binding-plan walks, launch plan,
    /// program validation, cooperative resolution.
    Prepare,
    /// `ptx_for_program_cached_with_key`: program clone, subgroup lowering,
    /// normalized digest, VSA fingerprint, cache lookup.
    Ptx,
    /// `module_cache_key_for_ptx_source_key`: digest of the PTX source key.
    ModuleKey,
    /// Binding resolution, borrowed staging uploads, output clears.
    Stage,
    /// `resolve_launch_function`: module cache lookup plus argument vector.
    Resolve,
    /// `lease_grid_barrier`: grid-sync detection, PTX barrier-marker scan,
    /// module-scope counter lookup, gate acquisition.
    Lease,
    /// The launch loop itself: per-iteration counter reset plus launch.
    LaunchLoop,
    /// `release_after_launch`: stream synchronize, arrival-count audit,
    /// gate release.
    Release,
    /// Output readback after the post-kernel fence.
    Readback,
}

impl Phase {
    /// Count of distinct phases, used to size the accumulator.
    const COUNT: usize = 9;

    const fn index(self) -> usize {
        match self {
            Self::Prepare => 0,
            Self::Ptx => 1,
            Self::ModuleKey => 2,
            Self::Stage => 3,
            Self::Resolve => 4,
            Self::Lease => 5,
            Self::LaunchLoop => 6,
            Self::Release => 7,
            Self::Readback => 8,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Prepare => "prepare_ns",
            Self::Ptx => "ptx_ns",
            Self::ModuleKey => "modkey_ns",
            Self::Stage => "stage_ns",
            Self::Resolve => "resolve_ns",
            Self::Lease => "lease_ns",
            Self::LaunchLoop => "launch_ns",
            Self::Release => "release_ns",
            Self::Readback => "readback_ns",
        }
    }
}

/// Sub-phases measured INSIDE another phase, so they are deliberately excluded
/// from `named_host_ns`.
///
/// `Phase` is a partition: its leaves are disjoint and sum to the attributed
/// host time, and adding a nested region to it would double count. These live
/// in a second array and print with a `sub_` prefix so a reader cannot mistake
/// one for a sibling of `ptx_ns`. Both current entries sit inside
/// `Phase::Ptx`, which is where the two whole-program walks on the cache-hit
/// path are: the normalized digest and the VSA fingerprint. Separating them is
/// the whole point, because the digest has a memo landing and the fingerprint
/// does not, so a combined figure cannot say which lane to fix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Nested {
    /// `try_normalized_program_cache_digest`: one whole-program walk.
    PtxDigest,
    /// `program_vsa_fingerprint_words`: a second whole-program walk.
    PtxVsa,
}

impl Nested {
    const COUNT: usize = 2;

    const fn index(self) -> usize {
        match self {
            Self::PtxDigest => 0,
            Self::PtxVsa => 1,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::PtxDigest => "sub_ptx_digest_ns",
            Self::PtxVsa => "sub_ptx_vsa_ns",
        }
    }
}

/// One dispatch's host phase times and the counted quantities beside them.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DispatchPhases {
    host_ns: [u64; Phase::COUNT],
    nested_ns: [u64; Nested::COUNT],
    /// IR nodes reachable from the program entry, counted recursively.
    ///
    /// The axis every whole-program walk on the dispatch path should scale
    /// with, and the one that decides whether "cjk carries a larger program"
    /// explains its per-dispatch cost.
    pub(crate) nodes: u64,
    /// Lowered PTX text length, the load-independent proxy for program size.
    pub(crate) ptx_bytes: u64,
    /// Buffer declarations on the program.
    pub(crate) buffers: u64,
    /// Bound resources handed to this dispatch.
    pub(crate) bindings: u64,
    /// Kernel replays inside this one dispatch.
    pub(crate) fixpoint_iterations: u64,
    /// Launch grid blocks, x times y times z.
    pub(crate) grid_blocks: u64,
    /// Device time between events recorded immediately around the launch loop.
    pub(crate) kernel_ns: u64,
}

impl DispatchPhases {
    /// Sum of the named host phases. Not the window wall: the caller prints
    /// both so the residue stays visible instead of being spread over rows.
    fn named_host_ns(&self) -> u64 {
        self.host_ns.iter().copied().sum()
    }
}

thread_local! {
    static CURRENT: RefCell<DispatchPhases> = const { RefCell::new(DispatchPhases::new_zeroed()) };
}

impl DispatchPhases {
    const fn new_zeroed() -> Self {
        Self {
            host_ns: [0; Phase::COUNT],
            nested_ns: [0; Nested::COUNT],
            nodes: 0,
            ptx_bytes: 0,
            buffers: 0,
            bindings: 0,
            fixpoint_iterations: 0,
            grid_blocks: 0,
            kernel_ns: 0,
        }
    }
}

static PROBE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VYRE_CUDA_DISPATCH_PHASE_PROBE").is_some());

/// Whether per-dispatch phase attribution is collecting.
#[inline]
pub(crate) fn enabled() -> bool {
    *PROBE_ENABLED
}

/// Run `work`, charging its host duration to `phase`.
///
/// Disabled, this is one bool load and a direct call, with no `Instant` read,
/// because two clock reads per phase across nine phases is a real cost on a
/// path whose whole problem is per-dispatch host overhead.
#[inline]
pub(crate) fn measure<T>(phase: Phase, work: impl FnOnce() -> T) -> T {
    if !enabled() {
        return work();
    }
    let started = Instant::now();
    let out = work();
    add_host_ns(phase, saturating_elapsed_ns(started));
    out
}

/// Charge `ns` to `phase`, accumulating when a phase is entered more than once.
pub(crate) fn add_host_ns(phase: Phase, ns: u64) {
    if !enabled() {
        return;
    }
    let _ = CURRENT.try_with(|current| {
        if let Ok(mut current) = current.try_borrow_mut() {
            let slot = &mut current.host_ns[phase.index()];
            *slot = slot.saturating_add(ns);
        }
    });
}

/// Run `work`, charging its host duration to the nested region `nested`.
///
/// Deliberately NOT `measure`: a nested charge must never land in `host_ns`,
/// because that array is a partition whose sum is reported as the attributed
/// host time. A region measured inside `Phase::Ptx` is already counted there,
/// so charging it again would inflate the total by its own duration and make
/// the unattributed residue read negative.
#[inline]
pub(crate) fn measure_nested<T>(nested: Nested, work: impl FnOnce() -> T) -> T {
    if !enabled() {
        return work();
    }
    let started = Instant::now();
    let out = work();
    let ns = saturating_elapsed_ns(started);
    let _ = CURRENT.try_with(|current| {
        if let Ok(mut current) = current.try_borrow_mut() {
            let slot = &mut current.nested_ns[nested.index()];
            *slot = slot.saturating_add(ns);
        }
    });
    out
}

/// Start a host region. `None` when the probe is off, so the caller pays no
/// clock read.
#[inline]
pub(crate) fn mark() -> Option<Instant> {
    enabled().then(Instant::now)
}

/// Charge the time since `started` to `phase`.
#[inline]
pub(crate) fn charge(phase: Phase, started: Option<Instant>) {
    if let Some(started) = started {
        add_host_ns(phase, saturating_elapsed_ns(started));
    }
}

/// Charge the time since `started` to `phase`, where the caller already holds
/// an `Instant` taken for another purpose.
#[inline]
pub(crate) fn charge_since(phase: Phase, started: Instant) {
    if !enabled() {
        return;
    }
    add_host_ns(phase, saturating_elapsed_ns(started));
}

/// Charge the time since `started` to `outer`, less what `inner` already holds.
///
/// Used where a wrapper and the region it wraps are both named phases, so the
/// two stay disjoint leaves that add rather than one containing the other.
/// `inner` MUST already have been charged for this dispatch.
#[inline]
pub(crate) fn charge_remainder(outer: Phase, started: Option<Instant>, inner: Phase) {
    let Some(started) = started else {
        return;
    };
    let total = saturating_elapsed_ns(started);
    add_host_ns(outer, total.saturating_sub(phase_ns(inner)));
}

/// Host nanoseconds charged to `phase` for the dispatch now being measured.
fn phase_ns(phase: Phase) -> u64 {
    CURRENT
        .try_with(|current| {
            current
                .try_borrow()
                .map(|current| current.host_ns[phase.index()])
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

/// Record the counted quantities for the dispatch now being measured.
///
/// The recursive node walk runs only under the probe: it is the one axis every
/// whole-program cost on this path should scale with, and it is far too
/// expensive to take on a release dispatch.
pub(crate) fn record_counts(
    program: &Program,
    ptx_bytes: usize,
    bindings: usize,
    fixpoint_iterations: usize,
    grid: [u32; 3],
) {
    if !enabled() {
        return;
    }
    let mut nodes = 0u64;
    vyre_foundation::transform::visit::walk_nodes(program, |_| {
        nodes = nodes.saturating_add(1);
    });
    let blocks = u64::from(grid[0])
        .saturating_mul(u64::from(grid[1]))
        .saturating_mul(u64::from(grid[2]));
    let _ = CURRENT.try_with(|current| {
        if let Ok(mut current) = current.try_borrow_mut() {
            current.nodes = nodes;
            current.ptx_bytes = ptx_bytes as u64;
            current.buffers = program.buffers().len() as u64;
            current.bindings = bindings as u64;
            current.fixpoint_iterations = fixpoint_iterations as u64;
            current.grid_blocks = blocks;
        }
    });
}

/// Record kernel-only device time from the inner event pair.
pub(crate) fn record_kernel_ns(kernel_ns: u64) {
    if !enabled() {
        return;
    }
    let _ = CURRENT.try_with(|current| {
        if let Ok(mut current) = current.try_borrow_mut() {
            current.kernel_ns = kernel_ns;
        }
    });
}

/// Elapsed nanoseconds, saturating rather than fallible.
///
/// A probe must not introduce a failure mode on the dispatch path it is
/// measuring: a clock that outran `u64` nanoseconds would turn a measurement
/// into a dispatch error, which is strictly worse than a clamped sample.
fn saturating_elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Emit one dispatch's decomposition and clear the accumulator.
///
/// `wall_ns`, `enqueue_ns`, `wait_ns` and `device_window_ns` come from the
/// dispatch's own telemetry so the line can be reconciled against the counters
/// `exatok` reads, and the unattributed residue is printed rather than folded
/// into a named phase.
pub(crate) fn emit(
    sequence: u64,
    wall_ns: u64,
    enqueue_ns: u64,
    wait_ns: u64,
    device_window_ns: Option<u64>,
    ptx_cache_hits: u64,
    ptx_cache_misses: u64,
) {
    if !enabled() {
        return;
    }
    let Ok(phases) = CURRENT.try_with(|current| {
        let taken = current.try_borrow_mut().map(|mut current| {
            let taken = *current;
            *current = DispatchPhases::new_zeroed();
            taken
        });
        taken.unwrap_or_default()
    }) else {
        return;
    };
    let mut line = String::with_capacity(512);
    line.push_str("vyre-cuda-dispatch-phase");
    push_field(&mut line, "seq", sequence);
    push_field(&mut line, "nodes", phases.nodes);
    push_field(&mut line, "ptx_bytes", phases.ptx_bytes);
    push_field(&mut line, "buffers", phases.buffers);
    push_field(&mut line, "bindings", phases.bindings);
    push_field(&mut line, "fixpoint", phases.fixpoint_iterations);
    push_field(&mut line, "grid_blocks", phases.grid_blocks);
    push_field(&mut line, "wall_ns", wall_ns);
    push_field(&mut line, "enqueue_ns", enqueue_ns);
    push_field(&mut line, "wait_ns", wait_ns);
    push_field(&mut line, "device_window_ns", device_window_ns.unwrap_or(0));
    push_field(&mut line, "kernel_ns", phases.kernel_ns);
    for phase in [
        Phase::Prepare,
        Phase::Ptx,
        Phase::ModuleKey,
        Phase::Stage,
        Phase::Resolve,
        Phase::Lease,
        Phase::LaunchLoop,
        Phase::Release,
        Phase::Readback,
    ] {
        push_field(&mut line, phase.label(), phases.host_ns[phase.index()]);
    }
    for nested in [Nested::PtxDigest, Nested::PtxVsa] {
        push_field(&mut line, nested.label(), phases.nested_ns[nested.index()]);
    }
    push_field(&mut line, "named_host_ns", phases.named_host_ns());
    push_field(
        &mut line,
        "unattributed_ns",
        wall_ns.saturating_sub(phases.named_host_ns()),
    );
    push_field(&mut line, "ptx_cache_hits", ptx_cache_hits);
    push_field(&mut line, "ptx_cache_misses", ptx_cache_misses);
    eprintln!("{line}");
}

fn push_field(line: &mut String, name: &str, value: u64) {
    use std::fmt::Write;

    // A formatting failure on a `String` is unreachable, and a probe must not
    // turn one into a dispatch error, so the field is dropped rather than
    // propagated.
    let _ = write!(line, " {name}={value}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks out: a phase added to [`Phase`] without widening the accumulator.
    ///
    /// [`Phase::index`] indexes a fixed-length array, so a tenth variant whose
    /// index is 9 would index out of bounds on the first `measure` call. That
    /// is a panic on the dispatch path, reachable only with the probe enabled,
    /// which is exactly the configuration nobody runs in CI.
    #[test]
    fn every_phase_index_is_inside_the_accumulator() {
        let phases = [
            Phase::Prepare,
            Phase::Ptx,
            Phase::ModuleKey,
            Phase::Stage,
            Phase::Resolve,
            Phase::Lease,
            Phase::LaunchLoop,
            Phase::Release,
            Phase::Readback,
        ];
        assert_eq!(phases.len(), Phase::COUNT);
        let mut seen = [false; Phase::COUNT];
        for phase in phases {
            let index = phase.index();
            assert!(index < Phase::COUNT, "{phase:?} indexes {index}");
            assert!(!seen[index], "{phase:?} reuses index {index}");
            seen[index] = true;
        }
        assert!(seen.iter().all(|slot| *slot));
    }

    /// Locks out: two phases sharing a printed label.
    ///
    /// The emitted line is parsed by field name. Two phases printing the same
    /// name silently merges them, and a merged phase is the one thing this
    /// module exists to prevent.
    #[test]
    fn phase_labels_are_distinct() {
        let labels = [
            Phase::Prepare.label(),
            Phase::Ptx.label(),
            Phase::ModuleKey.label(),
            Phase::Stage.label(),
            Phase::Resolve.label(),
            Phase::Lease.label(),
            Phase::LaunchLoop.label(),
            Phase::Release.label(),
            Phase::Readback.label(),
        ];
        let mut sorted = labels;
        sorted.sort_unstable();
        let mut deduped = sorted.to_vec();
        deduped.dedup();
        assert_eq!(deduped.len(), labels.len(), "duplicate phase label");
    }

    /// Locks out: `measure` costing a clock read when the probe is off.
    ///
    /// The probe measures per-dispatch host overhead, so an always-on pair of
    /// `Instant::now` calls per phase would be a real regression on the very
    /// path under study. The disabled path must return the closure's value
    /// with nothing recorded.
    #[test]
    fn disabled_measure_records_nothing() {
        if enabled() {
            return;
        }
        let value = measure(Phase::Prepare, || 7u32);
        assert_eq!(value, 7);
        add_host_ns(Phase::Prepare, 1_000_000);
        let observed = CURRENT.with(|current| current.borrow().named_host_ns());
        assert_eq!(observed, 0);
    }

    /// Locks out: the named phases silently absorbing the residue.
    ///
    /// `named_host_ns` must be the plain sum so the emitted
    /// `unattributed_ns` is a real remainder. A phase double-counted into the
    /// sum would make the residue negative and be clamped to zero by the
    /// saturating subtraction, hiding the defect.
    #[test]
    fn named_total_is_the_plain_sum_of_phases() {
        let mut phases = DispatchPhases::new_zeroed();
        phases.host_ns[Phase::Prepare.index()] = 11;
        phases.host_ns[Phase::Ptx.index()] = 22;
        phases.host_ns[Phase::Readback.index()] = 33;
        assert_eq!(phases.named_host_ns(), 66);
    }
}
