//! Per-dispatch HOST cost attribution for the CUDA dispatch path.
//!
//! # Why this exists
//!
//! `exatok` measured a prose encode whose `EncodePhase::Dispatch` share was
//! 30.7 percent, against an ENQUEUE-dominated timed dispatch window: vyre's own
//! counters put 99.8 percent of that window in enqueue and 0.004 ms in wait, and
//! a later re-measurement puts enqueue at 2.4956 ms of an 11.0041 ms encode
//! (22.7 percent) with wait at 0.0028 ms. The GPU is not the thing being waited
//! on, so the cost is host launch preparation repeated per dispatch. Nobody had
//! decomposed it, and five optimization passes were being planned against an
//! undecomposed number.
//!
//! TWO ABSOLUTES FROM THE ORIGINAL CAPTURE ARE WITHDRAWN, and they are recorded
//! here as withdrawn rather than deleted so nobody re-derives from them. That
//! capture read a 19.656 ms prose wall against 11.0041 ms once the box was
//! quieter, so it was roughly 1.8x contended and its ABSOLUTES do not carry; the
//! phase SHARES do. And the companion "GPU did 0.38 ms" is withdrawn outright:
//! it predates its own instrument's switch from single-draw counter deltas to a
//! median of 15, made because single draws swung 43 to 86 percent on some shapes,
//! and the reading was never re-taken afterwards. Prose device time now measures
//! 0.9146 to 1.2983 ms, which is 8.3 to 11.8 percent of the encode, so the old
//! figure understated the GPU by roughly 2.4x to 3.4x. Repairing an instrument
//! does not re-derive the figures it already published.
//!
//! The host-bound reading survives the correction on prose and short pretokens
//! (2.4 to 3.3 percent device) but NOT as a blanket claim: cjk measures 22.3 to
//! 23.5 percent and code 14.7 to 17.5 percent, so neither is host bound and this
//! module's premise should not be extended to them without re-measuring.
//!
//! This module decomposes it. It is a measurement instrument, not a gate: the
//! attribution is the deliverable even when a phase turns out to be
//! irreducible, because a phase measured at 40 ns is a phase nobody needs to
//! optimize.
//!
//! # Method
//!
//! The estimator rules come from `exatok`'s bench harness (owner: `Bench`,
//! `exatok::bench_support`). That harness cannot be linked from here:
//! `exatok/Cargo.toml` declares `cuda = ["dep:vyre-driver-cuda"]`, so exatok
//! depends on this crate and a dev-dependency back would be a package cycle.
//! The rules are therefore reimplemented, deliberately identically:
//!
//! * **Absolute cells quote the MINIMUM** of the timed repetitions. Contention
//!   on this box is strictly additive, so the fastest observation is the least
//!   contaminated one. `rel_stddev` is reported beside every point estimate:
//!   above [`RELIABLE_REL_STDDEV`] the cell is advisory, above
//!   [`VOIDING_REL_STDDEV`] it is void.
//! * **Paired comparisons quote the MEDIAN ratio**, never the best. Taking the
//!   best ratio selects the round where noise favoured the new code, and across
//!   a multi-phase decomposition that is close to guaranteed to manufacture a
//!   win.
//! * **Inner repetitions are calibrated after a warm pass** so every timed
//!   region reaches [`CALIBRATION_TARGET_SECS`]. Calibrating cold underestimates
//!   the repetition count and yields regions dominated by per-call overhead.
//! * **Paired rounds interleave and alternate**: A,B,B,A,A,B,B,A rather than
//!   A,B,A,B. Plain alternation gives whichever arm runs second a systematic
//!   icache and lookup-cache advantage, which matters here because several
//!   phases ARE cache lookups.
//! * **Busy-weighted CPU MHz is sampled at both region boundaries.** These
//!   phases are pure host CPU, so a core ramping between arms is a swing that
//!   survives a median and reads as a win. A between-arm gap above
//!   [`MAX_CLOCK_DRIFT_FRACTION`] disqualifies the comparison.
//!
//! Every timed closure returns a `u64` digest of its own result, and the timed
//! loop folds those digests into an accumulator that is handed to
//! [`std::hint::black_box`]. Without that, LLVM deletes loop-invariant phases
//! outright at these repetition counts, and a deleted phase measures as zero
//! and reads as a total win.
//!
//! `nvidia-smi` is NOT sampled inside a timed region: `pmon` costs hundreds of
//! milliseconds and would be most of a 0.5 s region. Device load is the
//! harness's business at the region boundaries.
//!
//! # Running it
//!
//! These are `#[ignore]`d so `cargo test -p vyre-driver-cuda --lib` stays fast.
//!
//! ```text
//! cargo test -p vyre-driver-cuda --lib -- --ignored --nocapture host_cost
//! ```

use std::hint::black_box;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use vyre_driver::BindingPlan;
use vyre_driver::{BackendError, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::backend::dispatch::CudaBackend;
use crate::backend::resident::{resident_bindings_from_handles, CudaResidentBuffer};

/// Duration a timed region is folded up to before it is trusted. Below this,
/// clock resolution and frequency ramp dominate the sample.
const CALIBRATION_TARGET_SECS: f64 = 0.5;
/// Timed repetitions per absolute cell. Main's floor is 5.
const CELL_REPS: usize = 5;
/// Relative standard deviation above which a cell is advisory only.
const RELIABLE_REL_STDDEV: f64 = 0.05;
/// Relative standard deviation above which a cell is void, not data.
const VOIDING_REL_STDDEV: f64 = 0.20;
/// Between-arm busy-weighted clock gap above which a paired result is
/// disqualified as frequency ramp rather than reported as a delta.
const MAX_CLOCK_DRIFT_FRACTION: f64 = 0.02;
/// Shortest calibration probe that is long enough to resolve against the
/// monotonic clock before the repetition count is extrapolated from it.
const CALIBRATION_PROBE_SECS: f64 = 0.05;
/// Maximum `/proc/stat` bytes accepted by the host attribution probe.
const MAX_PROC_STAT_BYTES: u64 = 1_048_576;
/// Maximum bytes accepted from one CPU frequency sysfs node.
const MAX_CPU_FREQ_BYTES: u64 = 64;

/// Busy-weighted mean CPU frequency in MHz across one region.
///
/// Weighting by per-CPU non-idle jiffies rather than taking the whole-machine
/// mean is deliberate: on a 16-core box with a handful of busy cores the
/// unweighted mean tracks how many cores are parked, not what frequency the
/// working core ran at.
fn busy_weighted_mhz(before: &CpuSample, after: &CpuSample) -> f64 {
    let mut weighted = 0.0_f64;
    let mut weight = 0.0_f64;
    for index in 0..before.busy_jiffies.len().min(after.busy_jiffies.len()) {
        let busy = after.busy_jiffies[index].saturating_sub(before.busy_jiffies[index]) as f64;
        if busy <= 0.0 {
            continue;
        }
        let mhz_before = before.mhz.get(index).copied().unwrap_or(0.0);
        let mhz_after = after.mhz.get(index).copied().unwrap_or(0.0);
        let mhz = if mhz_before > 0.0 && mhz_after > 0.0 {
            (mhz_before + mhz_after) / 2.0
        } else {
            mhz_before.max(mhz_after)
        };
        if mhz <= 0.0 {
            continue;
        }
        weighted += mhz * busy;
        weight += busy;
    }
    if weight <= 0.0 {
        return 0.0;
    }
    weighted / weight
}

fn read_host_text_bounded(path: impl AsRef<Path>, max_bytes: u64) -> Option<String> {
    let mut text = String::new();
    std::fs::File::open(path)
        .ok()?
        .take(max_bytes.saturating_add(1))
        .read_to_string(&mut text)
        .ok()?;
    (text.len() as u64 <= max_bytes).then_some(text)
}

/// Per-CPU busy time and current frequency at one instant.
struct CpuSample {
    busy_jiffies: Vec<u64>,
    mhz: Vec<f64>,
}

impl CpuSample {
    /// Read `/proc/stat` and the cpufreq sysfs nodes.
    ///
    /// Both are best-effort: a host without `scaling_cur_freq` reports zeroed
    /// frequencies, which makes [`busy_weighted_mhz`] return 0.0 and the drift
    /// check inconclusive rather than wrong.
    fn now() -> Self {
        let mut busy_jiffies = Vec::new();
        if let Some(stat) = read_host_text_bounded("/proc/stat", MAX_PROC_STAT_BYTES) {
            for line in stat.lines() {
                let Some(rest) = line.strip_prefix("cpu") else {
                    continue;
                };
                if !rest.starts_with(|c: char| c.is_ascii_digit()) {
                    continue;
                }
                let mut fields = rest.split_whitespace();
                let _cpu_id = fields.next();
                let values: Vec<u64> = fields.filter_map(|f| f.parse::<u64>().ok()).collect();
                // user nice system idle iowait irq softirq ...
                let total: u64 = values.iter().sum();
                let idle =
                    values.get(3).copied().unwrap_or(0) + values.get(4).copied().unwrap_or(0);
                busy_jiffies.push(total.saturating_sub(idle));
            }
        }
        let mut mhz = Vec::with_capacity(busy_jiffies.len());
        for index in 0..busy_jiffies.len() {
            let path = format!("/sys/devices/system/cpu/cpu{index}/cpufreq/scaling_cur_freq");
            let khz = read_host_text_bounded(&path, MAX_CPU_FREQ_BYTES)
                .and_then(|text| text.trim().parse::<f64>().ok())
                .unwrap_or(0.0);
            mhz.push(khz / 1000.0);
        }
        Self { busy_jiffies, mhz }
    }
}

/// One absolute measurement cell: N timed regions over the same folded work.
struct Cell {
    label: String,
    /// Inner repetitions folded into one timed region.
    units_per_round: u64,
    /// Seconds per single unit, one entry per timed region.
    unit_secs: Vec<f64>,
    /// Busy-weighted MHz across each timed region.
    mhz: Vec<f64>,
}

impl Cell {
    /// Point estimate: the fastest observation, in nanoseconds per unit.
    fn min_ns(&self) -> f64 {
        self.unit_secs.iter().copied().fold(f64::INFINITY, f64::min) * 1e9
    }

    fn mean_secs(&self) -> f64 {
        if self.unit_secs.is_empty() {
            return 0.0;
        }
        self.unit_secs.iter().sum::<f64>() / self.unit_secs.len() as f64
    }

    /// Spread of the samples relative to their mean.
    fn rel_stddev(&self) -> f64 {
        let mean = self.mean_secs();
        if mean <= 0.0 || self.unit_secs.len() < 2 {
            return 0.0;
        }
        let variance = self
            .unit_secs
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (self.unit_secs.len() - 1) as f64;
        variance.sqrt() / mean
    }

    fn clock_mhz(&self) -> f64 {
        if self.mhz.is_empty() {
            return 0.0;
        }
        self.mhz.iter().sum::<f64>() / self.mhz.len() as f64
    }

    fn verdict(&self) -> &'static str {
        let spread = self.rel_stddev();
        if spread > VOIDING_REL_STDDEV {
            "VOID"
        } else if spread > RELIABLE_REL_STDDEV {
            "advisory"
        } else {
            "ok"
        }
    }

    fn render(&self) -> String {
        format!(
            "{:<44} {:>13.1} ns  rel_stddev {:>6.2}%  {:<8} folded {:>9} reps  {:>6.0} MHz",
            self.label,
            self.min_ns(),
            self.rel_stddev() * 100.0,
            self.verdict(),
            self.units_per_round,
            self.clock_mhz(),
        )
    }
}

/// Fold `work` until one timed region reaches [`CALIBRATION_TARGET_SECS`], then
/// time it [`CELL_REPS`] times.
///
/// `work` MUST return a digest of whatever it computed. The digest is folded
/// into an accumulator that escapes through [`black_box`], which is what stops
/// LLVM from hoisting or deleting a loop-invariant phase. A phase that measures
/// at single-digit nanoseconds has almost certainly been deleted rather than
/// found to be fast, so treat such a figure as a defect in the probe.
fn measure_cell(
    label: impl Into<String>,
    mut work: impl FnMut() -> Result<u64, BackendError>,
) -> Result<Cell, BackendError> {
    let label = label.into();
    // Warm pass first. Calibrating cold underestimates the repetition count,
    // which produced sub-0.2 s regions that measured per-call overhead rather
    // than the phase.
    for _ in 0..3 {
        black_box(work()?);
    }
    let mut probe_reps = 1_u64;
    let units_per_round = loop {
        let started = Instant::now();
        let mut digest = 0_u64;
        for _ in 0..probe_reps {
            digest = digest.wrapping_add(work()?);
        }
        black_box(digest);
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed >= CALIBRATION_PROBE_SECS {
            let scaled = (probe_reps as f64) * (CALIBRATION_TARGET_SECS / elapsed);
            break (scaled.ceil() as u64).max(1);
        }
        let Some(next) = probe_reps.checked_mul(4) else {
            break probe_reps;
        };
        probe_reps = next;
    };

    let mut unit_secs = Vec::with_capacity(CELL_REPS);
    let mut mhz = Vec::with_capacity(CELL_REPS);
    for _ in 0..CELL_REPS {
        let cpu_before = CpuSample::now();
        let started = Instant::now();
        let mut digest = 0_u64;
        for _ in 0..units_per_round {
            digest = digest.wrapping_add(work()?);
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(digest);
        let cpu_after = CpuSample::now();
        unit_secs.push(elapsed / units_per_round as f64);
        mhz.push(busy_weighted_mhz(&cpu_before, &cpu_after));
    }

    Ok(Cell {
        label,
        units_per_round,
        unit_secs,
        mhz,
    })
}

/// A against B, interleaved and order-alternated within one process.
struct Paired {
    label_a: String,
    label_b: String,
    units_per_round: u64,
    a_unit_secs: Vec<f64>,
    b_unit_secs: Vec<f64>,
    mhz_a: Vec<f64>,
    mhz_b: Vec<f64>,
}

impl Paired {
    /// Median of the per-round ratios.
    ///
    /// Median rather than best: the contaminant has already largely cancelled
    /// inside a round, so the best ratio only selects the round where residual
    /// noise favoured B.
    fn median_speedup(&self) -> f64 {
        let mut ratios: Vec<f64> = self
            .a_unit_secs
            .iter()
            .zip(&self.b_unit_secs)
            .filter(|(_, b)| **b > 0.0)
            .map(|(a, b)| a / b)
            .collect();
        median(&mut ratios)
    }

    fn speedup_rel_stddev(&self) -> f64 {
        let ratios: Vec<f64> = self
            .a_unit_secs
            .iter()
            .zip(&self.b_unit_secs)
            .filter(|(_, b)| **b > 0.0)
            .map(|(a, b)| a / b)
            .collect();
        rel_stddev(&ratios)
    }

    /// Median of the per-round absolute deltas, in nanoseconds per unit.
    fn median_delta_ns(&self) -> f64 {
        let mut deltas: Vec<f64> = self
            .a_unit_secs
            .iter()
            .zip(&self.b_unit_secs)
            .map(|(a, b)| (a - b) * 1e9)
            .collect();
        median(&mut deltas)
    }

    fn clock_drift_fraction(&self) -> f64 {
        let a = mean(&self.mhz_a);
        let b = mean(&self.mhz_b);
        if a <= 0.0 || b <= 0.0 {
            return 0.0;
        }
        ((a - b) / a).abs()
    }

    /// Whether this comparison may be quoted.
    fn is_publishable(&self) -> bool {
        self.a_unit_secs.len() >= MIN_PAIRED_ROUNDS
            && self.speedup_rel_stddev() <= VOIDING_REL_STDDEV
            && self.clock_drift_fraction() <= MAX_CLOCK_DRIFT_FRACTION
    }

    fn render(&self) -> String {
        format!(
            "PAIRED {} -> {}\n  \
             A {:>12.1} ns/unit (min)   B {:>12.1} ns/unit (min)\n  \
             median speedup {:.4}x   median delta {:+.1} ns/unit   \
             speedup_rel_stddev {:.2}%\n  \
             rounds {}   folded {} reps   clock drift {:.2}%   {}",
            self.label_a,
            self.label_b,
            min_of(&self.a_unit_secs) * 1e9,
            min_of(&self.b_unit_secs) * 1e9,
            self.median_speedup(),
            self.median_delta_ns(),
            self.speedup_rel_stddev() * 100.0,
            self.a_unit_secs.len(),
            self.units_per_round,
            self.clock_drift_fraction() * 100.0,
            if self.is_publishable() {
                "PUBLISHABLE"
            } else {
                "NOT PUBLISHABLE"
            },
        )
    }

    /// Raw samples in the schema `Bench` folds into the shared table.
    ///
    /// Seconds, not nanoseconds, so nothing downstream has to guess a unit.
    fn to_json(&self, foreign_resident_baseline_mib: u64, foreign_sm_percent_max: u64) -> String {
        format!(
            concat!(
                "{{\n",
                "  \"label_a\": \"{}\",\n",
                "  \"label_b\": \"{}\",\n",
                "  \"a_unit_secs\": [{}],\n",
                "  \"b_unit_secs\": [{}],\n",
                "  \"units_per_round\": {},\n",
                "  \"foreign_resident_baseline_mib\": {},\n",
                "  \"foreign_sm_percent_max\": {},\n",
                "  \"busy_weighted_mhz_a\": [{}],\n",
                "  \"busy_weighted_mhz_b\": [{}]\n",
                "}}\n"
            ),
            self.label_a,
            self.label_b,
            join_f64(&self.a_unit_secs),
            join_f64(&self.b_unit_secs),
            self.units_per_round,
            foreign_resident_baseline_mib,
            foreign_sm_percent_max,
            join_f64(&self.mhz_a),
            join_f64(&self.mhz_b),
        )
    }
}

/// Paired rounds below this are refused before any work runs.
const MIN_PAIRED_ROUNDS: usize = 8;

/// Run `a` against `b`, interleaved, alternating which arm leads each round.
///
/// ABBA rather than ABAB. Under plain alternation the trailing arm always runs
/// on state the leading arm just warmed, which is a systematic advantage when
/// the phase under test IS a cache lookup, and several of these are.
fn compare_cells(
    label_a: impl Into<String>,
    mut a: impl FnMut() -> Result<u64, BackendError>,
    label_b: impl Into<String>,
    mut b: impl FnMut() -> Result<u64, BackendError>,
    rounds: usize,
) -> Result<Paired, BackendError> {
    assert!(
        rounds >= MIN_PAIRED_ROUNDS,
        "Fix: a paired comparison needs at least {MIN_PAIRED_ROUNDS} rounds; \
         fewer cannot establish a spread against a bursty contaminant."
    );
    for _ in 0..3 {
        black_box(a()?);
        black_box(b()?);
    }
    // Calibrate on the SLOWER arm so both regions clear the target.
    let units_per_round = calibrate(&mut a)?.max(calibrate(&mut b)?);

    let mut a_unit_secs = Vec::with_capacity(rounds);
    let mut b_unit_secs = Vec::with_capacity(rounds);
    let mut mhz_a = Vec::with_capacity(rounds);
    let mut mhz_b = Vec::with_capacity(rounds);
    for round in 0..rounds {
        // ABBA: the leading arm alternates every round.
        if round % 2 == 0 {
            let (secs, clock) = timed_region(&mut a, units_per_round)?;
            a_unit_secs.push(secs);
            mhz_a.push(clock);
            let (secs, clock) = timed_region(&mut b, units_per_round)?;
            b_unit_secs.push(secs);
            mhz_b.push(clock);
        } else {
            let (secs, clock) = timed_region(&mut b, units_per_round)?;
            b_unit_secs.push(secs);
            mhz_b.push(clock);
            let (secs, clock) = timed_region(&mut a, units_per_round)?;
            a_unit_secs.push(secs);
            mhz_a.push(clock);
        }
    }

    Ok(Paired {
        label_a: label_a.into(),
        label_b: label_b.into(),
        units_per_round,
        a_unit_secs,
        b_unit_secs,
        mhz_a,
        mhz_b,
    })
}

fn calibrate(work: &mut impl FnMut() -> Result<u64, BackendError>) -> Result<u64, BackendError> {
    let mut probe_reps = 1_u64;
    loop {
        let started = Instant::now();
        let mut digest = 0_u64;
        for _ in 0..probe_reps {
            digest = digest.wrapping_add(work()?);
        }
        black_box(digest);
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed >= CALIBRATION_PROBE_SECS {
            let scaled = (probe_reps as f64) * (CALIBRATION_TARGET_SECS / elapsed);
            return Ok((scaled.ceil() as u64).max(1));
        }
        let Some(next) = probe_reps.checked_mul(4) else {
            return Ok(probe_reps);
        };
        probe_reps = next;
    }
}

/// One timed region: seconds per unit, plus the busy-weighted clock across it.
fn timed_region(
    work: &mut impl FnMut() -> Result<u64, BackendError>,
    units: u64,
) -> Result<(f64, f64), BackendError> {
    let cpu_before = CpuSample::now();
    let started = Instant::now();
    let mut digest = 0_u64;
    for _ in 0..units {
        digest = digest.wrapping_add(work()?);
    }
    let elapsed = started.elapsed().as_secs_f64();
    black_box(digest);
    let cpu_after = CpuSample::now();
    Ok((
        elapsed / units as f64,
        busy_weighted_mhz(&cpu_before, &cpu_after),
    ))
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn rel_stddev(values: &[f64]) -> f64 {
    let mean = mean(values);
    if mean <= 0.0 || values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt() / mean
}

fn join_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.12e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Write raw samples where `Bench` collects them for the shared table.
fn write_sidecar(phase: &str, body: &str) {
    let Ok(home) = std::env::var("HOME") else {
        return;
    };
    let dir = std::path::Path::new(&home).join(".cache/exatok-bench/reports");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let path = dir.join(format!("enqueue-{phase}-{stamp}.json"));
    if let Err(error) = std::fs::write(&path, body) {
        println!("[host-cost] could not persist samples to {path:?}: {error}");
        return;
    }
    println!("[host-cost] raw samples: {}", path.display());
}

/// A program whose IR node count is `chain_len` and whose buffer count is
/// `buffer_count`, launched over one thread.
///
/// Both axes matter separately. `try_normalized_program_cache_digest` walks
/// buffers AND the whole node list; `LaunchPlan::from_bindings` and
/// `BindingPlan::build` walk buffers only. Sweeping them independently is what
/// separates "fixed per dispatch" from "scales with program size", which is the
/// question `GpuPretokWiring` needs answered to compute a break-even.
fn attribution_program(buffer_count: usize, chain_len: usize) -> Program {
    let buffer_count = buffer_count.max(2);
    let chain_len = chain_len.max(1);
    let mut buffers = vec![
        BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
        BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
    ];
    let scratch_names: Vec<String> = (2..buffer_count)
        .map(|index| format!("scratch{index}"))
        .collect();
    for (offset, name) in scratch_names.iter().enumerate() {
        buffers.push(
            BufferDecl::storage(
                name.as_str(),
                (offset + 2) as u32,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(4),
        );
    }

    let accumulator_names: Vec<String> =
        (0..chain_len).map(|index| format!("acc{index}")).collect();
    let mut body = vec![Node::let_bind(
        accumulator_names[0].as_str(),
        Expr::load("in", Expr::u32(0)),
    )];
    for index in 1..chain_len {
        body.push(Node::let_bind(
            accumulator_names[index].as_str(),
            Expr::add(
                Expr::var(accumulator_names[index - 1].as_str()),
                Expr::u32(index as u32),
            ),
        ));
    }
    body.push(Node::store(
        "out",
        Expr::u32(0),
        Expr::var(accumulator_names[chain_len - 1].as_str()),
    ));

    Program::wrapped(buffers, [1, 1, 1], body)
}

/// Fold a byte digest into the accumulator so the phase cannot be deleted.
fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut acc = 0_u64;
    for (index, byte) in bytes.iter().enumerate() {
        acc = acc.wrapping_add((*byte as u64).wrapping_mul(index as u64 + 1));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend plus resident handles for one program shape, allocated once.
    ///
    /// Allocation, upload and the cold PTX lowering all stay outside every
    /// timed closure.
    struct Fixture {
        program: Program,
        handles: Vec<CudaResidentBuffer>,
        config: DispatchConfig,
    }

    impl Fixture {
        fn new(backend: &CudaBackend, buffer_count: usize, chain_len: usize) -> Self {
            let program = attribution_program(buffer_count, chain_len);
            let handles: Vec<CudaResidentBuffer> = program
                .buffers()
                .iter()
                .map(|_| {
                    backend
                        .allocate_resident(16)
                        .expect("Fix: attribution fixture resident allocation must succeed.")
                })
                .collect();
            let config = DispatchConfig::default();
            // Warm the PTX source cache, module cache, and transient pool.
            backend
                .dispatch_resident_timed(&program, &handles, &config)
                .expect("Fix: attribution fixture warm dispatch must succeed.");
            Self {
                program,
                handles,
                config,
            }
        }

        fn free(self, backend: &CudaBackend) {
            for handle in self.handles {
                let _ = backend.free_resident(handle);
            }
        }
    }

    /// Host telemetry accepts a file exactly at its configured byte cap.
    #[test]
    fn bounded_host_read_accepts_exact_cap() {
        let dir = tempfile::tempdir().expect("Fix: host-read fixture directory must exist");
        let path = dir.path().join("stat");
        std::fs::write(&path, "12345678").expect("Fix: host-read fixture must be writable");

        assert_eq!(
            read_host_text_bounded(&path, 8).as_deref(),
            Some("12345678")
        );
    }

    /// Host telemetry rejects oversized pseudo-files instead of allocating without a bound.
    #[test]
    fn bounded_host_read_rejects_oversized_input() {
        let dir = tempfile::tempdir().expect("Fix: host-read fixture directory must exist");
        let path = dir.path().join("stat");
        std::fs::write(&path, "123456789").expect("Fix: host-read fixture must be writable");

        assert_eq!(read_host_text_bounded(&path, 8), None);
    }

    /// Missing host telemetry remains a best-effort absence rather than a dispatch failure.
    #[test]
    fn bounded_host_read_reports_missing_input_as_none() {
        let dir = tempfile::tempdir().expect("Fix: host-read fixture directory must exist");

        assert_eq!(read_host_text_bounded(dir.path().join("missing"), 8), None);
    }

    /// Attribute the fixed per-dispatch host cost of the resident path, phase
    /// by phase, in nanoseconds.
    ///
    /// The resident path is `exatok`'s path: it calls `dispatch_resident_timed`
    /// per dispatch, redispatching the same program shape out of a shape-keyed
    /// plan cache.
    ///
    /// Cells R0..R4 are CUMULATIVE PREFIXES of that call, so each phase is the
    /// difference of two adjacent minima and every phase is measured in a real
    /// dispatch context rather than synthetically. Cells K* and E* isolate the
    /// individual pieces so a prefix difference can be checked against the sum
    /// of its parts; where they disagree, the prefix difference is the honest
    /// figure and the gap is reported rather than hidden.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_resident_path() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");
        // 9 buffers matches GpuPretokWiring's declared pass and is close to
        // exatok's device loop program (7 harness buffers plus caller bindings).
        let fixture = Fixture::new(&backend, 9, 64);
        let program = &fixture.program;
        let handles = &fixture.handles;
        let config = &fixture.config;

        let mut cells = Vec::new();

        cells.push(
            measure_cell("R0 resident_bindings_from_handles", || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                Ok(bindings.len() as u64)
            })
            .expect("Fix: R0 cell must measure."),
        );

        cells.push(
            measure_cell("R1 + prepare_resident_dispatch", || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                let prepared =
                    backend.prepare_resident_dispatch(black_box(program), &bindings, config)?;
                Ok(prepared.bindings.bindings.len() as u64 + prepared.launch.element_count as u64)
            })
            .expect("Fix: R1 cell must measure."),
        );

        cells.push(
            measure_cell("R2 + ptx_for_program_cached_with_key", || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                let prepared =
                    backend.prepare_resident_dispatch(black_box(program), &bindings, config)?;
                let (ptx, key) =
                    backend.ptx_for_program_cached_with_key(black_box(program), config)?;
                Ok(prepared.bindings.bindings.len() as u64
                    + ptx.len() as u64
                    + digest_bytes(key.as_bytes()))
            })
            .expect("Fix: R2 cell must measure."),
        );

        cells.push(
            measure_cell("R3 + module_cache_key_for_ptx_source_key", || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                let prepared =
                    backend.prepare_resident_dispatch(black_box(program), &bindings, config)?;
                let (ptx, key) =
                    backend.ptx_for_program_cached_with_key(black_box(program), config)?;
                let module_key = backend.module_cache_key_for_ptx_source_key(key)?;
                Ok(prepared.bindings.bindings.len() as u64
                    + ptx.len() as u64
                    + digest_bytes(&module_key.0))
            })
            .expect("Fix: R3 cell must measure."),
        );

        cells.push(
            measure_cell("R4 dispatch_resident_timed (whole call)", || {
                let timed = backend.dispatch_resident_timed(
                    black_box(program),
                    black_box(handles),
                    config,
                )?;
                Ok(timed.wall_ns + timed.outputs.len() as u64)
            })
            .expect("Fix: R4 cell must measure."),
        );

        println!("\n=== CUDA resident dispatch: cumulative prefix cells ===");
        println!("program: {} buffers, {} body nodes", 9, 64);
        for cell in &cells {
            println!("{}", cell.render());
        }
        println!("\n--- phase costs (adjacent prefix differences) ---");
        let names = [
            "P1 resident_bindings_from_handles",
            "P2 prepare_resident_dispatch",
            "P3 ptx_for_program_cached_with_key",
            "P4 module_cache_key_for_ptx_source_key",
            "P5 enqueue + launch + sync + readback",
        ];
        let mut previous = 0.0_f64;
        for (index, name) in names.iter().enumerate() {
            let current = cells[index].min_ns();
            println!("{name:<44} {:>13.1} ns", current - previous);
            previous = current;
        }
        println!(
            "{:<44} {:>13.1} ns",
            "TOTAL per dispatch",
            cells[4].min_ns()
        );

        for cell in &cells {
            assert!(
                cell.min_ns() > 1.0,
                "Fix: cell `{}` measured at {:.2} ns, which is below the floor a real \
                 host phase can reach. The probe was optimized away rather than the phase \
                 being fast; consume the result through black_box.",
                cell.label,
                cell.min_ns()
            );
        }

        fixture.free(&backend);
    }

    /// Decompose the PTX-source cache key derivation, the phase that turned out
    /// to dominate.
    ///
    /// Every one of these runs on every dispatch of an already-lowered,
    /// already-cached program, purely to rediscover a key the caller could have
    /// kept.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_ptx_key_components() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");
        let fixture = Fixture::new(&backend, 9, 64);
        let program = &fixture.program;
        let config = &fixture.config;
        let adapter_caps = backend.caps.to_adapter_caps();
        let subgroup_size = backend
            .warp_size()
            .expect("Fix: CUDA warp size probe must report a hardware warp size.");
        let feature_flags = backend.pipeline_feature_flags();
        let ptx_target_sm = backend.ptx_target_sm();

        let mut cells = Vec::new();

        cells.push(
            measure_cell("K0 Program::clone", || {
                let cloned = black_box(program).clone();
                Ok(cloned.buffers().len() as u64 + cloned.entry().len() as u64)
            })
            .expect("Fix: K0 cell must measure."),
        );

        cells.push(
            measure_cell("K1 lower_subgroup_reductions(clone)", || {
                let lowered = vyre_foundation::lower::lower_subgroup_reductions(
                    black_box(program).clone(),
                    black_box(&adapter_caps),
                );
                Ok(lowered.buffers().len() as u64 + lowered.entry().len() as u64)
            })
            .expect("Fix: K1 cell must measure."),
        );

        cells.push(
            measure_cell("K2 try_normalized_program_cache_digest", || {
                let digest = vyre_driver::try_normalized_program_cache_digest(black_box(program))
                    .map_err(BackendError::new)?;
                Ok(digest_bytes(&digest))
            })
            .expect("Fix: K2 cell must measure."),
        );

        cells.push(
            measure_cell("K3 vsa fingerprint on a FRESH clone", || {
                let lowered = vyre_foundation::lower::lower_subgroup_reductions(
                    black_box(program).clone(),
                    black_box(&adapter_caps),
                );
                let words = vyre_driver::program_vsa_fingerprint_words(&lowered);
                Ok(words.iter().map(|w| u64::from(*w)).sum())
            })
            .expect("Fix: K3 cell must measure."),
        );

        cells.push(
            measure_cell("K3b vsa fingerprint on a REUSED program", || {
                let words = vyre_driver::program_vsa_fingerprint_words(black_box(program));
                Ok(words.iter().map(|w| u64::from(*w)).sum())
            })
            .expect("Fix: K3b cell must measure."),
        );

        cells.push(
            measure_cell("K4 dispatch_policy_cache_digest", || {
                let digest = vyre_driver::dispatch_policy_cache_digest(black_box(config));
                Ok(digest_bytes(&digest))
            })
            .expect("Fix: K4 cell must measure."),
        );

        cells.push(
            measure_cell("K5 ptx_source_cache.key_for_program", || {
                let key = backend.ptx_source_cache.key_for_program(
                    black_box(program),
                    config,
                    ptx_target_sm,
                    subgroup_size,
                    feature_flags,
                )?;
                Ok(digest_bytes(key.as_bytes()))
            })
            .expect("Fix: K5 cell must measure."),
        );

        cells.push(
            measure_cell("K6 whole ptx_for_program_cached_with_key", || {
                let (ptx, key) =
                    backend.ptx_for_program_cached_with_key(black_box(program), config)?;
                Ok(ptx.len() as u64 + digest_bytes(key.as_bytes()))
            })
            .expect("Fix: K6 cell must measure."),
        );

        println!("\n=== PTX source cache key derivation, per dispatch ===");
        for cell in &cells {
            println!("{}", cell.render());
        }

        fixture.free(&backend);
    }

    /// Decompose the enqueue itself: the pieces that run between a resolved
    /// plan and a synchronized stream.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_enqueue_components() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");
        let fixture = Fixture::new(&backend, 9, 64);
        let program = &fixture.program;
        let handles = &fixture.handles;
        let config = &fixture.config;

        let bindings = resident_bindings_from_handles(handles)
            .expect("Fix: attribution bindings must resolve.");
        let prepared = backend
            .prepare_resident_dispatch(program, &bindings, config)
            .expect("Fix: attribution plan must prepare.");
        let (ptx_src, ptx_key) = backend
            .ptx_for_program_cached_with_key(program, config)
            .expect("Fix: attribution PTX must lower.");
        let module_key = backend
            .module_cache_key_for_ptx_source_key(ptx_key)
            .expect("Fix: attribution module key must derive.");

        let mut cells = Vec::new();

        cells.push(
            measure_cell("E0 warmup (context bind)", || {
                backend.warmup()?;
                Ok(1)
            })
            .expect("Fix: E0 cell must measure."),
        );

        cells.push(
            measure_cell("E1 acquire_stream + release_stream", || {
                let stream = backend.launch_resources.acquire_stream()?;
                let raw = stream.raw() as usize as u64;
                backend.launch_resources.release_stream(stream);
                Ok(raw)
            })
            .expect("Fix: E1 cell must measure."),
        );

        cells.push(
            measure_cell("E2 acquire/release timing event pair", || {
                let (start, end) = backend.launch_resources.acquire_timing_event_pair()?;
                backend.launch_resources.release_timing_event(start);
                backend.launch_resources.release_timing_event(end);
                Ok(1)
            })
            .expect("Fix: E2 cell must measure."),
        );

        cells.push(
            measure_cell("E3 resolve_launch_function", || {
                let func = backend.resolve_launch_function(
                    black_box(&ptx_src),
                    module_key,
                    &prepared.launch,
                    prepared.cooperative,
                )?;
                Ok(func as usize as u64)
            })
            .expect("Fix: E3 cell must measure."),
        );

        {
            let stream = backend
                .launch_resources
                .acquire_stream()
                .expect("Fix: attribution stream must acquire.");
            let raw = stream.raw();
            cells.push(
                measure_cell("E4 cuStreamSynchronize on an idle stream", || {
                    crate::stream::synchronize_raw_stream(raw, "host-cost attribution sync")?;
                    Ok(1)
                })
                .expect("Fix: E4 cell must measure."),
            );
            backend.launch_resources.release_stream(stream);
        }

        cells.push(
            measure_cell("E5 BindingPlan::build", || {
                let plan = BindingPlan::build(black_box(program))?;
                Ok(plan.bindings.len() as u64)
            })
            .expect("Fix: E5 cell must measure."),
        );

        {
            let static_bindings =
                BindingPlan::build(program).expect("Fix: attribution binding plan must build.");
            let limits = backend.launch_limits();
            cells.push(
                measure_cell("E6 LaunchPlan::from_bindings", || {
                    let plan = LaunchPlan::from_bindings(
                        black_box(program),
                        &static_bindings.bindings,
                        config,
                        limits,
                    )?;
                    Ok(plan.element_count as u64 + plan.param_words.len() as u64)
                })
                .expect("Fix: E6 cell must measure."),
            );
        }

        cells.push(
            measure_cell("E7 validate_program_cached", || {
                backend.validate_program_cached(black_box(program))?;
                Ok(1)
            })
            .expect("Fix: E7 cell must measure."),
        );

        cells.push(
            measure_cell("E8 contains_grid_sync walk", || {
                Ok(u64::from(vyre_driver::grid_sync::contains_grid_sync(
                    black_box(program),
                )))
            })
            .expect("Fix: E8 cell must measure."),
        );

        println!("\n=== Enqueue components, per dispatch ===");
        for cell in &cells {
            println!("{}", cell.render());
        }

        fixture.free(&backend);
    }

    /// Does the per-dispatch host cost SCALE, and with what?
    ///
    /// `GpuPretokWiring` needs this to know whether their break-even is a
    /// constant or a function of input size, and `TransferVolume` needs it to
    /// know whether reducing transfer volume or reducing dispatch count is the
    /// better lever. Sweeps IR node count and buffer count independently.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_scaling_sweep() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");

        println!("\n=== Scaling with IR node count (9 buffers fixed) ===");
        println!(
            "{:<12} {:>14} {:>14} {:>14}",
            "body nodes", "ptx key ns", "prepare ns", "full dispatch ns"
        );
        for chain_len in [1_usize, 16, 128, 1024] {
            let fixture = Fixture::new(&backend, 9, chain_len);
            let program = &fixture.program;
            let handles = &fixture.handles;
            let config = &fixture.config;

            let key_cell = measure_cell(format!("ptx-key n={chain_len}"), || {
                let (ptx, key) =
                    backend.ptx_for_program_cached_with_key(black_box(program), config)?;
                Ok(ptx.len() as u64 + digest_bytes(key.as_bytes()))
            })
            .expect("Fix: sweep ptx-key cell must measure.");
            let prepare_cell = measure_cell(format!("prepare n={chain_len}"), || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                let prepared =
                    backend.prepare_resident_dispatch(black_box(program), &bindings, config)?;
                Ok(prepared.bindings.bindings.len() as u64)
            })
            .expect("Fix: sweep prepare cell must measure.");
            let full_cell = measure_cell(format!("dispatch n={chain_len}"), || {
                let timed = backend.dispatch_resident_timed(
                    black_box(program),
                    black_box(handles),
                    config,
                )?;
                Ok(timed.wall_ns)
            })
            .expect("Fix: sweep dispatch cell must measure.");

            println!(
                "{:<12} {:>14.1} {:>14.1} {:>14.1}",
                chain_len,
                key_cell.min_ns(),
                prepare_cell.min_ns(),
                full_cell.min_ns()
            );
            fixture.free(&backend);
        }

        println!("\n=== Scaling with buffer count (64 body nodes fixed) ===");
        println!(
            "{:<12} {:>14} {:>14} {:>14}",
            "buffers", "ptx key ns", "prepare ns", "full dispatch ns"
        );
        for buffer_count in [2_usize, 5, 9, 17] {
            let fixture = Fixture::new(&backend, buffer_count, 64);
            let program = &fixture.program;
            let handles = &fixture.handles;
            let config = &fixture.config;

            let key_cell = measure_cell(format!("ptx-key b={buffer_count}"), || {
                let (ptx, key) =
                    backend.ptx_for_program_cached_with_key(black_box(program), config)?;
                Ok(ptx.len() as u64 + digest_bytes(key.as_bytes()))
            })
            .expect("Fix: sweep ptx-key cell must measure.");
            let prepare_cell = measure_cell(format!("prepare b={buffer_count}"), || {
                let bindings = resident_bindings_from_handles(black_box(handles))?;
                let prepared =
                    backend.prepare_resident_dispatch(black_box(program), &bindings, config)?;
                Ok(prepared.bindings.bindings.len() as u64)
            })
            .expect("Fix: sweep prepare cell must measure.");
            let full_cell = measure_cell(format!("dispatch b={buffer_count}"), || {
                let timed = backend.dispatch_resident_timed(
                    black_box(program),
                    black_box(handles),
                    config,
                )?;
                Ok(timed.wall_ns)
            })
            .expect("Fix: sweep dispatch cell must measure.");

            println!(
                "{:<12} {:>14.1} {:>14.1} {:>14.1}",
                buffer_count,
                key_cell.min_ns(),
                prepare_cell.min_ns(),
                full_cell.min_ns()
            );
            fixture.free(&backend);
        }
    }

    /// Is the per-output readback cost per BYTE or per OPERATION, and would
    /// batching the copies onto the stream remove it?
    ///
    /// The buffer-count sweep put the marginal cost of one more output binding
    /// at roughly 6.4 us on 16-byte buffers, which cannot be bandwidth. The
    /// resident path issues one BLOCKING `cuMemcpyDtoH_v2` per output after it
    /// has already synchronized the stream, so the suspect is the per-call
    /// cost of a blocking copy. This isolates that: N blocking copies against
    /// N stream-ordered async copies followed by ONE synchronize, over
    /// identical pinned destinations and identical device sources.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_readback_strategy() {
        const COPY_BYTES: usize = 16;

        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");

        println!("\n=== Blocking readback cost against copy count ({COPY_BYTES}-byte copies) ===");
        println!(
            "{:<10} {:>16} {:>16} {:>14}",
            "copies", "N blocking ns", "N async+1 sync", "ratio"
        );

        for copy_count in [1_usize, 4, 8, 16] {
            let handles: Vec<CudaResidentBuffer> = (0..copy_count)
                .map(|_| {
                    backend
                        .allocate_resident(COPY_BYTES)
                        .expect("Fix: readback fixture allocation must succeed.")
                })
                .collect();
            let device_ptrs: Vec<u64> = handles
                .iter()
                .map(|handle| {
                    backend
                        .resident_store
                        .view(*handle)
                        .expect("Fix: readback fixture view must resolve.")
                        .ptr
                })
                .collect();
            let mut host_slots: Vec<crate::backend::allocations::PinnedHostAllocation> = (0
                ..copy_count)
                .map(|_| {
                    backend
                        .host_pool
                        .acquire(COPY_BYTES)
                        .expect("Fix: readback fixture pinned host allocation must succeed.")
                })
                .collect();
            let host_ptrs: Vec<*mut std::ffi::c_void> = host_slots
                .iter_mut()
                .map(crate::backend::allocations::PinnedHostAllocation::as_mut_ptr)
                .collect();
            let stream = backend
                .launch_resources
                .acquire_stream()
                .expect("Fix: readback fixture stream must acquire.");
            let stream_raw = stream.raw();

            let paired = compare_cells(
                format!("{copy_count} blocking cuMemcpyDtoH_v2"),
                || {
                    for index in 0..copy_count {
                        // SAFETY: both pointers are live for the whole cell:
                        // the device buffers are resident allocations held in
                        // `handles` and the destinations are pinned host slots
                        // held in `host_slots`, all of COPY_BYTES.
                        unsafe {
                            crate::backend::copy::d2h_sync_checked_with_label(
                                black_box(host_ptrs[index]),
                                black_box(device_ptrs[index]),
                                COPY_BYTES,
                                "attribution blocking readback",
                            )?;
                        }
                    }
                    Ok(copy_count as u64)
                },
                format!("{copy_count} async cuMemcpyDtoHAsync_v2 + 1 sync"),
                || {
                    for index in 0..copy_count {
                        // SAFETY: same pointers and lifetimes as the blocking
                        // arm, and the synchronize below completes every copy
                        // before the cell returns.
                        unsafe {
                            crate::backend::copy::d2h_async_checked_with_label(
                                black_box(host_ptrs[index]),
                                black_box(device_ptrs[index]),
                                COPY_BYTES,
                                stream_raw,
                                "attribution async readback",
                            )?;
                        }
                    }
                    crate::stream::synchronize_raw_stream(
                        stream_raw,
                        "attribution async readback sync",
                    )?;
                    Ok(copy_count as u64)
                },
                MIN_PAIRED_ROUNDS,
            )
            .expect("Fix: readback strategy comparison must measure.");

            println!(
                "{:<10} {:>16.1} {:>16.1} {:>13.2}x",
                copy_count,
                min_of(&paired.a_unit_secs) * 1e9,
                min_of(&paired.b_unit_secs) * 1e9,
                paired.median_speedup()
            );
            println!("  {}", paired.render().replace('\n', "\n  "));
            write_sidecar(
                &format!("readback-strategy-n{copy_count}"),
                &paired.to_json(0, 0),
            );

            backend.launch_resources.release_stream(stream);
            for slot in host_slots {
                backend.host_pool.release(slot);
            }
            for handle in handles {
                let _ = backend.free_resident(handle);
            }
        }
    }
    /// Size the device-signature TOML parse that both P2 and P3 pay for.
    ///
    /// `DeviceSignatureTable::builtins()` takes no arguments and reads no
    /// files: it parses the compiled-in `BUILTIN_BLACKWELL_120` string, sorts,
    /// and wraps. It is therefore a pure function of compile-time constants,
    /// and every dispatch calls it TWICE, once through
    /// `validation_options -> to_device_profile` in P2 and once through
    /// `ptx_for_program_cached_with_key -> to_adapter_caps` in P3.
    ///
    /// D5 is the `Program` deep clone that P3 makes before lowering, which is
    /// what puts the per-IR-node slope on the PTX key phase.
    #[test]
    #[ignore = "measurement instrument: run with --ignored --nocapture"]
    fn host_cost_attribution_device_profile() {
        let backend = CudaBackend::acquire()
            .expect("Fix: CUDA backend acquisition must succeed on a GPU-required host.");
        let fixture = Fixture::new(&backend, 9, 64);
        let program = &fixture.program;

        let mut cells = Vec::new();

        cells.push(
            measure_cell("D1 DeviceSignatureTable::builtins", || {
                let table = vyre_driver::DeviceSignatureTable::builtins()
                    .map_err(|error| BackendError::InvalidProgram { fix: error })?;
                Ok(black_box(table).signatures().len() as u64)
            })
            .expect("Fix: D1 cell must measure."),
        );

        cells.push(
            measure_cell("D2 caps.to_device_profile", || {
                let profile = black_box(&backend.caps).to_device_profile();
                Ok(u64::from(black_box(profile).compute_units))
            })
            .expect("Fix: D2 cell must measure."),
        );

        cells.push(
            measure_cell("D3 caps.to_adapter_caps", || {
                let adapter = black_box(&backend.caps).to_adapter_caps();
                Ok(u64::from(black_box(adapter).subgroup_size))
            })
            .expect("Fix: D3 cell must measure."),
        );

        cells.push(
            measure_cell("D4 backend.validation_options", || {
                let options = black_box(&backend).validation_options();
                Ok(u64::from(
                    black_box(&options)
                        .backend_capabilities
                        .is_some_and(|caps| caps.has_shared_memory),
                ))
            })
            .expect("Fix: D4 cell must measure."),
        );

        cells.push(
            measure_cell("D5 program.clone", || {
                let cloned = black_box(program).clone();
                Ok(black_box(&cloned).buffers.len() as u64)
            })
            .expect("Fix: D5 cell must measure."),
        );

        println!("\n=== Device-profile derivation cost (per call) ===");
        for cell in &cells {
            println!("{}", cell.render());
        }
        println!(
            "\nbuiltins() is called TWICE per dispatch (P2 and P3): {:.1} ns per dispatch.",
            cells[0].min_ns() * 2.0
        );

        // What the memo actually removed. The A arm is the parse that used to
        // run on every `builtins()` call; the B arm is the memoized call as it
        // ships, whose remaining cost is the table clone.
        let paired = compare_cells(
            "uncached DeviceSignature::from_toml_str".to_string(),
            || {
                let signature = vyre_driver::DeviceSignature::from_toml_str(black_box(
                    vyre_driver::DeviceSignature::BUILTIN_BLACKWELL_120,
                ))
                .map_err(|error| BackendError::InvalidProgram { fix: error })?;
                Ok(u64::from(black_box(signature).max_sm))
            },
            "memoized DeviceSignatureTable::builtins".to_string(),
            || {
                let table = vyre_driver::DeviceSignatureTable::builtins()
                    .map_err(|error| BackendError::InvalidProgram { fix: error })?;
                Ok(black_box(table).signatures().len() as u64)
            },
            MIN_PAIRED_ROUNDS,
        )
        .expect("Fix: signature-table memo comparison must measure.");
        println!("\n{}", paired.render());
        write_sidecar("device-signature-memo", &paired.to_json(0, 0));

        for cell in &cells {
            assert!(
                cell.min_ns() > 1.0,
                "Fix: cell `{}` measured at {:.2} ns, which is below the floor a real \
                 host phase can reach. The probe was optimized away rather than the phase \
                 being fast; consume the result through black_box.",
                cell.label,
                cell.min_ns()
            );
        }

        fixture.free(&backend);
    }
}
