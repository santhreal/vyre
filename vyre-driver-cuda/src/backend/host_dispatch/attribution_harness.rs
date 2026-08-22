use std::hint::black_box;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

use vyre_driver::BackendError;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Duration a timed region is folded up to before it is trusted. Below this,
/// clock resolution and frequency ramp dominate the sample.
pub(crate) const CALIBRATION_TARGET_SECS: f64 = 0.5;
/// Timed repetitions per absolute cell. Main's floor is 5.
pub(crate) const CELL_REPS: usize = 5;
/// Relative standard deviation above which a cell is advisory only.
pub(crate) const RELIABLE_REL_STDDEV: f64 = 0.05;
/// Relative standard deviation above which a cell is void, not data.
pub(crate) const VOIDING_REL_STDDEV: f64 = 0.20;
/// Between-arm busy-weighted clock gap above which a paired result is
/// disqualified as frequency ramp rather than reported as a delta.
pub(crate) const MAX_CLOCK_DRIFT_FRACTION: f64 = 0.02;
/// Shortest calibration probe that is long enough to resolve against the
/// monotonic clock before the repetition count is extrapolated from it.
pub(crate) const CALIBRATION_PROBE_SECS: f64 = 0.05;
/// Maximum `/proc/stat` bytes accepted by the host attribution probe.
pub(crate) const MAX_PROC_STAT_BYTES: u64 = 1_048_576;
/// Maximum bytes accepted from one CPU frequency sysfs node.
pub(crate) const MAX_CPU_FREQ_BYTES: u64 = 64;

/// Busy-weighted mean CPU frequency in MHz across one region.
///
/// Weighting by per-CPU non-idle jiffies rather than taking the whole-machine
/// mean is deliberate: on a 16-core box with a handful of busy cores the
/// unweighted mean tracks how many cores are parked, not what frequency the
/// working core ran at.
pub(crate) fn busy_weighted_mhz(before: &CpuSample, after: &CpuSample) -> f64 {
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

pub(crate) fn read_host_text_bounded(path: impl AsRef<Path>, max_bytes: u64) -> Option<String> {
    let mut text = String::new();
    std::fs::File::open(path)
        .ok()?
        .take(max_bytes.saturating_add(1))
        .read_to_string(&mut text)
        .ok()?;
    (text.len() as u64 <= max_bytes).then_some(text)
}

/// Per-CPU busy time and current frequency at one instant.
pub(crate) struct CpuSample {
    pub(crate) busy_jiffies: Vec<u64>,
    pub(crate) mhz: Vec<f64>,
}

impl CpuSample {
    /// Read `/proc/stat` and the cpufreq sysfs nodes.
    ///
    /// Both are best-effort: a host without `scaling_cur_freq` reports zeroed
    /// frequencies, which makes [`busy_weighted_mhz`] return 0.0 and the drift
    /// check inconclusive rather than wrong.
    pub(crate) fn now() -> Self {
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
pub(crate) struct Cell {
    pub(crate) label: String,
    /// Inner repetitions folded into one timed region.
    pub(crate) units_per_round: u64,
    /// Seconds per single unit, one entry per timed region.
    pub(crate) unit_secs: Vec<f64>,
    /// Busy-weighted MHz across each timed region.
    pub(crate) mhz: Vec<f64>,
}

impl Cell {
    /// Point estimate: the fastest observation, in nanoseconds per unit.
    pub(crate) fn min_ns(&self) -> f64 {
        self.unit_secs.iter().copied().fold(f64::INFINITY, f64::min) * 1e9
    }

    pub(crate) fn mean_secs(&self) -> f64 {
        if self.unit_secs.is_empty() {
            return 0.0;
        }
        self.unit_secs.iter().sum::<f64>() / self.unit_secs.len() as f64
    }

    /// Spread of the samples relative to their mean.
    pub(crate) fn rel_stddev(&self) -> f64 {
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

    pub(crate) fn clock_mhz(&self) -> f64 {
        if self.mhz.is_empty() {
            return 0.0;
        }
        self.mhz.iter().sum::<f64>() / self.mhz.len() as f64
    }

    pub(crate) fn verdict(&self) -> &'static str {
        let spread = self.rel_stddev();
        if spread > VOIDING_REL_STDDEV {
            "VOID"
        } else if spread > RELIABLE_REL_STDDEV {
            "advisory"
        } else {
            "ok"
        }
    }

    pub(crate) fn render(&self) -> String {
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
pub(crate) fn measure_cell(
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
pub(crate) struct Paired {
    pub(crate) label_a: String,
    pub(crate) label_b: String,
    pub(crate) units_per_round: u64,
    pub(crate) a_unit_secs: Vec<f64>,
    pub(crate) b_unit_secs: Vec<f64>,
    pub(crate) mhz_a: Vec<f64>,
    pub(crate) mhz_b: Vec<f64>,
}

impl Paired {
    /// Median of the per-round ratios.
    ///
    /// Median rather than best: the contaminant has already largely cancelled
    /// inside a round, so the best ratio only selects the round where residual
    /// noise favoured B.
    pub(crate) fn median_speedup(&self) -> f64 {
        let mut ratios: Vec<f64> = self
            .a_unit_secs
            .iter()
            .zip(&self.b_unit_secs)
            .filter(|(_, b)| **b > 0.0)
            .map(|(a, b)| a / b)
            .collect();
        median(&mut ratios)
    }

    pub(crate) fn speedup_rel_stddev(&self) -> f64 {
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
    pub(crate) fn median_delta_ns(&self) -> f64 {
        let mut deltas: Vec<f64> = self
            .a_unit_secs
            .iter()
            .zip(&self.b_unit_secs)
            .map(|(a, b)| (a - b) * 1e9)
            .collect();
        median(&mut deltas)
    }

    pub(crate) fn clock_drift_fraction(&self) -> f64 {
        let a = mean(&self.mhz_a);
        let b = mean(&self.mhz_b);
        if a <= 0.0 || b <= 0.0 {
            return 0.0;
        }
        ((a - b) / a).abs()
    }

    /// Whether this comparison may be quoted.
    pub(crate) fn is_publishable(&self) -> bool {
        self.a_unit_secs.len() >= MIN_PAIRED_ROUNDS
            && self.speedup_rel_stddev() <= VOIDING_REL_STDDEV
            && self.clock_drift_fraction() <= MAX_CLOCK_DRIFT_FRACTION
    }

    pub(crate) fn render(&self) -> String {
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
    pub(crate) fn to_json(
        &self,
        foreign_resident_baseline_mib: u64,
        foreign_sm_percent_max: u64,
    ) -> String {
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
pub(crate) const MIN_PAIRED_ROUNDS: usize = 8;

/// Run `a` against `b`, interleaved, alternating which arm leads each round.
///
/// ABBA rather than ABAB. Under plain alternation the trailing arm always runs
/// on state the leading arm just warmed, which is a systematic advantage when
/// the phase under test IS a cache lookup, and several of these are.
pub(crate) fn compare_cells(
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

pub(crate) fn calibrate(
    work: &mut impl FnMut() -> Result<u64, BackendError>,
) -> Result<u64, BackendError> {
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
pub(crate) fn timed_region(
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

pub(crate) fn median(values: &mut [f64]) -> f64 {
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

pub(crate) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

pub(crate) fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

pub(crate) fn rel_stddev(values: &[f64]) -> f64 {
    let mean = mean(values);
    if mean <= 0.0 || values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt() / mean
}

pub(crate) fn join_f64(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("{v:.12e}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Write raw samples where `Bench` collects them for the shared table.
pub(crate) fn write_sidecar(phase: &str, body: &str) {
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
pub(crate) fn attribution_program(buffer_count: usize, chain_len: usize) -> Program {
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
pub(crate) fn digest_bytes(bytes: &[u8]) -> u64 {
    let mut acc = 0_u64;
    for (index, byte) in bytes.iter().enumerate() {
        acc = acc.wrapping_add((*byte as u64).wrapping_mul(index as u64 + 1));
    }
    acc
}
