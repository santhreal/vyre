//! W7-2: the head-to-head the whole plan is judged by, vyre's GPU literal-set
//! scan vs the best CPU multi-pattern matcher (`aho-corasick`), END-TO-END with
//! staging included, on a consumer-shaped corpus.
//!
//! The claim vyre must sustain is not "fast kernel" but "beats the best CPU path
//! end-to-end, staging included." This case runs the SAME pattern set over the
//! SAME corpus through (a) vyre's resident GPU scan (tables uploaded once; every
//! timed dispatch re-stages the haystack and reads the matches back, the real
//! per-scan consumer cost) and (b) the `aho-corasick` crate built for the same
//! all-overlapping semantics vyre uses (`MatchKind::Standard` +
//! `find_overlapping_iter`; the automaton is built once, like vyre's matcher). It
//! reports the end-to-end speedup and, crucially. HARD-GATES that the two
//! produce the identical match set (a fast wrong answer is no answer). The
//! performance delta is REPORTED, never gated: today this comparison can favor the
//! CPU at the flagship consumer, and this bench exists to make that gap and its
//! closure visible per release, not to assert a win that isn't there.

use std::time::Instant;

use aho_corasick::{AhoCorasick, MatchKind};

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;
use crate::cases::scan_ac_irregular::support::{build_irregular_haystack, encode_match_triples};
use crate::cases::scan_ac_irregular::PATTERNS;
use vyre::VyreBackend;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::GpuLiteralSet;

/// A consumer-shaped mixed corpus, large enough that the GPU's staging (upload +
/// readback) is a real fraction of wall time, so the comparison is honestly
/// end-to-end and not a kernel-only microbenchmark.
const CORPUS_BYTES: usize = 8 * 1024 * 1024;
/// Steady-state iterations timed for each engine (a loop, not a cold shot).
const ITERS: usize = 16;
const MATCH_TRIPLE_BYTES: u64 = 12;
const SUITES: &[SuiteKind] = &[SuiteKind::Gpu, SuiteKind::Deep, SuiteKind::Honest];

pub struct LiteralSetVsCpu;

struct VsCpuPrepared {
    engine: GpuLiteralSet,
    aho: AhoCorasick,
    corpus: Vec<u8>,
    max_matches: u32,
    expected: u32,
    corpus_bytes: u64,
}

/// One head-to-head measurement: the steady-state wall time of each engine, the
/// GPU device-kernel time (if the backend has a timer), and the matches each
/// produced so the caller can prove they agree.
struct VsCpuMeasurement {
    vyre_wall_ns: u64,
    vyre_device_ns: Option<u64>,
    aho_wall_ns: u64,
    vyre_matches: Vec<Match>,
    aho_matches: Vec<Match>,
}

/// Build the CPU baseline automaton with the SAME all-overlapping semantics vyre's
/// literal set uses: `MatchKind::Standard` reports every pattern occurrence at
/// every position (via AC output chaining), which is exactly what vyre's DFA emits
/// (`reference_scan`). Patterns are inserted in `PATTERNS` order so the CPU pattern
/// index equals vyre's `pattern_id`.
fn build_aho_corasick() -> Result<AhoCorasick, BenchError> {
    AhoCorasick::builder()
        .match_kind(MatchKind::Standard)
        .build(PATTERNS.iter().copied())
        .map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "vs-cpu baseline failed to build aho-corasick automaton: {error}"
            ))
        })
}

/// Collect ALL overlapping matches from the CPU automaton, mapped into vyre's
/// `Match` triples and sorted into the canonical `(pattern_id, start, end)` order
/// vyre's decode also produces, so the two match vectors compare by plain
/// equality.
fn aho_corasick_matches(aho: &AhoCorasick, haystack: &[u8]) -> Vec<Match> {
    let mut matches: Vec<Match> = aho
        .find_overlapping_iter(haystack)
        .map(|hit| {
            Match::new(
                hit.pattern().as_u32(),
                hit.start() as u32,
                hit.end() as u32,
            )
        })
        .collect();
    matches.sort_unstable();
    matches
}

/// Run both engines over the corpus for `iters` steady-state iterations. The vyre
/// side uses a resident session (GPU-only. `CpuRefBackend` has no resident
/// allocation), so this helper is exercised by the bench on the Gpu suite; the CPU
/// baseline's correctness is unit-tested against `reference_scan` without a GPU.
fn run_vs_cpu(
    backend: &dyn VyreBackend,
    engine: &GpuLiteralSet,
    aho: &AhoCorasick,
    corpus: &[u8],
    max_matches: u32,
    iters: usize,
) -> Result<VsCpuMeasurement, vyre::BackendError> {
    // vyre: resident tables (uploaded once), then a timed re-dispatch loop.
    let session = engine.prepare_resident_scan(backend, corpus.len() + 64, max_matches)?;
    let mut vyre_matches = Vec::new();
    let mut scratch = Vec::new();
    session.scan_into(backend, corpus, &mut vyre_matches, &mut scratch)?; // warm
    let mut vyre_device_ns = None;
    let vyre_start = Instant::now();
    for _ in 0..iters {
        let timed = session.scan_into_timed(backend, corpus, &mut vyre_matches, &mut scratch)?;
        vyre_device_ns = timed.device_ns;
    }
    let vyre_wall_ns = clamp_ns(vyre_start.elapsed());
    session.free(backend)?;

    // aho-corasick: warm once, then a timed collect loop (the automaton itself is
    // built once in prepare, analogous to vyre's one-time matcher compile).
    let mut aho_matches = aho_corasick_matches(aho, corpus);
    let aho_start = Instant::now();
    for _ in 0..iters {
        aho_matches = aho_corasick_matches(aho, corpus);
    }
    let aho_wall_ns = clamp_ns(aho_start.elapsed());

    Ok(VsCpuMeasurement {
        vyre_wall_ns,
        vyre_device_ns,
        aho_wall_ns,
        vyre_matches,
        aho_matches,
    })
}

fn clamp_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// `aho / vyre`, scaled by 1000 so it survives the integer metric channel (1000 =
/// parity, >1000 = the GPU is faster end-to-end, <1000 = the CPU wins). Guards a
/// zero vyre wall (returns 0, an obviously-degenerate value the report surfaces).
fn speedup_x1000(aho_wall_ns: u64, vyre_wall_ns: u64) -> u64 {
    if vyre_wall_ns == 0 {
        return 0;
    }
    (u128::from(aho_wall_ns) * 1000 / u128::from(vyre_wall_ns)).min(u128::from(u64::MAX)) as u64
}

fn encode_matches(matches: &[Match]) -> [Vec<u8>; 2] {
    let count = u32::try_from(matches.len()).unwrap_or(u32::MAX);
    [count.to_le_bytes().to_vec(), encode_match_triples(matches)]
}

impl BenchCase for LiteralSetVsCpu {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.vs_cpu_aho_corasick".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet vs CPU aho-corasick (end-to-end)".to_string(),
            description: "Head-to-head of vyre's resident GPU literal-set scan against the aho-corasick crate on the same patterns and corpus, end-to-end with staging included, reporting the speedup and hard-gating that both engines produce the identical match set".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "literal-set".to_string(),
                "head-to-head".to_string(),
                "aho-corasick".to_string(),
                "cpu-baseline".to_string(),
            ],
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-libs".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(128 * 1024 * 1024),
            min_input_bytes: Some(CORPUS_BYTES as u64),
            feature_set: vec![
                "matching-dfa".to_string(),
                "literal-set".to_string(),
                "resident".to_string(),
                "cpu-baseline".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<VsCpuPrepared>()
            .map(|prepared| {
                let output_bytes = 4 + u64::from(prepared.expected) * MATCH_TRIPLE_BYTES;
                (prepared.corpus_bytes, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (corpus, _planted) = build_irregular_haystack(CORPUS_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "vs-cpu fixture failed to compile literal set: {error}"
            ))
        })?;
        let aho = build_aho_corasick()?;

        // Ground truth AND baseline pre-check: the CPU automaton must reproduce the
        // engine's own reference scan exactly, or the head-to-head is comparing
        // against a mis-built baseline (fail loudly here, not as a scan mismatch).
        let reference = engine.reference_scan(&corpus);
        let aho_reference = aho_corasick_matches(&aho, &corpus);
        if aho_reference != reference {
            return Err(BenchError::EnvironmentInvalid(format!(
                "vs-cpu aho-corasick baseline disagrees with reference_scan ({} vs {} matches); the CPU baseline semantics are mis-configured. Fix: MatchKind/overlapping selection in build_aho_corasick.",
                aho_reference.len(),
                reference.len()
            )));
        }
        let expected = u32::try_from(reference.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "vs-cpu corpus produced more matches than u32 can hold. Fix: shrink CORPUS_BYTES."
                    .to_string(),
            )
        })?;
        if expected == 0 {
            return Err(BenchError::EnvironmentInvalid(
                "vs-cpu corpus produced zero matches; the head-to-head must compare real output. Fix: change CORPUS_BYTES or PATTERNS.".to_string(),
            ));
        }
        let max_matches = expected.max(1);
        let corpus_bytes = corpus.len() as u64;

        Ok(Box::new(VsCpuPrepared {
            engine,
            aho,
            corpus,
            max_matches,
            expected,
            corpus_bytes,
        }))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared.downcast_ref::<VsCpuPrepared>().ok_or_else(|| {
            BenchError::ExecutionFailed("prepared vs-cpu payload had the wrong type".to_string())
        })?;

        let measurement = run_vs_cpu(
            ctx.preferred_backend.as_ref(),
            &prepared.engine,
            &prepared.aho,
            &prepared.corpus,
            prepared.max_matches,
            ITERS,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

        // Correctness output: the vyre GPU matches. Baseline output: the CPU
        // aho-corasick matches. `verify_exact_outputs` HARD-GATES that they are
        // identical (a fast wrong answer fails the case).
        let outputs = encode_matches(&measurement.vyre_matches).to_vec();
        let baseline_outputs = encode_matches(&measurement.aho_matches).to_vec();
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;

        let vyre_per_iter_ns = measurement.vyre_wall_ns / ITERS as u64;
        let aho_per_iter_ns = measurement.aho_wall_ns / ITERS as u64;
        let speedup = speedup_x1000(measurement.aho_wall_ns, measurement.vyre_wall_ns);

        let mut custom = vec![
            // >1000 = GPU faster end-to-end; <1000 = CPU wins. REPORTED, not gated.
            metric("vs_cpu_speedup_x1000", speedup),
            metric("vs_cpu_vyre_per_iter_ns", vyre_per_iter_ns),
            metric("vs_cpu_aho_per_iter_ns", aho_per_iter_ns),
            metric("vs_cpu_iters", ITERS as u64),
            metric("vs_cpu_matches", u64::from(prepared.expected)),
            metric("vs_cpu_corpus_bytes", prepared.corpus_bytes),
        ];
        if let Some(device_ns) = measurement.vyre_device_ns {
            custom.push(metric("vs_cpu_vyre_device_ns", device_ns));
            // Staging/readback = the GPU per-iteration wall not spent in the kernel 
            // the exact overhead the "end-to-end, staging included" claim turns on.
            let staging_ns = vyre_per_iter_ns.saturating_sub(device_ns);
            custom.push(metric("vs_cpu_vyre_staging_ns", staging_ns));
        }

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(vyre_per_iter_ns),
                dispatch_ns: measurement.vyre_device_ns,
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(aho_per_iter_ns),
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom: vec![metric("vs_cpu_aho_matches", u64::from(prepared.expected))],
                ..Default::default()
            }),
            outputs,
            baseline_outputs: Some(baseline_outputs),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

fn metric(name: &str, value: u64) -> MetricPoint {
    MetricPoint {
        name: name.to_string(),
        value,
    }
}

inventory::submit! {
    &LiteralSetVsCpu as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speedup_scales_and_guards_zero() {
        assert_eq!(speedup_x1000(3000, 1000), 3000); // GPU 3x faster
        assert_eq!(speedup_x1000(1000, 1000), 1000); // parity
        assert_eq!(speedup_x1000(500, 1000), 500); // CPU faster
        assert_eq!(speedup_x1000(1000, 0), 0); // degenerate
    }

    #[test]
    fn encode_matches_carries_count_then_triples() {
        let matches = [Match::new(0, 1, 4), Match::new(1, 8, 12)];
        let [count, triples] = encode_matches(&matches);
        assert_eq!(count, 2u32.to_le_bytes().to_vec());
        assert_eq!(triples.len(), 2 * MATCH_TRIPLE_BYTES as usize);
    }

    /// The load-bearing correctness property, runnable without a GPU: the CPU
    /// aho-corasick baseline (built with the all-overlapping `MatchKind::Standard`)
    /// produces EXACTLY the engine's `reference_scan` match set. This is what makes
    /// the head-to-head a fair comparison, the CPU side is the same answer, so the
    /// bench's GPU-side `verify_exact_outputs` (vyre == aho) transitively pins vyre
    /// to the reference too. A mis-chosen `MatchKind` (leftmost, non-overlapping)
    /// would fail here loudly.
    #[test]
    fn aho_corasick_baseline_equals_reference_scan() {
        let (corpus, _) = build_irregular_haystack(64 * 1024);
        let engine = GpuLiteralSet::try_compile(PATTERNS).expect("compile literal set");
        let aho = build_aho_corasick().expect("build aho-corasick");

        let reference = engine.reference_scan(&corpus);
        let aho_matches = aho_corasick_matches(&aho, &corpus);

        // Non-vacuous: the corpus actually produces matches.
        assert!(
            !reference.is_empty(),
            "the corpus must produce matches for the baseline check to be meaningful"
        );
        assert_eq!(
            aho_matches, reference,
            "aho-corasick MatchKind::Standard overlapping must reproduce the vyre reference scan set exactly"
        );
    }
}
