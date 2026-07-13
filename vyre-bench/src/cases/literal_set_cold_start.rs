//! W7-1: the COLD-START shape of the literal-set position scan.
//!
//! Every other scan bench measures steady state, a warmed engine re-dispatching
//! into resident device buffers. But a consumer that scans one corpus and exits
//! (keyhog's `scan` of a directory tree, a CI secret sweep) pays a DIFFERENT
//! cost: build the matcher from patterns, allocate + upload its immutable tables
//! for the first time, and run the first dispatch with cold compile/queue caches.
//! That first-touch cost is invisible to a steady-state loop yet dominates a
//! one-shot scan. This case times the full cold path: `try_compile` +the first
//! `scan_into`: against the warm steady-state per-iteration cost, and reports the
//! cold-start overhead factor plus the compile-vs-first-dispatch split. It VERIFIES
//! the cold matches equal the independent CPU reference (Law 10, a cold cache
//! changes no result bit) and that the warm matches equal the cold ones.

use std::time::Instant;

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

/// A consumer-shaped single corpus. Big enough that the first upload + dispatch is
/// a real cost, small enough that the cold path is dominated by build/first-touch,
/// not raw scan volume.
const CORPUS_BYTES: usize = 4 * 1024 * 1024;
/// Steady-state iterations timed for the warm baseline (a loop, not a cold shot).
const WARM_ITERS: usize = 32;
const MATCH_TRIPLE_BYTES: u64 = 12;
const SUITES: &[SuiteKind] = &[SuiteKind::Gpu, SuiteKind::Deep, SuiteKind::Honest];

pub struct LiteralSetColdStart;

struct ColdStartPrepared {
    corpus: Vec<u8>,
    reference: Vec<Match>,
    max_matches: u32,
    expected: u32,
    planted: u32,
    corpus_bytes: u64,
}

/// One cold-vs-warm measurement: the cold path's compile + first-dispatch split,
/// the warm steady-state per-iteration cost, and the matches decoded from each so
/// a caller can prove cold == warm == reference.
struct ColdStartMeasurement {
    compile_ns: u64,
    first_scan_ns: u64,
    cold_wall_ns: u64,
    warm_total_ns: u64,
    /// Device-kernel time of a single warm dispatch, when the backend exposes a
    /// device timer (`None` on a backend without one, a loud absence, never a
    /// fabricated zero).
    warm_device_ns: Option<u64>,
    cold_matches: Vec<Match>,
    warm_matches: Vec<Match>,
}

/// Run the cold path (build + first dispatch) then the warm steady-state loop.
/// Factored out of `run` so the correctness property (cold == warm) is
/// unit-testable on `CpuRefBackend` without a GPU. `backend` is `?Sized` so both a
/// `&dyn` and a concrete backend work.
fn run_cold_then_warm<B: VyreBackend + ?Sized>(
    backend: &B,
    corpus: &[u8],
    max_matches: u32,
) -> Result<ColdStartMeasurement, vyre::BackendError> {
    let mut cold_matches = Vec::new();
    let mut warm_matches = Vec::new();

    // COLD path: everything a one-shot consumer pays. Time the matcher build and
    // the first dispatch separately so the split (compile vs first-touch upload +
    // dispatch) is attributable, and sum them for the total cold wall.
    let compile_start = Instant::now();
    let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
        vyre::BackendError::new(format!("cold-start fixture failed to compile literal set: {error}"))
    })?;
    let compile_ns = clamp_ns(compile_start.elapsed());

    let first_scan_start = Instant::now();
    engine.scan_into(backend, corpus, max_matches, &mut cold_matches)?;
    let first_scan_ns = clamp_ns(first_scan_start.elapsed());
    let cold_wall_ns = compile_ns.saturating_add(first_scan_ns);

    // WARM path: the engine is now built and its caches/queues are hot. Time the
    // steady-state per-dispatch cost, what a resident/looping consumer sees after
    // the cold shot is paid once.
    let mut warm_device_ns = None;
    let warm_start = Instant::now();
    for _ in 0..WARM_ITERS {
        let timed = engine.scan_into_timed(backend, corpus, max_matches, &mut warm_matches)?;
        warm_device_ns = timed.device_ns;
    }
    let warm_total_ns = clamp_ns(warm_start.elapsed());

    Ok(ColdStartMeasurement {
        compile_ns,
        first_scan_ns,
        cold_wall_ns,
        warm_total_ns,
        warm_device_ns,
        cold_matches,
        warm_matches,
    })
}

fn clamp_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// `cold_wall / warm_per_iter`, scaled by 1000 so it survives the integer metric
/// channel (1000 = the cold shot costs one warm iteration; >1000 = the cold shot
/// is that many warm iterations of overhead). Guards a zero warm per-iter (returns
/// 0, an obviously-degenerate value the report surfaces).
fn cold_start_overhead_x1000(cold_wall_ns: u64, warm_total_ns: u64, warm_iters: u64) -> u64 {
    let warm_per_iter = warm_total_ns / warm_iters.max(1);
    if warm_per_iter == 0 {
        return 0;
    }
    (u128::from(cold_wall_ns) * 1000 / u128::from(warm_per_iter)).min(u128::from(u64::MAX)) as u64
}

fn encode_matches(matches: &[Match]) -> [Vec<u8>; 2] {
    let count = u32::try_from(matches.len()).unwrap_or(u32::MAX);
    [count.to_le_bytes().to_vec(), encode_match_triples(matches)]
}

impl BenchCase for LiteralSetColdStart {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.cold_start".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet Cold Start vs Warm".to_string(),
            description: "Times the full cold-start path (matcher build + first upload + first dispatch) of a one-shot literal-set scan against the warm steady-state per-dispatch cost, and verifies the cold matches equal the independent CPU reference".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "literal-set".to_string(),
                "cold-start".to_string(),
                "first-touch".to_string(),
                "compile".to_string(),
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
            min_vram_bytes: Some(64 * 1024 * 1024),
            min_input_bytes: Some(CORPUS_BYTES as u64),
            feature_set: vec![
                "matching-dfa".to_string(),
                "literal-set".to_string(),
                "cold-start".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<ColdStartPrepared>()
            .map(|prepared| {
                let output_bytes = 4 + u64::from(prepared.expected) * MATCH_TRIPLE_BYTES;
                (prepared.corpus_bytes, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (corpus, planted) = build_irregular_haystack(CORPUS_BYTES);
        // The reference matcher build here is NOT part of the timed cold path (that
        // build happens inside `run`); this one only computes ground truth.
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "cold-start fixture failed to compile literal set: {error}"
            ))
        })?;
        let reference = engine.reference_scan(&corpus);
        let expected = u32::try_from(reference.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "cold-start corpus produced more matches than u32 can hold. Fix: shrink CORPUS_BYTES."
                    .to_string(),
            )
        })?;
        // Non-vacuous: a cold scan that finds nothing measures the wrong thing.
        if expected == 0 {
            return Err(BenchError::EnvironmentInvalid(
                "cold-start corpus produced zero matches; the cold path must exercise real output. Fix: change CORPUS_BYTES or PATTERNS.".to_string(),
            ));
        }
        let max_matches = expected.max(1);
        let corpus_bytes = corpus.len() as u64;

        Ok(Box::new(ColdStartPrepared {
            corpus,
            reference,
            max_matches,
            expected,
            planted,
            corpus_bytes,
        }))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared.downcast_ref::<ColdStartPrepared>().ok_or_else(|| {
            BenchError::ExecutionFailed("prepared cold-start payload had the wrong type".to_string())
        })?;

        let measurement =
            run_cold_then_warm(ctx.preferred_backend.as_ref(), &prepared.corpus, prepared.max_matches)
                .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

        // The warm re-dispatch must reproduce the cold scan bit-for-bit (Law 10 
        // warming caches changes no result), fail the case loudly if not.
        if measurement.warm_matches != measurement.cold_matches {
            return Err(BenchError::CorrectnessViolation(
                "cold-start warm re-dispatch produced different matches than the cold dispatch"
                    .to_string(),
            ));
        }

        // Correctness output: the COLD matches. Baseline output: the independent
        // CPU `reference_scan` set. `verify_exact_outputs` then proves the cold GPU
        // scan equals the CPU oracle (not merely that it equals itself).
        let outputs = encode_matches(&measurement.cold_matches).to_vec();
        let baseline_outputs = encode_matches(&prepared.reference).to_vec();
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;

        let overhead = cold_start_overhead_x1000(
            measurement.cold_wall_ns,
            measurement.warm_total_ns,
            WARM_ITERS as u64,
        );
        let warm_per_iter_ns = measurement.warm_total_ns / WARM_ITERS as u64;

        let mut custom = vec![
            metric("cold_start_overhead_x1000", overhead),
            metric("cold_start_cold_wall_ns", measurement.cold_wall_ns),
            metric("cold_start_compile_ns", measurement.compile_ns),
            metric("cold_start_first_scan_ns", measurement.first_scan_ns),
            metric("cold_start_warm_total_ns", measurement.warm_total_ns),
            metric("cold_start_warm_per_iter_ns", warm_per_iter_ns),
            metric("cold_start_warm_iters", WARM_ITERS as u64),
            metric("cold_start_corpus_bytes", prepared.corpus_bytes),
            metric("cold_start_matches", u64::from(prepared.expected)),
            metric("cold_start_planted", u64::from(prepared.planted)),
        ];
        if let Some(device_ns) = measurement.warm_device_ns {
            custom.push(metric("cold_start_warm_device_ns", device_ns));
            // First-touch upload/host cost of the cold shot = cold wall not spent in
            // the matcher build nor a warm device dispatch.
            let first_touch_ns = measurement
                .first_scan_ns
                .saturating_sub(device_ns);
            custom.push(metric("cold_start_first_touch_host_ns", first_touch_ns));
        }

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(measurement.cold_wall_ns),
                dispatch_ns: measurement.warm_device_ns,
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(measurement.warm_total_ns / WARM_ITERS as u64),
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom: vec![metric(
                    "cold_start_reference_matches",
                    u64::from(prepared.expected),
                )],
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
    &LiteralSetColdStart as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_driver_reference::CpuRefBackend;

    #[test]
    fn overhead_scales_and_guards_zero() {
        // cold 10_000 ns, warm total 8_000 over 8 iters -> 1_000 per iter -> 10x.
        assert_eq!(cold_start_overhead_x1000(10_000, 8_000, 8), 10_000);
        // cold == one warm iter.
        assert_eq!(cold_start_overhead_x1000(1_000, 8_000, 8), 1_000);
        // zero warm per-iter -> degenerate 0.
        assert_eq!(cold_start_overhead_x1000(1_000, 0, 8), 0);
    }

    #[test]
    fn encode_matches_carries_count_then_triples() {
        let matches = [Match::new(0, 1, 4), Match::new(1, 8, 12)];
        let [count, triples] = encode_matches(&matches);
        assert_eq!(count, 2u32.to_le_bytes().to_vec());
        assert_eq!(triples.len(), 2 * MATCH_TRIPLE_BYTES as usize);
    }

    /// The correctness property this bench asserts, runnable without a GPU: on the
    /// serialized `CpuRefBackend`, the warm re-dispatch must reproduce the cold
    /// dispatch bit-for-bit (warming caches changes no result. Law 10). A small
    /// corpus keeps the CpuRef scan fully covered so the check is non-vacuous. The
    /// cold-vs-reference equality is asserted on the real GPU by the bench's own
    /// `verify_exact_outputs`; here we prove the load-bearing cold==warm property.
    #[test]
    fn cold_equals_warm_on_cpu_reference() {
        let backend = CpuRefBackend;
        let (corpus, _) = build_irregular_haystack(32 * 1024);
        let engine = GpuLiteralSet::try_compile(PATTERNS).expect("compile literal set");
        let reference = engine.reference_scan(&corpus);
        let max_matches = u32::try_from(reference.len().max(1)).unwrap();

        let measurement =
            run_cold_then_warm(&backend, &corpus, max_matches).expect("cold-then-warm on cpu ref");

        assert!(
            !measurement.cold_matches.is_empty(),
            "the cold corpus must produce matches for the check to be meaningful"
        );
        assert_eq!(
            measurement.cold_matches, measurement.warm_matches,
            "warm re-dispatch must reproduce the cold dispatch bit-for-bit"
        );
        // On a small, fully-covered corpus the cold GPU-path scan equals the CPU
        // AC-walk oracle too (the property the GPU suite asserts at full size).
        assert_eq!(
            measurement.cold_matches, reference,
            "cold scan of a fully-covered corpus must equal the reference AC walk"
        );
    }
}
