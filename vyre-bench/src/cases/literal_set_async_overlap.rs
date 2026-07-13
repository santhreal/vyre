//! W3-2 / W7-1: a QUANTITATIVE overlap-factor benchmark for the asynchronous
//! literal-set position scan.
//!
//! The async twin (`GpuLiteralSet::scan_into_async`) returns a `PendingMatches`
//! handle the instant the GPU dispatch is submitted, so a consumer can keep two
//! batches in flight at once and overlap the second batch's host staging /
//! upload with the first batch's device execution. The correctness of that
//! pipeline (no cross-handle corruption, order-independent decode) is locked by
//! `vyre-libs/tests/literal_set_async_two_batch_pipeline.rs`; what was MISSING
//! was a measurement of the win. This case runs a two-batch pipeline both
//! sequentially (submit → await → submit → await) and overlapped (submit A →
//! submit B → await A → await B) over a consumer-shaped corpus and reports the
//! overlap factor plus the timed kernel-vs-staging split, and it VERIFIES that
//! the overlapped matches are byte-identical to the sequential ones (Law 10 
//! overlap changes no result bit).

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

/// Batch A and batch B use DIFFERENT sizes so their content (and match sets)
/// differ, a cross-handle buffer mixup in the pipeline would then be
/// detectable, not masked by identical batches.
const BATCH_A_BYTES: usize = 2 * 1024 * 1024;
const BATCH_B_BYTES: usize = 3 * 1024 * 1024;
/// Steady-state iterations timed for each path (a loop, not a cold single shot).
const ITERS: usize = 16;
const MATCH_TRIPLE_BYTES: u64 = 12;
const SUITES: &[SuiteKind] = &[SuiteKind::Gpu, SuiteKind::Deep, SuiteKind::Honest];

pub struct LiteralSetAsyncOverlap;

struct AsyncOverlapPrepared {
    engine: GpuLiteralSet,
    batch_a: Vec<u8>,
    batch_b: Vec<u8>,
    max_matches: u32,
    expected_a: u32,
    expected_b: u32,
    planted_a: u32,
    planted_b: u32,
    encoded_input_bytes: u64,
}

/// The outcome of one two-batch overlap measurement: the two timing paths plus
/// the decoded matches from each, so a caller can both report the overlap factor
/// AND prove the overlapped result equals the sequential one.
struct OverlapMeasurement {
    sequential_wall_ns: u64,
    async_wall_ns: u64,
    /// Summed device-kernel time of the two sequential batches on the last
    /// iteration, when the backend exposes a device timer (`None` on a backend
    /// without one (a loud absence, never a fabricated zero)).
    sequential_device_ns: Option<u64>,
    matches_a_sequential: Vec<Match>,
    matches_b_sequential: Vec<Match>,
    matches_a_async: Vec<Match>,
    matches_b_async: Vec<Match>,
}

/// Run the two-batch pipeline both sequentially and overlapped. Factored out of
/// `run` so the correctness property (overlapped == sequential) is unit-testable
/// on `CpuRefBackend` without a GPU. `backend` is `?Sized` so both a `&dyn` and a
/// concrete backend work.
fn run_two_batch_overlap<B: VyreBackend + ?Sized>(
    backend: &B,
    engine: &GpuLiteralSet,
    batch_a: &[u8],
    batch_b: &[u8],
    max_matches: u32,
) -> Result<OverlapMeasurement, vyre::BackendError> {
    let mut a_seq = Vec::new();
    let mut b_seq = Vec::new();
    let mut a_async = Vec::new();
    let mut b_async = Vec::new();

    // Warm up caches/queues so the timed loops measure steady state, not first
    // touch (compile caches, device buffers).
    engine.scan_into(backend, batch_a, max_matches, &mut a_seq)?;
    engine.scan_into(backend, batch_b, max_matches, &mut b_seq)?;

    // Sequential path: each batch is submitted AND awaited before the next, so
    // the second batch's staging cannot overlap the first batch's execution.
    let mut sequential_device_ns = None;
    let seq_start = Instant::now();
    for _ in 0..ITERS {
        let timed_a = engine.scan_into_timed(backend, batch_a, max_matches, &mut a_seq)?;
        let timed_b = engine.scan_into_timed(backend, batch_b, max_matches, &mut b_seq)?;
        sequential_device_ns = match (timed_a.device_ns, timed_b.device_ns) {
            (Some(a), Some(b)) => Some(a.saturating_add(b)),
            _ => None,
        };
    }
    let sequential_wall_ns = clamp_ns(seq_start.elapsed());

    // Overlapped path: BOTH batches are submitted before either is awaited, so
    // on a pipelining backend batch B's upload overlaps batch A's device scan.
    let async_start = Instant::now();
    for _ in 0..ITERS {
        let pending_a = engine.scan_into_async(backend, batch_a, max_matches)?;
        let pending_b = engine.scan_into_async(backend, batch_b, max_matches)?;
        // Await in submit order; the retained owned inputs keep both uploads
        // valid until their decode.
        pending_a.await_into(&mut a_async)?;
        pending_b.await_into(&mut b_async)?;
    }
    let async_wall_ns = clamp_ns(async_start.elapsed());

    Ok(OverlapMeasurement {
        sequential_wall_ns,
        async_wall_ns,
        sequential_device_ns,
        matches_a_sequential: a_seq,
        matches_b_sequential: b_seq,
        matches_a_async: a_async,
        matches_b_async: b_async,
    })
}

fn clamp_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// `sequential / async`, scaled by 1000 so it survives the integer metric
/// channel (1000 = parity, >1000 = the overlap is faster). Guards a zero async
/// wall (returns 0, an obviously-degenerate value the report surfaces).
fn overlap_factor_x1000(sequential_wall_ns: u64, async_wall_ns: u64) -> u64 {
    if async_wall_ns == 0 {
        return 0;
    }
    (u128::from(sequential_wall_ns) * 1000 / u128::from(async_wall_ns)).min(u128::from(u64::MAX))
        as u64
}

fn encode_batch_outputs(matches: &[Match]) -> [Vec<u8>; 2] {
    let count = u32::try_from(matches.len()).unwrap_or(u32::MAX);
    [count.to_le_bytes().to_vec(), encode_match_triples(matches)]
}

impl BenchCase for LiteralSetAsyncOverlap {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.async_overlap.2batch".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet Async Two-Batch Overlap".to_string(),
            description: "Quantifies the overlap factor of the asynchronous literal-set position scan by pipelining two consumer-shaped batches vs running them sequentially, and verifies the overlapped matches equal the sequential ones".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "literal-set".to_string(),
                "async".to_string(),
                "pipeline".to_string(),
                "overlap".to_string(),
                "transfer".to_string(),
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
            min_input_bytes: Some((BATCH_A_BYTES + BATCH_B_BYTES) as u64),
            feature_set: vec![
                "matching-dfa".to_string(),
                "literal-set".to_string(),
                "async-pipeline".to_string(),
                "transfer-overlap".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<AsyncOverlapPrepared>()
            .map(|prepared| {
                let output_bytes = 8
                    + (u64::from(prepared.expected_a) + u64::from(prepared.expected_b))
                        * MATCH_TRIPLE_BYTES;
                (prepared.encoded_input_bytes, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (batch_a, planted_a) = build_irregular_haystack(BATCH_A_BYTES);
        let (batch_b, planted_b) = build_irregular_haystack(BATCH_B_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "async-overlap fixture failed to compile literal set: {error}"
            ))
        })?;

        let reference_a = engine.reference_scan(&batch_a);
        let reference_b = engine.reference_scan(&batch_b);
        let expected_a = u32::try_from(reference_a.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "async-overlap batch A produced more matches than u32 can hold. Fix: shrink BATCH_A_BYTES.".to_string(),
            )
        })?;
        let expected_b = u32::try_from(reference_b.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "async-overlap batch B produced more matches than u32 can hold. Fix: shrink BATCH_B_BYTES.".to_string(),
            )
        })?;
        // Distinct batches are load-bearing: identical match sets would mask a
        // cross-handle buffer mixup in the pipeline.
        if reference_a == reference_b {
            return Err(BenchError::EnvironmentInvalid(
                "async-overlap batches A and B produced identical matches; the pipeline mixup guard needs distinct batches. Fix: change BATCH_A_BYTES/BATCH_B_BYTES.".to_string(),
            ));
        }
        let max_matches = expected_a.max(expected_b).max(1);
        let encoded_input_bytes = (batch_a.len() + batch_b.len()) as u64;

        Ok(Box::new(AsyncOverlapPrepared {
            engine,
            batch_a,
            batch_b,
            max_matches,
            expected_a,
            expected_b,
            planted_a,
            planted_b,
            encoded_input_bytes,
        }))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<AsyncOverlapPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared async-overlap payload had the wrong type".to_string(),
                )
            })?;

        let measurement = run_two_batch_overlap(
            ctx.preferred_backend.as_ref(),
            &prepared.engine,
            &prepared.batch_a,
            &prepared.batch_b,
            prepared.max_matches,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

        // Correctness output: the OVERLAPPED matches. Baseline output: the
        // SEQUENTIAL matches. `verify_exact_outputs` then proves overlap changed
        // no result bit for either batch (Law 10).
        let mut outputs = Vec::with_capacity(4);
        outputs.extend(encode_batch_outputs(&measurement.matches_a_async));
        outputs.extend(encode_batch_outputs(&measurement.matches_b_async));
        let mut baseline_outputs = Vec::with_capacity(4);
        baseline_outputs.extend(encode_batch_outputs(&measurement.matches_a_sequential));
        baseline_outputs.extend(encode_batch_outputs(&measurement.matches_b_sequential));

        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let factor =
            overlap_factor_x1000(measurement.sequential_wall_ns, measurement.async_wall_ns);

        let mut custom = vec![
            metric("async_overlap_factor_x1000", factor),
            metric(
                "async_overlap_sequential_wall_ns",
                measurement.sequential_wall_ns,
            ),
            metric("async_overlap_async_wall_ns", measurement.async_wall_ns),
            metric("async_overlap_iters", ITERS as u64),
            metric("async_overlap_batch_a_bytes", prepared.batch_a.len() as u64),
            metric("async_overlap_batch_b_bytes", prepared.batch_b.len() as u64),
            metric(
                "async_overlap_batch_a_matches",
                u64::from(prepared.expected_a),
            ),
            metric(
                "async_overlap_batch_b_matches",
                u64::from(prepared.expected_b),
            ),
            metric(
                "async_overlap_batch_a_planted",
                u64::from(prepared.planted_a),
            ),
            metric(
                "async_overlap_batch_b_planted",
                u64::from(prepared.planted_b),
            ),
        ];
        if let Some(device_ns) = measurement.sequential_device_ns {
            custom.push(metric("async_overlap_sequential_device_ns", device_ns));
            // Host staging/readback = wall not spent in the two device kernels.
            let staging_ns = measurement
                .sequential_wall_ns
                .saturating_sub(device_ns.saturating_mul(ITERS as u64));
            custom.push(metric(
                "async_overlap_sequential_host_staging_ns",
                staging_ns,
            ));
        }

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(measurement.async_wall_ns),
                dispatch_ns: measurement.sequential_device_ns,
                input_bytes: Some(prepared.encoded_input_bytes),
                output_bytes: Some(output_bytes),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(measurement.sequential_wall_ns),
                input_bytes: Some(prepared.encoded_input_bytes),
                output_bytes: Some(output_bytes),
                custom: vec![metric(
                    "async_overlap_reference_total_matches",
                    u64::from(prepared.expected_a) + u64::from(prepared.expected_b),
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
    &LiteralSetAsyncOverlap as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_driver_reference::CpuRefBackend;

    #[test]
    fn overlap_factor_scales_and_guards_zero() {
        assert_eq!(overlap_factor_x1000(2000, 1000), 2000);
        assert_eq!(overlap_factor_x1000(1000, 1000), 1000);
        assert_eq!(overlap_factor_x1000(1000, 0), 0);
    }

    #[test]
    fn irregular_batches_of_different_sizes_are_distinct() {
        let (a, _) = build_irregular_haystack(BATCH_A_BYTES);
        let (b, _) = build_irregular_haystack(BATCH_B_BYTES);
        assert_ne!(a.len(), b.len());
        // The engine's reference scan over each must differ (distinct content),
        // which is what makes a cross-handle pipeline mixup detectable.
        let engine = GpuLiteralSet::try_compile(PATTERNS).expect("compile literal set");
        assert_ne!(engine.reference_scan(&a), engine.reference_scan(&b));
    }

    #[test]
    fn encode_batch_outputs_carries_count_then_triples() {
        let matches = [Match::new(0, 1, 4), Match::new(1, 8, 12)];
        let [count, triples] = encode_batch_outputs(&matches);
        assert_eq!(count, 2u32.to_le_bytes().to_vec());
        assert_eq!(triples.len(), 2 * MATCH_TRIPLE_BYTES as usize);
    }

    /// The pipeline's CORRECTNESS property, runnable without a GPU: on the
    /// serialized `CpuRefBackend`, the overlapped two-batch matches must equal
    /// the sequential ones for BOTH batches (the degraded path changes no bits 
    /// Law 10). This is the pipeline property this bench exists to prove; the
    /// separate question of `scan_into` vs the engine's own `reference_scan`
    /// AC-walk is out of scope here (that CPU-reference parity is covered by the
    /// literal-set scan tests). A generous match cap keeps the fixed-size decode
    /// from being the variable. Uses small batches so it runs quickly.
    #[test]
    fn overlapped_matches_equal_sequential_on_cpu_reference() {
        let backend = CpuRefBackend;
        // Small distinct batches (different sizes -> different content).
        let (batch_a, _) = build_irregular_haystack(64 * 1024);
        let (batch_b, _) = build_irregular_haystack(96 * 1024);
        let engine = GpuLiteralSet::try_compile(PATTERNS).expect("compile literal set");
        let max_matches = 100_000u32;

        let measurement = run_two_batch_overlap(&backend, &engine, &batch_a, &batch_b, max_matches)
            .expect("two-batch overlap on cpu reference");

        // Non-vacuous: both batches actually produce matches.
        assert!(
            !measurement.matches_a_async.is_empty() && !measurement.matches_b_async.is_empty(),
            "both batches must produce matches for the pipeline check to be meaningful"
        );
        // The load-bearing property: overlapped == sequential for BOTH batches.
        assert_eq!(
            measurement.matches_a_async, measurement.matches_a_sequential,
            "batch A: overlapped matches must equal sequential (overlap changes no bits)"
        );
        assert_eq!(
            measurement.matches_b_async, measurement.matches_b_sequential,
            "batch B: overlapped matches must equal sequential (overlap changes no bits)"
        );
        // (Batch distinctness, the cross-handle mixup guard, is asserted at full
        // coverage in `irregular_batches_of_different_sizes_are_distinct` and in
        // the bench's own `prepare`; it is not re-checked here because the
        // `CpuRefBackend` scan under-covers a haystack larger than its max buffer
        // element count, so A and B could share an identical covered prefix. That
        // under-coverage is a CpuRef-only reference-oracle limitation, orthogonal
        // to the overlap-equals-sequential property this test proves.)
    }
}
