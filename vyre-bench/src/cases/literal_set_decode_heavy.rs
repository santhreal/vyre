//! W7-1: the DECODE-HEAVY re-dispatch shape of the literal-set position scan.
//!
//! The async-overlap and cold-start cases measure a sparse-match corpus where the
//! win is transfer overlap or first-touch amortization. This case measures the
//! OPPOSITE regime a consumer hits on a dense corpus (a config dump, a minified
//! bundle, a secret-dense log): the kernel finds tens of thousands of matches, so
//! the per-dispatch cost is dominated by writing every `(pattern_id, start, end)`
//! triple to device memory, reading them back, and decoding them on the host 
//! not by table upload. To isolate that, the corpus is scanned through a RESIDENT
//! session (`prepare_resident_scan`), so the seven immutable tables upload ONCE
//! and every timed dispatch re-stages only the haystack + a 4-byte counter reset;
//! what remains in the steady-state cost is the match write-out + readback +
//! decode. The case reports the device-vs-host-decode split and verifies the
//! resident matches equal the independent CPU reference (Law 10).

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

/// A consumer-shaped corpus, tiled densely so the scan produces many thousands of
/// matches and the host readback/decode is the load-bearing cost.
const CORPUS_BYTES: usize = 4 * 1024 * 1024;
/// Spacing between planted matches, smaller = denser = more decode. 128 B over a
/// 4 MiB corpus is ~32k matches, a genuinely decode-bound dispatch.
const MATCH_PERIOD_BYTES: usize = 128;
/// Steady-state re-dispatch iterations timed (a loop, not a cold shot).
const ITERS: usize = 32;
const MATCH_TRIPLE_BYTES: u64 = 12;
const SUITES: &[SuiteKind] = &[SuiteKind::Gpu, SuiteKind::Deep, SuiteKind::Honest];

pub struct LiteralSetDecodeHeavy;

struct DecodeHeavyPrepared {
    engine: GpuLiteralSet,
    corpus: Vec<u8>,
    reference: Vec<Match>,
    max_matches: u32,
    expected: u32,
    planted: u32,
    corpus_bytes: u64,
}

/// One decode-heavy measurement: the steady-state re-dispatch wall time, the
/// single-dispatch device-kernel time (if the backend has a timer), and the
/// matches decoded so a caller can prove resident == reference.
struct DecodeHeavyMeasurement {
    wall_ns: u64,
    /// Device-kernel time of a single steady dispatch, when the backend exposes a
    /// device timer (`None` on a backend without one, a loud absence, never a
    /// fabricated zero).
    device_ns: Option<u64>,
    matches: Vec<Match>,
}

/// Build a dense-match corpus: an irregular background with the SHORTEST pattern
/// tiled every `period` bytes, so the scan finds a match roughly every `period`
/// bytes. Returns the corpus and the planted count (the true match total is the
/// reference scan, which also picks up incidental background hits).
fn build_dense_match_haystack(len: usize, period: usize) -> (Vec<u8>, u32) {
    let (mut haystack, _) = build_irregular_haystack(len);
    let shortest = PATTERNS
        .iter()
        .min_by_key(|pattern| pattern.len())
        .expect("PATTERNS is non-empty");
    let step = period.max(shortest.len());
    let mut planted = 0_u32;
    let mut offset = 0_usize;
    while offset + shortest.len() <= len {
        haystack[offset..offset + shortest.len()].copy_from_slice(shortest);
        planted += 1;
        offset += step;
    }
    (haystack, planted)
}

/// Prepare a resident session, warm it, then run the timed steady-state
/// re-dispatch loop. Factored out of `run` so it is unit-testable on
/// `CpuRefBackend` without a GPU.
fn run_decode_heavy(
    backend: &dyn VyreBackend,
    engine: &GpuLiteralSet,
    corpus: &[u8],
    max_matches: u32,
    iters: usize,
) -> Result<DecodeHeavyMeasurement, vyre::BackendError> {
    // Resident session: the seven immutable tables upload ONCE here, so the timed
    // loop below re-stages only the haystack + counter reset, the residual cost is
    // the dense match write-out + readback + decode.
    let session = engine.prepare_resident_scan(backend, corpus.len() + 64, max_matches)?;
    let mut matches = Vec::new();
    let mut scratch = Vec::new();

    // Warm the resident buffers/queues so the timed loop measures steady state.
    session.scan_into(backend, corpus, &mut matches, &mut scratch)?;

    let mut device_ns = None;
    let start = Instant::now();
    for _ in 0..iters {
        let timed = session.scan_into_timed(backend, corpus, &mut matches, &mut scratch)?;
        device_ns = timed.device_ns;
    }
    let wall_ns = clamp_ns(start.elapsed());

    session.free(backend)?;

    Ok(DecodeHeavyMeasurement {
        wall_ns,
        device_ns,
        matches,
    })
}

fn clamp_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn encode_matches(matches: &[Match]) -> [Vec<u8>; 2] {
    let count = u32::try_from(matches.len()).unwrap_or(u32::MAX);
    [count.to_le_bytes().to_vec(), encode_match_triples(matches)]
}

impl BenchCase for LiteralSetDecodeHeavy {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.decode_heavy".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet Decode-Heavy Re-Dispatch".to_string(),
            description: "Measures the decode-bound regime of the resident literal-set scan on a dense-match corpus (tables uploaded once; every dispatch dominated by match write-out, readback, and host decode), splitting device vs host-decode time and verifying the resident matches equal the CPU reference".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "literal-set".to_string(),
                "resident".to_string(),
                "decode".to_string(),
                "readback".to_string(),
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
                "decode-heavy".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<DecodeHeavyPrepared>()
            .map(|prepared| {
                let output_bytes = 4 + u64::from(prepared.expected) * MATCH_TRIPLE_BYTES;
                (prepared.corpus_bytes, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (corpus, planted) = build_dense_match_haystack(CORPUS_BYTES, MATCH_PERIOD_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "decode-heavy fixture failed to compile literal set: {error}"
            ))
        })?;
        let reference = engine.reference_scan(&corpus);
        let expected = u32::try_from(reference.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "decode-heavy corpus produced more matches than u32 can hold. Fix: shrink CORPUS_BYTES or raise MATCH_PERIOD_BYTES.".to_string(),
            )
        })?;
        // The whole point is a decode-heavy scan; assert the fixture is actually
        // dense (a sparse corpus would measure the wrong regime).
        let min_dense = (CORPUS_BYTES / (MATCH_PERIOD_BYTES * 2)) as u32;
        if expected < min_dense {
            return Err(BenchError::EnvironmentInvalid(format!(
                "decode-heavy corpus produced only {expected} matches, need at least {min_dense} for a decode-bound dispatch. Fix: lower MATCH_PERIOD_BYTES."
            )));
        }
        let max_matches = expected.max(1);
        let corpus_bytes = corpus.len() as u64;

        Ok(Box::new(DecodeHeavyPrepared {
            engine,
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
        let prepared = prepared
            .downcast_ref::<DecodeHeavyPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared decode-heavy payload had the wrong type".to_string(),
                )
            })?;

        let measurement = run_decode_heavy(
            ctx.preferred_backend.as_ref(),
            &prepared.engine,
            &prepared.corpus,
            prepared.max_matches,
            ITERS,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

        // Correctness output: the RESIDENT dense matches. Baseline: the independent
        // CPU `reference_scan` set. `verify_exact_outputs` proves the resident scan
        // reproduced the oracle over the whole dense corpus (Law 10).
        let outputs = encode_matches(&measurement.matches).to_vec();
        let baseline_outputs = encode_matches(&prepared.reference).to_vec();
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;

        let per_iter_ns = measurement.wall_ns / ITERS as u64;
        let readback_bytes = u64::from(prepared.expected) * MATCH_TRIPLE_BYTES;

        let mut custom = vec![
            metric("decode_heavy_wall_ns", measurement.wall_ns),
            metric("decode_heavy_per_iter_ns", per_iter_ns),
            metric("decode_heavy_iters", ITERS as u64),
            metric("decode_heavy_matches", u64::from(prepared.expected)),
            metric("decode_heavy_planted", u64::from(prepared.planted)),
            metric("decode_heavy_corpus_bytes", prepared.corpus_bytes),
            metric("decode_heavy_readback_bytes", readback_bytes),
        ];
        if let Some(device_ns) = measurement.device_ns {
            custom.push(metric("decode_heavy_device_ns", device_ns));
            // Host decode/staging = per-iteration wall not spent in the device
            // kernel (the cost this case exists to expose).
            let host_decode_ns = per_iter_ns.saturating_sub(device_ns);
            custom.push(metric("decode_heavy_host_decode_ns", host_decode_ns));
        }

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(per_iter_ns),
                dispatch_ns: measurement.device_ns,
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(per_iter_ns),
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom: vec![metric(
                    "decode_heavy_reference_matches",
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
    &LiteralSetDecodeHeavy as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_driver_reference::CpuRefBackend;

    #[test]
    fn dense_haystack_is_actually_dense() {
        let (corpus, planted) = build_dense_match_haystack(64 * 1024, MATCH_PERIOD_BYTES);
        assert_eq!(corpus.len(), 64 * 1024);
        // ~ len / period planted matches, minus the boundary tail.
        let expected_floor = (64 * 1024 / (MATCH_PERIOD_BYTES * 2)) as u32;
        assert!(
            planted >= expected_floor,
            "planted {planted} matches, expected at least {expected_floor} for a dense corpus"
        );
    }

    #[test]
    fn encode_matches_carries_count_then_triples() {
        let matches = [Match::new(0, 1, 4), Match::new(1, 8, 12)];
        let [count, triples] = encode_matches(&matches);
        assert_eq!(count, 2u32.to_le_bytes().to_vec());
        assert_eq!(triples.len(), 2 * MATCH_TRIPLE_BYTES as usize);
    }

    /// The correctness property, runnable without a GPU: the dense-match corpus is
    /// really decode-heavy AND the BORROWED `scan_into` of a small fully-covered
    /// dense corpus equals the CPU AC-walk oracle (`reference_scan`), so the dense
    /// fixture is matchable and the count-then-triples decode is correct. The
    /// RESIDENT path itself is GPU-only (`CpuRefBackend` does not implement
    /// `allocate_resident`), so the resident-equals-reference property over the
    /// full dense corpus is asserted on the real GPU by the bench's own
    /// `verify_exact_outputs`; here we prove the borrowed-path oracle relationship
    /// the resident scan is proven byte-identical to (`ResidentLiteralScan`
    /// GPU parity tests) without needing a device.
    #[test]
    fn dense_borrowed_scan_matches_reference_on_cpu_reference() {
        let backend = CpuRefBackend;
        let (corpus, _) = build_dense_match_haystack(16 * 1024, MATCH_PERIOD_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).expect("compile literal set");
        let reference = engine.reference_scan(&corpus);
        let max_matches = u32::try_from(reference.len().max(1)).unwrap();

        let mut borrowed = Vec::new();
        engine
            .scan_into(&backend, &corpus, max_matches, &mut borrowed)
            .expect("borrowed dense scan");

        // Non-vacuous: the dense corpus really is decode-heavy.
        assert!(
            borrowed.len() >= 50,
            "the dense corpus must produce many matches; got {}",
            borrowed.len()
        );
        // On a small fully-covered corpus the borrowed scan equals the AC-walk
        // oracle (the property the resident GPU scan is proven identical to).
        assert_eq!(
            borrowed, reference,
            "borrowed dense scan of a fully-covered corpus must equal the reference AC walk"
        );
    }
}
