//! W7-1 (paged corpus shape) + W2-4: throughput of the PAGED fused scan on a
//! corpus split into many small files that exceed one window budget.
//!
//! `scan_paged_fused` scans a corpus larger than one resident window as a sequence
//! of window dispatches, with host RSS bounded by one window rather than the whole
//! corpus. This case builds a consumer-shaped corpus of thousands of small files,
//! scans it paged with a budget that forces many windows, and reports throughput
//! and the sync-vs-async-pipeline overlap factor. Correctness is HARD-GATED two
//! ways: the async pipelined result must be byte-identical to the sync result (Law
//! 10, overlap changes nothing), verified by `verify_exact_outputs`; and the paged
//! matches must equal an independent CPU `reference_scan` of the concatenated corpus
//! (a truth check, not just self-consistency), asserted in `run`.

use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::suite::SuiteKind;
use crate::cases::scan_ac_irregular::support::build_irregular_haystack;
use crate::cases::scan_ac_irregular::PATTERNS;
use vyre_libs::scan::{scan_paged_fused, scan_paged_fused_async, GlobalMatch, GpuLiteralSet};

/// Total corpus bytes, a few MiB so the single-shot CPU reference is tractable
/// while the paging mechanism runs many windows.
const CORPUS_BYTES: usize = 8 * 1024 * 1024;
/// Each coalesced "file" (region) is this many bytes.
const FILE_BYTES: usize = 4 * 1024;
/// Per-window byte budget, much smaller than the corpus, so paging runs many
/// windows (host RSS is bounded by this, not by CORPUS_BYTES).
const WINDOW_BUDGET_BYTES: usize = 512 * 1024;
const ITERS: usize = 8;
const SUITES: &[SuiteKind] = &[SuiteKind::Gpu, SuiteKind::Deep, SuiteKind::Honest];

pub struct LiteralSetPagedCorpus;

struct PagedPrepared {
    engine: GpuLiteralSet,
    haystack: Vec<u8>,
    expected: Vec<GlobalMatch>,
    max_matches: u32,
    file_count: usize,
    corpus_bytes: u64,
}

/// Build the `&[&[u8]]` file view over the stored corpus (fixed-size chunks; the
/// last chunk may be short).
fn file_view(haystack: &[u8]) -> Vec<&[u8]> {
    haystack.chunks(FILE_BYTES).collect()
}

/// The region id a global position falls in, given fixed-size files.
fn region_of(position: u32) -> u32 {
    position / FILE_BYTES as u32
}

fn encode_paged(presence: &[u32], matches: &[GlobalMatch]) -> Vec<Vec<u8>> {
    let mut presence_bytes = Vec::with_capacity(presence.len() * 4);
    for word in presence {
        presence_bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut match_bytes = Vec::with_capacity(matches.len() * 24);
    for hit in matches {
        match_bytes.extend_from_slice(&hit.pattern_id.to_le_bytes());
        match_bytes.extend_from_slice(&hit.region_id.to_le_bytes());
        match_bytes.extend_from_slice(&hit.start.to_le_bytes());
        match_bytes.extend_from_slice(&hit.end.to_le_bytes());
    }
    vec![presence_bytes, match_bytes]
}

fn overlap_factor_x1000(sync_ns: u64, async_ns: u64) -> u64 {
    if async_ns == 0 {
        return 0;
    }
    (u128::from(sync_ns) * 1000 / u128::from(async_ns)).min(u128::from(u64::MAX)) as u64
}

impl BenchCase for LiteralSetPagedCorpus {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.paged_corpus".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet Paged Corpus (many-window)".to_string(),
            description: "Scans a corpus of thousands of small files that exceeds the window budget via scan_paged_fused (sync and async), reporting throughput and the pipeline overlap factor, and hard-gating async==sync and paged==CPU-reference".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "literal-set".to_string(),
                "paged".to_string(),
                "corpus".to_string(),
                "streaming".to_string(),
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
                "paged-corpus".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<PagedPrepared>()
            .map(|prepared| {
                let output_bytes = prepared.expected.len() as u64 * 24;
                (prepared.corpus_bytes, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, _ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (haystack, _planted) = build_irregular_haystack(CORPUS_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "paged-corpus fixture failed to compile literal set: {error}"
            ))
        })?;

        // Independent CPU truth: reference_scan over the concatenated corpus,
        // globalized to fixed-size-file regions.
        let reference = engine.reference_scan(&haystack);
        let mut expected: Vec<GlobalMatch> = reference
            .iter()
            .map(|hit| GlobalMatch {
                pattern_id: hit.pattern_id,
                region_id: region_of(hit.start),
                start: u64::from(hit.start),
                end: u64::from(hit.end),
            })
            .collect();
        expected.sort_unstable_by_key(|hit| (hit.region_id, hit.start, hit.end, hit.pattern_id));

        let file_count = haystack.len().div_ceil(FILE_BYTES);
        let max_matches = u32::try_from(reference.len().max(1)).map_err(|_| {
            BenchError::EnvironmentInvalid(
                "paged-corpus reference produced more matches than u32 can hold. Fix: shrink CORPUS_BYTES.".to_string(),
            )
        })?;
        if expected.is_empty() {
            return Err(BenchError::EnvironmentInvalid(
                "paged-corpus produced zero matches; the paged scan must exercise real output. Fix: change CORPUS_BYTES or PATTERNS.".to_string(),
            ));
        }
        // Non-trivial paging: the corpus must span multiple windows.
        if CORPUS_BYTES <= WINDOW_BUDGET_BYTES {
            return Err(BenchError::EnvironmentInvalid(
                "paged-corpus fits one window; the case must exercise many windows. Fix: raise CORPUS_BYTES or lower WINDOW_BUDGET_BYTES.".to_string(),
            ));
        }
        let corpus_bytes = haystack.len() as u64;

        Ok(Box::new(PagedPrepared {
            engine,
            haystack,
            expected,
            max_matches,
            file_count,
            corpus_bytes,
        }))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared.downcast_ref::<PagedPrepared>().ok_or_else(|| {
            BenchError::ExecutionFailed(
                "prepared paged-corpus payload had the wrong type".to_string(),
            )
        })?;
        let backend = ctx.preferred_backend.as_ref();
        let files = file_view(&prepared.haystack);

        // Warm, then time the SYNC paged scan.
        let mut sync_result = scan_paged_fused(
            &prepared.engine,
            backend,
            &files,
            WINDOW_BUDGET_BYTES,
            prepared.max_matches,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        let sync_start = Instant::now();
        for _ in 0..ITERS {
            sync_result = scan_paged_fused(
                &prepared.engine,
                backend,
                &files,
                WINDOW_BUDGET_BYTES,
                prepared.max_matches,
            )
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        }
        let sync_wall_ns = clamp_ns(sync_start.elapsed());

        // Time the ASYNC pipelined paged scan.
        let mut async_result = scan_paged_fused_async(
            &prepared.engine,
            backend,
            &files,
            WINDOW_BUDGET_BYTES,
            prepared.max_matches,
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        let async_start = Instant::now();
        for _ in 0..ITERS {
            async_result = scan_paged_fused_async(
                &prepared.engine,
                backend,
                &files,
                WINDOW_BUDGET_BYTES,
                prepared.max_matches,
            )
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
        }
        let async_wall_ns = clamp_ns(async_start.elapsed());

        // Truth gate: the paged matches must equal the independent CPU reference.
        if sync_result.matches != prepared.expected {
            return Err(BenchError::CorrectnessViolation(format!(
                "paged scan produced {} matches but the CPU reference has {}",
                sync_result.matches.len(),
                prepared.expected.len()
            )));
        }

        // Law-10 gate at the framework level: async == sync, byte-for-byte.
        let outputs = encode_paged(&async_result.presence, &async_result.matches);
        let baseline_outputs = encode_paged(&sync_result.presence, &sync_result.matches);
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;

        let overlap = overlap_factor_x1000(sync_wall_ns, async_wall_ns);
        let windows_approx = CORPUS_BYTES.div_ceil(WINDOW_BUDGET_BYTES) as u64;

        let custom = vec![
            metric("paged_sync_wall_ns", sync_wall_ns),
            metric("paged_async_wall_ns", async_wall_ns),
            metric("paged_async_overlap_x1000", overlap),
            metric("paged_iters", ITERS as u64),
            metric("paged_corpus_bytes", prepared.corpus_bytes),
            metric("paged_window_budget_bytes", WINDOW_BUDGET_BYTES as u64),
            metric("paged_windows_approx", windows_approx),
            metric("paged_files", prepared.file_count as u64),
            metric("paged_matches", sync_result.matches.len() as u64),
            metric("paged_region_count", u64::from(sync_result.region_count)),
        ];

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(async_wall_ns / ITERS as u64),
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(sync_wall_ns / ITERS as u64),
                input_bytes: Some(prepared.corpus_bytes),
                output_bytes: Some(output_bytes),
                custom: vec![metric(
                    "paged_reference_matches",
                    prepared.expected.len() as u64,
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

fn clamp_ns(duration: std::time::Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

fn metric(name: &str, value: u64) -> MetricPoint {
    MetricPoint {
        name: name.to_string(),
        value,
    }
}

inventory::submit! {
    &LiteralSetPagedCorpus as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_factor_scales_and_guards_zero() {
        assert_eq!(overlap_factor_x1000(2000, 1000), 2000);
        assert_eq!(overlap_factor_x1000(1000, 0), 0);
    }

    #[test]
    fn region_of_maps_position_to_fixed_size_file() {
        assert_eq!(region_of(0), 0);
        assert_eq!(region_of(FILE_BYTES as u32 - 1), 0);
        assert_eq!(region_of(FILE_BYTES as u32), 1);
        assert_eq!(region_of(FILE_BYTES as u32 * 3 + 7), 3);
    }

    #[test]
    fn file_view_splits_the_corpus_into_many_files() {
        let (haystack, _) = build_irregular_haystack(CORPUS_BYTES);
        let files = file_view(&haystack);
        assert_eq!(files.len(), CORPUS_BYTES.div_ceil(FILE_BYTES));
        assert!(
            files.len() > CORPUS_BYTES / WINDOW_BUDGET_BYTES,
            "the corpus must split into more files than windows"
        );
        // Every file is FILE_BYTES except possibly the last.
        assert!(files[..files.len() - 1]
            .iter()
            .all(|f| f.len() == FILE_BYTES));
    }

    #[test]
    fn encode_paged_carries_presence_then_matches() {
        let presence = [1u32, 0u32];
        let matches = [GlobalMatch {
            pattern_id: 2,
            region_id: 3,
            start: 4,
            end: 9,
        }];
        let encoded = encode_paged(&presence, &matches);
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0].len(), 8); // 2 u32 words
        assert_eq!(encoded[1].len(), 24); // one GlobalMatch
    }
}
