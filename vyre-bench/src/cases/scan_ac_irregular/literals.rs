//! The 4 MiB irregular literal scan: every match emitted as a byte range.
//!
//! Sibling of [`super::count`], which runs the same haystack through a
//! cardinality-only kernel. Both take their sample through [`super::sample`].

use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, PerformanceContract, PreparedCase,
};
use crate::api::metric::elapsed_ns;
use crate::api::resident::{input_bytes_total, u32_counter_reset_program, ResidentInputSet};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;
use vyre_libs::pattern::classic_ac::{
    classic_ac_compile,
    try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce,
    ClassicAcAutomaton, CLASSIC_AC_SUFFIX2_MASK_WORDS,
};
use vyre_libs::pattern::pack_haystack_u32;
use vyre_primitives::wire::pack_u32_slice;

use super::baseline::cpu_aho_overlapping_matches;
use super::haystack::{build_irregular_haystack, pattern_lengths};
use super::match_triples::{
    decode_scan_outputs, encode_match_triples, match_triples_output_bytes,
    match_triples_readback_bytes, selected_scan_output_bytes, with_matches_readback_range,
};
use super::metrics::{scan_ac_bounded_ranges_metric_points, ScanAcStats};
use super::sample::{
    dispatch_reset_then_scan, scan_bench_run, take_scan_sample, ResetThenScan, ScanSample,
};
use super::{
    CANDIDATE_END_MASK_INPUT_INDEX, CANDIDATE_SUFFIX2_MASK_INPUT_INDEX,
    CANDIDATE_SUFFIX3_BLOOM_INPUT_INDEX, HAYSTACK_BYTES, MAX_MATCHES, PATTERNS, SUITES,
};

pub(super) const MATCH_COUNT_INPUT_INDEX: usize = 6;
pub(super) const MATCHES_RESOURCE_INDEX: usize = 10;
const RESET_RESOURCE_INDICES: [usize; 1] = [MATCH_COUNT_INPUT_INDEX];
pub(super) const SCAN_RESOURCE_INDICES: [usize; 11] = [
    0,
    1,
    2,
    3,
    4,
    5,
    MATCH_COUNT_INPUT_INDEX,
    CANDIDATE_END_MASK_INPUT_INDEX,
    CANDIDATE_SUFFIX2_MASK_INPUT_INDEX,
    CANDIDATE_SUFFIX3_BLOOM_INPUT_INDEX,
    MATCHES_RESOURCE_INDEX,
];

/// Index of `match_count` inside `SCAN_RESOURCE_INDICES`, for readback.
const SCAN_MATCH_COUNT_SLOT: usize = 6;
/// Index of `matches` inside `SCAN_RESOURCE_INDICES`, for readback.
const SCAN_MATCHES_SLOT: usize = 10;

pub struct ScanAcIrregularLiterals;

pub(super) struct ScanAcIrregularPrepared {
    pub(super) program: Program,
    pub(super) reset_program: Program,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    baseline_outputs: Vec<Vec<u8>>,
    baseline_wall_ns: u64,
    pub(super) stats: ScanAcStats,
    resident: Option<ResidentInputSet>,
}

impl BenchCase for ScanAcIrregularLiterals {
    fn id(&self) -> BenchId {
        BenchId("scan.ac.irregular_literals.4m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        super::scan_ac_metadata(
            self.id(),
            "Aho-Corasick Irregular Literal Scan 4M",
            "Packed-byte AC bounded-ranges scan over unaligned, varied-length security/parser literals in a noisy 4 MiB haystack",
            false,
        )
    }

    fn suites(&self) -> &'static [SuiteKind] {
        SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        super::scan_ac_requirements()
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "Packed-byte Aho-Corasick irregular literal scan",
            "vyre-libs",
            "aho-corasick 1.1 overlapping CPU automaton",
            1.0,
        ))
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<ScanAcIrregularPrepared>()
            .map(|prepared| {
                let output_bytes = selected_scan_output_bytes(prepared.stats);
                (prepared.input_bytes_total, output_bytes)
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_scan_ac_irregular(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<ScanAcIrregularPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared_as::<ScanAcIrregularPrepared>(prepared, "irregular AC scan")?;
        let ctx: &BenchContext = ctx;

        let resident_sequence = prepared.resident.as_ref().map(|resident| {
            move |workgroup: [u32; 3]| -> Result<(Vec<Vec<u8>>, u64), BenchError> {
                let match_output_bytes =
                    match_triples_readback_bytes(prepared.stats.expected_matches)?;
                dispatch_reset_then_scan(
                    ctx,
                    resident,
                    workgroup,
                    ResetThenScan {
                        reset_program: &prepared.reset_program,
                        scan_program: &prepared.program,
                        reset_indices: &RESET_RESOURCE_INDICES,
                        scan_indices: &SCAN_RESOURCE_INDICES,
                        label: "irregular AC scan",
                        kind: "scan",
                        scan_resources_context: "irregular AC scan sequence",
                        haystack_bytes: prepared.stats.haystack_bytes,
                    },
                    &[
                        (SCAN_MATCH_COUNT_SLOT, prepared.baseline_outputs[0].len()),
                        (SCAN_MATCHES_SLOT, match_output_bytes),
                    ],
                )
            }
        });

        let sample: ScanSample = take_scan_sample(
            ctx,
            "irregular AC scan",
            &prepared.program,
            &prepared.inputs,
            prepared.stats.haystack_bytes,
            resident_sequence,
        )?;

        let resident_reset_bytes = 0;
        let custom = scan_ac_bounded_ranges_metric_points(
            prepared.stats,
            prepared.baseline_wall_ns,
            sample.wall_ns,
            sample.resident_used,
            resident_reset_bytes,
            sample.device_reset_sequence,
            sample.workgroup_x,
        );
        Ok(scan_bench_run(
            sample,
            prepared.input_bytes_total,
            prepared.baseline_wall_ns,
            prepared.stats,
            custom,
            prepared.baseline_outputs.clone(),
        ))
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        let baseline = run.baseline_outputs.as_ref().ok_or_else(|| {
            BenchError::CorrectnessViolation(
                "irregular AC scan did not capture baseline outputs".to_string(),
            )
        })?;
        let expected = decode_scan_outputs(baseline, "baseline irregular AC scan")?;
        let actual = decode_scan_outputs(&run.outputs, "GPU irregular AC scan")?;
        if actual != expected {
            return Err(BenchError::CorrectnessViolation(format!(
                "irregular AC scan decoded match mismatch: expected {} matches, got {}",
                expected.len(),
                actual.len()
            )));
        }
        Ok(Correctness::Certificate {
            digest: *blake3::hash(&encode_match_triples(&actual)).as_bytes(),
        })
    }
}

pub(super) fn prepare_scan_ac_irregular(
    ctx: Option<&BenchContext>,
) -> Result<ScanAcIrregularPrepared, BenchError> {
    let (haystack, planted_matches) = build_irregular_haystack(HAYSTACK_BYTES);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths()?;
    let (candidate_end_mask, candidate_suffix2_mask, candidate_suffix3_bloom) =
        super::scan_ac_candidate_masks(&ac);
    let reset_program = u32_counter_reset_program("match_count");

    let baseline_start = std::time::Instant::now();
    let expected_matches = cpu_aho_overlapping_matches(PATTERNS, &haystack)?;
    let baseline_wall_ns = elapsed_ns(baseline_start);
    if expected_matches.len() > MAX_MATCHES as usize {
        return Err(BenchError::EnvironmentInvalid(format!(
            "irregular AC scan fixture produced {} matches, above MAX_MATCHES={MAX_MATCHES}. Fix: lower fixture density or raise output capacity.",
            expected_matches.len()
        )));
    }
    let expected_match_count = expected_matches.len() as u32;
    let program = try_build_ac_bounded_ranges_suffix3_prefilter_program_with_subgroup_coalesce(
        &ac.dfa,
        pattern_lengths.len() as u32,
        MAX_MATCHES,
        false,
    )
    .map_err(BenchError::ExecutionFailed)
    .and_then(|program| with_matches_readback_range(program, expected_match_count))?;

    let inputs = scan_ac_inputs(
        &ac,
        &pattern_lengths,
        &haystack,
        &candidate_end_mask,
        &candidate_suffix2_mask,
        &candidate_suffix3_bloom,
    );
    let input_bytes_total = input_bytes_total(&inputs);
    let resident_output_sizes = [match_triples_output_bytes(MAX_MATCHES)?];
    let resident = ctx
        .map(|ctx| {
            ResidentInputSet::upload_with_zeroed_outputs_optional(
                ctx,
                &inputs,
                &resident_output_sizes,
                "irregular AC scan",
            )
        })
        .transpose()?
        .flatten();
    let baseline_outputs = vec![
        pack_u32_slice(&[expected_match_count]),
        encode_match_triples(&expected_matches),
    ];
    let stats = ScanAcStats::from_fixture_and_masks(
        &haystack,
        planted_matches,
        expected_match_count,
        &ac,
        &candidate_end_mask,
        &candidate_suffix2_mask,
        &candidate_suffix3_bloom,
    );

    Ok(ScanAcIrregularPrepared {
        program,
        reset_program,
        inputs,
        input_bytes_total,
        baseline_outputs,
        baseline_wall_ns,
        stats,
        resident,
    })
}

pub(super) fn scan_ac_inputs(
    ac: &ClassicAcAutomaton,
    pattern_lengths: &[u32],
    haystack: &[u8],
    candidate_end_mask: &[u32; 8],
    candidate_suffix2_mask: &[u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
    candidate_suffix3_bloom: &[u32],
) -> Vec<Vec<u8>> {
    vec![
        pack_haystack_u32(haystack),
        pack_u32_slice(&ac.dfa.transitions),
        pack_u32_slice(&ac.dfa.output_offsets),
        pack_u32_slice(&ac.dfa.output_records),
        pack_u32_slice(pattern_lengths),
        pack_u32_slice(&[haystack.len() as u32]),
        pack_u32_slice(&[0]),
        pack_u32_slice(candidate_end_mask),
        pack_u32_slice(candidate_suffix2_mask),
        pack_u32_slice(candidate_suffix3_bloom),
    ]
}

inventory::submit! {
    &ScanAcIrregularLiterals as &'static dyn BenchCase
}
