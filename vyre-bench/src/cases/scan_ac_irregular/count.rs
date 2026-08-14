use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::resident::{input_bytes_total, u32_counter_reset_program, ResidentInputSet};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;
use vyre_libs::scan::classic_ac::{
    build_ac_bounded_count_suffix3_prefilter_program, classic_ac_candidate_end_byte_mask_words,
    classic_ac_candidate_suffix2_mask_words, classic_ac_candidate_suffix3_bloom_words,
    classic_ac_compile, classic_ac_suffix3_bloom_contains, ClassicAcAutomaton,
    CLASSIC_AC_SUFFIX2_MASK_WORDS, CLASSIC_AC_SUFFIX3_BLOOM_WORDS,
};
use vyre_libs::scan::pack_haystack_u32;
use vyre_primitives::wire::pack_u32_slice;

use super::baseline::cpu_aho_overlapping_matches;
use super::haystack::{build_irregular_haystack, pattern_lengths};
use super::metrics::{scan_ac_count_metric_points, ScanAcStats};
use super::sample::{dispatch_reset_then_scan, scan_bench_run, take_scan_sample, ResetThenScan};
use super::{HAYSTACK_BYTES, MAX_MATCHES, PATTERNS, SUITES};

const COUNT_CANDIDATE_END_MASK_INPUT_INDEX: usize = 3;
const COUNT_CANDIDATE_SUFFIX2_MASK_INPUT_INDEX: usize = 4;
const COUNT_CANDIDATE_SUFFIX3_BLOOM_INPUT_INDEX: usize = 5;
const COUNT_HAYSTACK_LEN_INPUT_INDEX: usize = 6;
const COUNT_MATCH_COUNT_INPUT_INDEX: usize = 7;
const COUNT_RESET_RESOURCE_INDICES: [usize; 1] = [COUNT_MATCH_COUNT_INPUT_INDEX];
const COUNT_SCAN_RESOURCE_INDICES: [usize; 8] = [
    0,
    1,
    2,
    COUNT_CANDIDATE_END_MASK_INPUT_INDEX,
    COUNT_CANDIDATE_SUFFIX2_MASK_INPUT_INDEX,
    COUNT_CANDIDATE_SUFFIX3_BLOOM_INPUT_INDEX,
    COUNT_HAYSTACK_LEN_INPUT_INDEX,
    COUNT_MATCH_COUNT_INPUT_INDEX,
];
/// Index of `match_count` inside `COUNT_SCAN_RESOURCE_INDICES`, for readback.
const COUNT_MATCH_COUNT_RESOURCE_SLOT: usize = 7;

pub(super) struct ScanAcIrregularCountPrepared {
    pub(super) program: Program,
    reset_program: Program,
    pub(super) inputs: Vec<Vec<u8>>,
    pub(super) input_bytes_total: u64,
    pub(super) baseline_output: Vec<u8>,
    baseline_wall_ns: u64,
    pub(super) stats: ScanAcStats,
    resident: Option<ResidentInputSet>,
}

/// Count-only irregular AC preflight for exact match cardinality.
pub(super) struct ScanAcIrregularCount;

impl BenchCase for ScanAcIrregularCount {
    fn id(&self) -> BenchId {
        BenchId("scan.ac.irregular_count.4m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Aho-Corasick Irregular Count 4M".to_string(),
            description: "GPU-only match cardinality preflight over unaligned, varied-length security/parser literals in a noisy 4 MiB haystack".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "dfa".to_string(),
                "aho-corasick".to_string(),
                "packed-byte".to_string(),
                "count-only".to_string(),
                "irregular".to_string(),
                "release".to_string(),
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
            min_vram_bytes: Some(32 * 1024 * 1024),
            min_input_bytes: Some(HAYSTACK_BYTES as u64),
            feature_set: vec![
                "matching-dfa".to_string(),
                "packed-byte".to_string(),
                "aho-corasick".to_string(),
                "count-only".to_string(),
            ],
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "Packed-byte Aho-Corasick irregular count preflight",
            "vyre-libs",
            "aho-corasick 1.1 overlapping CPU automaton",
            1.0,
        ))
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<ScanAcIrregularCountPrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared.baseline_output.len() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_scan_ac_irregular_count(Some(ctx))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<ScanAcIrregularCountPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<ScanAcIrregularCountPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared irregular AC count payload had the wrong type".to_string(),
                )
            })?;
        let ctx: &BenchContext = ctx;

        let resident_sequence = prepared.resident.as_ref().map(|resident| {
            move |workgroup: [u32; 3]| -> Result<(Vec<Vec<u8>>, u64), BenchError> {
                dispatch_reset_then_scan(
                    ctx,
                    resident,
                    workgroup,
                    ResetThenScan {
                        reset_program: &prepared.reset_program,
                        scan_program: &prepared.program,
                        reset_indices: &COUNT_RESET_RESOURCE_INDICES,
                        scan_indices: &COUNT_SCAN_RESOURCE_INDICES,
                        label: "irregular AC count",
                        kind: "count",
                        scan_resources_context: "irregular AC count scan",
                        haystack_bytes: prepared.stats.haystack_bytes,
                    },
                    &[(
                        COUNT_MATCH_COUNT_RESOURCE_SLOT,
                        prepared.baseline_output.len(),
                    )],
                )
            }
        });

        let sample = take_scan_sample(
            ctx,
            "irregular AC count",
            &prepared.program,
            &prepared.inputs,
            prepared.stats.haystack_bytes,
            resident_sequence,
        )?;

        let custom = scan_ac_count_metric_points(
            prepared.stats,
            prepared.baseline_wall_ns,
            sample.wall_ns,
            sample.resident_used,
            sample.device_reset_sequence,
            sample.workgroup_x,
        );
        Ok(scan_bench_run(
            sample,
            prepared.input_bytes_total,
            prepared.baseline_wall_ns,
            prepared.stats,
            custom,
            vec![prepared.baseline_output.clone()],
        ))
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

pub(super) fn prepare_scan_ac_irregular_count(
    ctx: Option<&BenchContext>,
) -> Result<ScanAcIrregularCountPrepared, BenchError> {
    let (haystack, planted_matches) = build_irregular_haystack(HAYSTACK_BYTES);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths()?;
    let reset_program = u32_counter_reset_program("match_count");

    let baseline_start = Instant::now();
    let expected_match_count = cpu_aho_overlapping_matches(PATTERNS, &haystack)?.len() as u32;
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let candidate_end_mask = classic_ac_candidate_end_byte_mask_words(&ac.dfa);
    let candidate_suffix2_mask = classic_ac_candidate_suffix2_mask_words(&ac.dfa);
    let candidate_suffix3_bloom = classic_ac_candidate_suffix3_bloom_words(PATTERNS);
    let program = build_ac_bounded_count_suffix3_prefilter_program(&ac.dfa);
    let inputs = scan_ac_count_inputs_with_masks(
        &ac,
        &haystack,
        &candidate_end_mask,
        &candidate_suffix2_mask,
        &candidate_suffix3_bloom,
    );
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "irregular AC count"))
        .transpose()?
        .flatten();
    let stats = ScanAcStats {
        haystack_bytes: HAYSTACK_BYTES as u32,
        packed_haystack_words: HAYSTACK_BYTES.div_ceil(4) as u32,
        patterns: PATTERNS.len() as u32,
        dfa_states: ac.dfa.state_count,
        max_pattern_len: ac.dfa.max_pattern_len,
        output_records: ac.dfa.output_records.len() as u32,
        expected_matches: expected_match_count,
        max_matches: MAX_MATCHES,
        planted_matches,
        candidate_end_bytes: candidate_end_byte_count(&candidate_end_mask),
        candidate_end_lanes: candidate_end_lane_count(&haystack, &candidate_end_mask),
        candidate_suffix2_lanes: candidate_suffix2_lane_count(
            &haystack,
            &candidate_end_mask,
            &candidate_suffix2_mask,
        ),
        candidate_suffix3_lanes: candidate_suffix3_lane_count(
            &haystack,
            &candidate_end_mask,
            &candidate_suffix2_mask,
            &candidate_suffix3_bloom,
        ),
    };
    if stats.max_pattern_len != pattern_lengths.iter().copied().max().unwrap_or_default() {
        return Err(BenchError::EnvironmentInvalid(
            "irregular AC count DFA max pattern length disagreed with fixture pattern lengths. Fix: rebuild the DFA and count program from the same pattern set."
                .to_string(),
        ));
    }

    Ok(ScanAcIrregularCountPrepared {
        program,
        reset_program,
        inputs,
        input_bytes_total,
        baseline_output: pack_u32_slice(&[expected_match_count]),
        baseline_wall_ns,
        stats,
        resident,
    })
}

pub(super) fn scan_ac_count_inputs(ac: &ClassicAcAutomaton, haystack: &[u8]) -> Vec<Vec<u8>> {
    let candidate_end_mask = classic_ac_candidate_end_byte_mask_words(&ac.dfa);
    let candidate_suffix2_mask = classic_ac_candidate_suffix2_mask_words(&ac.dfa);
    let candidate_suffix3_bloom = classic_ac_candidate_suffix3_bloom_words(PATTERNS);
    scan_ac_count_inputs_with_masks(
        ac,
        haystack,
        &candidate_end_mask,
        &candidate_suffix2_mask,
        &candidate_suffix3_bloom,
    )
}

pub(super) fn scan_ac_count_inputs_with_masks(
    ac: &ClassicAcAutomaton,
    haystack: &[u8],
    candidate_end_mask: &[u32; 8],
    candidate_suffix2_mask: &[u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
    candidate_suffix3_bloom: &[u32],
) -> Vec<Vec<u8>> {
    debug_assert_eq!(
        candidate_suffix3_bloom.len(),
        CLASSIC_AC_SUFFIX3_BLOOM_WORDS
    );
    vec![
        pack_haystack_u32(haystack),
        pack_u32_slice(&ac.dfa.transitions),
        pack_u32_slice(&ac.dfa.output_offsets),
        pack_u32_slice(candidate_end_mask),
        pack_u32_slice(candidate_suffix2_mask),
        pack_u32_slice(candidate_suffix3_bloom),
        pack_u32_slice(&[haystack.len() as u32]),
        pack_u32_slice(&[0]),
    ]
}

pub(super) fn candidate_end_byte_count(mask: &[u32; 8]) -> u32 {
    mask.iter().map(|word| word.count_ones()).sum()
}

pub(super) fn candidate_end_lane_count(haystack: &[u8], mask: &[u32; 8]) -> u32 {
    haystack
        .iter()
        .filter(|byte| byte_is_candidate_end(**byte, mask))
        .count()
        .min(u32::MAX as usize) as u32
}

pub(super) fn byte_is_candidate_end(byte: u8, mask: &[u32; 8]) -> bool {
    (mask[byte as usize / 32] & (1_u32 << (byte as usize % 32))) != 0
}

pub(super) fn candidate_suffix2_lane_count(
    haystack: &[u8],
    end_mask: &[u32; 8],
    suffix2_mask: &[u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
) -> u32 {
    if haystack.is_empty() {
        return 0;
    }

    let lanes = u32::from(byte_is_candidate_end(haystack[0], end_mask));
    let suffix2_lanes = haystack
        .windows(2)
        .filter(|pair| suffix2_pair_is_candidate(pair[0], pair[1], suffix2_mask))
        .count()
        .min(u32::MAX as usize) as u32;
    lanes.saturating_add(suffix2_lanes)
}

pub(super) fn suffix2_pair_is_candidate(
    previous: u8,
    current: u8,
    mask: &[u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
) -> bool {
    let suffix = ((previous as usize) << 8) | current as usize;
    (mask[suffix / 32] & (1_u32 << (suffix % 32))) != 0
}

pub(super) fn candidate_suffix3_lane_count(
    haystack: &[u8],
    end_mask: &[u32; 8],
    suffix2_mask: &[u32; CLASSIC_AC_SUFFIX2_MASK_WORDS],
    suffix3_bloom: &[u32],
) -> u32 {
    if haystack.is_empty() {
        return 0;
    }

    let first_lane = u32::from(byte_is_candidate_end(haystack[0], end_mask));
    let second_lane = haystack
        .get(1)
        .copied()
        .filter(|current| {
            byte_is_candidate_end(*current, end_mask)
                && suffix2_pair_is_candidate(haystack[0], *current, suffix2_mask)
        })
        .map_or(0, |_| 1_u32);
    let suffix3_lanes = haystack
        .windows(3)
        .filter(|triple| {
            byte_is_candidate_end(triple[2], end_mask)
                && suffix2_pair_is_candidate(triple[1], triple[2], suffix2_mask)
                && suffix3_triple_is_candidate(triple[0], triple[1], triple[2], suffix3_bloom)
        })
        .count()
        .min(u32::MAX as usize) as u32;
    first_lane
        .saturating_add(second_lane)
        .saturating_add(suffix3_lanes)
}

pub(super) fn suffix3_triple_is_candidate(
    previous2: u8,
    previous: u8,
    current: u8,
    mask: &[u32],
) -> bool {
    classic_ac_suffix3_bloom_contains(mask, previous2, previous, current)
}

inventory::submit! {
    &ScanAcIrregularCount as &'static dyn BenchCase
}
