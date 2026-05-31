use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting, ResidentInputSet,
};
use crate::api::suite::SuiteKind;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::match_result::Match;
use vyre_libs::scan::classic_ac::{
    classic_ac_compile, try_build_ac_bounded_ranges_program_ext, ClassicAcAutomaton,
};
use vyre_libs::scan::dispatch_io::try_unpack_match_triples;
use vyre_libs::scan::{pack_haystack_u32, pack_u32_slice};

mod baseline;
mod metrics;

use baseline::cpu_aho_overlapping_matches;
#[cfg(test)]
use baseline::cpu_bounded_range_matches;
use metrics::{scan_ac_baseline_metric_points, scan_ac_metric_points, ScanAcStats};

#[cfg(test)]
mod tests;

const HAYSTACK_BYTES: usize = 4 * 1024 * 1024;
const MAX_MATCHES: u32 = 65_536;
const MATCH_COUNT_INPUT_INDEX: usize = 6;
const MATCHES_RESOURCE_INDEX: usize = 7;
const RESET_RESOURCE_INDICES: [usize; 1] = [MATCH_COUNT_INPUT_INDEX];
const SCAN_RESOURCE_INDICES: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, MATCHES_RESOURCE_INDEX];
const MATCH_TRIPLE_WORDS: usize = 3;
const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

const PATTERNS: &[&[u8]] = &[
    b"AKIA",
    b"ghp_",
    b"Authorization: Bearer ",
    b"password=",
    b"api_key",
    b"secret=",
    b"BEGIN RSA PRIVATE KEY",
    b"BEGIN OPENSSH PRIVATE KEY",
    b"eval(",
    b"strcpy(",
    b"memcpy(",
    b"TODO:",
    b"unsafe {",
    b"__attribute__((",
    b"container_of(",
    b"ioread32(",
];

pub struct ScanAcIrregularLiterals;

struct ScanAcIrregularPrepared {
    program: Program,
    reset_program: Program,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    baseline_outputs: Vec<Vec<u8>>,
    baseline_wall_ns: u64,
    stats: ScanAcStats,
    resident: Option<ResidentInputSet>,
}

impl BenchCase for ScanAcIrregularLiterals {
    fn id(&self) -> BenchId {
        BenchId("scan.ac.irregular_literals.4m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Aho-Corasick Irregular Literal Scan 4M".to_string(),
            description: "Packed-byte AC bounded-ranges scan over unaligned, varied-length security/parser literals in a noisy 4 MiB haystack".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "dfa".to_string(),
                "aho-corasick".to_string(),
                "packed-byte".to_string(),
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
            ],
        }
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
                let output_bytes = u64::from(prepared.stats.max_matches) * 3 * 4 + 4;
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
        let prepared = prepared
            .downcast_ref::<ScanAcIrregularPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared irregular AC scan payload had the wrong type".to_string(),
                )
            })?;

        let mut dispatch_config = ctx.dispatch_config.clone();
        let program_workgroup = prepared.program.workgroup_size();
        let workgroup = dispatch_config
            .workgroup_override
            .unwrap_or(program_workgroup);
        if workgroup.contains(&0) {
            return Err(BenchError::ExecutionFailed(format!(
                "irregular AC scan received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
        }
        dispatch_config.grid_override.get_or_insert([
            prepared.stats.haystack_bytes.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ]);

        let (outputs, wall_ns, dispatch_ns, resident_used, device_reset_sequence) =
            if let Some(resident) = prepared.resident.as_ref() {
                let sequence =
                    dispatch_resident_scan_sequence(ctx, prepared, resident, program_workgroup)?;
                (sequence.outputs, sequence.wall_ns, None, true, true)
            } else {
                let dispatch = dispatch_program_timed(
                    ctx,
                    &prepared.program,
                    None,
                    &prepared.inputs,
                    &dispatch_config,
                )?;
                let timed = dispatch.timed;
                (
                    timed.outputs,
                    timed.wall_ns,
                    timed.device_ns,
                    dispatch.resident_used,
                    false,
                )
            };

        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let resident_reset_bytes = 0;
        let accounting =
            transfer_accounting(prepared.input_bytes_total, output_bytes, resident_used);

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(wall_ns),
                dispatch_ns,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes),
                bytes_read: Some(accounting.bytes_read),
                bytes_written: Some(accounting.bytes_written),
                bytes_touched: Some(accounting.bytes_touched),
                custom: scan_ac_metric_points(
                    prepared.stats,
                    prepared.baseline_wall_ns,
                    wall_ns,
                    resident_used,
                    resident_reset_bytes,
                    device_reset_sequence,
                    workgroup[0],
                ),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(
                    prepared
                        .baseline_outputs
                        .iter()
                        .map(Vec::len)
                        .sum::<usize>() as u64,
                ),
                custom: scan_ac_baseline_metric_points(prepared.stats),
                ..Default::default()
            }),
            outputs,
            baseline_outputs: Some(prepared.baseline_outputs.clone()),
        })
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

fn prepare_scan_ac_irregular(
    ctx: Option<&BenchContext>,
) -> Result<ScanAcIrregularPrepared, BenchError> {
    let (haystack, planted_matches) = build_irregular_haystack(HAYSTACK_BYTES);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths()?;
    let program = try_build_ac_bounded_ranges_program_ext(
        &ac.dfa,
        pattern_lengths.len() as u32,
        MAX_MATCHES,
        false,
    )
    .map_err(BenchError::ExecutionFailed)?;
    let reset_program = scan_ac_match_count_reset_program();

    let baseline_start = std::time::Instant::now();
    let expected_matches = cpu_aho_overlapping_matches(PATTERNS, &haystack)?;
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    if expected_matches.len() > MAX_MATCHES as usize {
        return Err(BenchError::EnvironmentInvalid(format!(
            "irregular AC scan fixture produced {} matches, above MAX_MATCHES={MAX_MATCHES}. Fix: lower fixture density or raise output capacity.",
            expected_matches.len()
        )));
    }

    let inputs = scan_ac_inputs(&ac, &pattern_lengths, &haystack);
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
        pack_u32_slice(&[expected_matches.len() as u32]),
        encode_match_triples(&expected_matches),
    ];
    let stats = ScanAcStats {
        haystack_bytes: HAYSTACK_BYTES as u32,
        packed_haystack_words: HAYSTACK_BYTES.div_ceil(4) as u32,
        patterns: PATTERNS.len() as u32,
        dfa_states: ac.dfa.state_count,
        max_pattern_len: ac.dfa.max_pattern_len,
        output_records: ac.dfa.output_records.len() as u32,
        expected_matches: expected_matches.len() as u32,
        max_matches: MAX_MATCHES,
        planted_matches,
    };

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

fn scan_ac_inputs(
    ac: &ClassicAcAutomaton,
    pattern_lengths: &[u32],
    haystack: &[u8],
) -> Vec<Vec<u8>> {
    vec![
        pack_haystack_u32(haystack),
        pack_u32_slice(&ac.dfa.transitions),
        pack_u32_slice(&ac.dfa.output_offsets),
        pack_u32_slice(&ac.dfa.output_records),
        pack_u32_slice(pattern_lengths),
        pack_u32_slice(&[haystack.len() as u32]),
        pack_u32_slice(&[0]),
    ]
}

fn pattern_lengths() -> Result<Vec<u32>, BenchError> {
    PATTERNS
        .iter()
        .map(|pattern| {
            u32::try_from(pattern.len()).map_err(|_| {
                BenchError::EnvironmentInvalid(
                    "irregular AC pattern length exceeded u32. Fix: split oversized literals."
                        .to_string(),
                )
            })
        })
        .collect()
}

fn build_irregular_haystack(len: usize) -> (Vec<u8>, u32) {
    let mut haystack = vec![0_u8; len];
    for (index, byte) in haystack.iter_mut().enumerate() {
        let mixed = mix32(index as u32);
        *byte = 33 + (mixed % 90) as u8;
    }

    let mut planted = 0_u32;
    for (pattern_index, pattern) in PATTERNS.iter().enumerate() {
        let stride = 8_191 + pattern_index * 271;
        let phase = 17 + pattern_index * 113;
        let mut offset = phase;
        while offset + pattern.len() <= haystack.len() {
            if (offset & 31) != 0 {
                haystack[offset..offset + pattern.len()].copy_from_slice(pattern);
                planted += 1;
            }
            offset += stride;
        }
    }
    (haystack, planted)
}

fn decode_scan_outputs(outputs: &[Vec<u8>], context: &str) -> Result<Vec<Match>, BenchError> {
    let count_bytes = outputs.first().ok_or_else(|| {
        BenchError::CorrectnessViolation(format!("{context} did not produce match_count"))
    })?;
    if count_bytes.len() < 4 {
        return Err(BenchError::CorrectnessViolation(format!(
            "{context} match_count buffer was {} bytes, expected at least 4",
            count_bytes.len()
        )));
    }
    let count = u32::from_le_bytes([
        count_bytes[0],
        count_bytes[1],
        count_bytes[2],
        count_bytes[3],
    ]);
    let triples = outputs.get(1).ok_or_else(|| {
        BenchError::CorrectnessViolation(format!("{context} did not produce match triples"))
    })?;
    try_unpack_match_triples(triples, count).map_err(|error| {
        BenchError::CorrectnessViolation(format!(
            "{context} match triples failed to decode: {error}"
        ))
    })
}

fn encode_match_triples(matches: &[Match]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(matches.len() * 12);
    for hit in matches {
        encoded.extend_from_slice(&hit.pattern_id.to_le_bytes());
        encoded.extend_from_slice(&hit.start.to_le_bytes());
        encoded.extend_from_slice(&hit.end.to_le_bytes());
    }
    encoded
}

fn match_triples_output_bytes(max_matches: u32) -> Result<usize, BenchError> {
    usize::try_from(max_matches)
        .ok()
        .and_then(|matches| matches.checked_mul(MATCH_TRIPLE_WORDS))
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| {
            BenchError::EnvironmentInvalid(format!(
                "irregular AC scan max_matches={max_matches} overflows resident output byte sizing. Fix: split the scan output into smaller shards."
            ))
        })
}

struct ScanResidentSequenceRun {
    outputs: Vec<Vec<u8>>,
    wall_ns: u64,
}

fn dispatch_resident_scan_sequence(
    ctx: &BenchContext,
    prepared: &ScanAcIrregularPrepared,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<ScanResidentSequenceRun, BenchError> {
    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "irregular AC scan resident sequence uses program workgroup {:?}, but received override {:?}. Fix: run the resident scan sequence without a workgroup override or rebuild the resident sequence program.",
                workgroup, override_workgroup
            )));
        }
    }

    let reset_resources = resident
        .resources_for_indices(&RESET_RESOURCE_INDICES, "irregular AC scan reset sequence")?;
    let scan_resources =
        resident.resources_for_indices(&SCAN_RESOURCE_INDICES, "irregular AC scan sequence")?;
    let reset_step = ResidentDispatchStep {
        program: &prepared.reset_program,
        resources: &reset_resources,
        grid_override: Some([1, 1, 1]),
    };
    let scan_step = ResidentDispatchStep {
        program: &prepared.program,
        resources: &scan_resources,
        grid_override: Some([
            prepared.stats.haystack_bytes.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ]),
    };
    let match_output_bytes = match_triples_output_bytes(prepared.stats.max_matches)?;
    let read_ranges = [
        ResidentReadRange {
            resource: &scan_resources[MATCH_COUNT_INPUT_INDEX],
            byte_offset: 0,
            byte_len: prepared.baseline_outputs[0].len(),
        },
        ResidentReadRange {
            resource: &scan_resources[MATCHES_RESOURCE_INDEX],
            byte_offset: 0,
            byte_len: match_output_bytes,
        },
    ];

    let mut count_output = Vec::with_capacity(prepared.baseline_outputs[0].len());
    let mut matches_output = Vec::with_capacity(match_output_bytes);
    let started = Instant::now();
    ctx.preferred_backend
        .dispatch_resident_sequence_read_ranges_into(
            &[reset_step, scan_step],
            &read_ranges,
            &mut [&mut count_output, &mut matches_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;

    Ok(ScanResidentSequenceRun {
        outputs: vec![count_output, matches_output],
        wall_ns,
    })
}

fn scan_ac_match_count_reset_program() -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("match_count", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![Node::if_then(
            Expr::eq(idx, Expr::u32(0)),
            vec![Node::store("match_count", Expr::u32(0), Expr::u32(0))],
        )],
    )
}

fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846C_A68B);
    value ^ (value >> 16)
}

inventory::submit! {
    &ScanAcIrregularLiterals as &'static dyn BenchCase
}
