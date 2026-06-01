use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{transfer_accounting, u32_counter_reset_program, ResidentInputSet};
use crate::api::suite::SuiteKind;
use crate::cases::scan_ac_irregular::support::{build_irregular_haystack, encode_match_triples};
use crate::cases::scan_ac_irregular::PATTERNS;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::Program;
use vyre_foundation::match_result::Match;
use vyre_libs::scan::classic_ac::{CLASSIC_AC_SUFFIX2_MASK_WORDS, CLASSIC_AC_SUFFIX3_BLOOM_WORDS};
use vyre_libs::scan::{
    GpuLiteralSet, LiteralSetPreparedScan, LiteralSetScanScratch,
    LITERAL_SET_MATCHES_RESOURCE_INDEX, LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX,
    LITERAL_SET_RESET_RESOURCE_INDICES, LITERAL_SET_SCAN_RESOURCE_INDICES,
};

const HAYSTACK_BYTES: usize = 4 * 1024 * 1024;
const DEFAULT_LITERAL_SET_MATCH_CAP: u32 = 10_000;
const MATCH_TRIPLE_BYTES: u64 = 12;
const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

pub struct LiteralSetIrregularHotloop;

struct LiteralSetIrregularPrepared {
    engine: GpuLiteralSet,
    haystack: Vec<u8>,
    matches: Vec<Match>,
    scratch: LiteralSetScanScratch,
    prepared_scan: LiteralSetPreparedScan,
    reset_program: Program,
    resident: Option<ResidentInputSet>,
    baseline_outputs: Vec<Vec<u8>>,
    baseline_wall_ns: u64,
    expected_matches: u32,
    max_matches: u32,
    planted_matches: u32,
    encoded_input_bytes: u64,
    output_bytes: u64,
}

impl BenchCase for LiteralSetIrregularHotloop {
    fn id(&self) -> BenchId {
        BenchId("scan.literal_set.irregular_hotloop.4m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "GpuLiteralSet Irregular Hot Loop 4M".to_string(),
            description: "Public GpuLiteralSet prepared-dispatch hot loop over unaligned security/parser literals with resident input reuse when supported".to_string(),
            tags: vec![
                "scan".to_string(),
                "pattern".to_string(),
                "dfa".to_string(),
                "literal-set".to_string(),
                "irregular".to_string(),
                "hot-loop".to_string(),
                "resident".to_string(),
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
                "literal-set".to_string(),
                "public-api-hot-loop".to_string(),
                "resident-prepared-dispatch".to_string(),
            ],
        }
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "GpuLiteralSet irregular public scan",
            "vyre-libs",
            "vyre-libs DFA reference_scan",
            1.0,
        ))
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<LiteralSetIrregularPrepared>()
            .map(|prepared| (prepared.encoded_input_bytes, prepared.output_bytes))
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let (haystack, planted_matches) = build_irregular_haystack(HAYSTACK_BYTES);
        let engine = GpuLiteralSet::try_compile(PATTERNS).map_err(|error| {
            BenchError::EnvironmentInvalid(format!(
                "literal-set irregular fixture failed to compile: {error}"
            ))
        })?;

        let baseline_start = Instant::now();
        let baseline_matches = engine.reference_scan(&haystack);
        let baseline_wall_ns = baseline_start
            .elapsed()
            .as_nanos()
            .min(u128::from(u64::MAX)) as u64;
        let expected_matches = u32::try_from(baseline_matches.len()).map_err(|_| {
            BenchError::EnvironmentInvalid(format!(
                "literal-set irregular fixture produced {} matches, above u32 capacity. Fix: lower fixture density or shard the scan.",
                baseline_matches.len()
            ))
        })?;
        let max_matches = expected_matches.max(1);
        let encoded_matches = encode_match_triples(&baseline_matches);
        let output_bytes = 4_u64.saturating_add(encoded_matches.len() as u64);
        let baseline_outputs = vec![expected_matches.to_le_bytes().to_vec(), encoded_matches];
        let mut scratch = LiteralSetScanScratch::default();
        engine
            .prepare_literal_scratch(max_matches, &mut scratch)
            .map_err(|error| {
                BenchError::ExecutionFailed(format!(
                    "literal-set irregular hot-loop scratch preparation failed: {error}"
                ))
            })?;
        let prepared_scan = engine
            .prepare_scan_dispatch(&haystack, max_matches)
            .map_err(|error| {
                BenchError::ExecutionFailed(format!(
                    "literal-set irregular prepared dispatch failed: {error}"
                ))
            })?;
        let reset_program = u32_counter_reset_program("match_count");
        let resident = ResidentInputSet::upload_with_zeroed_outputs_optional(
            ctx,
            &prepared_scan.inputs,
            &[prepared_scan.matches_output_bytes],
            "literal-set irregular hot-loop",
        )?;
        let encoded_input_bytes = prepared_scan.encoded_input_bytes;

        Ok(Box::new(LiteralSetIrregularPrepared {
            engine,
            haystack,
            matches: Vec::with_capacity(expected_matches as usize),
            scratch,
            prepared_scan,
            reset_program,
            resident,
            baseline_outputs,
            baseline_wall_ns,
            expected_matches,
            max_matches,
            planted_matches,
            encoded_input_bytes,
            output_bytes,
        }))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<LiteralSetIrregularPrepared>()
            .map(|prepared| &prepared.prepared_scan.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_mut::<LiteralSetIrregularPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared literal-set irregular payload had the wrong type".to_string(),
                )
            })?;

        let (outputs, wall_ns, resident_used, device_reset_sequence) =
            if let Some(resident) = prepared.resident.as_ref() {
                let started = Instant::now();
                let sequence = dispatch_literal_set_resident_sequence(ctx, prepared, resident)?;
                prepared
                    .prepared_scan
                    .decode_outputs_into(&sequence.outputs, &mut prepared.matches)
                    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
                let encoded_matches = encode_match_triples(&prepared.matches);
                let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                (
                    vec![
                        (prepared.matches.len() as u32).to_le_bytes().to_vec(),
                        encoded_matches,
                    ],
                    wall_ns,
                    true,
                    true,
                )
            } else {
                let started = Instant::now();
                prepared
                    .engine
                    .scan_into_with_literal_scratch(
                        ctx.preferred_backend.as_ref(),
                        &prepared.haystack,
                        prepared.max_matches,
                        &mut prepared.matches,
                        &mut prepared.scratch,
                    )
                    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
                let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
                let encoded_matches = encode_match_triples(&prepared.matches);
                (
                    vec![
                        (prepared.matches.len() as u32).to_le_bytes().to_vec(),
                        encoded_matches,
                    ],
                    wall_ns,
                    false,
                    false,
                )
            };
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting =
            transfer_accounting(prepared.encoded_input_bytes, output_bytes, resident_used);

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(wall_ns),
                dispatch_ns: None,
                input_bytes: Some(prepared.encoded_input_bytes),
                output_bytes: Some(output_bytes),
                bytes_read: Some(accounting.bytes_read),
                bytes_written: Some(accounting.bytes_written),
                bytes_touched: Some(accounting.bytes_touched),
                custom: literal_set_metric_points(
                    prepared,
                    wall_ns,
                    resident_used,
                    device_reset_sequence,
                ),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.haystack.len() as u64),
                output_bytes: Some(prepared.output_bytes),
                custom: vec![metric(
                    "literal_set_irregular_reference_matches",
                    u64::from(prepared.expected_matches),
                )],
                ..Default::default()
            }),
            outputs,
            baseline_outputs: Some(prepared.baseline_outputs.clone()),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

fn literal_set_metric_points(
    prepared: &LiteralSetIrregularPrepared,
    wall_ns: u64,
    resident_used: bool,
    device_reset_sequence: bool,
) -> Vec<MetricPoint> {
    let avoided_default_matches =
        DEFAULT_LITERAL_SET_MATCH_CAP.saturating_sub(prepared.max_matches);
    let mut metrics = vec![
        metric(
            "literal_set_irregular_haystack_bytes",
            prepared.haystack.len() as u64,
        ),
        metric("literal_set_irregular_patterns", PATTERNS.len() as u64),
        metric(
            "literal_set_irregular_pattern_bytes",
            prepared.engine.pattern_bytes.len() as u64,
        ),
        metric(
            "literal_set_irregular_dfa_states",
            u64::from(prepared.engine.dfa.state_count),
        ),
        metric(
            "literal_set_irregular_dfa_table_bytes",
            ((prepared.engine.dfa.transitions.len()
                + prepared.engine.dfa.output_offsets.len()
                + prepared.engine.dfa.output_records.len()) as u64)
                .saturating_mul(4),
        ),
        metric(
            "literal_set_irregular_dfa_output_records",
            prepared.engine.dfa.output_records.len() as u64,
        ),
        metric(
            "literal_set_irregular_prefilter_mask_bytes",
            ((8 + CLASSIC_AC_SUFFIX2_MASK_WORDS + CLASSIC_AC_SUFFIX3_BLOOM_WORDS) as u64)
                .saturating_mul(4),
        ),
        metric(
            "literal_set_irregular_resident_used",
            u64::from(resident_used),
        ),
        metric(
            "literal_set_irregular_device_reset_sequence",
            u64::from(device_reset_sequence),
        ),
        metric(
            "literal_set_irregular_expected_matches",
            u64::from(prepared.expected_matches),
        ),
        metric(
            "literal_set_irregular_max_matches",
            u64::from(prepared.max_matches),
        ),
        metric(
            "literal_set_irregular_planted_matches",
            u64::from(prepared.planted_matches),
        ),
        metric(
            "literal_set_irregular_cap_specific_scratch_program_cache",
            u64::from(prepared.max_matches != DEFAULT_LITERAL_SET_MATCH_CAP),
        ),
        metric(
            "literal_set_irregular_avoided_default_readback_bytes",
            u64::from(avoided_default_matches).saturating_mul(MATCH_TRIPLE_BYTES),
        ),
    ];
    if wall_ns > 0 {
        metrics.push(metric(
            "literal_set_irregular_speedup_x1000",
            (u128::from(prepared.baseline_wall_ns) * 1000 / u128::from(wall_ns))
                .min(u128::from(u64::MAX)) as u64,
        ));
    }
    metrics
}

fn metric(name: &str, value: u64) -> MetricPoint {
    MetricPoint {
        name: name.to_string(),
        value,
    }
}

struct LiteralSetResidentSequenceRun {
    outputs: Vec<Vec<u8>>,
}

fn dispatch_literal_set_resident_sequence(
    ctx: &BenchContext,
    prepared: &LiteralSetIrregularPrepared,
    resident: &ResidentInputSet,
) -> Result<LiteralSetResidentSequenceRun, BenchError> {
    let program_workgroup = prepared.prepared_scan.program.workgroup_size();
    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != program_workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "literal-set irregular resident sequence uses program workgroup {:?}, but received override {:?}. Fix: run the resident scan sequence without a workgroup override or rebuild the prepared dispatch program.",
                program_workgroup, override_workgroup
            )));
        }
    }

    let reset_resources = resident.resources_for_indices(
        &LITERAL_SET_RESET_RESOURCE_INDICES,
        "literal-set irregular reset sequence",
    )?;
    let scan_resources = resident.resources_for_indices(
        &LITERAL_SET_SCAN_RESOURCE_INDICES,
        "literal-set irregular scan sequence",
    )?;
    let reset_step = ResidentDispatchStep {
        program: &prepared.reset_program,
        resources: &reset_resources,
        grid_override: Some([1, 1, 1]),
    };
    let scan_step = ResidentDispatchStep {
        program: &prepared.prepared_scan.program,
        resources: &scan_resources,
        grid_override: prepared.prepared_scan.dispatch_config.grid_override,
    };
    let match_output_bytes = prepared
        .prepared_scan
        .match_triples_readback_bytes(prepared.expected_matches)
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let read_ranges = [
        ResidentReadRange {
            resource: &scan_resources[LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX],
            byte_offset: 0,
            byte_len: prepared.prepared_scan.match_count_readback_bytes(),
        },
        ResidentReadRange {
            resource: &scan_resources[LITERAL_SET_MATCHES_RESOURCE_INDEX],
            byte_offset: 0,
            byte_len: match_output_bytes,
        },
    ];

    let mut count_output = Vec::with_capacity(prepared.prepared_scan.match_count_readback_bytes());
    let mut matches_output = Vec::with_capacity(match_output_bytes);
    ctx.preferred_backend
        .dispatch_resident_sequence_read_ranges_into(
            &[reset_step, scan_step],
            &read_ranges,
            &mut [&mut count_output, &mut matches_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

    Ok(LiteralSetResidentSequenceRun {
        outputs: vec![count_output, matches_output],
    })
}

inventory::submit! {
    &LiteralSetIrregularHotloop as &'static dyn BenchCase
}
