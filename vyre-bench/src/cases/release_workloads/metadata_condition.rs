//! Metadata condition batch release case over filesize, header, and entropy metadata
//! columns.

use super::registration::{gpu_requirements, RELEASE_SUITES};
use super::resident_batch::{
    batch_metric_points, dispatch_batch_or_single, BatchPlan, SingleSample,
};
use super::run_assembly::{
    bench_run_from_timed_with_accounting, encode_u32_words, resident_reset_transfer_accounting,
};
use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata,
    BenchRequirements, BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase,
    WorkloadClass,
};
use crate::api::metric::MetricPoint;
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, ResidentInputPool, ResidentInputSet,
};
use crate::cases::reference_sample::timed_reference;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

const METADATA_CONDITION_RESIDENT_BATCH_SIZE: usize = 16;

pub struct MetadataConditionBatch;

struct MetadataConditionPrepared {
    program: Program,
    filesize: Vec<u32>,
    header: Vec<u32>,
    entropy: Vec<u32>,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    expected_count: u32,
    resident: Option<ResidentInputSet>,
    resident_batch: Option<ResidentInputPool>,
}

pub(super) const METADATA_RECORDS: u32 = 1_048_576;

pub(super) const METADATA_WORKGROUP_SIZE: u32 = 256;

pub(super) const METADATA_OUTPUT_RESET_BYTES: u64 = 4;
impl BenchCase for MetadataConditionBatch {
    fn id(&self) -> BenchId {
        BenchId("metadata.condition.filesize_header.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Metadata Condition File/Header 1M".to_string(),
            description: "File metadata and PE/header-style condition evaluation over 1M records"
                .to_string(),
            tags: vec![
                "metadata".to_string(),
                "condition".to_string(),
                "filesize".to_string(),
                "header".to_string(),
                "pe".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-libs".to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        RELEASE_SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        gpu_requirements((METADATA_RECORDS as u64 * 12) + 4)
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            "metadata condition evaluation",
            "vyre-libs",
            "optimized CPU PE-header predicate evaluator",
            50.0,
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let program = metadata_condition_program();
        let mut filesize = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut header = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut entropy = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut packed_records = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut expected = 0u32;
        for index in 0..METADATA_RECORDS {
            let size = 1024 + (index.wrapping_mul(13) % 131_072);
            let hdr = if index % 5 == 0 {
                0x0000_4550
            } else {
                0x464C_457F
            };
            let ent = 5000 + (index.wrapping_mul(17) % 4500);
            expected += u32::from(size > 4096 && hdr == 0x0000_4550 && ent > 7200);
            filesize.push(size);
            header.push(hdr);
            entropy.push(ent);

            let size_offset = size - 1024;
            let entropy_offset = ent - 5000;
            let is_pe = u32::from(hdr == 0x0000_4550);
            let packed = size_offset | (entropy_offset << 17) | (is_pe << 30);
            packed_records.push(packed);
        }
        let inputs = vec![encode_u32_words(&packed_records)];
        let input_bytes_total = input_bytes_total(&inputs);
        let resident = ResidentInputSet::upload_program_ordered_with_zeroed_outputs_optional(
            ctx,
            &program,
            &inputs,
            "metadata condition bench",
        )?;
        let resident_batch =
            ResidentInputPool::upload_program_ordered_with_zeroed_outputs_optional(
                ctx,
                &program,
                &inputs,
                METADATA_CONDITION_RESIDENT_BATCH_SIZE,
                "metadata condition bench batch",
            )?;
        Ok(Box::new(MetadataConditionPrepared {
            program,
            filesize,
            header,
            entropy,
            inputs,
            input_bytes_total,
            expected_count: expected,
            resident,
            resident_batch,
        }))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<MetadataConditionPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared_as::<MetadataConditionPrepared>(prepared, "metadata condition")?;
        let sample = dispatch_batch_or_single(
            ctx,
            &prepared.program,
            &prepared.inputs,
            prepared.resident_batch.as_ref(),
            &BatchPlan {
                label: "metadata condition",
                batch_size: METADATA_CONDITION_RESIDENT_BATCH_SIZE,
                reset_resource: 0,
                reset_resource_kind: "counter",
                reset_payload: &0u32.to_le_bytes(),
            },
            || dispatch_single_metadata_resident(ctx, prepared),
        )?;

        let (cpu_count, baseline_wall) = timed_reference(|| {
            let mut cpu_count = 0u32;
            for index in 0..prepared.filesize.len() {
                cpu_count += u32::from(
                    prepared.filesize[index] > 4096
                        && prepared.header[index] == 0x0000_4550
                        && prepared.entropy[index] > 7200,
                );
            }
            cpu_count
        });
        if cpu_count != prepared.expected_count {
            return Err(BenchError::CorrectnessViolation(
                "metadata CPU baseline count disagreed with generator expectation".to_string(),
            ));
        }
        let baseline_outputs = vec![cpu_count.to_le_bytes().to_vec()];
        let output_bytes = sample.timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting = resident_reset_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            sample.resident_used,
            sample.reset_bytes,
        );
        let logical_bytes_touched = (u64::from(METADATA_RECORDS) * 12).saturating_add(output_bytes);
        let mut custom = vec![
            MetricPoint {
                name: "metadata_records".to_string(),
                value: u64::from(METADATA_RECORDS),
            },
            MetricPoint {
                name: "metadata_expected_matches".to_string(),
                value: u64::from(prepared.expected_count),
            },
        ];
        custom.append(&mut batch_metric_points("metadata", &sample));
        let mut run = bench_run_from_timed_with_accounting(
            sample.timed,
            prepared.input_bytes_total,
            baseline_outputs,
            baseline_wall,
            "metadata_records",
            METADATA_RECORDS,
            logical_bytes_touched,
            accounting,
        )?;
        run.metrics.custom = custom;
        Ok(run)
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<MetadataConditionPrepared>()
            .map(|_| {
                (
                    u64::from(METADATA_RECORDS) * 12,
                    METADATA_OUTPUT_RESET_BYTES,
                )
            })
            .unwrap_or((
                u64::from(METADATA_RECORDS) * 12,
                METADATA_OUTPUT_RESET_BYTES,
            ))
    }
}

fn dispatch_single_metadata_resident(
    ctx: &BenchContext,
    prepared: &MetadataConditionPrepared,
) -> Result<SingleSample, BenchError> {
    let reset_bytes = if let Some(resident) = prepared.resident.as_ref() {
        let payload = 0u32.to_le_bytes();
        resident.upload_resource(0, &payload, "metadata condition resident counter reset")?;
        payload.len() as u64
    } else {
        0
    };
    let dispatch = dispatch_program_timed(
        ctx,
        &prepared.program,
        prepared.resident.as_ref(),
        &prepared.inputs,
        &ctx.dispatch_config,
    )?;
    Ok(SingleSample {
        timed: dispatch.timed,
        resident_used: dispatch.resident_used,
        reset_bytes,
    })
}

#[must_use]
pub(super) fn metadata_condition_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out_count", 0, DataType::U32).with_count(1),
            BufferDecl::storage("metadata_records", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(METADATA_RECORDS),
            BufferDecl::workgroup("warp_scratch", 1024, DataType::U32),
        ],
        [METADATA_WORKGROUP_SIZE, 1, 1],
        super::synthetic_programs::warp_reduction_count_nodes(
            METADATA_WORKGROUP_SIZE,
            METADATA_RECORDS,
            Expr::and(
                Expr::var("in_bounds"),
                Expr::and(
                    Expr::var("is_pe"),
                    Expr::and(
                        Expr::gt(Expr::var("size_offset"), Expr::u32(3072)),
                        Expr::gt(Expr::var("entropy_offset"), Expr::u32(2200)),
                    ),
                ),
            ),
            vec![
                Node::let_bind(
                    "packed",
                    Expr::load("metadata_records", Expr::var("safe_idx")),
                ),
                Node::let_bind(
                    "size_offset",
                    Expr::bitand(Expr::var("packed"), Expr::u32(0x0001_FFFF)),
                ),
                Node::let_bind(
                    "entropy_offset",
                    Expr::bitand(
                        Expr::shr(Expr::var("packed"), Expr::u32(17)),
                        Expr::u32(0x0000_1FFF),
                    ),
                ),
                Node::let_bind(
                    "is_pe",
                    Expr::ne(
                        Expr::bitand(Expr::var("packed"), Expr::u32(0x4000_0000)),
                        Expr::u32(0),
                    ),
                ),
            ],
        ),
    )
}
