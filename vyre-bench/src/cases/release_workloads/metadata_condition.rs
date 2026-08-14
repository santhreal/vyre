//! Metadata condition batch release case over filesize, header, and entropy metadata
//! columns.

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase, WorkloadClass,
};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, ResidentInputPool, ResidentInputSet,
};
use super::registration::{gpu_requirements, RELEASE_SUITES};
use super::run_assembly::{encode_u32_words, resident_reset_transfer_accounting};
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
        let program = Program::wrapped(
            vec![
                BufferDecl::output("out_count", 0, DataType::U32).with_count(1),
                BufferDecl::storage("filesize", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(METADATA_RECORDS),
                BufferDecl::storage("header", 2, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(METADATA_RECORDS),
                BufferDecl::storage("entropy_x1000", 3, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(METADATA_RECORDS),
            ],
            [256, 1, 1],
            vec![
                Node::let_bind("idx", Expr::gid_x()),
                Node::if_then(
                    Expr::and(
                        Expr::lt(Expr::var("idx"), Expr::u32(METADATA_RECORDS)),
                        Expr::and(
                            Expr::gt(Expr::load("filesize", Expr::var("idx")), Expr::u32(4096)),
                            Expr::and(
                                Expr::eq(
                                    Expr::load("header", Expr::var("idx")),
                                    Expr::u32(0x0000_4550),
                                ),
                                Expr::gt(
                                    Expr::load("entropy_x1000", Expr::var("idx")),
                                    Expr::u32(7200),
                                ),
                            ),
                        ),
                    ),
                    vec![Node::let_bind(
                        "_slot",
                        Expr::atomic_add("out_count", Expr::u32(0), Expr::u32(1)),
                    )],
                ),
            ],
        );
        let mut filesize = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut header = Vec::with_capacity(METADATA_RECORDS as usize);
        let mut entropy = Vec::with_capacity(METADATA_RECORDS as usize);
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
        }
        let inputs = vec![
            encode_u32_words(&filesize),
            encode_u32_words(&header),
            encode_u32_words(&entropy),
        ];
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
        let prepared = prepared
            .downcast_ref::<MetadataConditionPrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "metadata condition prepared payload type mismatch".to_string(),
                )
            })?;
        let mut metadata_batch_wall_ns = None;
        let mut metadata_batch_len = None;
        let (timed, resident_used, resident_reset_bytes) = if let Some(resident_batch) =
            prepared.resident_batch.as_ref()
        {
            resident_batch.upload_resource_to_all_sets(
                0,
                &0u32.to_le_bytes(),
                "metadata condition resident batch counter reset",
            )?;
            let config = crate::api::case::dispatch_config_with_inferred_grid(
                &prepared.program,
                &prepared.inputs,
                &ctx.dispatch_config,
            )
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
            match resident_batch.dispatch_artifact_batch_timed(
                ctx,
                &prepared.program,
                METADATA_CONDITION_RESIDENT_BATCH_SIZE,
                &config,
            ) {
                Ok(batch) => {
                    if batch.outputs.len() != METADATA_CONDITION_RESIDENT_BATCH_SIZE {
                        return Err(BenchError::ExecutionFailed(format!(
                            "metadata condition resident batch returned {} output row(s), expected {}",
                            batch.outputs.len(),
                            METADATA_CONDITION_RESIDENT_BATCH_SIZE
                        )));
                    }
                    let first_outputs = batch.outputs.first().cloned().ok_or_else(|| {
                        BenchError::ExecutionFailed(
                            "metadata condition resident batch returned no output rows".to_string(),
                        )
                    })?;
                    if let Some((index, _)) = batch
                        .outputs
                        .iter()
                        .enumerate()
                        .find(|(_, outputs)| **outputs != first_outputs)
                    {
                        return Err(BenchError::CorrectnessViolation(format!(
                            "metadata condition resident batch output row {index} disagreed with row 0"
                        )));
                    }
                    metadata_batch_wall_ns = Some(batch.wall_ns_total);
                    metadata_batch_len = Some(batch.batch_len as u64);
                    (
                        vyre_driver::TimedDispatchResult {
                            outputs: first_outputs,
                            wall_ns: batch.per_item_wall_ns(),
                            device_ns: batch.per_item_device_ns(),
                            enqueue_ns: None,
                            wait_ns: None,
                        },
                        true,
                        METADATA_OUTPUT_RESET_BYTES,
                    )
                }
                Err(vyre_driver::BackendError::UnsupportedFeature { .. }) => {
                    dispatch_single_metadata_resident(ctx, prepared)?
                }
                Err(error) => return Err(BenchError::BackendFailed(error.to_string())),
            }
        } else {
            dispatch_single_metadata_resident(ctx, prepared)?
        };
        let outputs = timed.outputs.clone();
        let baseline_start = std::time::Instant::now();
        let mut cpu_count = 0u32;
        for index in 0..prepared.filesize.len() {
            cpu_count += u32::from(
                prepared.filesize[index] > 4096
                    && prepared.header[index] == 0x0000_4550
                    && prepared.entropy[index] > 7200,
            );
        }
        let baseline_wall = baseline_start.elapsed().as_nanos() as u64;
        if cpu_count != prepared.expected_count {
            return Err(BenchError::CorrectnessViolation(
                "metadata CPU baseline count disagreed with generator expectation".to_string(),
            ));
        }
        let baseline_outputs = vec![cpu_count.to_le_bytes().to_vec()];
        let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let accounting = resident_reset_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            resident_used,
            resident_reset_bytes,
        );
        let logical_bytes_touched = prepared.input_bytes_total.saturating_add(output_bytes);
        let mut custom = vec![
            MetricPoint {
                name: "metadata_records".to_string(),
                value: u64::from(METADATA_RECORDS),
            },
            MetricPoint {
                name: "metadata_expected_matches".to_string(),
                value: u64::from(prepared.expected_count),
            },
            MetricPoint {
                name: "metadata_resident_buffers".to_string(),
                value: u64::from(resident_used),
            },
            MetricPoint {
                name: "metadata_resident_reset_bytes".to_string(),
                value: resident_reset_bytes,
            },
        ];
        if let Some(batch_wall_ns) = metadata_batch_wall_ns {
            custom.push(MetricPoint {
                name: "metadata_resident_batch_wall_ns".to_string(),
                value: batch_wall_ns,
            });
        }
        if let Some(batch_len) = metadata_batch_len {
            custom.push(MetricPoint {
                name: "metadata_resident_batch_len".to_string(),
                value: batch_len,
            });
        }
        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(timed.wall_ns),
                dispatch_ns: timed.device_ns,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes),
                bytes_read: Some(accounting.bytes_read),
                bytes_written: Some(accounting.bytes_written),
                bytes_touched: Some(logical_bytes_touched),
                custom,
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(baseline_wall),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64),
                bytes_touched:
                    Some(
                        prepared.input_bytes_total.saturating_add(
                            baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64,
                        ),
                    ),
                ..Default::default()
            }),
            outputs: timed.outputs,
            baseline_outputs: Some(baseline_outputs),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<MetadataConditionPrepared>()
            .map(|prepared| (prepared.input_bytes_total, METADATA_OUTPUT_RESET_BYTES))
            .unwrap_or((
                u64::from(METADATA_RECORDS) * 12,
                METADATA_OUTPUT_RESET_BYTES,
            ))
    }
}

fn dispatch_single_metadata_resident(
    ctx: &BenchContext,
    prepared: &MetadataConditionPrepared,
) -> Result<(vyre_driver::TimedDispatchResult, bool, u64), BenchError> {
    let resident_reset_bytes = if let Some(resident) = prepared.resident.as_ref() {
        resident.upload_resource(
            0,
            &0u32.to_le_bytes(),
            "metadata condition resident counter reset",
        )?;
        METADATA_OUTPUT_RESET_BYTES
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
    Ok((dispatch.timed, dispatch.resident_used, resident_reset_bytes))
}
