//! Synthetic count workload descriptor, its pattern discriminant, and the bench case
//! that dispatches one release macro pattern.

use super::macro_registry::ReleaseMacroFamily;
use super::registration::{gpu_requirements, RELEASE_SUITES};
use super::run_assembly::{
    add_release_alias_metrics, bench_run_from_timed_with_accounting, encode_u32_words,
    resident_reset_transfer_accounting,
};
use super::synthetic_oracle::{
    pattern_input_count, string_bitmap_scatter_expected_words, string_bitmap_scatter_inputs,
    synthetic_baseline_label, synthetic_cpu_count_over_inputs, synthetic_inputs,
    synthetic_logical_output_bytes, synthetic_output_reset_bytes,
};
use super::synthetic_programs::{
    build_synthetic_release_program, string_bitmap_scatter_release_program,
};
use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata,
    BenchRequirements, BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase,
    WorkloadClass,
};
use crate::api::metric::{elapsed_ns, MetricPoint};
use crate::api::resident::{dispatch_program_timed, input_bytes_total, ResidentInputSet};
use vyre::ir::Program;

pub(super) const STRING_BITMAP_RESIDENT_BATCH_SIZE: usize = 16;

type SyntheticDispatchResult = (
    vyre_driver::TimedDispatchResult,
    bool,
    u64,
    Option<u64>,
    Option<u64>,
);

pub(super) struct SyntheticCountWorkload {
    pub(super) id: &'static str,
    pub(super) name: &'static str,
    pub(super) description: &'static str,
    pub(super) tags: &'static [&'static str],
    pub(super) owner_crate: &'static str,
    pub(super) primitive: &'static str,
    pub(super) metric_name: &'static str,
    pub(super) family: ReleaseMacroFamily,
    pub(super) records: u32,
    pub(super) min_speedup_x: f64,
    pub(super) pattern: SyntheticPattern,
}

struct SyntheticCountPrepared {
    program: Program,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    logical_output_bytes: u64,
    output_reset_payload: Vec<u8>,
    baseline: SyntheticBaseline,
    resident: Option<ResidentInputSet>,
}

enum SyntheticBaseline {
    Count {
        expected: u32,
    },
    StringBitmap {
        pattern_bitmap: Vec<u32>,
        rule_bitmap: Vec<u32>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SyntheticPattern {
    ConditionEval,
    StringBitmapScatter,
    OffsetCountAggregation,
    EntropyWindow,
    QuantifiedLoops,
    AliasReachingDef,
    IfdsWitness,
    AstMotifTraversal,
    MegakernelQueuedBatch,
    EgraphSaturation,
}

fn dispatch_single_synthetic_resident(
    ctx: &BenchContext,
    prepared: &SyntheticCountPrepared,
) -> Result<SyntheticDispatchResult, BenchError> {
    if let Some(resident) = prepared.resident.as_ref() {
        if !prepared.output_reset_payload.is_empty() {
            resident.upload_resource(
                0,
                &prepared.output_reset_payload,
                "synthetic release resident output reset",
            )?;
        }
        let dispatch = dispatch_program_timed(
            ctx,
            &prepared.program,
            Some(resident),
            &prepared.inputs,
            &ctx.dispatch_config,
        )?;
        return Ok((
            dispatch.timed,
            dispatch.resident_used,
            prepared.output_reset_payload.len() as u64,
            None,
            None,
        ));
    }

    let dispatch = dispatch_program_timed(
        ctx,
        &prepared.program,
        None,
        &prepared.inputs,
        &ctx.dispatch_config,
    )?;
    Ok((dispatch.timed, dispatch.resident_used, 0, None, None))
}

impl BenchCase for SyntheticCountWorkload {
    fn id(&self) -> BenchId {
        BenchId(self.id.to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        let mut tags = self
            .tags
            .iter()
            .map(|tag| (*tag).to_string())
            .collect::<Vec<_>>();
        tags.push("release".to_string());
        BenchMetadata {
            id: self.id(),
            name: self.name.to_string(),
            description: self.description.to_string(),
            tags,
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: self.owner_crate.to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        RELEASE_SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        let input_bytes = u64::from(self.records) * pattern_input_count(self.pattern) as u64 * 4;
        let output_bytes = if self.pattern == SyntheticPattern::StringBitmapScatter {
            u64::from(self.records.div_ceil(32))
                * std::mem::size_of::<u32>() as u64
                * STRING_BITMAP_RESIDENT_BATCH_SIZE as u64
        } else {
            4
        };
        gpu_requirements(input_bytes.saturating_add(output_bytes))
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_min_speedup(
            self.primitive,
            self.owner_crate,
            synthetic_baseline_label(self.pattern),
            self.min_speedup_x,
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let program = if self.pattern == SyntheticPattern::StringBitmapScatter {
            string_bitmap_scatter_release_program(self.records)
        } else {
            build_synthetic_release_program(self.pattern, self.records)
        };
        let (inputs, baseline) = match self.pattern {
            SyntheticPattern::StringBitmapScatter => {
                let mut generated = string_bitmap_scatter_inputs(self.records);
                generated.inputs[0].resize(
                    self.records.div_ceil(32) as usize
                        * std::mem::size_of::<u32>()
                        * STRING_BITMAP_RESIDENT_BATCH_SIZE,
                    0,
                );
                (
                    generated.inputs,
                    SyntheticBaseline::StringBitmap {
                        pattern_bitmap: generated.pattern_bitmap,
                        rule_bitmap: generated.rule_bitmap,
                    },
                )
            }
            pattern => {
                let generated = synthetic_inputs(pattern, self.records);
                (
                    generated.inputs,
                    SyntheticBaseline::Count {
                        expected: generated.expected,
                    },
                )
            }
        };
        let input_bytes_total = input_bytes_total(&inputs);
        let logical_output_bytes = synthetic_logical_output_bytes(self.pattern, self.records);
        let output_reset_payload =
            vec![0u8; synthetic_output_reset_bytes(self.pattern, self.records)];
        let resident = ResidentInputSet::upload_program_ordered_with_zeroed_outputs_optional(
            ctx,
            &program,
            &inputs,
            "synthetic release workload",
        )?;
        Ok(Box::new(SyntheticCountPrepared {
            program,
            inputs,
            input_bytes_total,
            logical_output_bytes,
            output_reset_payload,
            baseline,
            resident,
        }))
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared_as::<SyntheticCountPrepared>(prepared, "synthetic release")?;
        let mut dispatch_config = ctx.dispatch_config.clone();
        if self.pattern == SyntheticPattern::StringBitmapScatter {
            dispatch_config.grid_override = Some(
                vyre_driver::infer_dispatch_grid_for_count(
                    self.records,
                    prepared.program.workgroup_size(),
                )
                .map_err(|error| BenchError::BackendFailed(error.to_string()))?,
            );
        }
        if let Some(resident) = prepared.resident.as_ref() {
            if !prepared.output_reset_payload.is_empty() {
                resident.upload_resource(
                    0,
                    &prepared.output_reset_payload,
                    "synthetic release resident output reset",
                )?;
            }
        }
        let dispatch = dispatch_program_timed(
            ctx,
            &prepared.program,
            prepared.resident.as_ref(),
            &prepared.inputs,
            &dispatch_config,
        )?;
        let mut timed = dispatch.timed;
        let resident_used = dispatch.resident_used;
        let resident_reset_bytes = if resident_used {
            prepared.output_reset_payload.len() as u64
        } else {
            0
        };
        let (batch_wall_ns, batch_len) = if self.pattern == SyntheticPattern::StringBitmapScatter {
            let batch_len = STRING_BITMAP_RESIDENT_BATCH_SIZE as u64;
            let batch_wall_ns = timed.wall_ns;
            timed.wall_ns = timed.wall_ns.div_ceil(batch_len);
            timed.device_ns = timed.device_ns.map(|ns| ns.div_ceil(batch_len));
            (Some(batch_wall_ns), Some(batch_len))
        } else {
            (None, None)
        };
        let baseline_start = std::time::Instant::now();
        let (baseline_outputs, counted) = match &prepared.baseline {
            SyntheticBaseline::Count { .. } => {
                let cpu_count =
                    synthetic_cpu_count_over_inputs(self.pattern, &prepared.inputs, self.records)
                        .map_err(BenchError::CorrectnessViolation)?;
                (vec![cpu_count.to_le_bytes().to_vec()], Some(cpu_count))
            }
            SyntheticBaseline::StringBitmap {
                pattern_bitmap,
                rule_bitmap,
            } => {
                let baseline_row = encode_u32_words(&string_bitmap_scatter_expected_words(
                    pattern_bitmap,
                    rule_bitmap,
                    self.records,
                ));
                (
                    vec![baseline_row.repeat(STRING_BITMAP_RESIDENT_BATCH_SIZE)],
                    None,
                )
            }
        };
        let baseline_wall = elapsed_ns(baseline_start);
        if let (SyntheticBaseline::Count { expected }, Some(cpu_count)) =
            (&prepared.baseline, counted)
        {
            if cpu_count != *expected {
                return Err(BenchError::CorrectnessViolation(format!(
                    "{} CPU baseline counted {cpu_count} matching row(s) out of the uploaded inputs where the generator expected {expected}. Fix: regenerate the inputs for this pattern.",
                    self.id
                )));
            }
        }
        let output_bytes = timed
            .outputs
            .iter()
            .map(Vec::len)
            .sum::<usize>()
            .div_ceil(batch_len.unwrap_or(1) as usize) as u64;
        let accounting = resident_reset_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            resident_used,
            resident_reset_bytes,
        );
        let logical_bytes_touched = prepared
            .input_bytes_total
            .saturating_add(prepared.logical_output_bytes);
        let mut run = bench_run_from_timed_with_accounting(
            timed,
            prepared.input_bytes_total,
            baseline_outputs,
            baseline_wall,
            self.metric_name,
            self.records,
            logical_bytes_touched,
            accounting,
        )?;
        run.metrics.custom.push(MetricPoint {
            name: "synthetic_resident_buffers".to_string(),
            value: u64::from(resident_used),
        });
        run.metrics.custom.push(MetricPoint {
            name: "synthetic_resident_reset_bytes".to_string(),
            value: resident_reset_bytes,
        });
        if let Some(batch_wall_ns) = batch_wall_ns {
            run.metrics.custom.push(MetricPoint {
                name: "synthetic_resident_batch_wall_ns".to_string(),
                value: batch_wall_ns,
            });
        }
        if let Some(batch_len) = batch_len {
            run.metrics.custom.push(MetricPoint {
                name: "synthetic_resident_batch_len".to_string(),
                value: batch_len,
            });
        }
        match &prepared.baseline {
            SyntheticBaseline::Count { expected } => {
                add_release_alias_metrics(self.pattern, self.records, *expected, &mut run);
            }
            SyntheticBaseline::StringBitmap { .. } => {
                run.metrics.custom.push(MetricPoint {
                    name: "scatter_materialized_words".to_string(),
                    value: u64::from(self.records),
                });
            }
        }
        Ok(run)
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<SyntheticCountPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<SyntheticCountPrepared>()
            .map(|prepared| (prepared.input_bytes_total, prepared.logical_output_bytes))
            .unwrap_or((
                self.records as u64 * pattern_input_count(self.pattern) as u64 * 4,
                synthetic_logical_output_bytes(self.pattern, self.records),
            ))
    }
}
