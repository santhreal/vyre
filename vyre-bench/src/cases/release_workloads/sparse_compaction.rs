//! Sparse output compaction count release case: its IR program, candidate flag
//! predicate, and resident batch dispatch path.

use super::registration::{gpu_requirements, RELEASE_SUITES};
use super::resident_batch::{
    batch_metric_points, dispatch_batch_or_single, dispatch_single, BatchPlan,
};
use super::run_assembly::{
    bench_run_from_timed_with_accounting, encode_u32_words, resident_reset_transfer_accounting,
};
use super::synthetic_oracle::mixed_release_index;
use crate::api::case::{
    prepared_as, BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata,
    BenchRequirements, BenchRun, Correctness, DeterminismClass, PerformanceContract, PreparedCase,
    WorkloadClass,
};
use crate::api::resident::{input_bytes_total, ResidentInputPool};
use crate::cases::reference_sample::timed_reference;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub struct SparseOutputCompactionCount;

struct SparseOutputPrepared {
    program: Program,
    inputs: Vec<Vec<u8>>,
    input_bytes_total: u64,
    expected_count: u32,
    resident_batch: Option<ResidentInputPool>,
}

const SPARSE_ITEMS: u32 = 1_048_576;

const SPARSE_RESIDENT_BATCH_SIZE: usize = 16;

impl BenchCase for SparseOutputCompactionCount {
    fn id(&self) -> BenchId {
        BenchId("sparse.compaction.count.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Sparse Output Compaction Count 1M".to_string(),
            description:
                "Sparse hit counting front-end for GPU output compaction over a 1M candidate stream"
                    .to_string(),
            tags: vec![
                "sparse".to_string(),
                "compaction".to_string(),
                "compact".to_string(),
                "append".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Runtime,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-runtime".to_string(),
        }
    }

    fn suites(&self) -> &'static [crate::api::suite::SuiteKind] {
        RELEASE_SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        gpu_requirements(u64::from(SPARSE_ITEMS) * 4)
    }

    fn performance_contract(&self) -> Option<PerformanceContract> {
        Some(PerformanceContract::cpu_sota_100x(
            "sparse output compaction count",
            "vyre-runtime",
            "optimized CPU fired-rule collection over predicate masks",
        ))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        let program = sparse_output_compaction_count_program();
        let mut flags = Vec::with_capacity(SPARSE_ITEMS as usize);
        let mut expected_count = 0u32;
        for index in 0..SPARSE_ITEMS {
            let hit = sparse_compaction_flag(index) != 0;
            expected_count += u32::from(hit);
            flags.push(u32::from(hit));
        }
        let inputs = vec![encode_u32_words(&flags)];
        let input_bytes_total = input_bytes_total(&inputs);
        let resident_batch =
            ResidentInputPool::upload_program_ordered_with_zeroed_outputs_optional(
                ctx,
                &program,
                &inputs,
                SPARSE_RESIDENT_BATCH_SIZE,
                "sparse compaction batch",
            )?;

        Ok(Box::new(SparseOutputPrepared {
            program,
            inputs,
            input_bytes_total,
            expected_count,
            resident_batch,
        }))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<SparseOutputPrepared>()
            .map(|prepared| &prepared.program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared_as::<SparseOutputPrepared>(prepared, "sparse output")?;
        let sample = dispatch_batch_or_single(
            ctx,
            &prepared.program,
            &prepared.inputs,
            prepared.resident_batch.as_ref(),
            &BatchPlan {
                label: "sparse compaction",
                batch_size: SPARSE_RESIDENT_BATCH_SIZE,
                reset_resource: 0,
                reset_resource_kind: "counter",
                reset_payload: &0u32.to_le_bytes(),
                dispatch_config: &ctx.dispatch_config,
            },
            || dispatch_single(ctx, &prepared.program, &prepared.inputs),
        )?;

        let (fired_rules, baseline_wall) = timed_reference(|| {
            let mut fired_rules = Vec::new();
            for index in 0..SPARSE_ITEMS {
                if sparse_compaction_flag(index) != 0 {
                    fired_rules.push(index);
                }
            }
            fired_rules
        });
        let cpu_count = fired_rules.len() as u32;
        if cpu_count != prepared.expected_count {
            return Err(BenchError::CorrectnessViolation(
                "sparse CPU baseline count disagreed with generator expectation".to_string(),
            ));
        }

        let baseline_outputs = vec![cpu_count.to_le_bytes().to_vec()];
        let output_bytes = sample.timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
        let logical_bytes_touched = prepared.input_bytes_total.saturating_add(output_bytes);
        let accounting = resident_reset_transfer_accounting(
            prepared.input_bytes_total,
            output_bytes,
            sample.resident_used,
            sample.reset_bytes,
        );
        let mut custom = batch_metric_points("sparse", &sample);
        let mut run = bench_run_from_timed_with_accounting(
            sample.timed,
            prepared.input_bytes_total,
            baseline_outputs,
            baseline_wall,
            "sparse_items",
            SPARSE_ITEMS,
            logical_bytes_touched,
            accounting,
        )?;
        run.metrics.custom.append(&mut custom);
        Ok(run)
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

fn sparse_output_compaction_count_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out_count", 0, DataType::U32).with_count(1),
            BufferDecl::storage("flags", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(SPARSE_ITEMS),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("idx"), Expr::u32(SPARSE_ITEMS)),
                    Expr::ne(Expr::load("flags", Expr::var("idx")), Expr::u32(0)),
                ),
                vec![Node::let_bind(
                    "_slot",
                    Expr::atomic_add("out_count", Expr::u32(0), Expr::u32(1)),
                )],
            ),
        ],
    )
}

fn sparse_compaction_flag(index: u32) -> u32 {
    let hash = mixed_release_index(index, 18);
    u32::from(index % 97 == 0 || index % 4099 == 17 || hash == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_compaction_dispatch_inputs_match_program_abi() {
        let program = sparse_output_compaction_count_program();
        let input_lengths = [(SPARSE_ITEMS as usize) * 4];

        let plan = vyre_driver::BindingPlan::from_input_lengths(&program, &input_lengths)
            .expect("Fix: sparse compaction release workload inputs must match Program ABI.");
        assert_eq!(plan.input_indices, [1]);
        assert_eq!(
            plan.output_indices,
            [0],
            "Fix: sparse compaction count must expose out_count as an artifact-allocated output, not caller-initialized retained state."
        );
    }
}
