use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun, Correctness};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::ResidentInputSet;
use crate::cases::harness::{CaseOps, ContractDescription, HarnessCase, WorkloadDescription};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "foundation.optimizer.impact",
    name: "Optimizer Impact Analysis",
    summary: "Measures GPU speedup from CSE and constant folding",
    tags: &["compute", "optimizer"],
    contract: Some(ContractDescription {
        primitive: "canonical semantic Program optimization",
        baseline_crate: "vyre-bench",
        baseline_name: "the same Program submitted without semantic optimization",
        min_speedup_x: 1.0,
    }),
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<OptimizerImpactPrepared> = CaseOps {
    build: prepare,
    measure,
    verify,
    program: |prepared| Some(&prepared.program),
    fingerprint: None,
    bytes_touched: |_prepared| (BYTES_IN * 2, BYTES_IN),
};

pub(crate) static CASE: HarnessCase<OptimizerImpactPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

/// Bytes of one input buffer, and of the single output buffer.
const BYTES_IN: u64 = 1_000_000 * 4;

pub(crate) struct OptimizerImpactPrepared {
    program: Program,
    optimized_program: Program,
    inputs: Vec<Vec<u8>>,
    resident: Option<ResidentInputSet>,
    optimizer_input_nodes: u64,
    optimizer_output_nodes: u64,
}

fn prepare(ctx: &mut BenchContext) -> Result<OptimizerImpactPrepared, BenchError> {
    let program = redundant_program();
    let optimized_program = vyre_foundation::optimizer::optimize(program.clone())
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let inputs = input_buffers();
    let resident = ResidentInputSet::upload_program_ordered_with_zeroed_outputs_optional(
        ctx,
        &program,
        &inputs,
        "optimizer impact",
    )?;
    if matches!(ctx.preferred_backend.id(), "cuda" | "wgpu") && resident.is_none() {
        return Err(BenchError::BackendFailed(format!(
            "{} lacks resident buffer allocation required to measure optimizer execution without host-transfer noise",
            ctx.preferred_backend.id()
        )));
    }
    let optimizer_input_nodes = program.stats().node_count as u64;
    let optimizer_output_nodes = optimized_program.stats().node_count as u64;

    Ok(OptimizerImpactPrepared {
        program,
        optimized_program,
        inputs,
        resident,
        optimizer_input_nodes,
        optimizer_output_nodes,
    })
}

fn input_buffers() -> Vec<Vec<u8>> {
    let size = 1_000_000;
    let mut a_bytes = vec![0u8; size * 4];
    let mut b_bytes = vec![0u8; size * 4];
    for i in 0..size {
        let a_val = (i % 257) as f32;
        let b_val = (i % 131) as f32;
        a_bytes[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        b_bytes[i * 4..i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }
    vec![a_bytes, b_bytes]
}

/// A program with four identical subexpressions, two foldable constants, a
/// bit-or with zero and a bit-and with zero, so CSE, constant folding and the
/// strength-reduction peephole each have something to remove.
fn redundant_program() -> Program {
    let size = 1_000_000;

    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(size as u32),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(size as u32),
            BufferDecl::output("out", 2, DataType::F32).with_count(size as u32),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(size)),
                vec![
                    // Highly redundant program: CSE and Constant Folding opportunity
                    Node::let_bind("val_a", Expr::load("a", Expr::var("idx"))),
                    Node::let_bind("val_b", Expr::load("b", Expr::var("idx"))),
                    Node::let_bind("t1", Expr::add(Expr::var("val_a"), Expr::var("val_b"))),
                    Node::let_bind("t2", Expr::add(Expr::var("val_a"), Expr::var("val_b"))),
                    Node::let_bind("t3", Expr::add(Expr::var("val_a"), Expr::var("val_b"))),
                    Node::let_bind("t4", Expr::add(Expr::var("val_a"), Expr::var("val_b"))),
                    // DSL target: BitOr with 0 (should be eliminated by StrengthReduce peephole)
                    Node::let_bind("idx_bitor", Expr::bitor(Expr::var("idx"), Expr::u32(0))),
                    // DSL target: BitAnd with 0 (should become 0)
                    Node::let_bind("zero_mask", Expr::bitand(Expr::var("idx"), Expr::u32(0))),
                    Node::let_bind("c1", Expr::add(Expr::f32(1.0), Expr::f32(2.0))),
                    Node::let_bind("c2", Expr::mul(Expr::f32(1.0), Expr::f32(2.0))),
                    Node::let_bind("sum1", Expr::add(Expr::var("t1"), Expr::var("t2"))),
                    Node::let_bind("sum2", Expr::add(Expr::var("t3"), Expr::var("t4"))),
                    Node::let_bind(
                        "final",
                        Expr::add(
                            Expr::add(Expr::var("sum1"), Expr::var("sum2")),
                            Expr::add(Expr::var("c1"), Expr::var("c2")),
                        ),
                    ),
                    // Use the dummy vars so they aren't DCE'd before we can test strength reduction
                    Node::let_bind(
                        "final_masked",
                        Expr::add(
                            Expr::var("final"),
                            Expr::cast(DataType::F32, Expr::var("zero_mask")),
                        ),
                    ),
                    Node::store("out", Expr::var("idx_bitor"), Expr::var("final_masked")),
                ],
            ),
        ],
    )
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut OptimizerImpactPrepared,
) -> Result<BenchRun, BenchError> {
    let timed_unopt = dispatch_variant(ctx, &prepared.program, prepared)?;
    let elapsed_unopt = timed_unopt.wall_ns;
    let unopt_dispatch_ns = timed_unopt.device_ns;
    let unopt_outputs = timed_unopt.outputs;

    let timed_opt = dispatch_variant(ctx, &prepared.optimized_program, prepared)?;
    let elapsed_opt = timed_opt.wall_ns;
    let opt_dispatch_ns = timed_opt.device_ns;
    let outputs = timed_opt.outputs;
    let optimizer_nodes_eliminated = prepared
        .optimizer_input_nodes
        .saturating_sub(prepared.optimizer_output_nodes);

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(elapsed_opt),
            dispatch_ns: opt_dispatch_ns,
            input_bytes: Some(prepared.inputs.iter().map(Vec::len).sum::<usize>() as u64),
            output_bytes: Some(outputs.iter().map(Vec::len).sum::<usize>() as u64),
            custom: vec![
                MetricPoint {
                    name: "optimizer_input_nodes".to_string(),
                    value: prepared.optimizer_input_nodes,
                },
                MetricPoint {
                    name: "optimizer_output_nodes".to_string(),
                    value: prepared.optimizer_output_nodes,
                },
                MetricPoint {
                    name: "optimizer_nodes_eliminated".to_string(),
                    value: optimizer_nodes_eliminated,
                },
            ],
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(elapsed_unopt),
            dispatch_ns: unopt_dispatch_ns,
            input_bytes: Some(prepared.inputs.iter().map(Vec::len).sum::<usize>() as u64),
            output_bytes: Some(unopt_outputs.iter().map(Vec::len).sum::<usize>() as u64),
            ..Default::default()
        }),
        outputs,
        baseline_outputs: Some(unopt_outputs),
    })
}

fn dispatch_variant(
    ctx: &BenchContext,
    program: &Program,
    prepared: &OptimizerImpactPrepared,
) -> Result<vyre_driver::TimedDispatchResult, BenchError> {
    if let Some(resident) = &prepared.resident {
        return resident
            .dispatch_execution_timed(ctx, program, &ctx.dispatch_config)
            .map_err(|error| BenchError::BackendFailed(error.to_string()));
    }
    ctx.dispatch_timed(program, &prepared.inputs, &ctx.dispatch_config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))
}

fn verify(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::metric::MetricStats;
    use crate::runner::evaluate_contract;
    use std::collections::BTreeMap;

    fn stats(p50: u64) -> MetricStats {
        MetricStats::point(p50, p50 as f64, 0.0, 30)
    }

    #[test]
    fn harness_case_keeps_its_registered_identity_and_contract() {
        assert_eq!(CASE.id().0, "foundation.optimizer.impact");
        let contract = CASE
            .performance_contract()
            .expect("Fix: optimizer impact workload must declare an explicit performance contract");

        assert_eq!(
            contract.primitive,
            "canonical semantic Program optimization"
        );
        assert_eq!(contract.baselines.len(), 1);
        let baseline = &contract.baselines[0];
        assert_eq!(baseline.crate_name, "vyre-bench");
        assert_eq!(
            baseline.name,
            "the same Program submitted without semantic optimization"
        );
        assert_eq!(baseline.min_speedup_x, 1.0);

        for backend in ["cuda", "wgpu"] {
            assert!(
                baseline.backend_ids.iter().any(|b| b == backend),
                "Fix: optimizer impact performance contract must apply to `{backend}`"
            );
        }
    }

    #[test]
    fn contract_evaluation_passes_on_speedup_and_fails_on_regression() {
        let contract = CASE
            .performance_contract()
            .expect("Fix: optimizer impact workload must declare an explicit performance contract");

        for backend in ["cuda", "wgpu"] {
            let mut metrics = BTreeMap::new();
            metrics.insert("wall_ns".to_string(), stats(10_000_000));
            metrics.insert("baseline_wall_ns".to_string(), stats(11_000_000));

            let evaluation = evaluate_contract(&contract, &metrics, backend);
            assert!(
                evaluation.contract_passed,
                "Fix: contract must pass when optimized runtime achieves >= 1.0x speedup on `{backend}`; violations: {:?}",
                evaluation.violations
            );
            assert!(
                evaluation.speedup_x.is_some_and(|s| s >= 1.0),
                "Fix: speedup must be at least 1.0x on `{backend}`"
            );

            // Regressed case: optimized wall_ns is slower than baseline_wall_ns
            let mut regressed_metrics = BTreeMap::new();
            regressed_metrics.insert("wall_ns".to_string(), stats(12_000_000));
            regressed_metrics.insert("baseline_wall_ns".to_string(), stats(10_000_000));

            let regressed_eval = evaluate_contract(&contract, &regressed_metrics, backend);
            assert!(
                !regressed_eval.contract_passed,
                "Fix: contract must fail on `{backend}` when optimized runtime is slower than baseline"
            );
        }
    }
}
