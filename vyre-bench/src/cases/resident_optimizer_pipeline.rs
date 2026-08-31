//! The resident optimizer pipeline against the host optimizer pipeline.
//!
//! The claim this case carries is a speedup: running canonicalization, constant
//! folding, dead-code elimination and algebraic identities as device programs
//! over a resident IR image beats running the same passes on the host. That is
//! a release-path performance claim, so it is a registered case with a declared
//! host baseline and a recorded floor rather than an assertion inside a device
//! test.
//!
//! Both arms optimize the same fixture and both results are dispatched, so the
//! measurement is only admitted when the two pipelines agree on what the
//! program computes.

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::cases::harness::{
    verify_exact, CaseOps, ContractDescription, HarnessCase, WorkloadDescription,
};
use std::collections::BTreeMap;
use std::time::Instant;
use vyre_driver::self_optimizer_bench::{timed_cpu_pipeline_on_oracle_stack, wide_program};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, SemanticExecutionPolicy,
};
use vyre_pass_engine::optimizer::pipeline::gpu_optimize;
use vyre_runtime::RegisteredSemanticExecutor;

/// Independent `let` bindings in the fixture.
///
/// The wide family is where a level-wave pass has work to widen: every binding
/// is available in one level, so the device arm is not bounded by the fixture's
/// depth. A chain fixture measures the host's recursion instead of the device's
/// parallelism.
///
/// The size is part of the recorded ratio. A device stage pays encode, compile,
/// admit and dispatch cost per pipeline stage while the host pass pays per
/// binding, so a small fixture measures the fixed cost: 1000 bindings timed
/// 21.9 ms on the device against 1.84 ms on the host, a ratio of 0.08x.
/// Shrinking the fixture invalidates the recorded floor.
const LET_COUNT: usize = 50_000;

/// Bytes the fixture's single output buffer holds.
const OUTPUT_BYTES: u64 = 4;

/// Artifact ceiling the search budget admits, in bytes.
const ARTIFACT_BYTES_BOUND: u64 = 60_000;

/// Ratio of host pipeline time to device pipeline time the case admits.
///
/// This is a measured number, not a target. At 50 000 bindings the device
/// pipeline timed 307.7 ms p50 against 78.6 ms p50 through the host pipeline
/// over 30 samples, a ratio of 0.26x: the resident pipeline is slower than the
/// host pipeline at this size because each of the four stages re-encodes the
/// whole IR image, compiles an artifact and dispatches it, while the host
/// pipeline walks the tree once per pass. The recorded benchmark artifact
/// carries the device identity the ratio was taken from.
///
/// The floor sits below the measured ratio by a margin that covers run-to-run
/// spread, and what it detects is a regression in the device arm relative to
/// the host arm on the same fixture. A device arm that gets slower, or a host
/// arm that gets faster without the device arm following, turns the case red.
const MEASURED_RATIO_FLOOR: f64 = 0.22;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "release.optimizer.resident_pipeline",
    name: "Resident Optimizer Pipeline",
    summary: "Times the device optimizer pipeline against the host optimizer pipeline",
    tags: &["compute", "optimizer", "resident"],
    contract: Some(ContractDescription::cpu_sota(
        "semantic Program optimization pipeline",
        "vyre-foundation",
        "the registered host optimizer pipeline on an expanded-stack worker",
        MEASURED_RATIO_FLOOR,
    )),
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<ResidentOptimizerPrepared> = CaseOps {
    build: prepare,
    measure,
    verify: verify_exact,
    program: |prepared| Some(&prepared.program),
    fingerprint: None,
    bytes_touched: |_prepared| (0, OUTPUT_BYTES),
};

pub(crate) static CASE: HarnessCase<ResidentOptimizerPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

pub(crate) struct ResidentOptimizerPrepared {
    program: Program,
    policy: SemanticExecutionPolicy,
    executor: RegisteredSemanticExecutor,
}

fn prepare(ctx: &mut BenchContext) -> Result<ResidentOptimizerPrepared, BenchError> {
    let program = wide_program(LET_COUNT);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        ctx.preferred_backend.device_profile().compile_facts(),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, ARTIFACT_BYTES_BOUND),
        SearchBudget::new(128, 128, 0, 0, 128),
    );
    let executor = RegisteredSemanticExecutor::new(ctx.preferred_registration);
    let prepared = ResidentOptimizerPrepared {
        program,
        policy,
        executor,
    };
    // One untimed pass per arm, so neither measurement carries first-dispatch
    // compilation or a cold module cache.
    let _ = device_pipeline(&prepared)?;
    let _ = timed_cpu_pipeline_on_oracle_stack(prepared.program.clone());
    Ok(prepared)
}

/// Optimize the fixture through the device pipeline.
fn device_pipeline(prepared: &ResidentOptimizerPrepared) -> Result<Program, BenchError> {
    gpu_optimize(
        prepared.program.clone(),
        &prepared.executor,
        &prepared.policy,
    )
    .map_err(|error| BenchError::BackendFailed(error.to_string()))
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut ResidentOptimizerPrepared,
) -> Result<BenchRun, BenchError> {
    let started = Instant::now();
    let device_optimized = device_pipeline(prepared)?;
    let device_ns = elapsed_ns(started);

    let host_us = timed_cpu_pipeline_on_oracle_stack(prepared.program.clone());
    let host_ns = u64::try_from(host_us.saturating_mul(1000)).unwrap_or(u64::MAX);
    let host_optimized =
        vyre_driver::self_optimizer_bench::cpu_pipeline_on_oracle_stack(prepared.program.clone());

    let outputs = dispatch(ctx, &device_optimized, "device-optimized")?;
    let baseline_outputs = dispatch(ctx, &host_optimized, "host-optimized")?;

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(device_ns),
            dispatch_ns: None,
            input_bytes: Some(0),
            output_bytes: Some(outputs.iter().map(Vec::len).sum::<usize>() as u64),
            custom: vec![
                MetricPoint {
                    name: "optimizer_pipeline_device_ns".to_string(),
                    value: device_ns,
                },
                MetricPoint {
                    name: "optimizer_pipeline_host_ns".to_string(),
                    value: host_ns,
                },
                MetricPoint {
                    name: "optimizer_pipeline_let_count".to_string(),
                    value: LET_COUNT as u64,
                },
            ],
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(host_ns),
            dispatch_ns: None,
            input_bytes: Some(0),
            output_bytes: Some(baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64),
            ..Default::default()
        }),
        outputs,
        baseline_outputs: Some(baseline_outputs),
    })
}

/// Run one optimized fixture and return its output bytes.
fn dispatch(ctx: &BenchContext, program: &Program, arm: &str) -> Result<Vec<Vec<u8>>, BenchError> {
    let inputs: Vec<Vec<u8>> = Vec::new();
    ctx.preferred_backend
        .dispatch(program, &inputs, &ctx.dispatch_config)
        .map_err(|error| {
            BenchError::BackendFailed(format!("{arm} fixture dispatch failed: {error}"))
        })
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
