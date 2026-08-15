use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, Correctness, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::suite::SuiteKind;
use crate::cases::harness::{CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::reference_sample::timed_reference;
use rand::{RngExt, SeedableRng};
use vyre_foundation::ir::*;

/// Lanes, and so the element count of both buffers.
const LANES: u32 = 1024;
/// Independent live variables the register allocator has to keep alive at once.
const LIVE_VARIABLES: usize = 100;

const REGISTER_EXHAUSTION_SUITES: &[SuiteKind] =
    &[SuiteKind::Adversarial, SuiteKind::Deep, SuiteKind::Release];

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "adversarial.register_exhaustion.u32_1024",
    name: "Register Exhaustion",
    summary: "Generates a highly-nested set of independent live variables to stress-test register allocators",
    tags: &["adversarial", "compiler"],
    layer: BenchLayer::Backend,
    workload: WorkloadClass::Adversarial,
    suites: REGISTER_EXHAUSTION_SUITES,
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<RegisterExhaustionPrepared> = CaseOps {
    build: prepare,
    measure,
    verify,
    program: |prepared| Some(&prepared.program),
    fingerprint: None,
    bytes_touched: |prepared| crate::api::case::static_program_bytes_touched(&prepared.program),
};

pub(crate) static CASE: HarnessCase<RegisterExhaustionPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

pub(crate) struct RegisterExhaustionPrepared {
    program: Program,
    inputs: [Vec<u8>; 1],
}

fn prepare(_ctx: &mut BenchContext) -> Result<RegisterExhaustionPrepared, BenchError> {
    let mut body = Vec::with_capacity(LIVE_VARIABLES + 3);
    body.push(Node::let_bind("tid", Expr::gid_x()));

    // The independent live variables.
    for index in 0..LIVE_VARIABLES {
        body.push(Node::let_bind(
            format!("v{index}"),
            Expr::add(Expr::var("tid"), Expr::u32(index as u32)),
        ));
    }

    // A mixing loop so none of them is dead before the reduce below.
    let mut loop_body = Vec::new();
    for index in 0..LIVE_VARIABLES {
        let next = (index + 1) % LIVE_VARIABLES;
        loop_body.push(Node::assign(
            format!("v{index}"),
            Expr::add(
                Expr::var(format!("v{index}")),
                Expr::var(format!("v{next}")),
            ),
        ));
    }

    body.push(Node::Loop {
        var: "iter".into(),
        from: Expr::u32(0),
        to: Expr::u32(10),
        body: loop_body,
    });

    // A reduce tree over all of them, so none is eliminated as dead.
    let mut reduce_expr = Expr::var("v0");
    for index in 1..LIVE_VARIABLES {
        reduce_expr = Expr::add(reduce_expr, Expr::var(format!("v{index}")));
    }

    body.push(Node::store("out", Expr::var("tid"), reduce_expr));

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(LANES),
            BufferDecl::output("out", 1, DataType::U32).with_count(LANES),
        ],
        [256, 1, 1],
        body,
    );

    // Seeded, so the bytes are the same every sample. Generated once here rather
    // than re-filled per sample inside the measured loop.
    let mut input = vec![0u8; LANES as usize * 4];
    rand::rngs::StdRng::seed_from_u64(1337).fill(input.as_mut_slice());

    Ok(RegisterExhaustionPrepared {
        program,
        inputs: [input],
    })
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut RegisterExhaustionPrepared,
) -> Result<BenchRun, BenchError> {
    let program = &prepared.program;
    let input = &prepared.inputs[0];

    let timed = ctx
        .dispatch_timed(
            program,
            &prepared.inputs,
            &vyre_driver::DispatchConfig::default(),
        )
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;

    let (baseline, elapsed_ref) =
        timed_reference(|| cpu_register_exhaustion_outputs(LANES as usize));

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(timed.wall_ns),
            dispatch_ns: timed.device_ns,
            input_bytes: Some(input.len() as u64),
            output_bytes: Some(timed.outputs.iter().map(Vec::len).sum::<usize>() as u64),
            bytes_read: Some(input.len() as u64),
            bytes_written: Some(timed.outputs.iter().map(Vec::len).sum::<usize>() as u64),
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(elapsed_ref),
            ..Default::default()
        }),
        outputs: timed.outputs,
        baseline_outputs: Some(vec![baseline]),
    })
}

fn verify(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

fn cpu_register_exhaustion_outputs(lanes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(lanes * 4);
    for tid in 0..lanes {
        let tid = tid as u32;
        let mut values = [0u32; 100];
        for (i, value) in values.iter_mut().enumerate() {
            *value = tid.wrapping_add(i as u32);
        }
        for _ in 0..10 {
            for i in 0..100 {
                let next = (i + 1) % 100;
                values[i] = values[i].wrapping_add(values[next]);
            }
        }
        let reduced = values
            .iter()
            .copied()
            .fold(0u32, |acc, value| acc.wrapping_add(value));
        out.extend_from_slice(&reduced.to_le_bytes());
    }
    out
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
