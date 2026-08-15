use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchRun, Correctness, DeterminismClass,
};
use crate::api::metric::{elapsed_ns, BenchMetrics};
use crate::cases::harness::{no_program, CaseOps, HarnessCase, WorkloadDescription};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static FLAKY_RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "synthetic.flaky",
    name: "Flaky Synthetic",
    summary: "A case that fluctuates randomly",
    tags: &["synthetic"],
    determinism: DeterminismClass::NonDeterministic,
    custom_suites: &["flaky_test"],
    needs_gpu: false,
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<()> = CaseOps {
    build: |_ctx| Ok(()),
    measure,
    verify,
    program: no_program,
    fingerprint: None,
    bytes_touched: |_prepared| (0, 0),
};

pub(crate) static CASE: HarnessCase<()> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn measure(_ctx: &mut BenchContext, _prepared: &mut ()) -> Result<BenchRun, BenchError> {
    let baseline_start = Instant::now();
    let mut baseline_acc = 0u64;
    for i in 0..4_096u64 {
        baseline_acc = baseline_acc.wrapping_add(black_box(i.rotate_left((i % 31) as u32)));
    }
    black_box(baseline_acc);
    let baseline_wall_ns = elapsed_ns(baseline_start);

    let started = Instant::now();
    let mut acc = 0u64;
    let run_index = FLAKY_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let measured_block = run_index.saturating_sub(1) / 30;
    let iterations = if measured_block % 2 == 0 {
        8_192u64
    } else {
        262_144u64
    };
    for i in 0..iterations {
        acc = acc.wrapping_add(black_box(i.rotate_left((i % 31) as u32)));
    }
    black_box(acc);
    let wall_ns = elapsed_ns(started);

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(wall_ns.max(1)),
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(baseline_wall_ns.max(1)),
            ..Default::default()
        }),
        outputs: vec![],
        baseline_outputs: Some(vec![]),
    })
}

fn verify(run: &BenchRun) -> Result<Correctness, BenchError> {
    run.verify_exact_outputs()
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
