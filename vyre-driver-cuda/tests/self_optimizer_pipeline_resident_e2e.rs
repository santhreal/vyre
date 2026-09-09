//! End-to-end + scaling bench: persistent-resident pipeline on CUDA.
//!
//! Runs `gpu_optimize` (single encode, persistent buffers, all passes share
//! GPU state) on real CUDA hardware and compares against the foundation CPU pipeline.

#![cfg(all(test, feature = "device-tests"))]

use std::collections::BTreeMap;
use std::thread;
use std::time::Instant;

use vyre::ir::{Expr, Node, Program};
use vyre_driver_cuda::{registered_backend_id, CUDA_BACKEND_ID};
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, ObjectiveMetric, SearchBudget, SemanticExecutionPolicy,
};
use vyre_pass_engine::optimizer::pipeline::gpu_optimize;
use vyre_runtime::RegisteredSemanticExecutor;

fn synthetic_chain_program(n: usize) -> Program {
    let mut entry: Vec<Node> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let value = if i == 0 {
            Expr::mul(Expr::add(Expr::u32(1), Expr::u32(2)), Expr::u32(3))
        } else {
            let prev = format!("v{}", i - 1);
            Expr::mul(Expr::add(Expr::u32(5), Expr::var(prev)), Expr::u32(2))
        };
        entry.push(Node::let_bind(format!("v{i}"), value));
    }
    let last = format!("v{}", n.saturating_sub(1));
    entry.push(Node::store("buf", Expr::u32(0), Expr::var(last)));
    Program::wrapped(
        vec![vyre::ir::BufferDecl::output("buf", 0, vyre::ir::DataType::U32).with_count(1)],
        [1, 1, 1],
        entry,
    )
}

fn synthetic_wide_program(n: usize) -> Program {
    let mut entry: Vec<Node> = Vec::with_capacity(n + 1);
    for i in 0..n {
        let value = Expr::mul(
            Expr::add(
                Expr::u32(((i % 7) + 1) as u32),
                Expr::u32(((i % 13) + 1) as u32),
            ),
            Expr::u32(((i % 5) + 1) as u32),
        );
        entry.push(Node::let_bind(format!("v{i}"), value));
    }
    let last = format!("v{}", n.saturating_sub(1));
    entry.push(Node::store("buf", Expr::u32(0), Expr::var(last)));
    Program::wrapped(
        vec![vyre::ir::BufferDecl::output("buf", 0, vyre::ir::DataType::U32).with_count(1)],
        [1, 1, 1],
        entry,
    )
}

/// Tree shape: each let depends on its parent index `i/2`.
///
/// Diameter is `log2(n)`, so DCE BFS converges in O(log n) iterations even
/// though the program has n lets. Parallel-friendly fixture for level-wave passes.
fn synthetic_tree_program(n: usize) -> Program {
    assert!(n >= 2, "tree fixture needs at least 2 lets");
    let mut entry: Vec<Node> = Vec::with_capacity(n + 1);
    entry.push(Node::let_bind(
        "v0",
        Expr::mul(Expr::add(Expr::u32(1), Expr::u32(2)), Expr::u32(3)),
    ));
    for i in 1..n {
        let parent = format!("v{}", i / 2);
        let value = Expr::add(Expr::var(parent), Expr::u32(((i % 7) + 1) as u32));
        entry.push(Node::let_bind(format!("v{i}"), value));
    }
    let last = format!("v{}", n - 1);
    entry.push(Node::store("buf", Expr::u32(0), Expr::var(last)));
    Program::wrapped(
        vec![vyre::ir::BufferDecl::output("buf", 0, vyre::ir::DataType::U32).with_count(1)],
        [1, 1, 1],
        entry,
    )
}

fn run_cpu_pipeline(p: Program) -> Program {
    use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine::run as cpu_canonicalize;
    use vyre_foundation::optimizer::passes::fusion_cse::dce::dce as cpu_dce;
    let p = cpu_canonicalize(p);
    let p = vyre_foundation::optimizer::optimize(p).expect("registered optimizer must converge");
    cpu_dce(p)
}

fn acquire_cuda_resident_execution() -> (
    crate::harness::CudaBackend,
    RegisteredSemanticExecutor,
    SemanticExecutionPolicy,
) {
    let _ = registered_backend_id();
    let backend = crate::harness::live_backend();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let device = registration.acquire().expect("live CUDA backend");
    let executor = RegisteredSemanticExecutor::new(registration);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        device.device_profile().compile_facts(),
        CompileObjective::minimize_latency().with_bound(ObjectiveMetric::ArtifactBytes, 1_000_000),
        SearchBudget::new(128, 128, 0, 0, 128),
    );
    (backend, executor, policy)
}
#[test]
fn cuda_persistent_pipeline_correctness() {
    let (_backend, executor, policy) = acquire_cuda_resident_execution();

    // let dead = 99
    // let live = 1 + 2     // foldable to 3
    // store buf 0 (3 + live)   // canon swaps to (live + 3)
    let p = Program::wrapped(
        vec![vyre::ir::BufferDecl::output("buf", 0, vyre::ir::DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("dead", Expr::u32(99)),
            Node::let_bind("live", Expr::add(Expr::u32(1), Expr::u32(2))),
            Node::store(
                "buf",
                Expr::u32(0),
                Expr::add(Expr::u32(3), Expr::var("live")),
            ),
        ],
    );

    let out = gpu_optimize(p, &executor, &policy).expect("persistent pipeline runs");
    let body: Vec<Node> = match out.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    };

    assert_eq!(
        body.len(),
        1,
        "all lets dead after const-prop. body={body:?}"
    );
    match &body[0] {
        Node::Store { value, .. } => {
            assert!(
                matches!(value, Expr::LitU32(6)),
                "expected LitU32(6); got {value:?}"
            );
        }
        other => panic!("expected Store; got {other:?}"),
    }
}

#[test]
fn cuda_persistent_pipeline_reuses_static_buffers_on_warm_run() {
    let (backend, executor, policy) = acquire_cuda_resident_execution();
    let p = synthetic_wide_program(1_000);

    backend.reset_telemetry();
    let _ = gpu_optimize(p.clone(), &executor, &policy).expect("cold resident pipeline");
    let cold_h2d = backend.telemetry_snapshot().host_to_device_bytes;
    assert!(
        cold_h2d > 0,
        "Fix: cold CUDA resident optimizer run must report immutable + mutable H2D traffic."
    );

    backend.reset_telemetry();
    let _ = gpu_optimize(p, &executor, &policy).expect("warm resident pipeline");
    let warm_h2d = backend.telemetry_snapshot().host_to_device_bytes;
    assert!(
        warm_h2d > 0,
        "Fix: warm CUDA resident optimizer run must still report mutable scratch H2D traffic."
    );
    assert!(
        warm_h2d < cold_h2d,
        "Fix: warm CUDA resident optimizer run must reuse immutable static buffers; cold_h2d={cold_h2d}, warm_h2d={warm_h2d}."
    );
}

/// Node counts the scaling bench measures.
const SCALING_SIZES: [usize; 7] = [10, 100, 1000, 5000, 10_000, 20_000, 50_000];

/// Program shapes the scaling bench measures.
const SCALING_SHAPES: [(&str, fn(usize) -> Program); 3] = [
    ("chain", synthetic_chain_program),
    ("wide", synthetic_wide_program),
    ("tree", synthetic_tree_program),
];

/// Whether `shape` is measured at `n`.
///
/// Chain diameter grows with `n` and sequential per-source work explodes past
/// a thousand nodes on both CPU and GPU. Wide and tree stay O(1) / O(log n).
fn shape_is_measured_at(shape: &str, n: usize) -> bool {
    !(shape == "chain" && n >= 5000)
}

#[test]
fn cuda_persistent_pipeline_scaling_bench() {
    thread::Builder::new()
        .name("cuda_persistent_pipeline_scaling_bench_worker".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(cuda_persistent_pipeline_scaling_bench_body)
        .expect("scaling bench worker thread must spawn")
        .join()
        .expect("scaling bench worker thread must complete");
}

fn cuda_persistent_pipeline_scaling_bench_body() {
    let (backend, executor, policy) = acquire_cuda_resident_execution();

    println!("\n=== CUDA persistent-pipeline scaling vs CPU ===");
    println!(
        "{:>8} | {:>8} | {:>14} | {:>14} | {:>10} | {:>8} | {:>10} | {:>10} | {:>6} | {:>7} | {:>7} | {:>8}",
        "shape",
        "n",
        "gpu_us",
        "cpu_us",
        "gpu/cpu",
        "launch",
        "h2d_kib",
        "d2h_kib",
        "sync",
        "utilbp",
        "wastebp",
        "denbp"
    );
    println!("{}", "-".repeat(146));

    for &n in &SCALING_SIZES {
        for (shape, build) in SCALING_SHAPES {
            if !shape_is_measured_at(shape, n) {
                continue;
            }
            let p = build(n);
            // Warmup: cache pipeline compile + warm CUDA driver paths.
            let _ = gpu_optimize(p.clone(), &executor, &policy).expect("warmup gpu");
            let _ = run_cpu_pipeline(p.clone());

            backend.reset_telemetry();
            let t_gpu = Instant::now();
            let gpu_out = gpu_optimize(p.clone(), &executor, &policy).expect("gpu pipeline");
            let gpu_us = t_gpu.elapsed().as_micros();
            let telemetry = backend.telemetry_snapshot();

            let t_cpu = Instant::now();
            let cpu_out = run_cpu_pipeline(p);
            let cpu_us = t_cpu.elapsed().as_micros();

            let ratio = if cpu_us == 0 {
                f64::INFINITY
            } else {
                gpu_us as f64 / cpu_us as f64
            };

            println!(
                "{:>8} | {:>8} | {:>14} | {:>14} | {:>10.2}x | {:>8} | {:>10} | {:>10} | {:>6} | {:>7} | {:>7} | {:>8}",
                shape,
                n,
                gpu_us,
                cpu_us,
                ratio,
                telemetry.kernel_launches,
                telemetry.host_to_device_bytes / 1024,
                telemetry.readback_bytes / 1024,
                telemetry.sync_points,
                telemetry.logical_thread_utilization_bps,
                telemetry.logical_thread_waste_bps,
                telemetry.logical_elements_per_thread_slot_bps
            );

            // Residency invariants: verify the resident execution is active and telemetry is coherent.
            assert!(
                telemetry.kernel_launches > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose launch count for {shape}/{n}."
            );
            assert!(
                telemetry.host_to_device_bytes > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose H2D bytes for {shape}/{n}."
            );
            assert!(
                telemetry.readback_bytes > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose final readback bytes for {shape}/{n}."
            );
            assert!(
                telemetry.sync_points > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose synchronization pressure for {shape}/{n}."
            );
            assert!(
                telemetry.sync_points <= 3,
                "Fix: persistent CUDA pipeline must keep resident orchestration sync-collapsed; observed {} sync point(s) for {shape}/{n}, expected <= 3.",
                telemetry.sync_points
            );
            assert!(
                telemetry.logical_thread_utilization_bps > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose non-zero logical thread utilization for {shape}/{n}."
            );
            assert!(
                telemetry.logical_thread_waste_bps <= 10_000,
                "Fix: persistent CUDA pipeline waste telemetry must stay in basis points for {shape}/{n}."
            );
            assert!(
                telemetry.logical_elements_per_thread_slot_bps > 0,
                "Fix: persistent CUDA pipeline scaling evidence must expose logical element density for {shape}/{n}."
            );

            // Correctness parity: both pipelines must produce equivalent output shapes.
            assert_eq!(
                gpu_out.entry().len(),
                cpu_out.entry().len(),
                "Fix: GPU optimizer and CPU optimizer entry node counts must agree for {shape}/{n}"
            );
        }
    }
    println!();
}

/// A resident optimizer run reports the kernels it executed.
///
/// The resident pipeline dispatches through captured CUDA graphs, so every
/// kernel reaches the device through `cuGraphLaunch` rather than
/// `cuLaunchKernel`. Counting only the non-graph launch path left
/// `cuda_kernel_launches` at zero while the device ran seven graph replays per
/// run, which certified zero device work for a pipeline that was executing.
/// The counters are read through the registered backend facet, which is the
/// device the pipeline runs on, and as deltas, so a run that launches nothing
/// cannot satisfy the assertion by inheriting an earlier run's total.
///
/// This does not prove per-kernel occupancy: the graph path records no launch
/// geometry, so `logical_thread_utilization_bps` stays zero.
#[test]
fn cuda_resident_pipeline_reports_graph_dispatched_kernel_launches() {
    let (_backend, executor, policy) = acquire_cuda_resident_execution();
    let registration =
        vyre_driver::backend_registration(CUDA_BACKEND_ID).expect("registered CUDA backend");
    let registered = registration
        .acquire()
        .expect("registered CUDA device facet");
    let counter = |name: &str| -> u64 {
        let metrics: BTreeMap<&str, u64> =
            registered.backend_metric_snapshot().into_iter().collect();
        *metrics
            .get(name)
            .unwrap_or_else(|| panic!("Fix: registered CUDA facet must report `{name}`."))
    };

    let program = synthetic_chain_program(10);
    let _ = gpu_optimize(program.clone(), &executor, &policy).expect("warm resident pipeline");

    let launches_before = counter("cuda_kernel_launches");
    let graphs_before = counter("cuda_graph_launches");
    let _ = gpu_optimize(program, &executor, &policy).expect("measured resident pipeline");
    let launch_delta = counter("cuda_kernel_launches") - launches_before;
    let graph_delta = counter("cuda_graph_launches") - graphs_before;

    assert!(
        graph_delta > 0,
        "Fix: the resident optimizer pipeline must dispatch through captured CUDA graphs; observed {graph_delta} graph launches."
    );
    assert!(
        launch_delta >= graph_delta,
        "Fix: every CUDA graph replay executes at least one captured kernel, so `cuda_kernel_launches` must not fall below `cuda_graph_launches`; observed {launch_delta} kernel launches for {graph_delta} graph launches."
    );
}
