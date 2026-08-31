//! Per-phase dispatch-overhead breakdown.
//!
//! Measures end-to-end host-to-host latency on the simplest possible CUDA
//! dispatch and attributes the time to four phases:
//!
//!   1. **`backend_acquire_ns`**  -  CudaBackend::acquire (one-time, includes
//!      device probe + module-cache init + transient-pool bootstrap).
//!   2. **`compile_ns`**  -  first-call PTX compilation + module load (cold).
//!   3. **`compile_warm_ns`**  -  second-call dispatch with the module cached
//!      (warm  -  this is the per-dispatch floor for repeat-dispatch workloads).
//!   4. **`steady_state_ns`**  -  dispatch #3 onward, the per-dispatch floor we
//!      care about for the latency-bound corner of the bench.
//!
//! Outputs the breakdown via `println!` so cargo test --nocapture captures
//! the numbers. Asserts conservative ceilings so the test fails when latency
//! regresses past obviously-bad thresholds.
//!
//! The second case measures a strided workgroup-memory tile with and without
//! the selected bank-conflict mitigation and reports the classified conflict
//! count beside both times.

#![cfg(feature = "device-tests")]

use std::num::NonZeroU32;
use std::time::Instant;

mod harness;
use harness::no_op_program;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program};
use vyre_lower::analyses::{analyze_bank_conflict, BankConflictKind};
use vyre_lower::descriptor_builder::{binop, body, descriptor, lit, op, shared_rw};
use vyre_lower::{KernelOpKind, LiteralValue};

fn assert_noop_output(outputs: &[Vec<u8>], phase: &str) {
    let expected = 0u32.to_le_bytes();
    assert_eq!(
        outputs.len(),
        1,
        "{phase}: CUDA no-op dispatch must return exactly one output buffer"
    );
    assert_eq!(
        outputs[0].as_slice(),
        expected.as_slice(),
        "{phase}: CUDA no-op dispatch must write the expected zero word"
    );
}

#[test]
fn dispatch_overhead_breakdown_reports_per_phase_latency() {
    let backend_t0 = Instant::now();
    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let backend_acquire_ns = backend_t0.elapsed().as_nanos();

    let program = no_op_program();
    let inputs: Vec<Vec<u8>> = vec![vec![0u8; 4]];
    let config = DispatchConfig::default();

    // Cold dispatch: includes PTX compile + module load + first-launch overhead.
    let cold_t0 = Instant::now();
    let cold_outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("Fix: cuda no-op dispatch must succeed");
    let cold_ns = cold_t0.elapsed().as_nanos();
    assert_noop_output(&cold_outputs, "cold dispatch");

    // Warm dispatch (module cached, transient pool warm): the per-dispatch
    // floor for repeat-dispatch workloads.
    let warm_t0 = Instant::now();
    let warm_outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("warm dispatch must succeed");
    let warm_ns = warm_t0.elapsed().as_nanos();
    assert_noop_output(&warm_outputs, "warm dispatch");

    // Steady state: average over 100 dispatches after warmup. Per-dispatch
    // wall-clock is what the latency-bound corner of the bench cares about.
    const STEADY_RUNS: u32 = 100;
    let steady_t0 = Instant::now();
    for _ in 0..STEADY_RUNS {
        let steady_outputs = backend
            .dispatch(&program, &inputs, &config)
            .expect("steady-state dispatch must succeed");
        assert_noop_output(&steady_outputs, "steady-state dispatch");
    }
    let steady_total_ns = steady_t0.elapsed().as_nanos();
    let steady_per_dispatch_ns = steady_total_ns / u128::from(STEADY_RUNS);

    println!();
    println!("=== CUDA dispatch overhead breakdown ===");
    println!("backend_acquire_ns           {backend_acquire_ns:>12}  (one-time)");
    println!("cold_first_dispatch_ns       {cold_ns:>12}  (incl. PTX compile + module load)");
    println!("warm_second_dispatch_ns      {warm_ns:>12}  (module cached)");
    println!("steady_state_per_dispatch_ns {steady_per_dispatch_ns:>12}  ({STEADY_RUNS}-run avg)");
    println!("===");

    // Conservative ceilings  -  fail the test if latency regresses past these.
    // The numbers are the headline budget for the latency-bound corner.
    assert!(
        backend_acquire_ns < 5_000_000_000, // 5 seconds
        "backend acquire must complete in under 5s; observed {backend_acquire_ns}ns. \
         A regression here means PTX target probing or device init has broken."
    );
    assert!(
        cold_ns < 500_000_000, // 500 ms
        "cold first dispatch must complete in under 500ms; observed {cold_ns}ns. \
         A regression here means PTX compile or module load is broken."
    );
    assert!(
        warm_ns < 50_000_000, // 50 ms
        "warm dispatch must complete in under 50ms; observed {warm_ns}ns. \
         A regression here means module cache lookup or transient-pool reuse is broken."
    );
    assert!(
        steady_per_dispatch_ns < 10_000_000, // 10 ms
        "steady-state per-dispatch must complete in under 10ms; observed \
         {steady_per_dispatch_ns}ns. A regression here means the dispatch hot path picked \
         up a per-call allocation, lock contention, or readback stall."
    );
}

/// Rows of a tile staged in workgroup memory, one row per thread.
///
/// Row length equals the bank count, so lane `t` of a warp addresses element
/// `t * 32 + column`: every lane of the warp lands on the same bank and the
/// classifier states a 32-way conflict. Each thread writes its whole row, the
/// workgroup barriers, and each thread sums its row back, so the kernel is
/// shared traffic and one output word.
///
/// `mask_the_index` wraps every shared index in `& (extent - 1)`. The mask is
/// the identity for this launch geometry, because the largest index a thread
/// forms is `extent - 1`, and the classifier states no stride through a bitwise
/// and. The masked program is therefore the same kernel with the mitigation
/// unreachable, which is what a before measurement has to be.
fn row_tile_program(threads: u32, mask_the_index: bool) -> Program {
    let row_length = 32_u32;
    let extent = threads * row_length;
    let index = |column: u32| {
        let element = Expr::add(
            Expr::mul(Expr::gid_x(), Expr::u32(row_length)),
            Expr::u32(column),
        );
        if mask_the_index {
            Expr::bitand(element, Expr::u32(extent - 1))
        } else {
            element
        }
    };

    let mut nodes: Vec<Node> = Vec::new();
    for column in 0..row_length {
        nodes.push(Node::store(
            "tile",
            index(column),
            Expr::add(Expr::gid_x(), Expr::u32(column)),
        ));
    }
    nodes.push(Node::Barrier {
        ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
    });
    let mut sum = Expr::load("tile", index(0));
    for column in 1..row_length {
        sum = Expr::add(sum, Expr::load("tile", index(column)));
    }
    nodes.push(Node::store("out", Expr::gid_x(), sum));

    Program::wrapped(
        vec![
            BufferDecl::workgroup("tile", extent, DataType::U32),
            BufferDecl::output("out", 0, DataType::U32).with_count(threads),
        ],
        [threads, 1, 1],
        nodes,
    )
}

/// The classification for the row-tile access pattern, as the neutral analysis
/// states it for an equivalent descriptor.
///
/// One scalar store and one scalar load per column, both at stride
/// `row_length`, so the report carries one site per access and every site
/// classifies the same way.
fn classified_row_tile_conflicts(row_length: u32) -> Vec<BankConflictKind> {
    let banks = NonZeroU32::new(32).expect("Fix: 32 is not zero.");
    let descriptor = descriptor("row_tile")
        .slot(shared_rw(0, DataType::U32, row_length * 32, "tile"))
        .dispatch(32, 1, 1)
        .body(
            body()
                .op(op(KernelOpKind::LocalInvocationId, [0], 0))
                .op(lit(0, 1))
                .op(binop(BinOp::Mul, 0, 1, 2))
                .op(op(KernelOpKind::LoadShared, [0, 2], 3))
                .literal(LiteralValue::U32(row_length)),
        )
        .build();
    analyze_bank_conflict(&descriptor, banks)
        .sites
        .iter()
        .map(|site| site.conflict)
        .collect()
}

/// Record the classified conflict count and the measured device time of a
/// strided tile kernel with and without the mitigation applied.
///
/// The two programs are the same kernel: the masked one reaches the same
/// elements through an index the classifier cannot state a stride for, so the
/// emitter has nothing to permute. Both dispatches must therefore agree word
/// for word, and the emitted text of each is checked so a constant fold that
/// removed the mask would fail here instead of turning the baseline into a
/// second copy of the mitigated kernel.
#[test]
fn a_strided_tile_kernel_records_its_conflict_count_and_time_before_and_after() {
    const THREADS: u32 = 128;
    const ROW_LENGTH: u32 = 32;
    const RUNS: u32 = 200;

    let conflicts = classified_row_tile_conflicts(ROW_LENGTH);
    let thirty_two_way = conflicts
        .iter()
        .filter(|conflict| matches!(conflict, BankConflictKind::Conflict { way_count: 32 }))
        .count();
    assert_eq!(
        thirty_two_way,
        conflicts.len(),
        "Fix: a row length equal to the bank count is a 32-way conflict at \
         every site; classified {conflicts:?}"
    );

    let mitigated = row_tile_program(THREADS, false);
    let baseline = row_tile_program(THREADS, true);
    let config = DispatchConfig::default();

    let mitigated_text = vyre_driver_cuda::codegen::program_to_ptx(&mitigated, &config)
        .expect("Fix: the row-tile kernel must lower to PTX.");
    let baseline_text = vyre_driver_cuda::codegen::program_to_ptx(&baseline, &config)
        .expect("Fix: the masked row-tile kernel must lower to PTX.");
    assert!(
        mitigated_text.contains(", 33;"),
        "Fix: the mitigated kernel scales its row by the padded row length."
    );
    assert!(
        !baseline_text.contains(", 33;"),
        "Fix: the masked kernel states no stride, so nothing is permuted in \
         it. A permutation here means the mask was folded away and the before \
         measurement is the after measurement."
    );

    let backend = CudaBackend::acquire()
        .expect("Fix: CUDA backend acquisition must succeed on the GPU-required test host.");
    let inputs: Vec<Vec<u8>> = Vec::new();

    let measure = |program: &Program, what: &str| -> (u128, Vec<Vec<u8>>) {
        let first = backend
            .dispatch(program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: {what} row-tile dispatch failed: {error}"));
        let start = Instant::now();
        for _ in 0..RUNS {
            backend
                .dispatch(program, &inputs, &config)
                .unwrap_or_else(|error| panic!("Fix: {what} row-tile dispatch failed: {error}"));
        }
        (start.elapsed().as_nanos() / u128::from(RUNS), first)
    };
    let (baseline_ns, baseline_outputs) = measure(&baseline, "unmitigated");
    let (mitigated_ns, mitigated_outputs) = measure(&mitigated, "mitigated");

    assert_eq!(
        mitigated_outputs, baseline_outputs,
        "Fix: a shared index permutation must be one-to-one inside the extent \
         it declares, so the permuted kernel computes what the unpermuted one \
         computes."
    );

    println!();
    println!("=== strided tile shared-memory mitigation ===");
    println!("threads                      {THREADS:>12}");
    println!("row_length                   {ROW_LENGTH:>12}  (elements, = bank count)");
    println!("classified_sites             {:>12}", conflicts.len());
    println!("classified_32_way_sites      {thirty_two_way:>12}");
    println!("unmitigated_per_dispatch_ns  {baseline_ns:>12}  ({RUNS}-run avg)");
    println!("mitigated_per_dispatch_ns    {mitigated_ns:>12}  ({RUNS}-run avg)");
    println!("===");
}
