//! Backend differential gate for `branch_collapse`.
//!
//! A rewrite that changes program semantics is invisible to any single-backend
//! test, because the wrong answer is the only answer that backend produces.
//! Every test here runs the SAME `Program` on two independent executors and
//! asserts exact element-by-element equality of the output buffer, plus
//! equality with an independent host oracle so the two backends cannot agree
//! on a wrong value and pass vacuously.
//!
//! The executors are intentionally independent:
//!
//! - `CpuRefBackend` interprets the semantic `Program`.
//! - `CudaBackend` uses `lower_verified`, pure PTX emission, and device execution.
//!
//! Exact agreement with the host oracle proves that canonical descriptor cleanup
//! preserves observable semantics across constant and value-dependent guards.
//! Descriptor validity and cleanup fixpoint behavior are covered by
//! `branch_collapse_nested_assign_miscompile`.
//!
//! CUDA is REQUIRED, not optional. This host has an RTX 5090; a skip would
//! silently retire the only gate that can see a backend value divergence, so
//! `CudaBackend::acquire()` failing is a test failure.

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_cuda::CudaBackend;
use vyre_driver_reference::CpuRefBackend;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

mod shapes;
use shapes::*;

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Run `program` on both backends and assert the output buffer is
/// element-by-element identical on both AND equal to `expected`.
///
/// `expected` is an independent host oracle. Without it, two backends that
/// shared a wrong answer would pass; with it, all three have to agree.
fn assert_backends_agree(program: &Program, src: &[u32], expected: &[u32], case: &str) {
    let config = DispatchConfig::default();
    // `dispatch` takes one host buffer per READ declaration. Every shape here
    // declares exactly `src` (read, slot 0) and `out` (output, slot 1), so the
    // input list is `src` alone; passing a placeholder for `out` is rejected by
    // CUDA with "expected 1 input buffer(s) ... but received 2".
    let inputs = vec![u32_bytes(src)];

    let cpu_raw = CpuRefBackend
        .dispatch(program, &inputs, &config)
        .unwrap_or_else(|e| panic!("[{case}] CpuRefBackend dispatch must succeed: {e}"));

    let cuda = CudaBackend::acquire().unwrap_or_else(|e| {
        panic!(
            "[{case}] CudaBackend::acquire must succeed on this RTX 5090 host \
             (driver 570.211.01, CUDA 12.8). A CUDA skip retires the only gate \
             that can observe a CPU-versus-GPU divergence, so this is a \
             configuration failure, not an acceptable result: {e}"
        )
    });
    // `CudaBackend` is the concrete device handle and deliberately does not
    // implement `VyreBackend` (the trait object wrapper is
    // `CudaBackendRegistration`), so probe the device itself: a compute
    // capability is read off the acquired device and cannot be produced by a
    // host-side stand-in. This is what rules out the differential silently
    // degrading into a reference backend compared against itself.
    let (cc_major, cc_minor) = cuda.compute_capability();
    assert!(
        cc_major >= 7,
        "[{case}] expected a real CUDA device of compute capability 7.0 or \
         newer, got {cc_major}.{cc_minor}"
    );
    let cuda_raw = cuda
        .dispatch(program, &inputs, &config)
        .unwrap_or_else(|e| panic!("[{case}] CUDA dispatch must succeed: {e}"));

    // `dispatch` returns one buffer per OUTPUT declaration, not one per
    // declaration, so `out` (binding slot 1, the only output) is at index 0.
    let cpu = bytes_to_u32(&cpu_raw[0]);
    let gpu = bytes_to_u32(&cuda_raw[0]);

    assert_eq!(
        cpu.len(),
        expected.len(),
        "[{case}] reference output length must match the oracle"
    );
    assert_eq!(
        cpu, expected,
        "[{case}] CpuRefBackend disagrees with the host oracle"
    );
    assert_eq!(
        gpu, expected,
        "[{case}] CUDA disagrees with the host oracle; CPU produced {cpu:?}"
    );
    assert_eq!(
        cpu, gpu,
        "[{case}] CpuRefBackend and CUDA disagree element-by-element"
    );
}

/// The reported repro shape, end to end on both backends.
///
/// Locks out: a `branch_collapse` that inlines the always-true `if (end == 0)`
/// arm without merging that arm's literal pool into the parent's. Pre-fix this
/// program did not even reach a backend comparison; the pass emitted a
/// descriptor carrying `LiteralPoolOutOfRange { pool_idx: 2, pool_size: 2 }`
/// (and idx 3 and 4 likewise, 6 violations total).
#[test]
fn repro_shape_matches_across_backends() {
    let src = vec![0, 5, 0, 7, 9, 0, 3, 0];
    let expected = repro_oracle(&src);
    assert_backends_agree(&repro_program(REPRO_N), &src, &expected, "repro");
}

/// Locks out: treating a variable as still holding its initializer after a
/// `Node::assign` inside a LOOP body. `acc` is let-bound to 0, incremented
/// inside `loop_for`, and the guard `acc == 0` after the loop must be decided
/// at runtime, not folded to true.
#[test]
fn assign_inside_loop_body_matches_across_backends() {
    let src = vec![0, 1, 1, 0, 2, 0, 3, 1];
    let expected = loop_assign_oracle(&src);
    assert_backends_agree(
        &loop_assign_program(REPRO_N),
        &src,
        &expected,
        "assign-in-loop",
    );
}

/// Locks out: treating a variable as still holding its initializer after a
/// `Node::assign` inside an ELSE branch. `collect_carrier_names` unions the
/// then and otherwise arms; a collapse that only inspected the then arm would
/// miss this mutation entirely.
#[test]
fn assign_inside_else_branch_matches_across_backends() {
    let src = vec![0, 4, 0, 6, 0, 8, 1, 0];
    let expected = else_assign_oracle(&src);
    assert_backends_agree(
        &else_assign_program(REPRO_N),
        &src,
        &expected,
        "assign-in-else",
    );
}

/// Locks out: treating a variable as still holding its initializer after a
/// `Node::assign` inside a nested `Node::Region`. `Region` carries no
/// execution semantics but DOES carry a body, and the region-exit phi-merge
/// publishes the in-region value back to the parent.
#[test]
fn assign_inside_nested_region_matches_across_backends() {
    let src = vec![1, 0, 3, 0, 5, 0, 7, 0];
    let expected = region_assign_oracle(&src);
    assert_backends_agree(
        &region_assign_program(REPRO_N),
        &src,
        &expected,
        "assign-in-region",
    );
}

/// Locks out: the join case. `flag` is let-bound to a literal, reassigned in
/// exactly one arm of an if/else, and READ AFTER the join. The read must
/// observe the merged value, so no guard on it may be folded from the
/// pre-branch literal.
#[test]
fn assign_in_one_branch_read_after_join_matches_across_backends() {
    let src = vec![0, 9, 0, 9, 1, 1, 0, 0];
    let expected = join_oracle(&src);
    assert_backends_agree(&join_program(REPRO_N), &src, &expected, "one-branch-join");
}

/// The downstream tokenizer's shape, reported by Main from
/// `exatok/src/gpu_select.rs` (`rank_acc` at line 282, `index_acc` at 317):
/// a sentinel let-bound to a literal, mutated by a SELF-REFERENCING `min`
/// inside an `if_then` inside a `loop_for`, with an outer guard reading the
/// same variable.
///
/// Locks out: a regression that starts collapsing this guard. It is not
/// collapsible today and this test pins that by behaviour rather than by
/// argument, because the argument depends on what `lower` happens to emit.
#[test]
fn self_referencing_min_sentinel_matches_across_backends() {
    let src = vec![7, 3, 9, 1, 8, 4, 6, 2];
    let expected = sentinel_min_oracle(&src);
    assert_backends_agree(
        &sentinel_min_program(REPRO_N),
        &src,
        &expected,
        "sentinel-min",
    );
}

/// A program whose guard IS legitimately constant must still be collapsed and
/// still compute the right answer. Without this, "decline to collapse
/// everything" would pass the whole differential suite while silently
/// retiring the optimization.
#[test]
fn legitimately_constant_guard_still_collapses_and_matches() {
    let src = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let expected: Vec<u32> = src.iter().map(|v| v + 1).collect();
    let program = Program::wrapped(
        vec![
            BufferDecl::read("src", 0, DataType::U32).with_count(REPRO_N),
            BufferDecl::output("out", 1, DataType::U32).with_count(REPRO_N),
        ],
        [REPRO_N, 1, 1],
        vec![Node::if_then(
            Expr::lt(Expr::InvocationId { axis: 0 }, Expr::u32(REPRO_N)),
            vec![
                Node::let_bind("zero", Expr::u32(0)),
                // No assignment to `zero` anywhere, so `zero == 0` is a
                // genuine compile-time constant and collapsing it is sound.
                Node::if_then(
                    Expr::eq(Expr::var("zero"), Expr::u32(0)),
                    vec![Node::store(
                        "out",
                        Expr::InvocationId { axis: 0 },
                        Expr::add(
                            Expr::load("src", Expr::InvocationId { axis: 0 }),
                            Expr::u32(1),
                        ),
                    )],
                ),
            ],
        )],
    );
    assert_backends_agree(&program, &src, &expected, "constant-guard");
}
