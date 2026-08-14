#![allow(missing_docs)]
//! Criterion bench for the full IR optimizer pass pipeline.
//!
//! `optimize()` is the CPU-side pipeline every compile runs before any backend
//! lowering, so its wall time is compile latency for the whole product. The
//! bench measures the pipeline, not one pass: the scheduler's per-pass fixed
//! overhead (analysis walks, gate certificates, snapshots) dominates on small
//! programs, and the rewrite work itself dominates on large ones, so both
//! shapes are covered.
//!
//! Inputs:
//! - `release_corpus_families`: one program per semantic family from the
//!   shipped release corpus (`optimizer::corpus`), the same shapes the release
//!   optimization gate scores.
//! - `kernel_wide` / `kernel_loop_nest`: synthesized kernels sized like real
//!   generated code (long straight-line arithmetic with reused subexpressions,
//!   and a loop nest with invariant loads).
//!
//! Run with:
//! ```text
//! cargo bench -p vyre-foundation --bench optimizer_pipeline
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::corpus::generate_release_corpus;
use vyre_foundation::optimizer::optimize;

const WIDTH: u32 = 256;

fn buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(WIDTH),
        BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(WIDTH),
        BufferDecl::output("out", 2, DataType::U32).with_count(WIDTH),
    ]
}

fn index() -> Expr {
    Expr::bitand(Expr::gid_x(), Expr::u32(WIDTH - 1))
}

/// Wide straight-line kernel: `depth` dependent arithmetic bindings with
/// constant division, constant remainder, strength-reducible multiplies,
/// repeated subexpressions for CSE, and unused bindings for DCE.
fn kernel_wide(depth: u32) -> Program {
    let mut body = Vec::with_capacity(depth as usize * 2 + 3);
    body.push(Node::let_bind("idx", index()));
    body.push(Node::let_bind("v0", Expr::load("in", Expr::var("idx"))));
    for step in 0..depth {
        let previous = Expr::var(format!("v{step}"));
        let common = Expr::add(previous.clone(), Expr::u32(step % 17));
        let value = match step % 5 {
            0 => Expr::div(common, Expr::u32(7)),
            1 => Expr::rem(common, Expr::u32(9)),
            2 => Expr::mul(common, Expr::u32(8)),
            3 => Expr::add(Expr::mul(common.clone(), Expr::u32(3)), common),
            _ => Expr::add(common, Expr::mul(previous, Expr::u32(0))),
        };
        body.push(Node::let_bind(format!("v{}", step + 1), value));
        if step % 4 == 0 {
            body.push(Node::let_bind(
                format!("unused{step}"),
                Expr::mul(Expr::var(format!("v{step}")), Expr::u32(1)),
            ));
        }
    }
    body.push(Node::store(
        "out",
        Expr::var("idx"),
        Expr::var(format!("v{depth}")),
    ));
    Program::wrapped(buffers(), [64, 1, 1], body).with_entry_op_id("bench.optimizer.kernel_wide")
}

/// Loop nest with loop-invariant loads, a redundant bound expression, and a
/// guarded store: the shape loop passes (LICM, bound tightening, unrolling)
/// and memory passes all fire on.
fn kernel_loop_nest(rows: u32, columns: u32) -> Program {
    let inner = vec![
        Node::let_bind(
            "invariant",
            Expr::add(Expr::load("in", Expr::var("idx")), Expr::u32(11)),
        ),
        Node::let_bind(
            "acc",
            Expr::add(
                Expr::mul(Expr::var("invariant"), Expr::u32(4)),
                Expr::div(Expr::var("j"), Expr::u32(3)),
            ),
        ),
        Node::if_then_else(
            Expr::lt(Expr::var("acc"), Expr::u32(1_000_000)),
            vec![Node::store("scratch", Expr::var("idx"), Expr::var("acc"))],
            vec![Node::store("scratch", Expr::var("idx"), Expr::u32(0))],
        ),
    ];
    let body = vec![
        Node::let_bind("idx", index()),
        Node::Loop {
            var: "i".into(),
            from: Expr::u32(0),
            to: Expr::u32(rows),
            body: vec![Node::Loop {
                var: "j".into(),
                from: Expr::u32(0),
                to: Expr::u32(columns),
                body: inner,
            }],
        },
        Node::store(
            "out",
            Expr::var("idx"),
            Expr::load("scratch", Expr::var("idx")),
        ),
    ];
    Program::wrapped(buffers(), [64, 1, 1], body)
        .with_entry_op_id("bench.optimizer.kernel_loop_nest")
}

/// One program per semantic family, seed 0, from the shipped release corpus.
fn corpus_families() -> Vec<Program> {
    let cases = generate_release_corpus();
    let mut seen = Vec::new();
    let mut programs = Vec::new();
    for case in cases {
        if seen.contains(&case.family) {
            continue;
        }
        seen.push(case.family.clone());
        programs.push(case.program);
    }
    programs
}

fn checked(program: Program) -> Program {
    assert!(
        optimize(program.clone()).is_ok(),
        "bench input must converge through the release pipeline"
    );
    program
}

fn bench_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer/pipeline");

    let families = corpus_families();
    for program in &families {
        assert!(
            optimize(program.clone()).is_ok(),
            "release corpus program must converge"
        );
    }
    group.bench_function("release_corpus_families", |b| {
        b.iter(|| {
            for program in &families {
                let optimized = optimize(program.clone());
                let _ = black_box(&optimized);
            }
        });
    });

    for depth in [16u32, 64] {
        let program = checked(kernel_wide(depth));
        group.bench_function(format!("kernel_wide/{depth}"), |b| {
            b.iter_batched(
                || program.clone(),
                |program| black_box(optimize(program)),
                BatchSize::SmallInput,
            );
        });
    }

    let nest = checked(kernel_loop_nest(4, 8));
    group.bench_function("kernel_loop_nest/4x8", |b| {
        b.iter_batched(
            || nest.clone(),
            |program| black_box(optimize(program)),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_pipeline);
criterion_main!(benches);
