//! Canonical scaling-bench harness for the self-hosted GPU optimizer.
//!
//! Two concrete driver crates each measured the GPU optimizer against the CPU
//! optimizer on synthetic programs, and both carried their own copy of the
//! fixture builders, the CPU pipeline, the oracle worker thread and the result
//! table. The copies had already diverged: one measured a depth-bound chain
//! fixture and a parallelism-friendly wide fixture, the other only the chain,
//! and only one ran the CPU oracle on an enlarged stack, so the other was one
//! fixture size away from overflowing the default test stack on the recursive
//! CPU walk. The numbers were also printed in two column layouts, so the two
//! backends could not be read side by side.
//!
//! What stays with each backend is the part that is genuinely backend-typed:
//! acquiring its device and handing over a `ProgramDispatcher`.
//!
//! The fixtures are the two ends of the parallelism axis the GPU passes are
//! judged on. A chain program's lets each read the previous one, so a level-wave
//! kernel has one let per level and the pass cannot widen. A wide program's lets
//! are independent, so every let is available in one level.

use std::time::Instant;

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::program_dispatch::ProgramDispatcher;

/// Stack for the CPU oracle worker.
///
/// The CPU optimizer walks the `Expr` arena recursively, so a chain fixture's
/// depth is the recursion depth. The default 2 MiB test stack overflows before
/// the largest fixture size, and an overflow inside a bench reads as a device
/// fault rather than as a host recursion limit.
const CPU_ORACLE_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Fixture sizes, in `let` bindings, every backend's scaling bench reports.
///
/// One shared list, so two backends' tables have the same rows and can be read
/// against each other.
pub const SCALING_FIXTURE_SIZES: &[usize] = &[10, 100, 1000];

/// A named fixture family and its builder.
pub struct ScalingFixture {
    /// Column heading for this family's table.
    pub label: &'static str,
    /// Build the program for a given `let` count.
    pub build: fn(usize) -> Program,
}

/// Both ends of the parallelism axis, in the order every table reports them.
pub const SCALING_FIXTURES: &[ScalingFixture] = &[
    ScalingFixture {
        label: "chain fixture (depth-bound, worst case for parallelism)",
        build: chain_program,
    },
    ScalingFixture {
        label: "wide fixture (independent computations, parallelism-friendly)",
        build: wide_program,
    },
];

/// A chain of `lets`, each reading the one before it.
///
/// Node-graph depth is `count`; the `Expr` arena stays shallow because `Var` is
/// a leaf. Worst case for level parallelism inside one pass.
pub fn chain_program(count: usize) -> Program {
    let mut entry: Vec<Node> = Vec::with_capacity(count + 1);
    for index in 0..count {
        let value = if index == 0 {
            Expr::mul(Expr::add(Expr::u32(1), Expr::u32(2)), Expr::u32(3))
        } else {
            let previous = format!("v{}", index - 1);
            Expr::mul(Expr::add(Expr::u32(5), Expr::var(previous)), Expr::u32(2))
        };
        entry.push(Node::let_bind(format!("v{index}"), value));
    }
    Program::wrapped(Vec::new(), [1, 1, 1], terminated(entry, count))
}

/// `count` independent `lets` over literals only.
///
/// Const-fold collapses every one without ordering, canonicalize finds each
/// already literal-on-right, and DCE drops all but the one the store reads. Best
/// case for a parallel kernel.
pub fn wide_program(count: usize) -> Program {
    let mut entry: Vec<Node> = Vec::with_capacity(count + 1);
    for index in 0..count {
        let value = Expr::mul(
            Expr::add(
                Expr::u32(((index % 7) + 1) as u32),
                Expr::u32(((index % 13) + 1) as u32),
            ),
            Expr::u32(((index % 5) + 1) as u32),
        );
        entry.push(Node::let_bind(format!("v{index}"), value));
    }
    Program::wrapped(Vec::new(), [1, 1, 1], terminated(entry, count))
}

/// Append the store that keeps the last binding live.
///
/// Without it DCE deletes the whole body and every fixture size measures the
/// same empty program.
fn terminated(mut entry: Vec<Node>, count: usize) -> Vec<Node> {
    let last = format!("v{}", count.saturating_sub(1));
    entry.push(Node::store("buf", Expr::u32(0), Expr::var(last)));
    entry
}

/// The CPU optimizer stack the GPU pipeline is measured against.
pub fn cpu_pipeline(program: Program) -> Program {
    use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine::run as cpu_canonicalize;
    use vyre_foundation::optimizer::passes::fusion_cse::dce::dce as cpu_dce;
    let program = cpu_canonicalize(program);
    let program =
        vyre_foundation::optimizer::optimize(program).expect("registered optimizer must converge");
    cpu_dce(program)
}

/// Run the CPU oracle on a worker with a stack deep enough for every fixture.
pub fn cpu_pipeline_on_oracle_stack(program: Program) -> Program {
    on_oracle_stack("cpu-oracle", move || cpu_pipeline(program))
}

/// Microseconds the CPU oracle takes, measured on the worker that runs it.
pub fn timed_cpu_pipeline_on_oracle_stack(program: Program) -> u128 {
    on_oracle_stack("cpu-oracle-timer", move || {
        let started = Instant::now();
        let _ = cpu_pipeline(program);
        started.elapsed().as_micros()
    })
}

fn on_oracle_stack<T: Send + 'static>(name: &str, work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .name(format!("self-optimizer-bench-{name}"))
        .stack_size(CPU_ORACLE_STACK_BYTES)
        .spawn(work)
        .expect("Fix: the scaling bench must be able to spawn its CPU oracle worker")
        .join()
        .expect("Fix: the CPU optimizer oracle must not panic on a scaling-bench fixture")
}

/// Print one backend's scaling table over every fixture family and size.
///
/// `gpu_pipeline` is the measured path. It receives the same program the CPU
/// oracle gets, once to warm the path and once timed, so the reported figure
/// excludes first-dispatch compilation.
pub fn report_scaling(
    backend: &str,
    dispatcher: &dyn ProgramDispatcher,
    gpu_pipeline: fn(Program, &dyn ProgramDispatcher) -> Program,
) {
    println!("\n=== self-hosted {backend} optimizer scaling vs CPU pipeline ===");
    for fixture in SCALING_FIXTURES {
        println!("\n--- {} {} ---", backend, fixture.label);
        println!(
            "{:>8} | {:>14} | {:>14} | {:>10}",
            "n", "gpu_us", "cpu_us", "gpu/cpu"
        );
        println!("{}", "-".repeat(56));
        for &count in SCALING_FIXTURE_SIZES {
            let program = (fixture.build)(count);

            let _ = gpu_pipeline(program.clone(), dispatcher);
            let _ = cpu_pipeline_on_oracle_stack(program.clone());

            let started = Instant::now();
            let _ = gpu_pipeline(program.clone(), dispatcher);
            let gpu_us = started.elapsed().as_micros();

            let cpu_us = timed_cpu_pipeline_on_oracle_stack(program);

            let ratio = if cpu_us == 0 {
                f64::INFINITY
            } else {
                gpu_us as f64 / cpu_us as f64
            };
            println!("{count:>8} | {gpu_us:>14} | {cpu_us:>14} | {ratio:>10.2}x");
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the store is what keeps a fixture from optimizing to nothing. A
    /// builder that forgets it produces a program DCE empties, and every
    /// fixture size then measures the same work.
    #[test]
    fn every_fixture_keeps_its_last_binding_live() {
        for fixture in SCALING_FIXTURES {
            for &count in SCALING_FIXTURE_SIZES {
                let program = (fixture.build)(count);
                let rendered = format!("{:?}", program.entry());
                assert!(
                    rendered.contains(&format!("v{}", count - 1)),
                    "Fix: {} at n={count} does not read its last binding, so DCE empties it",
                    fixture.label
                );
            }
        }
    }

    /// WHY: the two families exist to be the two ends of the parallelism axis.
    /// If they build the same body the bench reports one measurement twice.
    #[test]
    fn the_fixture_families_are_not_the_same_program() {
        for &count in SCALING_FIXTURE_SIZES {
            assert_ne!(
                chain_program(count).fingerprint(),
                wide_program(count).fingerprint(),
                "Fix: the chain and wide fixtures collapsed to one program at n={count}"
            );
        }
    }

    /// WHY: a fixture whose size does not change its program measures dispatch
    /// overhead and calls it scaling.
    #[test]
    fn a_bigger_fixture_is_a_different_program() {
        for fixture in SCALING_FIXTURES {
            let mut seen = std::collections::BTreeSet::new();
            for &count in SCALING_FIXTURE_SIZES {
                assert!(
                    seen.insert((fixture.build)(count).fingerprint()),
                    "Fix: {} produces the same program at two sizes",
                    fixture.label
                );
            }
        }
    }
}
