//! Command-line entry point for the vyre benchmark runner.

#[cfg(test)]
use vyre_bench::probes;
use vyre_bench::{api, registry, release_matrix, report, runner, workloads};

// Per-thread counters: the runner measures one sample on one thread, and a
// process-wide counter would charge it for every other thread's allocations.
#[global_allocator]
static GLOBAL: vyre_alloc_probe::ThreadAlloc = vyre_alloc_probe::ThreadAlloc;

fn main() -> anyhow::Result<()> {
    cli::run_cli()
}

mod cli;
