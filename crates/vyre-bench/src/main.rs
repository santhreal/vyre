//! Command-line entry point for the vyre benchmark runner.

#[cfg(test)]
use vyre_bench::probes;
use vyre_bench::{api, registry, report, runner};

// Per-thread counters: the runner measures one sample on one thread, and a
// process-wide counter would charge it for every other thread's allocations.
#[global_allocator]
static GLOBAL: vyre_bench::alloc_probe::ThreadAlloc = vyre_bench::alloc_probe::ThreadAlloc;

fn main() -> anyhow::Result<()> {
    cli::run_cli()
}

mod cli;
