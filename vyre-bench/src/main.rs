//! Command-line entry point for the vyre benchmark runner.

#[cfg(test)]
use vyre_bench::probes;
use vyre_bench::{api, registry, release_matrix, report, runner};

#[global_allocator]
static GLOBAL: vyre_bench::probes::TrackingAllocator = vyre_bench::probes::TrackingAllocator;

fn main() -> anyhow::Result<()> {
    cli::run_cli()
}

mod cli;
