#![allow(missing_docs)]

#[cfg(test)]
use vyre_bench::probes;
use vyre_bench::{
    api, link_benchmark_backend_registrations, registry, release_matrix, report, runner,
};

#[global_allocator]
static GLOBAL: vyre_bench::probes::TrackingAllocator = vyre_bench::probes::TrackingAllocator;

fn main() -> anyhow::Result<()> {
    cli::run_cli()
}

mod cli;
