//! CLI entrypoint for the independent interactive graphics consumer.

use std::process::ExitCode;

use vyre_graphics_app::BenchmarkHarness;

fn main() -> ExitCode {
    let samples = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);

    println!("Running vyre-graphics-app benchmark ({samples} samples)...");
    match run(samples) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The measurement suite's report, rendered as pretty JSON.
fn run(samples: usize) -> Result<String, String> {
    let report = BenchmarkHarness::run_suite(samples)?;
    serde_json::to_string_pretty(&report).map_err(|error| format!("report serialize: {error}"))
}
