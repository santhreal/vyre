//! CLI entrypoint for the independent interactive graphics consumer.

use vyre_graphics_app::BenchmarkHarness;

fn main() {
    let samples = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);

    println!("Running vyre-graphics-app benchmark ({samples} samples)...");
    let report = BenchmarkHarness::run_suite(samples);
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    println!("{json}");
}
