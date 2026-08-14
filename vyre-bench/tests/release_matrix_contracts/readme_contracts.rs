use super::*;

#[test]
fn benchmark_documentation_defers_to_the_manifest_and_generated_evidence() {
    let workspace = workspace_root();
    let perf = std::fs::read_to_string(workspace.join("docs/PERF.md"))
        .expect("Fix: docs/PERF.md must remain readable.");

    assert!(
        perf.contains("docs/optimization/BENCH_TARGETS.toml")
            && perf.contains("release/evidence/benchmarks/"),
        "Fix: performance documentation must name the benchmark-target manifest and generated evidence authorities."
    );
}
