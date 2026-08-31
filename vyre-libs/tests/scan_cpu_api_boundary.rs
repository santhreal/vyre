//! API-boundary regression tests for production scan paths.

mod harness;

#[test]
fn scan_layer_does_not_export_cpu_named_execution_paths() {
    harness::assert_no_cpu_named_api_exports(
        "src/pattern",
        "pattern",
        &["scan_cpu"],
        "pattern-layer CPU-named APIs must be explicit reference/parity internals",
    );
}
