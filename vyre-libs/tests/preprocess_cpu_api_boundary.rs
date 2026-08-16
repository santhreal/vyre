//! API-boundary regression tests for production preprocess paths.

mod harness;

#[test]
fn preprocess_does_not_export_cpu_named_execution_paths() {
    harness::assert_no_cpu_named_api_exports(
        "src/parsing/c/preprocess",
        "preprocess",
        &[],
        "C preprocessor CPU-named APIs must stay private to explicit reference/parity tests",
    );
}
