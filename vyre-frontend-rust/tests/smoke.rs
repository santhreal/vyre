//! Smoke tests for the Rust nano-subset parser and pipeline boundary.

use vyre_frontend_rust::api::parse_rust_bytes;
use vyre_frontend_rust::pipeline::{RustPipeline, RustPipelineConfig};

#[test]
fn parse_trivial_function() {
    let source = "fn main() { let x: i32 = 5; }";
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
}

#[test]
fn parse_function_with_params() {
    let source = "fn add(a: i32, b: i32) -> i32 { return a + b; }";
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
    assert_eq!(summary.module.functions[0].params.len(), 2);
}

#[test]
fn parse_if_else() {
    let source = "fn max(a: i32, b: i32) -> i32 { if a < b { return b; } else { return a; }; }";
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
}

#[test]
fn parse_borrow() {
    let source = "fn borrow(x: &i32) -> i32 { return *x; }";
    let summary = parse_rust_bytes(source.as_bytes()).unwrap();
    assert_eq!(summary.module.functions.len(), 1);
}

#[test]
fn compile_pipeline_rejects_unwired_gpu_lex_without_silent_cpu_path() {
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: true,
        borrow_check: false,
        lower: false,
    });

    let error = pipeline
        .compile_unit(b"fn main() { let x: i32 = 5; }")
        .expect_err("Fix: GPU lexing must fail loudly until Rust GPU lexer dispatch is wired.");

    let message = error.to_string();
    assert!(message.contains("GPU backend unavailable"));
    assert!(message.contains("not wired yet"));
    assert!(message.contains("silently"));
}

#[test]
fn compile_pipeline_rejects_unwired_typeck_without_fake_program() {
    // Resolution now succeeds; the meaningful boundary is type checking, which
    // is unwired. compile_unit must fail loudly there, never return a success
    // that skipped type checking.
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: false,
        borrow_check: false,
        lower: false,
    });

    let error = pipeline
        .compile_unit(b"fn main() { let x: i32 = 5; }")
        .expect_err("Fix: compile_unit must not return success while type checking is not wired.");

    let message = error.to_string();
    assert!(message.contains("unsupported construct"));
    assert!(message.contains("type checking is not wired"));
}
