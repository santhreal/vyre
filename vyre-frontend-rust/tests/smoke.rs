//! Smoke tests for the Rust nano-subset parser and end-to-end pipeline.

use vyre_frontend_rust::api::parse_rust_bytes;
use vyre_frontend_rust::pipeline::{RustPipeline, RustPipelineConfig};
use vyre_frontend_rust::RustFrontendError;

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
fn compile_pipeline_gpu_lex_dispatches_real_lexer_ir() {
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: true,
        borrow_check: false,
        lower: false,
        lower_lane_count: None,
    });
    let unit = pipeline
        .compile_unit(b"fn main() { let x: i32 = 5; }")
        .expect("Fix: gpu_lex=true must dispatch the Rust lexer IR and feed parser/typeck");
    assert_eq!(unit.module.functions.len(), 1);
    assert!(
        unit.token_count > 8,
        "Fix: GPU lexer must return a real token stream, not an empty placeholder"
    );
    assert!(
        unit.program.is_none(),
        "lowering is off in this smoke path, so gpu_lex must not fabricate a Program"
    );
}

#[test]
fn compile_pipeline_gpu_lex_surfaces_unknown_byte_as_lex_error() {
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: true,
        borrow_check: false,
        lower: false,
        lower_lane_count: None,
    });
    let error = pipeline
        .compile_unit(b"fn main() { @ }")
        .expect_err("Fix: GPU lexer ERROR tokens must become frontend lex errors");
    assert!(matches!(error, RustFrontendError::Lex(_)), "got {error:?}");
}

#[test]
fn compile_unit_succeeds_on_well_typed_program() {
    let pipeline = RustPipeline::new(RustPipelineConfig::default());
    let unit = pipeline
        .compile_unit(b"fn add(a: i32, b: i32) -> i32 { return a + b; }")
        .expect("Fix: a well-typed nano-subset program must compile through resolve + typeck");
    assert!(
        unit.program.is_none(),
        "Fix: lowering is off by default, so there is no Program yet"
    );
}

#[test]
fn compile_unit_rejects_type_mismatch() {
    let pipeline = RustPipeline::new(RustPipelineConfig::default());
    let error = pipeline
        .compile_unit(b"fn f() -> i32 { return true; }")
        .expect_err("Fix: a return-type mismatch must fail type checking, not compile.");
    assert!(
        matches!(error, RustFrontendError::Typeck(_)),
        "got {error:?}"
    );
    assert!(error.to_string().contains("mismatched types"));
}

#[test]
fn compile_unit_borrow_check_catches_e0596() {
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: false,
        borrow_check: true,
        lower: false,
        lower_lane_count: None,
    });
    let error = pipeline
        .compile_unit(b"fn f() { let x: i32 = 0; let r: &mut i32 = &mut x; }")
        .expect_err("Fix: &mut of an immutable binding must fail borrow checking.");
    assert!(
        matches!(error, RustFrontendError::Borrow(_)),
        "got {error:?}"
    );
    assert!(error.to_string().contains("cannot borrow `x` as mutable"));
}

#[test]
fn compile_unit_borrow_check_accepts_clean_program() {
    // The conflicting-borrow rules are wired (CFG NLL dataflow engine), so a
    // conflict-free program borrow-checks successfully, matching rustc.
    let pipeline = RustPipeline::new(RustPipelineConfig {
        gpu_lex: false,
        borrow_check: true,
        lower: false,
        lower_lane_count: None,
    });
    let unit = pipeline
        .compile_unit(b"fn f() { let mut x: i32 = 0; let r: &mut i32 = &mut x; let _p: i32 = *r; }")
        .expect("Fix: a conflict-free program must pass the wired borrow check");
    assert!(
        unit.program.is_none(),
        "lowering is off by default, so there is no Program yet"
    );
}
