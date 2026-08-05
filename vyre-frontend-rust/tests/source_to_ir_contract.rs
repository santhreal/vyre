//! Source-to-typed-IR boundary contracts.
//!
//! Positive, negative, boundary, and hostile-source cases keep frontend
//! semantics independent from execution while pinning public diagnostics.

#![forbid(unsafe_code)]

use vyre_frontend_rust::pipeline::{RustPipeline, RustPipelineConfig};
use vyre_frontend_rust::RustFrontendError;
use vyre_reference::value::Value;

#[test]
fn positive_emitted_ir_is_consumed_by_reference_harness() {
    let pipeline = RustPipeline::new(RustPipelineConfig {
        borrow_check: true,
        lower: true,
        lower_lane_count: None,
    });
    let unit = pipeline
        .compile_unit(b"fn increment(value: i32) -> i32 { return value + 1; }")
        .expect("well-typed source must emit backend-neutral IR");
    let program = unit.program.expect("lower:true must emit a Program");

    let outputs = vyre_reference::reference_eval(&program, &[Value::I32(41)])
        .expect("the reference harness must consume frontend-emitted IR directly");
    let value = match outputs.as_slice() {
        [Value::I32(value)] => *value,
        [Value::U32(value)] => *value as i32,
        [Value::Bytes(bytes)] => {
            i32::from_le_bytes(bytes[..4].try_into().expect("one i32 output"))
        }
        other => panic!("unexpected reference output: {other:?}"),
    };
    assert_eq!(value, 42);
}

#[test]
fn negative_type_error_text_is_exact() {
    let error = RustPipeline::new(RustPipelineConfig::default())
        .compile_unit(b"fn f() -> i32 { return true; }")
        .expect_err("a return-type mismatch must be rejected");

    assert!(matches!(error, RustFrontendError::Typeck(_)));
    assert_eq!(
        error.to_string(),
        "Rust frontend type check failed: mismatched types in return value: expected `i32`, found `bool`. Fix: correct the types so they match."
    );
}

#[test]
fn boundary_empty_batch_preserves_an_empty_result() {
    let batch = RustPipeline::new(RustPipelineConfig::default())
        .compile_units(&[])
        .expect("an empty source batch is a valid boundary input");
    assert!(batch.units.is_empty());
}

#[test]
fn hostile_syntax_error_text_and_offset_are_exact() {
    let error = RustPipeline::new(RustPipelineConfig::default())
        .compile_unit(b"fn main() { @ }")
        .expect_err("an unsupported source byte must fail lexing");

    assert!(matches!(error, RustFrontendError::Lex(12)));
    assert_eq!(
        error.to_string(),
        "Rust frontend lex failed at byte 12. Fix: check for invalid UTF-8 or unsupported characters in the source."
    );
}
