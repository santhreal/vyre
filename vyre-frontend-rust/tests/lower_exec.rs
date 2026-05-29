//! End-to-end driver smoke test: `compile_unit` with `lower: true` produces a
//! Vyre `Program` that executes on the reference interpreter with the expected
//! result. Proves the full frontend path lex -> parse -> resolve -> typeck ->
//! borrow -> lower -> execute through the public driver API.

#![forbid(unsafe_code)]

use vyre_frontend_rust::pipeline::{RustPipeline, RustPipelineConfig};
use vyre_reference::value::Value;

fn run(src: &str, inputs: &[i32]) -> i32 {
    let config = RustPipelineConfig { gpu_lex: false, borrow_check: true, lower: true };
    let unit = RustPipeline::new(config)
        .compile_unit(src.as_bytes())
        .expect("Fix: a well-formed nano program must compile through the full pipeline");
    let program = unit.program.expect("Fix: lower:true must produce a Program");
    let values: Vec<Value> = inputs.iter().map(|&x| Value::I32(x)).collect();
    let out = vyre_reference::reference_eval(&program, &values).expect("reference_eval");
    match &out[0] {
        Value::I32(x) => *x,
        Value::U32(x) => *x as i32,
        Value::Bytes(b) => i32::from_le_bytes(b[..4].try_into().expect("4 bytes")),
        other => panic!("unexpected output {other:?}"),
    }
}

#[test]
fn compile_unit_lowers_and_executes_arithmetic() {
    assert_eq!(run("fn f(a: i32, b: i32) -> i32 { return a + b * 2; }", &[3, 4]), 11);
}

#[test]
fn compile_unit_lowers_and_executes_branch_and_call() {
    let src = "fn g(a: i32) -> i32 { return a + 1; } \
               fn f(a: i32, b: i32) -> i32 { if a < b { return g(b); } else { return a; } }";
    assert_eq!(run(src, &[2, 5]), 6);
    assert_eq!(run(src, &[9, 5]), 9);
}

#[test]
fn compile_unit_lowers_and_executes_refs() {
    assert_eq!(run("fn f(a: i32) -> i32 { let r: &i32 = &a; return *r + 1; }", &[6]), 7);
}
