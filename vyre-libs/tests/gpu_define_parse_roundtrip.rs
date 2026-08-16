//! GPU `#define` row parser reference roundtrip.
//!
//! Pins the kernel against ground-truth name/args/body byte spans for
//! object-like and function-like macros, including edge cases:
//! empty body, leading/trailing whitespace, function-like with no
//! args, function-like with multiple args, indented `#`.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod harness;

use harness::preprocess_stream::{run_directive_metadata_stage, unpack_u32};
use vyre_libs::parsing::c::preprocess::gpu_define_parse::gpu_define_parse;
use vyre_reference::value::Value;

#[derive(Debug, PartialEq, Eq)]
struct DefineRow {
    name: Vec<u8>,
    args: Vec<u8>,
    body: Vec<u8>,
    is_func: bool,
}

fn run_pipeline(source: &[u8]) -> Vec<Option<DefineRow>> {
    let stage = run_directive_metadata_stage(source);
    let n = stage.n;

    let prog_b = gpu_define_parse(n as u32, source.len() as u32);
    let outs = vyre_reference::reference_eval(
        &prog_b,
        &[
            Value::from(stage.tok_starts_bytes.clone()),
            Value::from(stage.tok_lens_bytes.clone()),
            Value::from(stage.directive_kinds_bytes.clone()),
            Value::from(stage.source_bytes.clone()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
            Value::from(stage.zero_column()),
        ],
    )
    .expect("17b.6 kernel eval");
    let name_s = unpack_u32(&outs[0].to_bytes());
    let name_l = unpack_u32(&outs[1].to_bytes());
    let args_s = unpack_u32(&outs[2].to_bytes());
    let args_l = unpack_u32(&outs[3].to_bytes());
    let body_s = unpack_u32(&outs[4].to_bytes());
    let body_l = unpack_u32(&outs[5].to_bytes());
    let is_f = unpack_u32(&outs[6].to_bytes());

    (0..n)
        .map(|i| {
            if name_l[i] == 0 {
                None
            } else {
                let nb = name_s[i] as usize;
                let nl = name_l[i] as usize;
                let ab = args_s[i] as usize;
                let al = args_l[i] as usize;
                let bb = body_s[i] as usize;
                let bl = body_l[i] as usize;
                Some(DefineRow {
                    name: source[nb..nb + nl].to_vec(),
                    args: if al == 0 {
                        Vec::new()
                    } else {
                        source[ab..ab + al].to_vec()
                    },
                    body: if bl == 0 {
                        Vec::new()
                    } else {
                        source[bb..bb + bl].to_vec()
                    },
                    is_func: is_f[i] == 1,
                })
            }
        })
        .collect()
}

fn first_define(source: &[u8]) -> DefineRow {
    let rows = run_pipeline(source);
    rows.into_iter()
        .flatten()
        .next()
        .expect("expected at least one #define row")
}

#[test]
fn object_like_simple() {
    let r = first_define(b"#define FOO 1\n");
    assert_eq!(r.name, b"FOO");
    assert!(r.args.is_empty());
    assert_eq!(r.body, b"1");
    assert!(!r.is_func);
}

#[test]
fn object_like_no_body() {
    let r = first_define(b"#define FOO\n");
    assert_eq!(r.name, b"FOO");
    assert!(r.args.is_empty());
    assert!(r.body.is_empty());
    assert!(!r.is_func);
}

#[test]
fn object_like_multiword_body() {
    let r = first_define(b"#define PI 3.14\n");
    assert_eq!(r.name, b"PI");
    assert_eq!(r.body, b"3.14");
}

#[test]
fn object_like_with_underscore_and_digits_in_name() {
    let r = first_define(b"#define HAVE_LIB_2 1\n");
    assert_eq!(r.name, b"HAVE_LIB_2");
}

#[test]
fn object_like_long_macro_name_is_not_truncated() {
    let name = format!("MACRO_{}", "A".repeat(160));
    let source = format!("#define {name} 1\n");

    let r = first_define(source.as_bytes());

    assert_eq!(r.name, name.as_bytes());
    assert_eq!(r.body, b"1");
}

#[test]
fn macro_name_starting_with_digit_is_rejected() {
    let rows = run_pipeline(b"#define 1BAD 9\n");

    assert!(
        rows.iter().all(Option::is_none),
        "C macro identifiers must not start with a digit"
    );
}

#[test]
fn function_like_no_args() {
    let r = first_define(b"#define FOO() 1\n");
    assert_eq!(r.name, b"FOO");
    assert!(r.args.is_empty());
    assert_eq!(r.body, b"1");
    assert!(r.is_func);
}

#[test]
fn function_like_one_arg() {
    let r = first_define(b"#define SQ(x) ((x)*(x))\n");
    assert_eq!(r.name, b"SQ");
    assert_eq!(r.args, b"x");
    assert_eq!(r.body, b"((x)*(x))");
    assert!(r.is_func);
}

#[test]
fn function_like_multi_arg() {
    let r = first_define(b"#define MAX(a,b) ((a)>(b)?(a):(b))\n");
    assert_eq!(r.name, b"MAX");
    assert_eq!(r.args, b"a,b");
    assert_eq!(r.body, b"((a)>(b)?(a):(b))");
    assert!(r.is_func);
}

#[test]
fn function_like_args_with_whitespace() {
    let r = first_define(b"#define ADD(a, b) (a+b)\n");
    assert_eq!(r.name, b"ADD");
    assert_eq!(r.args, b"a, b");
    assert_eq!(r.body, b"(a+b)");
}

#[test]
fn function_like_long_arg_list_is_not_truncated() {
    let args = (0..80)
        .map(|index| format!("a{index}"))
        .collect::<Vec<_>>()
        .join(",");
    let source = format!("#define MANY({args}) body\n");

    let r = first_define(source.as_bytes());

    assert_eq!(r.name, b"MANY");
    assert_eq!(r.args, args.as_bytes());
    assert_eq!(r.body, b"body");
    assert!(r.is_func);
}

#[test]
fn extra_whitespace_after_define_keyword() {
    let r = first_define(b"#define   FOO   42\n");
    assert_eq!(r.name, b"FOO");
    assert_eq!(r.body, b"42");
}

#[test]
fn indented_hash() {
    let r = first_define(b"   #define INDENTED 1\n");
    assert_eq!(r.name, b"INDENTED");
    assert_eq!(r.body, b"1");
}

#[test]
fn space_between_hash_and_define() {
    let r = first_define(b"# define SPACED 1\n");
    assert_eq!(r.name, b"SPACED");
    assert_eq!(r.body, b"1");
}

#[test]
fn body_with_trailing_whitespace_is_trimmed() {
    let r = first_define(b"#define X foo   \n");
    assert_eq!(r.body, b"foo");
}

#[test]
fn non_define_row_emits_zero_name_len() {
    let rows = run_pipeline(b"#include <stdio.h>\n#pragma once\n");
    assert!(rows.iter().all(|r| r.is_none()));
}

#[test]
fn mixed_directives_only_define_rows_have_names() {
    let rows = run_pipeline(b"#define A 1\n#include <foo.h>\n#define B 2\n");
    let defines: Vec<_> = rows.into_iter().flatten().collect();
    assert_eq!(defines.len(), 2);
    assert_eq!(defines[0].name, b"A");
    assert_eq!(defines[1].name, b"B");
}
