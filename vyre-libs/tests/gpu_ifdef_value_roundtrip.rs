//! GPU `#ifdef` / `#ifndef` evaluator reference roundtrip.
//!
//! Asserts the 17b.1 kernel emits `1`/`0` for each `ifdef`/`ifndef`
//! token matching what the CPU
//! `reference_c_preprocessor_directive_metadata` produces. Other
//! directive kinds must remain `0` in this kernel's output column.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod support;

use support::preprocess_stream::{
    column_words, cpu_kinds_and_values, padded_defined_macros, run_directive_metadata_stage,
};
use vyre_libs::parsing::c::lex::tokens::{TOK_PP_IFDEF, TOK_PP_IFNDEF};
use vyre_libs::parsing::c::preprocess::gpu_ifdef_value::gpu_ifdef_value;
use vyre_reference::value::Value;

fn run_full_pipeline(source: &[u8], defined_macros: &[&[u8]]) -> (Vec<u32>, Vec<u32>) {
    let stage = run_directive_metadata_stage(source);
    let (macro_names_padded, macro_offsets_bytes) = padded_defined_macros(defined_macros);

    let prog_b = gpu_ifdef_value(stage.n as u32, source.len() as u32);
    let outputs_b = vyre_reference::reference_eval(
        &prog_b,
        &[
            Value::from(stage.tok_starts_bytes.clone()),
            Value::from(stage.tok_lens_bytes.clone()),
            Value::from(stage.directive_kinds_bytes.clone()),
            Value::from(stage.source_bytes.clone()),
            Value::from(macro_names_padded),
            Value::from(macro_offsets_bytes),
            Value::from(stage.zero_column()),
        ],
    )
    .expect("17b.1 kernel eval");

    (
        column_words(&stage.directive_kinds_bytes, stage.n),
        column_words(&outputs_b[0].to_bytes(), stage.n),
    )
}

/// Filter values to only keep ifdef/ifndef rows so we don't get
/// confused by 17b.4 work that hasn't shipped (the Reference oracle
/// computes `if`/`elif` values too; this kernel returns 0 for them).
fn keep_only_ifdef_ifndef(kinds: &[u32], values: &[u32]) -> Vec<u32> {
    kinds
        .iter()
        .zip(values)
        .map(|(k, v)| {
            if *k == TOK_PP_IFDEF || *k == TOK_PP_IFNDEF {
                *v
            } else {
                0
            }
        })
        .collect()
}

#[test]
fn ifdef_returns_one_when_macro_is_defined() {
    let src = b"#ifdef FOO\n";
    let defined = [b"FOO".as_slice()];
    let (kinds, gpu_values) = run_full_pipeline(src, &defined);
    let (cpu_kinds, cpu_values) = cpu_kinds_and_values(src, &defined);
    assert_eq!(kinds, cpu_kinds);
    assert_eq!(gpu_values, keep_only_ifdef_ifndef(&cpu_kinds, &cpu_values));
    assert_eq!(gpu_values, vec![1]);
}

#[test]
fn ifdef_returns_zero_when_macro_is_undefined() {
    let src = b"#ifdef MISSING\n";
    let defined = [b"FOO".as_slice()];
    let (kinds, gpu_values) = run_full_pipeline(src, &defined);
    let (cpu_kinds, cpu_values) = cpu_kinds_and_values(src, &defined);
    assert_eq!(kinds, cpu_kinds);
    assert_eq!(gpu_values, keep_only_ifdef_ifndef(&cpu_kinds, &cpu_values));
    assert_eq!(gpu_values, vec![0]);
}

#[test]
fn ifndef_returns_one_when_macro_is_undefined() {
    let src = b"#ifndef NEWHEADER\n";
    let defined = [b"FOO".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![1]);
}

#[test]
fn ifndef_returns_zero_when_macro_is_defined() {
    let src = b"#ifndef FOO\n";
    let defined = [b"FOO".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![0]);
}

#[test]
fn macro_with_underscore_and_digits_is_matched_byte_for_byte() {
    let src = b"#ifdef HAVE_LIB_2\n";
    let defined = [b"HAVE_LIB_2".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![1]);
}

#[test]
fn long_ifdef_identifier_is_not_truncated() {
    let name = format!("CONFIG_{}_FEATURE", "LONG_".repeat(40));
    let source = format!("#ifdef {name}\n");
    let defined = [name.as_bytes()];

    let (_kinds, gpu_values) = run_full_pipeline(source.as_bytes(), &defined);

    assert_eq!(gpu_values, vec![1]);
}

#[test]
fn long_ifndef_identifier_matches_full_name_before_inverting() {
    let name = format!("HAVE_{}_HEADER", "GENERATED_".repeat(32));
    let source = format!("#ifndef {name}\n");
    let defined = [name.as_bytes()];

    let (_kinds, gpu_values) = run_full_pipeline(source.as_bytes(), &defined);

    assert_eq!(gpu_values, vec![0]);
}

#[test]
fn macro_substring_match_does_not_count_as_defined() {
    // The defined name FOO is NOT a substring match for FOOBAR  -  must
    // be a full byte-for-byte equality.
    let src = b"#ifdef FOOBAR\n";
    let defined = [b"FOO".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![0]);
}

#[test]
fn extra_horizontal_whitespace_between_directive_and_name() {
    let src = b"#ifdef    SPACED\n";
    let defined = [b"SPACED".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![1]);
}

#[test]
fn other_directive_kinds_emit_zero_in_value_column() {
    let src = b"#define X 1\n#include <foo.h>\n#pragma once\n";
    let defined: [&[u8]; 0] = [];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert!(
        gpu_values.iter().all(|&v| v == 0),
        "non-ifdef/ifndef rows must emit 0; got {gpu_values:?}"
    );
}

#[test]
fn dense_block_with_mixed_defined_undefined_macros() {
    let src = b"#ifdef A\n#ifndef B\n#ifdef C\n#ifndef D\n";
    let defined = [b"A".as_slice(), b"C".as_slice()];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    // A defined → 1; B undefined → 1; C defined → 1; D undefined → 1.
    assert_eq!(gpu_values, vec![1, 1, 1, 1]);
}

#[test]
fn empty_macro_table_means_every_ifdef_is_zero_and_ifndef_is_one() {
    let src = b"#ifdef X\n#ifndef Y\n";
    let defined: [&[u8]; 0] = [];
    let (_kinds, gpu_values) = run_full_pipeline(src, &defined);
    assert_eq!(gpu_values, vec![0, 1]);
}
