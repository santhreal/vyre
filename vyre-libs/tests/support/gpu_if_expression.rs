#![allow(deprecated)]

use super::preprocess_stream::{
    column_words, cpu_kinds_and_values, padded_defined_macros, run_directive_metadata_stage,
    unpack_u32,
};
use vyre_libs::parsing::c::lex::tokens::TOK_PP_IF;
use vyre_libs::parsing::c::parse::gnu_builtins::gpu_builtin_hash_table_words;
use vyre_libs::parsing::c::preprocess::gpu_if_expression::gpu_if_expression;
use vyre_primitives::wire::pack_u32_slice as pack_u32_le;
use vyre_reference::value::Value;

fn pack_macro_values_with_builtin_hashes(values: &[u32]) -> Vec<u8> {
    let mut words = Vec::with_capacity(values.len() + gpu_builtin_hash_table_words().len());
    words.extend_from_slice(&gpu_builtin_hash_table_words());
    words.extend_from_slice(values);
    pack_u32_le(&words)
}

fn pack_defined_macro_values(names: &[&[u8]]) -> Vec<u8> {
    let count = names.len().max(1);
    let values = vec![1u32; count];
    pack_u32_le(&values)
}

pub(crate) fn run_full_pipeline(source: &[u8], defined_macros: &[&[u8]]) -> (Vec<u32>, Vec<u32>) {
    let stage = run_directive_metadata_stage(source);
    let (macro_names_padded, macro_offsets_bytes) = padded_defined_macros(defined_macros);
    let macro_values_words = unpack_u32(&pack_defined_macro_values(defined_macros));
    let macro_values_bytes = pack_macro_values_with_builtin_hashes(&macro_values_words);

    let prog_b = gpu_if_expression(stage.n as u32, source.len() as u32);
    let outputs_b = vyre_reference::reference_eval(
        &prog_b,
        &[
            Value::from(stage.tok_starts_bytes.clone()),
            Value::from(stage.tok_lens_bytes.clone()),
            Value::from(stage.directive_kinds_bytes.clone()),
            Value::from(stage.source_bytes.clone()),
            Value::from(macro_names_padded),
            Value::from(macro_offsets_bytes),
            Value::from(macro_values_bytes),
            Value::from(stage.zero_column()),
        ],
    )
    .expect("17b.4 kernel eval");

    (
        column_words(&stage.directive_kinds_bytes, stage.n),
        column_words(&outputs_b[0].to_bytes(), stage.n),
    )
}

pub(crate) fn run_if_expression_with_macro_value(source: &[u8], name: &[u8], value: u32) -> u32 {
    let mut src = source.to_vec();
    src.resize((source.len().div_ceil(4) * 4).max(4), 0);
    let (macro_names, macro_offsets_bytes) = padded_defined_macros(&[name]);
    let prog = gpu_if_expression(1, 0);
    let outputs = vyre_reference::reference_eval(
        &prog,
        &[
            Value::from(pack_u32_le(&[0])),
            Value::from(pack_u32_le(&[source.len() as u32])),
            Value::from(pack_u32_le(&[TOK_PP_IF])),
            Value::from(src),
            Value::from(macro_names),
            Value::from(macro_offsets_bytes),
            Value::from(pack_macro_values_with_builtin_hashes(&[value])),
            Value::from(pack_u32_le(&[0])),
        ],
    )
    .expect("gpu_if_expression macro-value contract eval");
    unpack_u32(&outputs[0].to_bytes())[0]
}

fn keep_only_if_elif(kinds: &[u32], values: &[u32]) -> Vec<u32> {
    use vyre_libs::parsing::c::lex::tokens::{TOK_PP_ELIF, TOK_PP_IF};
    kinds
        .iter()
        .zip(values)
        .map(|(k, v)| {
            if *k == TOK_PP_IF || *k == TOK_PP_ELIF {
                *v
            } else {
                0
            }
        })
        .collect()
}

pub(crate) fn assert_gpu_matches_cpu(source: &[u8], defined: &[&[u8]]) {
    let (kinds, gpu_values) = run_full_pipeline(source, defined);
    let (cpu_kinds, cpu_values) = cpu_kinds_and_values(source, defined);
    assert_eq!(
        kinds,
        cpu_kinds,
        "directive_kinds mismatch on {:?}",
        std::str::from_utf8(source)
    );
    assert_eq!(
        gpu_values,
        keep_only_if_elif(&cpu_kinds, &cpu_values),
        "directive_values mismatch on {:?}",
        std::str::from_utf8(source),
    );
}
