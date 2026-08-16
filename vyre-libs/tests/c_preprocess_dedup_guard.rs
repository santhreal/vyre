//! Step 0 guard for the two C-preprocessor clone families collapsed by PR-08.
//!
//! Family 1, directive-line scan: `gpu_directive_metadata`, `gpu_ifdef_value`
//! and `gpu_if_expression` (plus their raw-`U8` twins) each carried their own
//! copy of the same walk - leading horizontal whitespace, `#`, whitespace,
//! keyword, whitespace, payload.
//!
//! Family 2, macro-expansion walk: `opt_named_macro_expansion` and
//! `opt_named_macro_expansion_materialized` each carried their own copy of the
//! dispatch walk and of the replacement walk under it.
//!
//! Two independent checks:
//!
//!   1. Cross-entry-point behaviour. Every directive entry point must agree
//!      with every other entry point on where the scan lands, across a
//!      whitespace and line-splice spread that brackets the kernels'
//!      `MAX_DIRECTIVE_WS_PREFIX` cap. Divergence here is drift, not noise.
//!   2. Shape. `Program::fingerprint()` for every public entry point, pinned
//!      from the pre-merge tree, so a merge that claims to be a pure rehome is
//!      checked against the pre-merge IR instead of trusted.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]

mod harness;

use harness::preprocess_stream::{
    build_token_stream_with, pack_defined_macros, unpack_u32, LineEnds,
};
use vyre_foundation::ir::{Expr, Program};
use vyre_libs::parsing::c::lex::tokens::{TOK_PP_ELIF, TOK_PP_IF, TOK_PP_IFDEF, TOK_PP_IFNDEF};
use vyre_libs::parsing::c::parse::gnu_builtins::gpu_builtin_hash_table_words;
use vyre_libs::parsing::c::preprocess::expansion::{
    opt_dynamic_macro_expansion, opt_named_macro_expansion, opt_named_macro_expansion_materialized,
};
use vyre_libs::parsing::c::preprocess::gpu_define_parse::{gpu_define_parse, gpu_define_parse_u8};
use vyre_libs::parsing::c::preprocess::gpu_directive_metadata::{
    gpu_directive_metadata, gpu_directive_metadata_u8,
};
use vyre_libs::parsing::c::preprocess::gpu_if_expression::{
    gpu_if_expression, gpu_if_expression_u8,
};
use vyre_libs::parsing::c::preprocess::gpu_ifdef_value::{gpu_ifdef_value, gpu_ifdef_value_u8};
use vyre_libs::parsing::c::preprocess::gpu_include_parse::{
    gpu_include_parse, gpu_include_parse_u8,
};
use vyre_libs::parsing::c::preprocess::gpu_undef_parse::{gpu_undef_parse, gpu_undef_parse_u8};
use vyre_libs::parsing::c::preprocess::reference_c_preprocessor_directive_metadata;
use vyre_primitives::wire::pack_u32_slice as pack_u32_le;
use vyre_reference::value::Value;

// ---------------------------------------------------------------- harness

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// Four source bytes per `U32` element.
    Packed,
    /// One source byte per `U8` element.
    RawU8,
}

fn byte_buffer(bytes: &[u8], layout: Layout) -> Vec<u8> {
    let mut out = bytes.to_vec();
    match layout {
        Layout::Packed => out.resize((bytes.len().div_ceil(4) * 4).max(4), 0),
        Layout::RawU8 => {
            if out.is_empty() {
                out.push(0);
            }
        }
    }
    out
}

fn macro_values_with_builtin_hashes(count: usize) -> Vec<u8> {
    let mut words = gpu_builtin_hash_table_words();
    words.extend(std::iter::repeat_n(1u32, count.max(1)));
    pack_u32_le(&words)
}

/// Directive kinds plus the `#ifdef`/`#ifndef` and `#if`/`#elif` value columns
/// produced by one layout's three entry points, chained exactly as the
/// pipeline chains them.
struct FamilyRun {
    kinds: Vec<u32>,
    ifdef_values: Vec<u32>,
    if_values: Vec<u32>,
}

fn run_family(source: &[u8], defined: &[&[u8]], layout: Layout) -> FamilyRun {
    let (tok_types, tok_starts, tok_lens) = build_token_stream_with(source, LineEnds::Tokenized);
    let n = tok_types.len();
    let n_padded = n.max(1);

    let mut tt = pack_u32_le(&tok_types);
    tt.resize(n_padded * 4, 0);
    let mut ts = pack_u32_le(&tok_starts);
    ts.resize(n_padded * 4, 0);
    let mut tl = pack_u32_le(&tok_lens);
    tl.resize(n_padded * 4, 0);
    let src = byte_buffer(source, layout);

    let metadata = match layout {
        Layout::Packed => gpu_directive_metadata(n as u32, source.len() as u32),
        Layout::RawU8 => gpu_directive_metadata_u8(n as u32, source.len() as u32),
    };
    let kinds_out = vyre_reference::reference_eval(
        &metadata,
        &[
            Value::from(tt),
            Value::from(ts.clone()),
            Value::from(tl.clone()),
            Value::from(src.clone()),
            Value::from(vec![0u8; n_padded * 4]),
            Value::from(vec![0u8; n_padded * 4]),
        ],
    )
    .expect("directive metadata eval");
    let mut kind_bytes = kinds_out[0].to_bytes().to_vec();
    kind_bytes.resize(n_padded * 4, 0);

    let (macro_names, macro_offsets) = pack_defined_macros(defined);
    let names_buffer = byte_buffer(&macro_names, layout);
    let offsets_bytes = pack_u32_le(&macro_offsets);

    let ifdef = match layout {
        Layout::Packed => gpu_ifdef_value(n as u32, source.len() as u32),
        Layout::RawU8 => gpu_ifdef_value_u8(n as u32, source.len() as u32),
    };
    let ifdef_out = vyre_reference::reference_eval(
        &ifdef,
        &[
            Value::from(ts.clone()),
            Value::from(tl.clone()),
            Value::from(kind_bytes.clone()),
            Value::from(src.clone()),
            Value::from(names_buffer.clone()),
            Value::from(offsets_bytes.clone()),
            Value::from(vec![0u8; n_padded * 4]),
        ],
    )
    .expect("ifdef value eval");

    let if_expr = match layout {
        Layout::Packed => gpu_if_expression(n as u32, source.len() as u32),
        Layout::RawU8 => gpu_if_expression_u8(n as u32, source.len() as u32),
    };
    let if_out = vyre_reference::reference_eval(
        &if_expr,
        &[
            Value::from(ts),
            Value::from(tl),
            Value::from(kind_bytes.clone()),
            Value::from(src),
            Value::from(names_buffer),
            Value::from(offsets_bytes),
            Value::from(macro_values_with_builtin_hashes(defined.len())),
            Value::from(vec![0u8; n_padded * 4]),
        ],
    )
    .expect("if expression eval");

    let mut kinds = unpack_u32(&kind_bytes);
    kinds.truncate(n);
    let mut ifdef_values = unpack_u32(&ifdef_out[0].to_bytes());
    ifdef_values.truncate(n);
    let mut if_values = unpack_u32(&if_out[0].to_bytes());
    if_values.truncate(n);
    FamilyRun {
        kinds,
        ifdef_values,
        if_values,
    }
}

fn cpu_oracle(source: &[u8], defined: &[&[u8]]) -> (Vec<u32>, Vec<u32>) {
    let (tok_types, tok_starts, tok_lens) = build_token_stream_with(source, LineEnds::Tokenized);
    reference_c_preprocessor_directive_metadata(&tok_types, &tok_starts, &tok_lens, source, defined)
        .expect("CPU directive oracle")
}

fn masked(kinds: &[u32], values: &[u32], wanted: &[u32]) -> Vec<u32> {
    kinds
        .iter()
        .zip(values)
        .map(|(k, v)| if wanted.contains(k) { *v } else { 0 })
        .collect()
}

/// Rows the kernels promise to handle: whitespace runs inside the documented
/// `MAX_DIRECTIVE_WS_PREFIX` cap of 4, and no phase-2 line splice left in the
/// row. GPU and CPU must agree exactly on these.
fn within_contract_cases() -> Vec<(String, Vec<&'static [u8]>)> {
    let mut cases: Vec<(String, Vec<&'static [u8]>)> = Vec::new();
    for lead in 0..=4usize {
        for after_hash in 0..=4usize {
            for after_kw in 1..=4usize {
                let l = " ".repeat(lead);
                let h = " ".repeat(after_hash);
                let k = " ".repeat(after_kw);
                cases.push((format!("{l}#{h}ifdef{k}FOO\n"), vec![b"FOO".as_slice()]));
                cases.push((format!("{l}#{h}ifndef{k}FOO\n"), vec![]));
                cases.push((format!("{l}#{h}if{k}1\n"), vec![]));
                cases.push((format!("{l}#{h}elif{k}0\n"), vec![]));
            }
        }
    }
    cases.push(("\t#\tifdef\tFOO\n".to_string(), vec![b"FOO".as_slice()]));
    cases.push((
        "\u{b}#\u{c}ifdef\u{b}FOO\n".to_string(),
        vec![b"FOO".as_slice()],
    ));
    cases.push(("#if 1 ? 2 : 3\n".to_string(), vec![]));
    cases.push(("#if defined(FOO)\n".to_string(), vec![b"FOO".as_slice()]));
    cases.push(("#\n".to_string(), vec![]));
    cases.push(("#define FOO 1\n#ifdef FOO\n#endif\n".to_string(), vec![]));
    cases
}

/// Rows outside the kernels' contract. Two classes, both pre-existing and both
/// uniform across the whole family:
///
///   - whitespace runs past `MAX_DIRECTIVE_WS_PREFIX`, which the straight-line
///     scan cannot see past;
///   - rows that still carry a phase-2 `\<newline>` splice, which the resident
///     pipeline removes before these kernels ever run - `gpu_directive_metadata`
///     documents `tok_lens` as already excluding phase-2 splices.
///
/// The CPU oracle handles both, so it is not a comparand here. What must hold
/// is that every entry point in the family lands in the same place, because a
/// second copy of the scan is exactly what would make one of them differ.
fn out_of_contract_cases() -> Vec<(String, Vec<&'static [u8]>)> {
    let mut cases: Vec<(String, Vec<&'static [u8]>)> = Vec::new();
    for run in 5..=8usize {
        let pad = " ".repeat(run);
        cases.push((format!("{pad}#ifdef FOO\n"), vec![b"FOO".as_slice()]));
        cases.push((format!("#{pad}ifdef FOO\n"), vec![b"FOO".as_slice()]));
        cases.push((format!("#ifdef{pad}FOO\n"), vec![b"FOO".as_slice()]));
        cases.push((format!("{pad}#if 1\n"), vec![]));
        cases.push((format!("#{pad}if 1\n"), vec![]));
        cases.push((format!("#if{pad}1\n"), vec![]));
    }
    cases.push(("#ifdef \\\n FOO\n".to_string(), vec![b"FOO".as_slice()]));
    cases.push(("#if \\\n 1\n".to_string(), vec![]));
    cases.push(("#if\\\n 1\n".to_string(), vec![]));
    cases
}

// ------------------------------------------------ family 1: behaviour

#[test]
fn directive_entry_points_agree_across_byte_layouts() {
    for (source, defined) in within_contract_cases()
        .into_iter()
        .chain(out_of_contract_cases())
    {
        let packed = run_family(source.as_bytes(), &defined, Layout::Packed);
        let raw = run_family(source.as_bytes(), &defined, Layout::RawU8);
        assert_eq!(packed.kinds, raw.kinds, "kinds diverge on {source:?}");
        assert_eq!(
            packed.ifdef_values, raw.ifdef_values,
            "ifdef values diverge on {source:?}"
        );
        assert_eq!(
            packed.if_values, raw.if_values,
            "if values diverge on {source:?}"
        );
    }
}

#[test]
fn directive_entry_points_agree_with_cpu_oracle_within_contract() {
    for (source, defined) in within_contract_cases() {
        let bytes = source.as_bytes();
        let gpu = run_family(bytes, &defined, Layout::Packed);
        let (cpu_kinds, cpu_values) = cpu_oracle(bytes, &defined);
        assert_eq!(gpu.kinds, cpu_kinds, "kinds diverge on {source:?}");
        assert_eq!(
            gpu.ifdef_values,
            masked(&cpu_kinds, &cpu_values, &[TOK_PP_IFDEF, TOK_PP_IFNDEF]),
            "ifdef values diverge on {source:?}"
        );
        assert_eq!(
            gpu.if_values,
            masked(&cpu_kinds, &cpu_values, &[TOK_PP_IF, TOK_PP_ELIF]),
            "if values diverge on {source:?}"
        );
    }
}

/// Pin what the family actually does on out-of-contract rows. Merging the
/// three scans into one must not silently move any of these.
#[test]
fn out_of_contract_rows_are_pinned() {
    let observed: Vec<(String, String)> = out_of_contract_cases()
        .into_iter()
        .map(|(source, defined)| {
            let run = run_family(source.as_bytes(), &defined, Layout::Packed);
            (
                format!("{source:?}"),
                format!("{:?}/{:?}/{:?}", run.kinds, run.ifdef_values, run.if_values),
            )
        })
        .collect();
    assert_pinned(&observed, OUT_OF_CONTRACT_PINS);
}

// ------------------------------------------------------- fingerprints

fn hex(program: &Program) -> String {
    program
        .fingerprint()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn directive_family_fingerprints() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (tokens, source_len) in [(0u32, 0u32), (1, 1), (3, 16), (64, 4096)] {
        for (name, program) in [
            (
                "gpu_directive_metadata",
                gpu_directive_metadata(tokens, source_len),
            ),
            (
                "gpu_directive_metadata_u8",
                gpu_directive_metadata_u8(tokens, source_len),
            ),
            ("gpu_ifdef_value", gpu_ifdef_value(tokens, source_len)),
            ("gpu_ifdef_value_u8", gpu_ifdef_value_u8(tokens, source_len)),
            ("gpu_if_expression", gpu_if_expression(tokens, source_len)),
            (
                "gpu_if_expression_u8",
                gpu_if_expression_u8(tokens, source_len),
            ),
            ("gpu_define_parse", gpu_define_parse(tokens, source_len)),
            (
                "gpu_define_parse_u8",
                gpu_define_parse_u8(tokens, source_len),
            ),
            ("gpu_undef_parse", gpu_undef_parse(tokens, source_len)),
            ("gpu_undef_parse_u8", gpu_undef_parse_u8(tokens, source_len)),
            ("gpu_include_parse", gpu_include_parse(tokens, source_len)),
            (
                "gpu_include_parse_u8",
                gpu_include_parse_u8(tokens, source_len),
            ),
        ] {
            out.push((format!("{name}({tokens},{source_len})"), hex(&program)));
        }
    }
    out
}

fn expansion_family_fingerprints() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for tokens in [1u32, 8, 64] {
        let named = opt_named_macro_expansion(
            "in_tok_types",
            "in_tok_starts",
            "in_tok_lens",
            "source_words",
            "macro_name_hashes",
            "macro_name_starts",
            "macro_name_lens",
            "macro_name_words",
            "macro_vals",
            "macro_sizes",
            "macro_kinds",
            "macro_param_counts",
            "macro_replacement_params",
            "out_tok_types",
            "out_tok_counts",
            Expr::u32(tokens),
            Expr::u32(tokens * 8),
            tokens * 4,
        );
        out.push((format!("opt_named_macro_expansion({tokens})"), hex(&named)));

        let materialized = opt_named_macro_expansion_materialized(
            "in_tok_types",
            "in_tok_starts",
            "in_tok_lens",
            "source_words",
            "macro_name_hashes",
            "macro_name_starts",
            "macro_name_lens",
            "macro_name_words",
            "macro_vals",
            "macro_sizes",
            "macro_kinds",
            "macro_param_counts",
            "macro_replacement_params",
            "macro_replacement_starts",
            "macro_replacement_lens",
            "macro_replacement_words",
            "runtime_counts",
            "out_tok_types",
            "out_tok_starts",
            "out_tok_lens",
            "out_source_words",
            "out_tok_counts",
            "out_source_counts",
            Expr::u32(tokens),
            Expr::u32(tokens * 8),
            Expr::u32(tokens * 16),
            tokens,
            tokens * 8,
            tokens * 16,
            tokens * 4,
            tokens * 32,
        );
        out.push((
            format!("opt_named_macro_expansion_materialized({tokens})"),
            hex(&materialized),
        ));

        let dynamic = opt_dynamic_macro_expansion(
            "in_tok_types",
            "macro_keys",
            "macro_vals",
            "macro_sizes",
            "out_tok_types",
            "out_tok_counts",
            Expr::u32(tokens),
            tokens * 4,
        );
        out.push((
            format!("opt_dynamic_macro_expansion({tokens})"),
            hex(&dynamic),
        ));
    }
    out
}

fn assert_pinned(actual: &[(String, String)], pinned: &[(&str, &str)]) {
    let rendered: String = actual
        .iter()
        .map(|(name, value)| format!("    ({name:?}, {value:?}),\n"))
        .collect();
    assert_eq!(
        actual.len(),
        pinned.len(),
        "pinned row count changed; observed table:\n{rendered}"
    );
    // Report every moved row, not the first. A merge that shifts one stage
    // shifts it in every consumer, and the set of consumers it reaches is the
    // whole finding.
    let moved: Vec<String> = actual
        .iter()
        .zip(pinned)
        .filter_map(|((name, value), (pin_name, pin_value))| {
            assert_eq!(name, pin_name, "pinned row order changed");
            (value != pin_value).then(|| format!("  {name}: {pin_value} -> {value}"))
        })
        .collect();
    assert!(
        moved.is_empty(),
        "{} of {} pinned rows moved from the pre-merge tree:\n{}\nobserved table:\n{rendered}",
        moved.len(),
        actual.len(),
        moved.join("\n"),
    );
}

/// Pinned from the pre-merge tree at 8cf10543e0, except for the 24 rows of
/// `gpu_directive_metadata*`, `gpu_ifdef_value*` and `gpu_if_expression*`
/// re-pinned when PR-08 resolved the directive-scanner drift. Those three
/// kernels carried their own spelling of a scan that `gpu_define_parse`,
/// `gpu_undef_parse` and `gpu_include_parse` already took from the shared
/// helper, and the helper's spelling won. The behavior tests above stayed green
/// across that move, and the define/undef/include rows never budged, which is
/// what distinguishes a reshape from a semantic change.
///
/// The 8 `gpu_if_expression*` rows moved a second time when the `#if`
/// evaluator's inline integer-literal scan was routed onto
/// `c_int_literal_grammar`. That move is a semantic fix, not a reshape: the
/// inline copy accumulated digits with wrapping `u32` arithmetic and consumed a
/// type suffix after a radix prefix that had no digits, where both the standalone
/// scanner and the CPU `consume_integer` saturate and reject. The behavior tests
/// above stayed green, and no other kernel's rows budged.
///
/// A merge that only rehomes code leaves every one of these untouched.
const DIRECTIVE_FAMILY_PINS: &[(&str, &str)] = &[
    (
        "gpu_directive_metadata(0,0)",
        "c11dd4784353b2f5bb86cccb9b878f7714e488ee7f6d4f1f3ba9144bc556fcfb",
    ),
    (
        "gpu_directive_metadata_u8(0,0)",
        "c76fe2279df7eaf499be620c032fa119ffb9e362b96ca6f4fc76aee8a1db3c04",
    ),
    (
        "gpu_ifdef_value(0,0)",
        "d86aa2c50c54095045cd7b8bec3b37019eb7165fa5bea2d80cb02e69d9a0f190",
    ),
    (
        "gpu_ifdef_value_u8(0,0)",
        "9e79a0c4d35e279f403a48ea8981ab33b08da820f3b939d591c5a4a43e8a54ab",
    ),
    (
        "gpu_if_expression(0,0)",
        "30c1f93866e8454b2014f0ab966e41dec5f8569b7a1b7b237a53faa9a1b4a080",
    ),
    (
        "gpu_if_expression_u8(0,0)",
        "b2efc06f7dcf0f4559391d937638a9efa05e48a0511bf0cc7e6a44d294ef74bd",
    ),
    (
        "gpu_define_parse(0,0)",
        "1e9a9b2c7e17e694932bc68549f8f5f2c5bbaa967d1ee2185951bb8166397413",
    ),
    (
        "gpu_define_parse_u8(0,0)",
        "47969df8f1e1f572fef70ffb0cc0462b451f46ada7f4128cf50223821c57d315",
    ),
    (
        "gpu_undef_parse(0,0)",
        "de4e1a78239a2ac1a6131c6e830fef1fe1d554ed8787f620065acaf1b28a7b59",
    ),
    (
        "gpu_undef_parse_u8(0,0)",
        "f381b89a3dd99a84d825778e64acb26969975c85c538f75111eafd08a430e13d",
    ),
    (
        "gpu_include_parse(0,0)",
        "d9a71b7ea6886fa79509c594d84f4490ea2f5c683e630837bdb4e67c3882ae85",
    ),
    (
        "gpu_include_parse_u8(0,0)",
        "5ded0e9b2fd01a5a111f58634ee9845ce9378054d65660c2fa551596ab48f54d",
    ),
    (
        "gpu_directive_metadata(1,1)",
        "c11dd4784353b2f5bb86cccb9b878f7714e488ee7f6d4f1f3ba9144bc556fcfb",
    ),
    (
        "gpu_directive_metadata_u8(1,1)",
        "c76fe2279df7eaf499be620c032fa119ffb9e362b96ca6f4fc76aee8a1db3c04",
    ),
    (
        "gpu_ifdef_value(1,1)",
        "c93336417a926d1add11b9e99b2b4dab596b69fd85a3d421c2f15a0056d7793d",
    ),
    (
        "gpu_ifdef_value_u8(1,1)",
        "72862072cdc9e63c56e3948bc21fbf6eecbc58530170db24939f277c249cbf3f",
    ),
    (
        "gpu_if_expression(1,1)",
        "3cd75d7769bf1efa3d1d3e343d09fc14285464850672c98225a0748498742003",
    ),
    (
        "gpu_if_expression_u8(1,1)",
        "0d4869eb74d5e6c2300a87da397875f30faa3ba080b1e27840dc44799caafc9a",
    ),
    (
        "gpu_define_parse(1,1)",
        "1e9a9b2c7e17e694932bc68549f8f5f2c5bbaa967d1ee2185951bb8166397413",
    ),
    (
        "gpu_define_parse_u8(1,1)",
        "47969df8f1e1f572fef70ffb0cc0462b451f46ada7f4128cf50223821c57d315",
    ),
    (
        "gpu_undef_parse(1,1)",
        "de4e1a78239a2ac1a6131c6e830fef1fe1d554ed8787f620065acaf1b28a7b59",
    ),
    (
        "gpu_undef_parse_u8(1,1)",
        "f381b89a3dd99a84d825778e64acb26969975c85c538f75111eafd08a430e13d",
    ),
    (
        "gpu_include_parse(1,1)",
        "d9a71b7ea6886fa79509c594d84f4490ea2f5c683e630837bdb4e67c3882ae85",
    ),
    (
        "gpu_include_parse_u8(1,1)",
        "5ded0e9b2fd01a5a111f58634ee9845ce9378054d65660c2fa551596ab48f54d",
    ),
    (
        "gpu_directive_metadata(3,16)",
        "772d8e3b34eea5fe1e9bf813e55c3f691ffa20da694abdc2d41fcdf73aa92119",
    ),
    (
        "gpu_directive_metadata_u8(3,16)",
        "a0e0f44e7e0ec55609abcb32bc57846768899a6c1ea103fd0bc0e081e82303d8",
    ),
    (
        "gpu_ifdef_value(3,16)",
        "c987f702a6c005f2c37e1695ef3127926030c27c3574d250ff5325bfdcb150c5",
    ),
    (
        "gpu_ifdef_value_u8(3,16)",
        "0d2ecda362d77f08c330d226cf3ee74e2e01d25416211fd0f89c921b6964987f",
    ),
    (
        "gpu_if_expression(3,16)",
        "4f161dc21a0f6d0b4194cc0591952e28536ebf5c9e7f45d8649abe017f7c074d",
    ),
    (
        "gpu_if_expression_u8(3,16)",
        "87234d3f1c35ade7b589041d76a06c1d303ad2b1dcb68094c09e2ae9c3943f5f",
    ),
    (
        "gpu_define_parse(3,16)",
        "e8554936dcfa7a5d6196858a0f67cdb1f2b98fac9f78961ebf8569b4cebca78c",
    ),
    (
        "gpu_define_parse_u8(3,16)",
        "671d0f21f97e6f75d516330616f01562083d15d34b83b0bc060c00621d9b1623",
    ),
    (
        "gpu_undef_parse(3,16)",
        "75f93ae7a0721e7e1f9bac15021e9b7d52817b92f264c4d085f401288aa5818e",
    ),
    (
        "gpu_undef_parse_u8(3,16)",
        "d4fbbb36aaf256e1f8306d835135e622e64ad719a5381a571259100b05dcda53",
    ),
    (
        "gpu_include_parse(3,16)",
        "fd333af13f6e4f2bda0eaa786fafd9c21d10ee76ff22516be7d91201c28c2626",
    ),
    (
        "gpu_include_parse_u8(3,16)",
        "247e179eb69bb215f717cbbdbdb9963a35d52de30da40236b60e0d83a9115d4e",
    ),
    (
        "gpu_directive_metadata(64,4096)",
        "9a3becb9e8e074cd24f82e07daf2bb0fadb50554485a071e1aa71f22dae55e17",
    ),
    (
        "gpu_directive_metadata_u8(64,4096)",
        "11b744c079652e64fe375bb62aeccc31e865fcfaa30f7d4694853bf906485843",
    ),
    (
        "gpu_ifdef_value(64,4096)",
        "484252505e8a635fec505c277a4cabc8b333f0f1a5e3fd571ce72170f84b5d38",
    ),
    (
        "gpu_ifdef_value_u8(64,4096)",
        "05f6ed003012cbc3c532b8993044b9563101a591d88d1738afd485a9e3f5354a",
    ),
    (
        "gpu_if_expression(64,4096)",
        "2929f7f857d4ba0e38c7e7ce20cde67b39413f7d3af3e23f99b06824a035813e",
    ),
    (
        "gpu_if_expression_u8(64,4096)",
        "8f48d832be74b94c2a2fa519d8b1d70c595b588013434ad085e38b82a9c538b6",
    ),
    (
        "gpu_define_parse(64,4096)",
        "6d245c5de6c62da2c6c9564e271c3260ea7199d4e2f717923409cf39576f6e67",
    ),
    (
        "gpu_define_parse_u8(64,4096)",
        "a1ef258950664e677f2ceef60b0ff0d776ac50c2381db324f3e2516c1d0003a2",
    ),
    (
        "gpu_undef_parse(64,4096)",
        "8ea3107048e8e51be8ffe53e7e51d23dbdf285ef6044973d36bbda23b627e58c",
    ),
    (
        "gpu_undef_parse_u8(64,4096)",
        "a41adc9cb6ee6c2a3f01c2e75c29c2a1a54caf67f2c1e1162fbab4bf7af7350d",
    ),
    (
        "gpu_include_parse(64,4096)",
        "df1624f421b1332e7d897596094c8b65a9291bc6ce70c179a2e5809898a4e5a0",
    ),
    (
        "gpu_include_parse_u8(64,4096)",
        "a58ad9768c847266d40ca275050c85fdf18c5533a7a3af468470ab473db95a92",
    ),
];

/// Pinned from the pre-merge tree at 8cf10543e0.
const EXPANSION_FAMILY_PINS: &[(&str, &str)] = &[
    (
        "opt_named_macro_expansion(1)",
        "6d3044945b37acb29c68753ae6317bbd137553250b3bb51ea28f87e37b86007f",
    ),
    (
        "opt_named_macro_expansion_materialized(1)",
        "7013c9a6bdb444ea5c1430b876faaa17d04848a130b3c8d15001301a380aef58",
    ),
    (
        "opt_dynamic_macro_expansion(1)",
        "d347fa4d1270d0065141c336226b8b29e81f2b8141783680beddc0747e24106b",
    ),
    (
        "opt_named_macro_expansion(8)",
        "69b6044eb228d4d23477654f93ac6e5490c108dfffefe608b06ed3822898123a",
    ),
    (
        "opt_named_macro_expansion_materialized(8)",
        "d95bc85499bcdfd51a5493f04b298c400cdc692b62fb23bc2f3e7e6dcc881b54",
    ),
    (
        "opt_dynamic_macro_expansion(8)",
        "4fb1f4adff1e2ab172a55383baa9fe1c5ac64343a76236e580834d094ac0eb42",
    ),
    (
        "opt_named_macro_expansion(64)",
        "70be92365700558f450620725b02df1a809e7529ce654bef248d2b6fa36bd191",
    ),
    (
        "opt_named_macro_expansion_materialized(64)",
        "ba0304c5f8655a266366d8cd5993fa6398c0f4ed6296c919f83afa740ba3dce3",
    ),
    (
        "opt_dynamic_macro_expansion(64)",
        "f881d9703c99c685a1e0d9888f39e7405725a8dae8b2fb95ee2a9c7d3762f59a",
    ),
];

/// `kinds/ifdef_values/if_values` per out-of-contract row, pinned from the
/// pre-merge tree at 8cf10543e0. `203` is `TOK_PP_NULL`, `207` is `TOK_PP_IF`,
/// `208` is `TOK_PP_IFDEF`.
const OUT_OF_CONTRACT_PINS: &[(&str, &str)] = &[
    ("\"     #ifdef FOO\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#     ifdef FOO\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#ifdef     FOO\\n\"", "[208, 0]/[0, 0]/[0, 0]"),
    ("\"     #if 1\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#     if 1\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#if     1\\n\"", "[207, 0]/[0, 0]/[1, 0]"),
    ("\"      #ifdef FOO\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#      ifdef FOO\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#ifdef      FOO\\n\"", "[208, 0]/[0, 0]/[0, 0]"),
    ("\"      #if 1\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#      if 1\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#if      1\\n\"", "[207, 0]/[0, 0]/[1, 0]"),
    ("\"       #ifdef FOO\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#       ifdef FOO\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#ifdef       FOO\\n\"", "[208, 0]/[0, 0]/[0, 0]"),
    ("\"       #if 1\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#       if 1\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#if       1\\n\"", "[207, 0]/[0, 0]/[1, 0]"),
    ("\"        #ifdef FOO\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#        ifdef FOO\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#ifdef        FOO\\n\"", "[208, 0]/[0, 0]/[0, 0]"),
    ("\"        #if 1\\n\"", "[0, 0]/[0, 0]/[0, 0]"),
    ("\"#        if 1\\n\"", "[203, 0]/[0, 0]/[0, 0]"),
    ("\"#if        1\\n\"", "[207, 0]/[0, 0]/[1, 0]"),
    ("\"#ifdef \\\\\\n FOO\\n\"", "[208, 0]/[0, 0]/[0, 0]"),
    ("\"#if \\\\\\n 1\\n\"", "[207, 0]/[0, 0]/[0, 0]"),
    ("\"#if\\\\\\n 1\\n\"", "[207, 0]/[0, 0]/[0, 0]"),
];

#[test]
fn directive_family_ir_is_unchanged() {
    assert_pinned(&directive_family_fingerprints(), DIRECTIVE_FAMILY_PINS);
}

#[test]
fn expansion_family_ir_is_unchanged() {
    assert_pinned(&expansion_family_fingerprints(), EXPANSION_FAMILY_PINS);
}
