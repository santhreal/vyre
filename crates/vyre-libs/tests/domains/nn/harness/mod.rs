//! Source-reading and enum-declaration helpers the `vyre-libs` `nn` contract
//! tests share.
//!
//! This module is compiled once per including test binary, and no binary uses
//! every helper: `scan_cpu_api_boundary` wants only
//! `assert_no_cpu_named_api_exports`, and
//! `nn_attention_clone_family_ir_invariance`,
//! `attention_layout_launch_domain`, `flash_attention_plan_shared_memory` and
//! `dedup_conv_ast_walk_family_guard` only the source readers. Each
//! unused-in-this-binary helper is live in a sibling binary, so `dead_code`
//! here reports the inclusion shape rather than an item with no caller. An
//! `expect` cannot state that: the lint fires in some binaries and not others,
//! and the fulfilled half would fail `unfulfilled_lint_expectations`.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

/// This crate's directory, resolved from the working directory at run time.
pub(crate) fn crate_dir() -> PathBuf {
    vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
}

pub(crate) fn crate_file(path: &str) -> String {
    fs::read_to_string(crate_dir().join(path)).unwrap_or_else(|error| {
        panic!("failed to read {path}: {error}");
    })
}

/// Variant names of one enum, read from the source that declares it.
///
/// `header` is the text that opens the declaration through its brace, for
/// example `enum IndexMap {`. Struct-like, tuple and unit variants all count.
/// A parser that recognises one of those three shapes lets a variant of
/// another shape join the enum without appearing in a coverage set, which
/// leaves the check passing while the axis it judges has grown.
pub(crate) fn declared_enum_variants(source: &str, header: &str) -> BTreeSet<String> {
    let body = source
        .split_once(header)
        .unwrap_or_else(|| panic!("Fix: the source no longer declares `{header}`"))
        .1
        .split_once("\n}")
        .unwrap_or_else(|| panic!("Fix: the `{header}` declaration is unterminated"))
        .0;
    body.lines()
        .filter_map(|line| {
            let line = line.trim();
            let name = line
                .split(|character: char| !character.is_alphanumeric() && character != '_')
                .next()
                .filter(|name| name.starts_with(char::is_uppercase))?;
            // What follows the name says which of the three shapes this is:
            // `{` a struct form, `(` a tuple form, `,` or nothing a unit form.
            // Anything else is a field, a type or an attribute, not a variant.
            let rest = line[name.len()..].trim_start();
            matches!(rest.chars().next(), None | Some('{' | '(' | ',')).then(|| name.to_string())
        })
        .collect()
}
