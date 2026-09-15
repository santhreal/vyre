//! Contracts for `is_test_source_path`, the predicate every source-scanning gate
//! uses to decide whether a file ships.
//!
//! WHY: closes the class "a gate judges test code as production, or excuses
//! production code as test code, because the predicate reads a name instead of
//! what the compiler does". A `tests.rs` module inside `src/` is skipped by
//! name, so the name has to imply `#[cfg(test)]` for every such file in the
//! workspace. Adding one that ships turns this red.
//!
//! Not covered: a `#[cfg(test)] mod` declared inline with a body, which the
//! per-line `cfg_test_lines` mask handles inside the file that declares it.

use std::path::{Path, PathBuf};

use crate::workspace_sources;

use workspace_sources::{sources_under, workspace_member_src_dirs, workspace_root};

use xtask::gates::use_paths::is_test_source_path;

/// Every `tests.rs` file under a workspace member's `src` directory.
fn tests_module_files(root: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = workspace_member_src_dirs(root)
        .iter()
        .flat_map(|dir| sources_under(dir, &["rs"]))
        .filter(|path| path.file_name().is_some_and(|name| name == "tests.rs"))
        .collect();
    files.sort();
    files
}

/// The module file that declares `mod tests;` for a `tests.rs` sibling.
///
/// A module file declares its children, so the declaration is in `mod.rs` next
/// to it, or in the `<name>.rs` that names the directory, or in the crate root.
fn declaring_module(root: &Path, tests_file: &Path) -> Option<PathBuf> {
    let dir = tests_file.parent()?;
    let beside = dir.join("mod.rs");
    if beside.is_file() {
        return Some(beside);
    }
    let named = dir.with_extension("rs");
    if named.is_file() {
        return Some(named);
    }
    for stem in ["lib.rs", "main.rs"] {
        let candidate = dir.join(stem);
        if candidate.is_file() && candidate.starts_with(root) {
            return Some(candidate);
        }
    }
    None
}

/// Whether `module` declares `mod tests;` under a `#[cfg(test)]` attribute.
fn declares_tests_under_cfg_test(module: &Path) -> bool {
    let text = std::fs::read_to_string(module).unwrap_or_default();
    let parsed = match syn::parse_file(&text) {
        Ok(parsed) => parsed,
        Err(_) => return false,
    };
    parsed.items.iter().any(|item| match item {
        syn::Item::Mod(module) if module.ident == "tests" && module.content.is_none() => module
            .attrs
            .iter()
            .any(|attribute| attribute_is_cfg_test(attribute)),
        _ => false,
    })
}

/// Whether an attribute is `#[cfg(test)]`.
fn attribute_is_cfg_test(attribute: &syn::Attribute) -> bool {
    if !attribute.path().is_ident("cfg") {
        return false;
    }
    attribute
        .parse_args::<syn::Meta>()
        .is_ok_and(|meta| meta.path().is_ident("test"))
}

/// Every `src/**/tests.rs` in the workspace is declared under `#[cfg(test)]`.
///
/// The predicate skips these files by name. That is only sound while no such
/// file compiles into a shipped artifact, so the set is derived from the
/// workspace at run time rather than listed here.
#[test]
fn tests_are_declared_under_cfg_test() {
    let root = workspace_root();
    let files = tests_module_files(&root);
    assert!(
        !files.is_empty(),
        "no `src/**/tests.rs` found under {}, so this contract covers nothing",
        root.display()
    );

    let shipped: Vec<String> = files
        .iter()
        .filter(|file| {
            declaring_module(&root, file)
                .is_none_or(|module| !declares_tests_under_cfg_test(&module))
        })
        .map(|file| {
            file.strip_prefix(&root)
                .unwrap_or(file)
                .display()
                .to_string()
        })
        .collect();

    assert!(
        shipped.is_empty(),
        "these `tests.rs` files are not declared `#[cfg(test)] mod tests;`, so \
         `is_test_source_path` excuses shipped code: {shipped:?}. Fix: gate the \
         declaration on `#[cfg(test)]`, or rename the file so it is not read as a \
         test module."
    );
}

/// A `tests.rs` module file is test source.
///
/// This is the case the predicate missed: `tests.rs` is neither a directory
/// component named `tests` nor a `test_`/`_test` stem, so an entire
/// `#[cfg(test)]` module read as production and every lock, unwrap, and panic
/// inside it was reported against shipped code.
#[test]
fn tests_module_file_is_test_source() {
    assert!(is_test_source_path(Path::new(
        "vyre-driver-wgpu/src/pipeline/disk_cache/tests.rs"
    )));
    assert!(is_test_source_path(Path::new("crate/src/tests.rs")));
}

/// A production file whose name merely contains `tests` is shipped source.
///
/// The predicate matches a whole stem, not a substring: widening it to
/// `contains("test")` would excuse `latest.rs`, `contest.rs`, and every
/// `attestation.rs` in the tree.
#[test]
fn production_names_near_tests_are_not_test_source() {
    for path in [
        "crate/src/latest.rs",
        "crate/src/contest.rs",
        "crate/src/attestation.rs",
        "crate/src/test_harness/registry.rs",
    ] {
        assert!(
            !is_test_source_path(Path::new(path)),
            "`{path}` is shipped source and was classified as test source"
        );
    }
}

/// The established cases keep their classification.
#[test]
fn directory_and_stem_cases_remain_test_source() {
    for path in [
        "crate/tests/thing.rs",
        "crate/src/thing/tests/case.rs",
        "crate/src/test_thing.rs",
        "crate/src/thing_test.rs",
    ] {
        assert!(
            is_test_source_path(Path::new(path)),
            "`{path}` is test source and was classified as shipped"
        );
    }
}
